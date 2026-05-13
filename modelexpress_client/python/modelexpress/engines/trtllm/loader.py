# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT-LLM checkpoint loader backed by ModelExpress strategies."""

from __future__ import annotations

import logging
import os
import sys
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional, Union

from ...load_strategy import LoadResult, LoadStrategyChain
from ...metadata.client_factory import resolve_metadata_server_url

logger = logging.getLogger("modelexpress.engines.trtllm.loader")


class _TrtllmStyleFormatter(logging.Formatter):
    _LEVEL_TAGS = {
        logging.DEBUG: "D",
        logging.INFO: "I",
        logging.WARNING: "W",
        logging.ERROR: "E",
        logging.CRITICAL: "C",
    }

    def format(self, record: logging.LogRecord) -> str:
        record.mx_level_tag = self._LEVEL_TAGS.get(record.levelno, record.levelname[0])
        return super().format(record)


@contextmanager
def _rank_log_scope(global_rank: int):
    """Mirror TRT-LLM worker logs to stderr and a line-buffered rank log."""
    log_dir = os.environ.get("MX_TRANSFER_LOG_DIR", "/tmp/mx_logs")
    os.makedirs(log_dir, exist_ok=True)
    rank_log = os.path.abspath(os.path.join(log_dir, f"rank{global_rank}.log"))

    mx_logger = logging.getLogger("modelexpress")
    mx_logger.setLevel(logging.INFO)
    for handler in list(mx_logger.handlers):
        if getattr(handler, "_mx_rank_log_path", None) == rank_log or getattr(
            handler, "baseFilename", None
        ) == rank_log:
            mx_logger.removeHandler(handler)
            handler.close()

    formatter = _TrtllmStyleFormatter(
        "[%(asctime)s] [ModelExpress] [%(mx_level_tag)s] %(message)s",
        datefmt="%m/%d/%Y-%H:%M:%S",
    )
    stream = open(rank_log, "a", buffering=1)
    file_handler = logging.StreamHandler(stream)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    file_handler._mx_rank_log_path = rank_log
    mx_logger.addHandler(file_handler)

    # TRT-LLM MPI workers do not reliably inherit a root handler that forwards
    # ModelExpress logs to the container log stream. Keep the durable rank log,
    # but also mirror transfer events to stderr so `kubectl logs` shows state.
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.INFO)
    stderr_handler.setFormatter(formatter)
    stderr_handler._mx_trtllm_stderr = True
    mx_logger.addHandler(stderr_handler)
    try:
        yield
    finally:
        try:
            try:
                file_handler.flush()
                stream.flush()
                os.fsync(stream.fileno())
                stderr_handler.flush()
            except Exception:
                pass
        finally:
            mx_logger.removeHandler(file_handler)
            mx_logger.removeHandler(stderr_handler)
            stream.close()


def _resolve_mx_model_name(
    model_name_arg: Optional[str],
    checkpoint_dir: Optional[str],
) -> str:
    """Resolve the model identity used for TRT-LLM ModelExpress source matching."""
    if model_name_arg:
        return str(model_name_arg)

    env_model_name = os.environ.get("MODEL_NAME")
    if env_model_name:
        return env_model_name

    if checkpoint_dir:
        path = os.path.normpath(str(checkpoint_dir))
        parts = path.split(os.sep)
        if (
            len(parts) >= 3
            and parts[-2] == "snapshots"
            and parts[-3].startswith("models--")
        ):
            return parts[-3].removeprefix("models--").replace("--", "/")
        return os.path.basename(path)

    return "unknown"


try:
    from tensorrt_llm._torch.models.checkpoints.base_config_loader import (
        BaseConfigLoader,
    )
    from tensorrt_llm._torch.models.checkpoints.base_weight_loader import (
        BaseWeightLoader,
        ConsumableWeightsDict,
    )
    from tensorrt_llm._torch.models.checkpoints.base_weight_mapper import (
        BaseWeightMapper,
    )
    from tensorrt_llm._torch.models.checkpoints.auto_mapper import (
        AutoCheckpointMapper,
    )
    from tensorrt_llm._torch.models.checkpoints.hf.checkpoint_loader import (
        HfCheckpointLoader,
    )
    from tensorrt_llm.mapping import Mapping
except Exception as exc:
    _TRTLLM_IMPORT_ERROR = exc

    class MXCheckpointLoader:
        """Placeholder used when TensorRT-LLM is not installed."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "modelexpress.engines.trtllm.loader.MXCheckpointLoader "
                "requires TensorRT-LLM to be installed."
            ) from _TRTLLM_IMPORT_ERROR

else:

    class MXCheckpointLoader(HfCheckpointLoader):
        """TRT-LLM checkpoint loader backed by ModelExpress load strategies."""

        def __init__(
            self,
            *,
            weight_loader: Optional[BaseWeightLoader] = None,
            weight_mapper: Optional[BaseWeightMapper] = None,
            config_loader: Optional[BaseConfigLoader] = None,
            mx_server_url: Optional[str] = None,
            model_name: Optional[Union[str, Path]] = None,
            query_timeout_s: Optional[int] = None,
        ):
            super().__init__(
                weight_loader=weight_loader,
                weight_mapper=weight_mapper,
                config_loader=config_loader,
            )
            self._checkpoint_format = "modelexpress"
            self._mx_server_url = mx_server_url
            self._model_name = str(model_name) if model_name is not None else None
            self._query_timeout_s = query_timeout_s
            self._p2p_succeeded = False
            self._last_load_ctx = None

        @property
        def checkpoint_format(self) -> str:
            return "modelexpress"

        @property
        def mx_server_url(self) -> Optional[str]:
            return self._mx_server_url

        @property
        def model_name(self) -> Optional[str]:
            return self._model_name

        @property
        def query_timeout_s(self) -> Optional[int]:
            return self._query_timeout_s

        @property
        def p2p_succeeded(self) -> bool:
            return self._p2p_succeeded

        def is_weights_preloaded(self) -> bool:
            return self._p2p_succeeded

        def load_weights(
            self,
            checkpoint_dir: str,
            mapping: Mapping,
            *,
            model=None,
            **kwargs,
        ) -> dict[str, Any]:
            """Load TRT-LLM weights through ModelExpress, falling back to native disk.

            `mapping` matches TRT-LLM's BaseCheckpointLoader contract. `model`
            is a ModelExpress-only extension passed by TRT-LLM's model loader
            so P2P can write directly into live parameter buffers. Keep `model`
            explicit instead of inside `kwargs` so native fallback loaders
            never receive an argument they do not understand.
            """
            fallback_loader = lambda: self._load_from_disk(
                checkpoint_dir,
                mapping,
                **kwargs,
            )

            if model is None:
                logger.info(
                    "TRT-LLM ModelExpress loader did not receive a model reference; "
                    "using native checkpoint loading."
                )
                return fallback_loader()

            self._p2p_succeeded = False
            self._last_load_ctx = None
            try:
                ctx = self._build_load_context(
                    model=model,
                    checkpoint_dir=checkpoint_dir,
                    mapping=mapping,
                    native_loader=fallback_loader,
                    source_query_timeout_s=self._query_timeout_s,
                )
                self._last_load_ctx = ctx
                with _rank_log_scope(ctx.global_rank):
                    result = LoadStrategyChain.run(model, ctx)
            except Exception:
                logger.warning(
                    "ModelExpress strategy chain failed; falling back to native checkpoint "
                    "loading.\n%s",
                    traceback.format_exc(),
                )
                return fallback_loader()

            selected_strategy = getattr(ctx, "selected_strategy", None)
            if selected_strategy != "rdma":
                if result is None:
                    result = {}
                if not isinstance(result, (dict, ConsumableWeightsDict)):
                    raise TypeError(
                        "TRT-LLM native fallback must return a weight dict "
                        "or ConsumableWeightsDict "
                        f"or None, got {type(result).__name__}"
                    )
                if result:
                    fallback_bytes = sum(
                        tensor.numel() * tensor.element_size()
                        for tensor in result.values()
                    )
                    logger.info(
                        "ModelExpress default strategy loaded %d fallback weights "
                        "(%.2f MiB) through TRT-LLM native checkpoint loading.",
                        len(result),
                        fallback_bytes / (1 << 20),
                    )
                return result

            self._p2p_succeeded = True
            logger.info(
                "ModelExpress P2P weight transfer succeeded from %s",
                self._mx_server_url,
            )
            return {}

        def get_initialized_weight_mapper(self, model, config):
            """Use TRT-LLM's HF mapper for ModelExpress disk fallback weights.

            ModelExpress is a transport format here. When the strategy chain
            falls back to disk, the checkpoint tensors are still HF-formatted,
            so asking TRT-LLM for a transport-specific mapper can miss
            architecture-specific mappers in released TRT-LLM images.
            """
            if self.weight_mapper is not None:
                self.weight_mapper.init_model_and_config(model, config)
                return self.weight_mapper

            if config.pretrained_config and config.pretrained_config.architectures:
                model_arch = config.pretrained_config.architectures[0]
            else:
                raise ValueError("Cannot determine model architecture from config")
            weight_mapper = AutoCheckpointMapper.get("HF", model_arch)
            weight_mapper.init_model_and_config(model, config)
            self.weight_mapper = weight_mapper
            return weight_mapper

        def _load_from_disk(
            self,
            checkpoint_dir: str,
            mapping: Mapping,
            **kwargs,
        ) -> dict[str, Any]:
            return super().load_weights(checkpoint_dir, mapping=mapping, **kwargs)

        def publish_as_source(
            self,
            model,
            checkpoint_dir: Optional[str] = None,
        ) -> None:
            self._publish_as_source(model, checkpoint_dir=checkpoint_dir)

        def _build_load_context(
            self,
            *,
            model,
            checkpoint_dir: Optional[str],
            mapping: Optional[Mapping] = None,
            native_loader=None,
            source_query_timeout_s: Optional[int] = None,
        ):
            from .adapter import build_trtllm_load_context

            server_url = resolve_metadata_server_url(self._mx_server_url)
            resolved_name = _resolve_mx_model_name(self._model_name, checkpoint_dir)
            return build_trtllm_load_context(
                model_name=resolved_name,
                checkpoint_dir=checkpoint_dir,
                model=model,
                mapping=mapping,
                server_url=server_url,
                native_loader=native_loader,
                source_query_timeout_s=source_query_timeout_s,
            )

        def _publish_as_source(
            self,
            model,
            *,
            checkpoint_dir: Optional[str] = None,
            mapping: Optional[Mapping] = None,
        ) -> None:
            self._publish_current_model(model)

        def _publish_current_model(self, model) -> None:
            ctx = self._last_load_ctx
            if ctx is None or getattr(ctx.adapter, "model", None) is not model:
                logger.warning(
                    "Skipping ModelExpress source publish because TRT-LLM did not provide "
                    "a matching load context for this model."
                )
                return
            try:
                from ...load_strategy import publish_loaded_model

                with _rank_log_scope(ctx.global_rank):
                    publish_loaded_model(LoadResult(value=model, model=model), ctx)
            except Exception:
                logger.warning(
                    "Failed to publish weights to ModelExpress server at %s.\n%s",
                    self._mx_server_url,
                    traceback.format_exc(),
                )

        def post_load_publish(
            self,
            model,
            *,
            checkpoint_dir: str,
            weights_preloaded: bool = False,
        ) -> None:
            """Publish only after TRT-LLM has finished its post-load path.

            load_weights() may return native fallback weights for TRT-LLM to
            merge through its own model.load_weights() path. Publishing inside
            the strategy chain would advertise stale tensors before TRT-LLM
            post-processing. RDMA receivers also publish here: after P2P writes
            weights into live buffers and TRT-LLM completes post-load setup,
            the receiver can safely become another source.
            """
            # TRT-LLM applies fallback weights after load_weights() returns.
            # Reusing the load context keeps disk fallback and RDMA receiver
            # publication under the same worker_id used by the load attempt.
            self._publish_current_model(model)


__all__ = [
    "MXCheckpointLoader",
]
