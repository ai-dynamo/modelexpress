# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM weight transfer engine for the canonical S3 delta path.

The engine owns vLLM's weight update lifecycle: it opens and closes the
layerwise reload window and keeps the CUDA-graph-captured model intact. The
ModelExpress delta protocol - revision catalog, S3 buckets, XOR application,
digest verification, crash-safe journal - belongs to the receiver it holds as a
client, so the two lifecycles stay separate. vLLM's ``start_weight_update``
(open the window) and the receiver's ``start_weight_update`` (download and XOR a
revision) are different operations that happen to share a name.

Registered with vLLM's own engine registry under ``modelexpress``, so a worker
opts in with ``--weight-transfer-config '{"backend": "modelexpress"}'``.
"""

from __future__ import annotations

import logging
import socket
import time
from dataclasses import asdict, dataclass

import torch

from modelexpress.refit.factory import RolloutBackend, build_delta_receiver
from modelexpress.refit.receiver import ModelExpressWeightReceiver, ReceiverConfig
from modelexpress.refit.timing import RefitTimingRecorder

from ..registration import MX_WEIGHT_TRANSFER_BACKEND

from vllm.config import VllmConfig
from vllm.config.weight_transfer import WeightTransferConfig
from vllm.distributed.weight_transfer.base import (
    WeightTransferEngine,
    WeightTransferInitInfo,
    WeightTransferUpdateInfo,
)

logger = logging.getLogger(__name__)

# The receiver reports its own phase timings; these are their normalized homes.
# Bucket download and XOR application share one thread pool and are not
# separable, so the pool time is reported as wire transfer and transformation is
# marked as combined with it.
_RECEIVER_TIMING_STAGES = {
    "perf/mx_receive_delta_index_download": "control_discovery",
    "perf/mx_receive_pool": "wire_transfer",
    "perf/mx_receive_install_time": "installation",
}
_INAPPLICABLE_TIMING_STAGES = (
    "setup_registration",
    "transfer_planning",
    "receive_sync",
    "rollout_readiness",
)


@dataclass
class MxDeltaInitInfo(WeightTransferInitInfo):
    """Everything the receiver needs to reach the catalog and its S3 buckets."""

    model_id: str
    catalog_endpoint: str
    initial_version: str
    preparation_cache_dir: str
    ready_timeout_seconds: float = 600.0
    s3_endpoint_url: str | None = None


@dataclass
class MxDeltaUpdateInfo(WeightTransferUpdateInfo):
    """The revision to install. vLLM's own update window carries no version."""

    version: str


class MxWeightTransferEngine(WeightTransferEngine[MxDeltaInitInfo, MxDeltaUpdateInfo]):
    """vLLM weight transfer engine driving a ModelExpress delta receiver."""

    init_info_cls = MxDeltaInitInfo
    update_info_cls = MxDeltaUpdateInfo

    # A receiver is bound to one catalog model id and one local checkpoint, so
    # it cannot serve a draft model; a draft would need its own receiver.
    supports_draft_weight_update = False

    def __init__(
        self,
        config: WeightTransferConfig,
        vllm_config: VllmConfig,
        device: torch.device,
        model: torch.nn.Module,
    ) -> None:
        super().__init__(config, vllm_config, device, model)
        self.receiver: ModelExpressWeightReceiver | None = None
        self._bare_tensors: dict = {}
        self._timing: RefitTimingRecorder | None = None

    def init_transfer_engine(self, init_info: MxDeltaInitInfo) -> None:
        rank = self.parallel_config.rank
        self.receiver = build_delta_receiver(
            RolloutBackend.VLLM,
            config=ReceiverConfig(**asdict(init_info)),
            receiver_id=f"{socket.gethostname()}:{rank}",
            engine=self,
        )
        self.receiver.initialize()
        logger.info(
            "[delta] receiver %s ready on revision %s",
            self.receiver.receiver_id,
            self.receiver.installed_version,
        )

    def start_weight_update(self) -> None:
        from vllm.config import set_current_vllm_config
        from vllm.model_executor.model_loader.reload import initialize_layerwise_reload

        receiver = self._require_receiver()
        self._timing = RefitTimingRecorder(
            backend=MX_WEIGHT_TRANSFER_BACKEND,
            version=receiver.installed_version or "unknown",
            rank=self.parallel_config.rank,
            tp_size=self.parallel_config.tensor_parallel_size,
        )
        for stage in _INAPPLICABLE_TIMING_STAGES:
            self._timing.mark_not_applicable(stage, reason="not in the delta path")
        self._timing.mark_not_applicable(
            "transformation", combined_with="wire_transfer"
        )
        self._bare_tensors = {}
        for module in self.model.modules():
            attrs = {
                name: tensor
                for name, tensor in module.__dict__.items()
                if isinstance(tensor, torch.Tensor)
            }
            if attrs:
                self._bare_tensors[module] = attrs
        with set_current_vllm_config(self.vllm_config):
            initialize_layerwise_reload(self.model)

    def receive_weights(self, update_info: MxDeltaUpdateInfo) -> None:
        receiver = self._require_receiver()
        if self._timing is not None:
            self._timing.version = update_info.version
        try:
            receiver.start_weight_update(update_info.version)
        finally:
            self._drain_receiver_timings(receiver)
        try:
            result = receiver.update_weights(defer_verification=True)
        finally:
            self._drain_receiver_timings(receiver)
        if not result.success:
            # vLLM's lifecycle signals by raising. The POISONED versus FAILED
            # distinction stays queryable through receiver.status().
            raise RuntimeError(
                f"ModelExpress delta install failed ({result.state.value}): "
                f"{result.detail}"
            )
        logger.info("[delta] worker loaded revision %s", update_info.version)

    def finish_weight_update(self) -> None:
        from vllm.config import set_current_vllm_config
        from vllm.model_executor.model_loader.reload import finalize_layerwise_reload

        receiver = self._require_receiver()
        started = time.perf_counter()
        try:
            with set_current_vllm_config(self.vllm_config):
                finalize_layerwise_reload(self.model, self.model_config)
            for module, attrs in self._bare_tensors.items():
                for name, boot_tensor in attrs.items():
                    current = module.__dict__.get(name)
                    if isinstance(current, torch.Tensor) and current is not boot_tensor:
                        if (
                            current.shape == boot_tensor.shape
                            and current.dtype == boot_tensor.dtype
                        ):
                            boot_tensor.data.copy_(current)
                        else:
                            logger.error(
                                "[refit] bare-attr %s.%s changed shape/dtype across "
                                "refit (cur %s/%s vs boot %s/%s); skipping copy, "
                                "re-attaching STALE boot tensor",
                                type(module).__name__,
                                name,
                                tuple(current.shape),
                                current.dtype,
                                tuple(boot_tensor.shape),
                                boot_tensor.dtype,
                            )
                    setattr(module, name, boot_tensor)

            for _name, module in self.model.named_modules():
                if not (
                    hasattr(module, "W_UV") or hasattr(module, "W_UK_T")
                ) or not hasattr(module, "kv_b_proj"):
                    continue
                out_dtype = (
                    module.W_UV.dtype
                    if hasattr(module, "W_UV")
                    else module.W_UK_T.dtype
                )
                kv_b_proj_weight = module.kv_b_proj.weight.view(
                    module.num_heads,
                    module.qk_nope_head_dim + module.v_head_dim,
                    -1,
                )
                w_uk, w_uv = kv_b_proj_weight.split(
                    [module.qk_nope_head_dim, module.v_head_dim], dim=1
                )
                if hasattr(module, "W_UV"):
                    module.W_UV.copy_(w_uv.transpose(0, 1).to(out_dtype))
                if hasattr(module, "W_UK_T"):
                    module.W_UK_T.copy_(w_uk.permute(1, 2, 0).to(out_dtype))

            meta = [
                name
                for name, parameter in self.model.named_parameters()
                if parameter.device.type == "meta"
            ]
            if meta:
                logger.error(
                    "[refit] POST-COMMIT META PARAMS (graph-hang risk): %d %s",
                    len(meta),
                    meta[:10],
                )
            result = receiver.mark_verified()
            logger.info(
                "[delta] worker now serving revision %s", result.installed_version
            )
        except Exception as error:
            receiver.mark_poisoned(f"vLLM finalization failed: {error}")
            raise
        finally:
            self._bare_tensors = {}
            if self._timing is not None:
                self._timing.add_duration("post_install", time.perf_counter() - started)
                self._timing.emit(logger)
                self._timing = None

    def shutdown(self) -> None:
        self.receiver = None

    @staticmethod
    def trainer_send_weights(iterator, trainer_args) -> None:
        del iterator, trainer_args
        raise NotImplementedError(
            "ModelExpress canonical S3 revisions are receiver-pulled"
        )

    def _require_receiver(self) -> ModelExpressWeightReceiver:
        if self.receiver is None:
            raise RuntimeError(
                "ModelExpress weight transfer used before init_transfer_engine()"
            )
        return self.receiver

    def _drain_receiver_timings(self, receiver: ModelExpressWeightReceiver) -> None:
        metrics = receiver.pop_metrics()
        if self._timing is None:
            return
        for key, seconds in metrics.items():
            stage = _RECEIVER_TIMING_STAGES.get(key)
            if stage is not None:
                self._timing.add_duration(stage, max(0.0, seconds))
        pool = metrics.get("perf/mx_receive_pool")
        if pool is not None:
            # A revision another rank on this node already prepared costs no
            # wire time, which is the difference between a cold and warm refit.
            self._timing.set_cold(pool > 0)
