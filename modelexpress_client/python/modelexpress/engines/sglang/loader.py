# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ModelExpress loader entrypoint for SGLang's remote_instance backend."""

from __future__ import annotations

import logging
import time
from importlib.metadata import PackageNotFoundError, version as pkg_version
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from ... import envs, p2p_pb2
from ...load_strategy import LoadContext, run_load_strategy_chain
from ...load_strategy.base import clear_exception_tracebacks
from ...load_strategy.context import LoadResult
from ...load_strategy.draft_gate import DraftPublicationGate
from ...metadata.publisher import PublisherThread
from ...metadata.payload import tensor_source_metadata, worker_tensor_descriptors
from ...metadata.publish import _heartbeat_threads
from ...metrics import enable_metrics
from ...nixl_transfer import NixlTransferManager
from ...metrics import metrics as selection_metrics
from ...source_selection import configured_policy_label, get_configured_selector
from .adapter import (
    _get_model_name,
    build_sglang_load_context,
    is_sglang_draft_model,
)
from .draft_weights import draft_tensor_namespace, draft_weight_adapter_for
from .artifacts import (
    _sglang_health_ready,
    install_sglang_cache_artifacts,
    schedule_sglang_cache_artifact_publish,
)

logger = logging.getLogger("modelexpress.engines.sglang.loader")

if TYPE_CHECKING:
    from sglang.srt.configs.device_config import DeviceConfig
    from sglang.srt.configs.load_config import LoadConfig
    from sglang.srt.configs.model_config import ModelConfig


_tensor_registry: dict[int, dict[str, torch.Tensor]] = {}
_nixl_managers: dict[int, NixlTransferManager] = {}
_loader_registry: dict[int, MxModelLoader] = {}
# Main-pass publication gates awaiting this device's speculative draft pass.
_draft_publication_gates: dict[int, DraftPublicationGate] = {}


def _sglang_version() -> str:
    try:
        return pkg_version("sglang")
    except PackageNotFoundError:
        return ""


def _speculative_server_args():
    """Return SGLang's speculative settings for this process, if available."""
    try:
        from sglang.srt.runtime_context import get_spec

        return get_spec()
    except Exception:
        pass
    try:
        from sglang.srt.server_args import get_global_server_args

        return get_global_server_args()
    except Exception:
        return None


def _expects_draft_pass() -> bool:
    """True when SGLang will run a draft ModelRunner load through this loader.

    NGRAM speculation has no draft model, and a draft with its own
    ``speculative_draft_load_format`` never reaches the remote_instance
    backend.
    """
    spec = _speculative_server_args()
    algorithm = getattr(spec, "speculative_algorithm", None)
    algorithm = str(getattr(algorithm, "name", algorithm) or "").upper()
    if algorithm in ("", "NONE", "NGRAM"):
        return False
    draft_format = getattr(spec, "speculative_draft_load_format", None)
    draft_format = getattr(draft_format, "value", draft_format)
    return draft_format is None or str(draft_format) == "remote_instance"


class MxModelLoader:
    """Unified ModelExpress loader for SGLang.

    SGLang instantiates this class from its ``remote_instance`` loader when
    ``remote-instance-weight-loader-backend=modelexpress``. The class receives
    the already-initialized SGLang model and delegates loading policy to the
    shared ModelExpress strategy chain.
    """

    def __init__(self, load_config: LoadConfig):
        self.load_config = load_config
        # Off the load path, but still before any transport or strategy choice:
        # a run that records nothing must leave the exporter up, so a scrape can
        # prove it came up. No-op unless enabled; never raises.
        enable_metrics()
        self._ctx: LoadContext | None = None
        self._strict_draft_publication = False
        self._draft_weight_adapter = None

    def load_model(
        self,
        *,
        model: nn.Module,
        model_config: ModelConfig,
        device_config: DeviceConfig,
    ) -> nn.Module:
        """Load model weights through the shared ModelExpress strategy chain."""
        transport = getattr(self.load_config, "modelexpress_transport", "nixl")
        # L0 wraps the dispatcher rather than each transport, so the
        # transfer_engine path -- which never touches the strategy chain -- still
        # reports a load duration. Instrumenting only the chain would leave that
        # whole deployment mode looking like it never loaded anything.
        #
        # SGLang's speculative draft ModelRunner loads through this same path
        # on the same worker; label it separately so the target's load-time
        # percentiles stay meaningful.
        is_draft = is_sglang_draft_model(model_config)
        if is_draft and envs.MX_LOAD_STRATEGY_CHAIN == "RL":
            raise ValueError(
                "RL initial loading does not support speculative draft models"
            )
        model_role = "draft" if is_draft else "main"
        # Same helper the load context uses, so the label matches the model name
        # every other client family reports for this process.
        model_id = _get_model_name(model_config)
        with selection_metrics.time_load("sglang", model_id, model_role):
            if transport == "nixl":
                return self._load_model_via_nixl(
                    model=model,
                    model_config=model_config,
                    device_config=device_config,
                )
            if transport == "transfer_engine":
                return self._load_model_via_transfer_engine(
                    model=model,
                    model_config=model_config,
                    device_config=device_config,
                )
            raise ValueError(
                "SGLang ModelExpress integration currently supports "
                f"modelexpress transports 'nixl' and 'transfer_engine', "
                f"got {transport!r}."
            )

    def _load_model_via_nixl(
        self,
        *,
        model: nn.Module,
        model_config: ModelConfig,
        device_config: DeviceConfig,
    ) -> nn.Module:
        load_start = time.perf_counter()
        ctx = build_sglang_load_context(
            self.load_config,
            model_config,
            device_config,
        )
        is_draft = is_sglang_draft_model(model_config)
        ctx.p2p_role = "draft" if is_draft else "main"
        draft_gate: DraftPublicationGate | None = None
        strict_draft = False
        if is_draft:
            main_loader = _loader_registry.get(ctx.device_id)
            main_ctx = main_loader._ctx if main_loader is not None else None
            draft_adapter = draft_weight_adapter_for(
                type(model).__name__, role="draft"
            )
            same_checkpoint = main_ctx is not None and main_ctx.identity == ctx.identity
            compatible_adapter = (
                same_checkpoint
                and draft_adapter is not None
                and main_loader._draft_weight_adapter is not None
                and draft_adapter.compatibility_tag
                == main_loader._draft_weight_adapter.compatibility_tag
            )
            namespace = (
                draft_tensor_namespace(
                    ctx.identity,
                    draft_adapter,
                    type(model).__name__,
                    _sglang_version(),
                    envs.MX_MODEL_URI or ctx.identity.model_name,
                )
                if compatible_adapter and ctx.identity.pipeline_parallel_size == 1
                else None
            )
            if namespace is not None:
                # Same checkpoint (MTP / NextN): join the main load's
                # publication through its NIXL agent instead of binding
                # MX_METADATA_PORT + device_id a second time.
                ctx.shared_nixl_manager = _nixl_managers.get(ctx.device_id)
                ctx.draft_tensor_namespace = namespace
                strict_draft = main_loader._strict_draft_publication
            else:
                # Unknown layouts, an unpinned checkpoint or a different
                # draft identity fall back to SGLang's storage loader.
                ctx.p2p_enabled = False
            draft_gate = _draft_publication_gates.pop(ctx.device_id, None)
        elif _expects_draft_pass():
            self._draft_weight_adapter = draft_weight_adapter_for(
                type(model).__name__, role="main"
            )
            self._strict_draft_publication = self._draft_weight_adapter is not None
            if self._strict_draft_publication or not envs.MX_ARTIFACT_READY_URL.strip():
                main_gate = DraftPublicationGate(
                    grace_secs=None if self._strict_draft_publication else 600.0
                )
                _draft_publication_gates[ctx.device_id] = main_gate
                if envs.MX_ARTIFACT_READY_URL.strip():
                    ctx.source_ready_fn = (
                        lambda: main_gate.is_open() and _sglang_health_ready(ctx)
                    )
                else:
                    ctx.source_ready_fn = main_gate.is_open
        elif not is_draft:
            self._draft_weight_adapter = draft_weight_adapter_for(
                type(model).__name__, role="main"
            )
        if not is_draft and ctx.source_ready_fn is None and envs.MX_ARTIFACT_READY_URL.strip():
            ctx.source_ready_fn = lambda: _sglang_health_ready(ctx)
        if not is_draft:
            self._ctx = ctx

        logger.info(
            "[Worker %s] SGLang MxModelLoader starting (model=%s, "
            "p2p_enabled=%s, p2p_role=%s)",
            ctx.global_rank,
            ctx.identity.model_name,
            ctx.p2p_enabled,
            ctx.p2p_role,
        )
        try:
            model = self._run_nixl_load(model, ctx)
        finally:
            if draft_gate is not None and (
                not strict_draft or ctx.draft_published
            ):
                draft_gate.release()
            main_gate_pending = _draft_publication_gates.get(ctx.device_id)
            if not is_draft and main_gate_pending is not None:
                main_gate_pending.arm()

        total_time = time.perf_counter() - load_start
        logger.info(
            "[Worker %s] SGLang MxModelLoader.load_model() COMPLETE in %.2fs",
            ctx.global_rank,
            total_time,
        )
        return model.eval()

    def _run_nixl_load(self, model: nn.Module, ctx: LoadContext) -> nn.Module:
        is_main = ctx.p2p_role == "main"
        # No model_init phase here, unlike vLLM: SGLang builds the module and
        # hands it in, so there is no initialization inside this window to time.
        # The phases still partition the load; this load simply has three.
        if is_main:
            with selection_metrics.time_load_phase(
                "sglang", ctx.identity.model_name, "artifact_install"
            ):
                install_sglang_cache_artifacts(ctx)
        with selection_metrics.time_load_phase(
            "sglang", ctx.identity.model_name, "chain"
        ):
            model = run_load_strategy_chain(model, ctx)

        if not is_main:
            if ctx.p2p_enabled and ctx.nixl_manager is not None:
                # Same checkpoint, same SourceIdentity: extend the main load's
                # manifest instead of registering a second source under a
                # colliding mx_source_id.
                _tensor_registry.setdefault(ctx.device_id, {}).update(ctx.tensors)
            return model

        _loader_registry[ctx.device_id] = self
        _tensor_registry[ctx.device_id] = ctx.tensors
        if ctx.nixl_manager is not None:
            _nixl_managers[ctx.device_id] = ctx.nixl_manager
        else:
            _nixl_managers.pop(ctx.device_id, None)

        with selection_metrics.time_load_phase(
            "sglang", ctx.identity.model_name, "publish"
        ):
            schedule_sglang_cache_artifact_publish(ctx)
        return model

    def _load_model_via_transfer_engine(
        self,
        *,
        model: nn.Module,
        model_config: ModelConfig,
        device_config: DeviceConfig,
    ) -> nn.Module:
        """Load SGLang weights via ModelExpress metadata and TransferEngine."""
        load_start = time.perf_counter()
        ctx = build_sglang_load_context(
            self.load_config,
            model_config,
            device_config,
        )
        if is_sglang_draft_model(model_config):
            # TransferEngine publishes a separate source per load, so a draft
            # would advertise its head under the target's SourceIdentity (same
            # checkpoint) and replace the target's heartbeat, and its
            # same-named tensors would match the target's manifest. SGLang's
            # own transfer_engine backend likewise never publishes the draft.
            # Load it natively; the nixl transport serves drafts over P2P.
            logger.info(
                "[Worker %s] Speculative draft on the transfer_engine "
                "transport; loading natively without source publication",
                ctx.global_rank,
            )
            result = ctx.adapter.load_via_native(
                LoadResult(value=model, model=model)
            )
            ctx.tensors = {}
            self.remote_instance_transfer_engine_weight_info = {}
            return result.model.eval()
        self._ctx = ctx

        transfer_engine = getattr(
            self.load_config, "remote_instance_weight_loader_transfer_engine", None
        )
        session_id = getattr(
            self.load_config,
            "remote_instance_weight_loader_transfer_engine_session_id",
            None,
        )
        if transfer_engine is None or not session_id:
            raise RuntimeError(
                "SGLang ModelExpress transfer_engine transport requires an "
                "initialized SGLang TransferEngine and session id."
            )

        logger.info(
            "[Worker %s] SGLang MxModelLoader starting transfer_engine "
            "(model=%s)",
            ctx.global_rank,
            ctx.identity.model_name,
        )

        result = LoadResult(value=model, model=model)
        source_worker = self._find_transfer_engine_source(ctx)
        weight_info = None
        if source_worker is None:
            logger.info(
                "[Worker %s] No TransferEngine source available, loading natively",
                ctx.global_rank,
            )
            result = ctx.adapter.load_via_native(result)
            tensors = ctx.adapter.discover_tensors(result)
        else:
            registered_tensors = None
            try:
                result = ctx.adapter.before_rdma_receive(result)
                tensors = ctx.adapter.discover_tensors(result)
                weight_info = self._register_transfer_engine_tensors(
                    tensors,
                    transfer_engine,
                )
                registered_tensors = tensors
                self._receive_via_transfer_engine(
                    tensors,
                    transfer_engine,
                    source_worker,
                    ctx,
                )
                result = ctx.adapter.after_rdma_receive(result)
            except Exception as exc:
                if registered_tensors is not None:
                    self._unregister_transfer_engine_tensors(
                        registered_tensors,
                        transfer_engine,
                    )
                logger.warning(
                    "[Worker %s] TransferEngine load failed, falling back "
                    "to native load: %s",
                    ctx.global_rank,
                    exc,
                    exc_info=True,
                )
                registered_tensors = None
                tensors = {}
                clear_exception_tracebacks(exc)
                result = ctx.adapter.reinit_for_retry(result)
                result = ctx.adapter.load_via_native(result)
                tensors = ctx.adapter.discover_tensors(result)
                weight_info = None

        if weight_info is None:
            try:
                weight_info = self._register_transfer_engine_tensors(
                    tensors,
                    transfer_engine,
                )
            except Exception as exc:
                logger.warning(
                    "[Worker %s] TransferEngine source registration failed; "
                    "model load will continue without source publication: %s",
                    ctx.global_rank,
                    exc,
                    exc_info=True,
                )
                ctx.tensors = tensors
                self.remote_instance_transfer_engine_weight_info = {}
                return result.model.eval()
        ctx.tensors = tensors
        self.remote_instance_transfer_engine_weight_info = weight_info
        publish_ok = self._publish_transfer_engine_source(
            ctx=ctx,
            session_id=session_id,
            weight_info=weight_info,
        )
        if not publish_ok:
            logger.warning(
                "[Worker %s] TransferEngine source advertisement failed; "
                "model load will continue",
                ctx.global_rank,
            )

        total_time = time.perf_counter() - load_start
        logger.info(
            "[Worker %s] SGLang MxModelLoader transfer_engine COMPLETE in %.2fs",
            ctx.global_rank,
            total_time,
        )
        return result.model.eval()

    def _find_transfer_engine_source(self, ctx: LoadContext):
        # This path bypasses LoadStrategyChain and RdmaStrategy entirely, so it
        # needs its own instrumentation: without it a mooncake/transfer_engine
        # pod exports mx_build_info and nothing else in every state, and a
        # metadata-backend outage is indistinguishable from a cluster that has
        # published no peers -- the exact pair the ListSources counter exists to
        # separate.
        policy = configured_policy_label()
        try:
            response = ctx.mx_client.list_sources(
                identity=ctx.identity,
                status_filter=p2p_pb2.SOURCE_STATUS_READY,
            )
        except Exception as exc:
            logger.warning(
                "[Worker %s] TransferEngine source discovery failed, "
                "falling back to native load: %s",
                ctx.global_rank,
                exc,
            )
            selection_metrics.record_list_sources(policy, "error")
            return None

        if not response.instances:
            selection_metrics.record_list_sources(policy, "empty")
            for stage in ("listed", "rank_matched"):
                selection_metrics.observe_candidates(policy, stage, 0)
            return None

        candidates = [
            inst for inst in response.instances if inst.worker_rank == ctx.worker_rank
        ]
        selection_metrics.record_list_sources(policy, "ok")
        selection_metrics.observe_candidates(policy, "listed", len(response.instances))
        selection_metrics.observe_candidates(policy, "rank_matched", len(candidates))
        # The nixl transport (and vLLM) order sources in the shared
        # RdmaStrategy; this is SGLang's separate transfer_engine path, which
        # discovers sources itself, so it applies the same selector here.
        selector = get_configured_selector()
        candidates = selector.order(candidates, ctx)
        logger.info(
            "[Worker %s] TransferEngine source selection: source_selector=%s "
            "source_candidates_total=%d source_candidates_rank_matched=%d",
            ctx.global_rank,
            selector.name,
            len(response.instances),
            len(candidates),
        )
        for source_ref in candidates:
            metadata = ctx.mx_client.get_metadata(
                mx_source_id=source_ref.mx_source_id,
                worker_id=source_ref.worker_id,
            )
            if not metadata.found:
                continue
            worker = metadata.worker
            if worker.WhichOneof("backend_metadata") == "transfer_engine_session_id":
                return worker
        return None

    @staticmethod
    def _byte_size(numel: int, element_size: int) -> int:
        return numel * element_size

    def _receive_via_transfer_engine(
        self,
        tensors: dict[str, torch.Tensor],
        transfer_engine,
        source_worker,
        ctx: LoadContext,
    ) -> None:
        seed_weight_info = {
            tensor.name: (tensor.addr, tensor.size)
            for tensor in worker_tensor_descriptors(source_worker)
        }

        seed_ptr_list = []
        client_ptr_list = []
        client_len_list = []
        for name, tensor in tensors.items():
            weight_info = seed_weight_info.get(name)
            if weight_info is None:
                raise RuntimeError(
                    f"ModelExpress transfer_engine: missing tensor {name!r} "
                    "in source metadata"
                )
            seed_ptr, seed_size = weight_info
            local_size = self._byte_size(tensor.numel(), tensor.element_size())
            if seed_size != local_size:
                raise RuntimeError(
                    f"ModelExpress transfer_engine: size mismatch for {name}: "
                    f"source={seed_size} bytes, local={local_size} bytes"
                )
            if local_size == 0:
                continue
            seed_ptr_list.append(seed_ptr)
            client_ptr_list.append(tensor.data_ptr())
            client_len_list.append(local_size)

        logger.info(
            "[Worker %s] Receiving %d tensors via TransferEngine",
            ctx.global_rank,
            len(seed_ptr_list),
        )
        ret = transfer_engine.batch_transfer_sync_read(
            source_worker.transfer_engine_session_id,
            client_ptr_list,
            seed_ptr_list,
            client_len_list,
        )
        if ret < 0:
            raise RuntimeError(
                f"ModelExpress transfer_engine: batch_transfer_sync_read failed "
                f"with error={ret}"
            )

    def _register_transfer_engine_tensors(
        self,
        tensors: dict[str, torch.Tensor],
        transfer_engine,
    ) -> dict[str, tuple[int, int, int]]:
        weight_info = {}
        registered_ptrs = set()
        try:
            for name, tensor in tensors.items():
                addr = tensor.data_ptr()
                numel = tensor.numel()
                element_size = tensor.element_size()
                size = self._byte_size(numel, element_size)
                weight_info[name] = (addr, numel, element_size)
                if size == 0:
                    continue
                if addr not in registered_ptrs:
                    ret = transfer_engine.register_memory(addr, size)
                    if ret != 0:
                        raise RuntimeError(
                            "ModelExpress transfer_engine: register_memory failed "
                            f"for tensor {name!r}, error={ret}"
                        )
                    registered_ptrs.add(addr)
        except Exception:
            self._unregister_transfer_engine_ptrs(
                registered_ptrs,
                transfer_engine,
            )
            raise
        return weight_info

    def _unregister_transfer_engine_tensors(
        self,
        tensors: dict[str, torch.Tensor],
        transfer_engine,
    ) -> None:
        registered_ptrs = {
            tensor.data_ptr()
            for tensor in tensors.values()
            if self._byte_size(tensor.numel(), tensor.element_size()) > 0
        }
        self._unregister_transfer_engine_ptrs(registered_ptrs, transfer_engine)

    def _unregister_transfer_engine_ptrs(
        self,
        registered_ptrs: set[int],
        transfer_engine,
    ) -> None:
        for addr in registered_ptrs:
            try:
                ret = transfer_engine.unregister_memory(addr)
                if ret != 0:
                    logger.warning(
                        "ModelExpress transfer_engine: unregister_memory failed "
                        "for address %s, error=%s",
                        addr,
                        ret,
                    )
            except Exception:
                logger.exception(
                    "ModelExpress transfer_engine: unregister_memory raised "
                    "for address %s",
                    addr,
                )

    def _publish_transfer_engine_source(
        self,
        *,
        ctx: LoadContext,
        session_id: str,
        weight_info,
    ) -> bool:
        try:
            tensors = [
                p2p_pb2.TensorDescriptor(
                    name=name,
                    addr=addr,
                    size=self._byte_size(numel, element_size),
                    device_id=ctx.device_id,
                )
                for name, (addr, numel, element_size) in weight_info.items()
            ]
            worker = p2p_pb2.WorkerMetadata(
                worker_rank=ctx.worker_rank,
                transfer_engine_session_id=session_id,
                tensor_source=tensor_source_metadata(tensors),
                accelerator=ctx.accelerator_backend.name,
            )
        except Exception:
            logger.exception(
                "[Worker %s] TransferEngine metadata payload build failed "
                "(worker_id=%s, worker_rank=%s)",
                ctx.global_rank,
                ctx.worker_id,
                ctx.worker_rank,
            )
            return False
        def publish_fn() -> str:
            mx_source_id = ctx.mx_client.publish_metadata(
                ctx.identity,
                worker,
                ctx.worker_id,
            )
            logger.info(
                "[Worker %s] Published TransferEngine metadata to MX server "
                "(mx_source_id=%s, worker_id=%s)",
                ctx.global_rank,
                mx_source_id,
                ctx.worker_id,
            )
            return mx_source_id

        try:
            heartbeat = PublisherThread(
                mx_client=ctx.mx_client,
                worker_id=ctx.worker_id,
                worker_rank=ctx.worker_rank,
                nixl_manager=None,
                publish_fn=publish_fn,
                ready_fn=(
                    (lambda: _sglang_health_ready(ctx))
                    if envs.MX_ARTIFACT_READY_URL.strip()
                    else None
                ),
            )
            heartbeat.start()
            _heartbeat_threads[ctx.worker_rank] = heartbeat
        except Exception:
            logger.exception(
                "[Worker %s] TransferEngine heartbeat startup failed "
                "(worker_id=%s, worker_rank=%s)",
                ctx.global_rank,
                ctx.worker_id,
                ctx.worker_rank,
            )
            return False
        logger.info(
            "[Worker %s] Scheduled TransferEngine source publication after "
            "engine readiness (worker_id=%s)",
            ctx.global_rank,
            ctx.worker_id,
        )
        return True

    @property
    def nixl_manager(self) -> NixlTransferManager | None:
        if self._ctx is not None:
            return self._ctx.nixl_manager
        return None

    @property
    def tensors(self) -> dict[str, torch.Tensor]:
        if self._ctx is not None:
            return self._ctx.tensors
        return {}
