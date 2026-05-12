# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ModelExpress loader entrypoint for SGLang's remote_instance backend."""

from __future__ import annotations

import logging
import random
import time
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from ... import p2p_pb2
from ...load_strategy import LoadContext, LoadStrategyChain
from ...load_strategy.context import LoadResult
from ...metadata.heartbeat import HeartbeatThread
from ...metadata.publish import _heartbeat_threads
from ...nixl_transfer import NixlTransferManager
from .adapter import build_sglang_load_context

logger = logging.getLogger("modelexpress.engines.sglang.loader")

if TYPE_CHECKING:
    from sglang.srt.configs.device_config import DeviceConfig
    from sglang.srt.configs.load_config import LoadConfig
    from sglang.srt.configs.model_config import ModelConfig


_tensor_registry: dict[int, dict[str, torch.Tensor]] = {}
_nixl_managers: dict[int, NixlTransferManager] = {}


class MxModelLoader:
    """Unified ModelExpress loader for SGLang.

    SGLang instantiates this class from its ``remote_instance`` loader when
    ``remote-instance-weight-loader-backend=modelexpress``. The class receives
    the already-initialized SGLang model and delegates loading policy to the
    shared ModelExpress strategy chain.
    """

    def __init__(self, load_config: LoadConfig):
        self.load_config = load_config
        self._ctx: LoadContext | None = None

    def load_model(
        self,
        *,
        model: nn.Module,
        model_config: ModelConfig,
        device_config: DeviceConfig,
    ) -> nn.Module:
        """Load model weights through the shared ModelExpress strategy chain."""
        transport = getattr(self.load_config, "modelexpress_transport", "nixl")
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
        self._ctx = ctx

        logger.info(
            "[Worker %s] SGLang MxModelLoader starting (model=%s)",
            ctx.global_rank,
            ctx.identity.model_name,
        )
        model = LoadStrategyChain.run(model, ctx)

        _tensor_registry[ctx.device_id] = ctx.tensors
        if ctx.nixl_manager is not None:
            _nixl_managers[ctx.device_id] = ctx.nixl_manager
        else:
            _nixl_managers.pop(ctx.device_id, None)

        total_time = time.perf_counter() - load_start
        logger.info(
            "[Worker %s] SGLang MxModelLoader.load_model() COMPLETE in %.2fs",
            ctx.global_rank,
            total_time,
        )
        return model.eval()

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
            result = ctx.adapter.before_rdma_receive(result)
            tensors = ctx.adapter.discover_tensors(result)
            weight_info = self._register_transfer_engine_tensors(
                tensors,
                transfer_engine,
            )
            self._receive_via_transfer_engine(
                tensors,
                transfer_engine,
                source_worker,
                ctx,
            )
            result = ctx.adapter.after_rdma_receive(result)

        if weight_info is None:
            weight_info = self._register_transfer_engine_tensors(
                tensors,
                transfer_engine,
            )
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
            return None

        candidates = [
            inst for inst in response.instances if inst.worker_rank == ctx.worker_rank
        ]
        random.shuffle(candidates)
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

    def _receive_via_transfer_engine(
        self,
        tensors: dict[str, torch.Tensor],
        transfer_engine,
        source_worker,
        ctx: LoadContext,
    ) -> None:
        seed_weight_info = {
            tensor.name: (tensor.addr, tensor.size)
            for tensor in source_worker.tensors
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
            local_size = tensor.numel() * tensor.element_size()
            if seed_size != local_size:
                raise RuntimeError(
                    f"ModelExpress transfer_engine: size mismatch for {name}: "
                    f"source={seed_size} bytes, local={local_size} bytes"
                )
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
        for name, tensor in tensors.items():
            addr = tensor.data_ptr()
            numel = tensor.numel()
            element_size = tensor.element_size()
            size = numel * element_size
            if addr not in registered_ptrs:
                ret = transfer_engine.register_memory(addr, size)
                if ret != 0:
                    raise RuntimeError(
                        "ModelExpress transfer_engine: register_memory failed "
                        f"for tensor {name!r}, error={ret}"
                    )
                registered_ptrs.add(addr)
            weight_info[name] = (addr, numel, element_size)
        return weight_info

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
                    size=numel * element_size,
                    device_id=ctx.device_id,
                )
                for name, (addr, numel, element_size) in weight_info.items()
            ]
            worker = p2p_pb2.WorkerMetadata(
                worker_rank=ctx.worker_rank,
                transfer_engine_session_id=session_id,
                tensors=tensors,
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
        try:
            mx_source_id = ctx.mx_client.publish_metadata(
                ctx.identity,
                worker,
                ctx.worker_id,
            )
        except Exception:
            logger.exception(
                "[Worker %s] TransferEngine publish_metadata failed "
                "(worker_id=%s, worker_rank=%s)",
                ctx.global_rank,
                ctx.worker_id,
                ctx.worker_rank,
            )
            return False
        try:
            ctx.mx_client.update_status(
                mx_source_id=mx_source_id,
                worker_id=ctx.worker_id,
                worker_rank=ctx.worker_rank,
                status=p2p_pb2.SOURCE_STATUS_READY,
            )
        except Exception:
            logger.exception(
                "[Worker %s] TransferEngine update_status failed "
                "(mx_source_id=%s, worker_id=%s, worker_rank=%s)",
                ctx.global_rank,
                mx_source_id,
                ctx.worker_id,
                ctx.worker_rank,
            )
            return False
        try:
            heartbeat = HeartbeatThread(
                mx_client=ctx.mx_client,
                mx_source_id=mx_source_id,
                worker_id=ctx.worker_id,
                worker_rank=ctx.worker_rank,
                nixl_manager=None,
            )
            heartbeat.start()
            _heartbeat_threads[ctx.worker_rank] = heartbeat
        except Exception:
            logger.exception(
                "[Worker %s] TransferEngine heartbeat startup failed "
                "(mx_source_id=%s, worker_id=%s, worker_rank=%s)",
                ctx.global_rank,
                mx_source_id,
                ctx.worker_id,
                ctx.worker_rank,
            )
            return False
        logger.info(
            "[Worker %s] Published TransferEngine metadata to MX server "
            "(mx_source_id=%s, worker_id=%s)",
            ctx.global_rank,
            mx_source_id,
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
