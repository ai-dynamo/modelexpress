# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Framework-agnostic transfer planner for mixed tensor-parallelism.

Computes RDMA transfer ops given source worker metadata and target parameter
descriptions. Handles four cases:
1. Legacy (no shard metadata): 1:1 rank match
2. Replicated (shard_dim == -1): read from any one source rank
3. Dim-0 sharded: byte-range overlap algorithm (contiguous row slicing)
4. Dim-1 sharded: transfer full source shard + GPU-side column slice

Usage::

    from modelexpress.transfer_planner import TransferPlanner, TargetParamInfo

    planner = TransferPlanner()
    ops, fixups = planner.compute_plan(source_workers, target_params, tp_rank, tp_size)
    planner.execute(transfer_engine, ops, rank_to_session)
    planner.apply_dim1_fixups(fixups)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import torch

logger = logging.getLogger("modelexpress.transfer_planner")


@dataclass
class SourceTensorInfo:
    """Source tensor descriptor (populated from WorkerMetadata.tensors)."""

    name: str
    addr: int
    size: int
    device_id: int
    full_shape: list[int]
    shard_dim: int
    effective_tp_size: int
    shard_index: int


@dataclass
class TargetParamInfo:
    """Target parameter descriptor (populated by framework adapter)."""

    name: str
    data_ptr: int
    numel: int
    element_size: int
    shape: list[int]
    dtype: torch.dtype
    device: torch.device
    param: torch.nn.Parameter
    shard_dim: int = -1
    shard_index: int = 0
    effective_tp: int = 1
    output_sizes: list[int] | None = None
    is_contiguous: bool = True


@dataclass
class TransferOp:
    """A single RDMA transfer operation."""

    src_rank: int
    src_addr: int
    src_offset: int
    dst_addr: int
    dst_offset: int
    length: int
    param_name: str


@dataclass
class Dim1Fixup:
    """Post-transfer GPU-side column slice for dim-1 sharded params."""

    param: torch.nn.Parameter
    name: str
    temp: torch.Tensor
    src_shape: list[int]
    src_col_start: int
    src_col_end: int
    dst_col_start: int
    dst_shape: list[int]
    dtype: torch.dtype
    elem_size: int


@dataclass
class SourceIndexResult:
    """Result of building a source index from WorkerMetadata."""

    source_index: dict[str, dict[int, SourceTensorInfo]]
    rank_to_session: dict[int, str] = field(default_factory=dict)
    rank_to_nixl_metadata: dict[int, bytes] = field(default_factory=dict)
    backend: str = "unknown"


def build_source_index(workers: list) -> SourceIndexResult:
    """Build source index from gRPC WorkerMetadata list.

    Auto-detects the backend type (transfer_engine or nixl) from the
    oneof backend_metadata field.

    Args:
        workers: List of WorkerMetadata protobuf objects.

    Returns:
        SourceIndexResult with source_index and backend-specific connection info.
    """
    source_index: dict[str, dict[int, SourceTensorInfo]] = defaultdict(dict)
    rank_to_session: dict[int, str] = {}
    rank_to_nixl_metadata: dict[int, bytes] = {}
    backend = "unknown"

    for w in workers:
        backend_field = w.WhichOneof("backend_metadata")
        if backend_field == "transfer_engine_session_id":
            rank_to_session[w.worker_rank] = w.transfer_engine_session_id
            backend = "transfer_engine"
        elif backend_field == "nixl_metadata":
            rank_to_nixl_metadata[w.worker_rank] = w.nixl_metadata
            backend = "nixl"

        for td in w.tensors:
            source_index[td.name][w.worker_rank] = SourceTensorInfo(
                name=td.name,
                addr=td.addr,
                size=td.size,
                device_id=td.device_id,
                full_shape=list(td.full_shape),
                shard_dim=td.shard_dim,
                effective_tp_size=td.effective_tp_size,
                shard_index=td.shard_index,
            )

    return SourceIndexResult(
        source_index=dict(source_index),
        rank_to_session=rank_to_session,
        rank_to_nixl_metadata=rank_to_nixl_metadata,
        backend=backend,
    )


class TransferPlanner:
    """Computes and executes RDMA transfer plans for mixed tensor-parallelism."""

    def compute_plan(
        self,
        source_index: dict[str, dict[int, SourceTensorInfo]],
        target_params: list[TargetParamInfo],
        tp_rank: int,
        tp_size: int,
    ) -> tuple[list[TransferOp], list[Dim1Fixup]]:
        """Compute transfer plan from source index and target parameters.

        Returns:
            (ops, dim1_fixups)
        """
        ops: list[TransferOp] = []
        dim1_fixups: list[Dim1Fixup] = []

        for tgt in target_params:
            if tgt.name not in source_index:
                raise RuntimeError(
                    f"ModelExpress: {tgt.name} not found in source metadata"
                )

            src_tensors = source_index[tgt.name]
            td0 = next(iter(src_tensors.values()))
            local_size = tgt.numel * tgt.element_size
            dst_addr = tgt.data_ptr

            # Case 1: Legacy (old source without shard metadata)
            if td0.effective_tp_size == 0:
                if tp_rank not in src_tensors:
                    raise RuntimeError(
                        f"ModelExpress: no source rank {tp_rank} for "
                        f"{tgt.name} (legacy mode)"
                    )
                src_td = src_tensors[tp_rank]
                if src_td.size != local_size:
                    raise RuntimeError(
                        f"ModelExpress: size mismatch for {tgt.name}: "
                        f"seed={src_td.size}, local={local_size}"
                    )
                ops.append(TransferOp(
                    tp_rank, src_td.addr, 0, dst_addr, 0, local_size, tgt.name,
                ))
                continue

            shard_dim = td0.shard_dim
            src_eff_tp = td0.effective_tp_size

            # Case 2: Replicated (norms, biases, scales)
            if shard_dim == -1:
                src_rank = min(src_tensors.keys())
                src_td = src_tensors[src_rank]
                if src_td.size != local_size:
                    new_data = torch.empty(
                        src_td.size // tgt.element_size,
                        dtype=tgt.dtype,
                        device=tgt.device,
                    )
                    tgt.param.data = new_data
                    dst_addr = new_data.data_ptr()
                    local_size = src_td.size
                    logger.info(
                        "ModelExpress: reallocated %s to %d bytes", tgt.name, src_td.size,
                    )
                ops.append(TransferOp(
                    src_rank, src_td.addr, 0, dst_addr, 0, local_size, tgt.name,
                ))
                continue

            tgt_eff_tp = tgt.effective_tp
            tgt_shard_index = tgt.shard_index

            full_dim = td0.full_shape[shard_dim]
            src_shard = full_dim // src_eff_tp
            tgt_shard = full_dim // tgt_eff_tp

            stride = td0.size // src_shard if src_shard > 0 else 0

            # Case 4: Inner-dimension sharding (shard_dim >= 1)
            if shard_dim >= 1 and len(td0.full_shape) >= 2:
                self._plan_dim1(
                    ops, dim1_fixups, tgt, src_tensors, td0,
                    src_shard, tgt_shard, tgt_shard_index,
                )
                continue

            # Case 3: Outer-dimension sharding (shard_dim == 0)
            expected_size = tgt_shard * stride
            if local_size != expected_size:
                new_data = torch.empty(
                    expected_size // tgt.element_size,
                    dtype=tgt.dtype,
                    device=tgt.device,
                )
                tgt.param.data = new_data
                dst_addr = new_data.data_ptr()
                local_size = expected_size
                logger.info(
                    "ModelExpress: reallocated sharded %s to %d bytes",
                    tgt.name, expected_size,
                )

            if tgt.output_sizes is not None:
                self._plan_dim0_merged(
                    ops, tgt, src_tensors, dst_addr,
                    src_eff_tp, tgt_eff_tp, tgt_shard_index, stride,
                )
            else:
                self._plan_dim0_simple(
                    ops, tgt, src_tensors, dst_addr,
                    src_shard, tgt_shard, tgt_shard_index, stride,
                )

        return ops, dim1_fixups

    def _plan_dim0_simple(
        self,
        ops: list[TransferOp],
        tgt: TargetParamInfo,
        src_tensors: dict[int, SourceTensorInfo],
        dst_addr: int,
        src_shard: int,
        tgt_shard: int,
        tgt_shard_index: int,
        stride: int,
    ) -> None:
        tgt_start = tgt_shard_index * tgt_shard
        tgt_end = tgt_start + tgt_shard

        for src_rank in sorted(src_tensors.keys()):
            src_td = src_tensors[src_rank]
            src_start = src_td.shard_index * src_shard
            src_end = src_start + src_shard

            ov_start = max(tgt_start, src_start)
            ov_end = min(tgt_end, src_end)

            if ov_start < ov_end:
                src_byte_off = (ov_start - src_start) * stride
                dst_byte_off = (ov_start - tgt_start) * stride
                length = (ov_end - ov_start) * stride

                ops.append(TransferOp(
                    src_rank, src_td.addr, src_byte_off,
                    dst_addr, dst_byte_off, length, tgt.name,
                ))

    def _plan_dim0_merged(
        self,
        ops: list[TransferOp],
        tgt: TargetParamInfo,
        src_tensors: dict[int, SourceTensorInfo],
        dst_addr: int,
        src_eff_tp: int,
        tgt_eff_tp: int,
        tgt_shard_index: int,
        stride: int,
    ) -> None:
        src_sub_offset = 0
        dst_sub_offset = 0
        for sub_full in tgt.output_sizes:
            sub_src_shard = sub_full // src_eff_tp
            sub_tgt_shard = sub_full // tgt_eff_tp

            sub_tgt_start = tgt_shard_index * sub_tgt_shard
            sub_tgt_end = sub_tgt_start + sub_tgt_shard

            for src_rank in sorted(src_tensors.keys()):
                src_td = src_tensors[src_rank]
                sub_src_start = src_td.shard_index * sub_src_shard
                sub_src_end = sub_src_start + sub_src_shard

                ov_start = max(sub_tgt_start, sub_src_start)
                ov_end = min(sub_tgt_end, sub_src_end)

                if ov_start < ov_end:
                    src_byte_off = (
                        src_sub_offset + (ov_start - sub_src_start)
                    ) * stride
                    dst_byte_off = (
                        dst_sub_offset + (ov_start - sub_tgt_start)
                    ) * stride
                    length = (ov_end - ov_start) * stride

                    ops.append(TransferOp(
                        src_rank, src_td.addr, src_byte_off,
                        dst_addr, dst_byte_off, length, tgt.name,
                    ))

            src_sub_offset += sub_src_shard
            dst_sub_offset += sub_tgt_shard

    def _plan_dim1(
        self,
        ops: list[TransferOp],
        dim1_fixups: list[Dim1Fixup],
        tgt: TargetParamInfo,
        src_tensors: dict[int, SourceTensorInfo],
        td0: SourceTensorInfo,
        src_shard: int,
        tgt_shard: int,
        tgt_shard_index: int,
    ) -> None:
        non_shard_dim = td0.full_shape[0]
        src_cols = src_shard
        tgt_cols = tgt_shard

        tgt_global_start = tgt_shard_index * tgt_cols
        tgt_global_end = tgt_global_start + tgt_cols

        for src_rank in sorted(src_tensors.keys()):
            src_td = src_tensors[src_rank]
            src_global_start = src_td.shard_index * src_cols
            src_global_end = src_global_start + src_cols

            ov_start = max(tgt_global_start, src_global_start)
            ov_end = min(tgt_global_end, src_global_end)

            if ov_start < ov_end:
                temp = torch.empty(
                    src_td.size, dtype=torch.uint8, device=tgt.device,
                )
                ops.append(TransferOp(
                    src_rank, src_td.addr, 0,
                    temp.data_ptr(), 0, src_td.size, tgt.name,
                ))
                dim1_fixups.append(Dim1Fixup(
                    param=tgt.param,
                    name=tgt.name,
                    temp=temp,
                    src_shape=[non_shard_dim, src_cols],
                    src_col_start=ov_start - src_global_start,
                    src_col_end=ov_end - src_global_start,
                    dst_col_start=ov_start - tgt_global_start,
                    dst_shape=[non_shard_dim, tgt_cols],
                    dtype=tgt.dtype,
                    elem_size=tgt.element_size,
                ))

    @staticmethod
    def execute_nixl(
        nixl_agent: Any,
        ops: list[TransferOp],
        rank_to_nixl_metadata: dict[int, bytes],
        device_id: int,
    ) -> None:
        """Execute RDMA transfer ops via NIXL, grouped by source rank.

        Uses NIXL agents and UCX backend for GPU-to-GPU RDMA. UCX auto-detects
        NVLink for intra-node transfers.

        Args:
            nixl_agent: Initialized nixl_agent instance.
            ops: Transfer operations from compute_plan().
            rank_to_nixl_metadata: {rank: nixl_metadata_bytes} from source workers.
            device_id: Local GPU device ID.
        """
        import time

        if not ops:
            return

        by_rank: dict[int, list[TransferOp]] = defaultdict(list)
        for op in ops:
            by_rank[op.src_rank].append(op)

        total_bytes = sum(op.length for op in ops)
        logger.info(
            "ModelExpress NIXL: %d ops, %.2f GB across %d source ranks",
            len(ops), total_bytes / 1e9, len(by_rank),
        )

        start_time = time.perf_counter()

        for src_rank in sorted(by_rank.keys()):
            rank_ops = by_rank[src_rank]
            nixl_metadata = rank_to_nixl_metadata.get(src_rank)
            if nixl_metadata is None:
                raise RuntimeError(
                    f"ModelExpress: no NIXL metadata for source rank {src_rank}"
                )

            remote_agent_name = nixl_agent.add_remote_agent(nixl_metadata)

            # Build descriptor lists: (addr, size, device_id) tuples
            remote_descs = [
                (op.src_addr + op.src_offset, op.length, 0)  # source device_id from metadata
                for op in rank_ops
            ]
            local_descs = [
                (op.dst_addr + op.dst_offset, op.length, device_id)
                for op in rank_ops
            ]

            src_prepped = nixl_agent.prep_xfer_dlist(
                agent_name=remote_agent_name,
                xfer_list=remote_descs,
                mem_type="cuda",
                backends=["UCX"],
            )
            dst_prepped = nixl_agent.prep_xfer_dlist(
                agent_name="",
                xfer_list=local_descs,
                mem_type="cuda",
                backends=["UCX"],
            )

            indices = list(range(len(rank_ops)))

            handle = nixl_agent.make_prepped_xfer(
                operation="READ",
                local_xfer_side=dst_prepped,
                local_indices=indices,
                remote_xfer_side=src_prepped,
                remote_indices=indices,
                backends=["UCX"],
            )
            nixl_agent.transfer(handle)

            # Wait for completion
            while True:
                status = nixl_agent.check_xfer_state(handle)
                if status in ("DONE", "SUCCESS"):
                    nixl_agent.release_xfer_handle(handle)
                    break
                if status in ("ERR", "ERROR", "FAIL"):
                    nixl_agent.release_xfer_handle(handle)
                    raise RuntimeError(
                        f"NIXL transfer from rank {src_rank} failed: {status}"
                    )
                time.sleep(0.001)

            rank_bytes = sum(op.length for op in rank_ops)
            logger.info(
                "ModelExpress NIXL: rank %d: %d ops (%.2f GB) done",
                src_rank, len(rank_ops), rank_bytes / 1e9,
            )

        torch.cuda.synchronize(device_id)

        duration = time.perf_counter() - start_time
        bandwidth_gbps = (total_bytes * 8) / (duration * 1e9) if duration > 0 else 0.0
        logger.info(
            "ModelExpress NIXL: transfer complete: %.2f GB in %.2fs (%.1f Gbps)",
            total_bytes / 1e9, duration, bandwidth_gbps,
        )

    @staticmethod
    def execute(
        transfer_engine: Any,
        ops: list[TransferOp],
        rank_to_session: dict[int, str],
    ) -> None:
        """Execute RDMA transfer ops via TransferEngine, grouped by source rank."""
        if not ops:
            return

        by_rank: dict[int, list[TransferOp]] = defaultdict(list)
        for op in ops:
            by_rank[op.src_rank].append(op)

        total_bytes = sum(op.length for op in ops)
        logger.info(
            "ModelExpress: RDMA %d ops, %.2f GB across %d source ranks",
            len(ops), total_bytes / 1e9, len(by_rank),
        )

        for src_rank in sorted(by_rank.keys()):
            rank_ops = by_rank[src_rank]
            session_id = rank_to_session.get(src_rank)
            if session_id is None:
                raise RuntimeError(
                    f"ModelExpress: no TransferEngine session for source rank {src_rank}"
                )

            rank_bytes = sum(op.length for op in rank_ops)
            logger.info(
                "ModelExpress: rank %d: %d ops (%.2f GB) session=%s",
                src_rank, len(rank_ops), rank_bytes / 1e9, session_id,
            )

            src_ptrs = [op.src_addr + op.src_offset for op in rank_ops]
            dst_ptrs = [op.dst_addr + op.dst_offset for op in rank_ops]
            lengths = [op.length for op in rank_ops]
            ret = transfer_engine.batch_transfer_sync_read(
                session_id, dst_ptrs, src_ptrs, lengths,
            )
            if ret < 0:
                raise RuntimeError(
                    f"ModelExpress: batch transfer from rank {src_rank} failed, error={ret}"
                )

    @staticmethod
    def apply_dim1_fixups(fixups: list[Dim1Fixup]) -> None:
        """Apply GPU-side column slicing for dim-1 sharded params."""
        if not fixups:
            return

        by_param: dict[str, list[Dim1Fixup]] = defaultdict(list)
        for f in fixups:
            by_param[f.name].append(f)

        for name, param_fixups in by_param.items():
            param = param_fixups[0].param
            dst_shape = param_fixups[0].dst_shape
            elem_size = param_fixups[0].elem_size
            dtype = param_fixups[0].dtype
            rows = dst_shape[0]
            tgt_cols = dst_shape[1]

            expected_elems = rows * tgt_cols
            if param.numel() != expected_elems:
                new_data = torch.empty(
                    expected_elems, dtype=dtype, device=param.device,
                )
                param.data = new_data

            dst = param.data.reshape(rows, tgt_cols)

            for f in param_fixups:
                src_shape = f.src_shape
                src_cols = src_shape[1]
                num_src_bytes = src_shape[0] * src_cols * elem_size

                src_tensor = f.temp[:num_src_bytes].view(dtype).reshape(
                    src_shape[0], src_cols
                )
                sliced = src_tensor[:, f.src_col_start:f.src_col_end]
                ncols = f.src_col_end - f.src_col_start
                dst_start = f.dst_col_start
                dst[:, dst_start:dst_start + ncols] = sliced

        logger.info(
            "ModelExpress: applied %d dim-1 fixups for %d params",
            len(fixups), len(by_param),
        )
