# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Three-GPU PP1->PP2 shared-source NCCL M2N regression.

Rank 0 is one trainer stage and owns both PP-pair communicators. Ranks 1 and 2
are separate generator stages. The trainer passes the exact same CUDA tensor
and data pointer to both M2N execution-context buckets in one outer group.
Repeated versions verify read-only cross-bucket reuse, serving-boundary
visibility, canonical submission, and distinct-stream GPU overlap.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
from modelexpress.refit.reshard.transport.nccl_m2n import (
    M2nPPGroupBootstrap,
    NcclM2nExecutor,
    ReshardParam,
)
from nccl import core as nccl


def _tensor(*, value: float, device: int) -> torch.Tensor:
    size_mib = int(os.environ.get("M2N_OVERLAP_MIB", "256"))
    elements = (
        size_mib * 1024 * 1024 // torch.empty((), dtype=torch.float32).element_size()
    )
    return torch.full(
        (elements,),
        value,
        dtype=torch.float32,
        device=f"cuda:{device}",
    )


def main() -> int:
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    source_rank = 0
    generator_stages = 2
    if world != generator_stages + 1:
        raise SystemExit(f"expected three ranks, got {world}")

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="gloo")

    unique_ids: list[bytes | None]
    if rank == source_rank:
        unique_ids = [
            bytes(nccl.get_unique_id().as_bytes) for _ in range(generator_stages)
        ]
    else:
        unique_ids = [None] * generator_stages
    dist.broadcast_object_list(unique_ids, src=source_rank)
    if any(unique_id is None for unique_id in unique_ids):
        raise RuntimeError("failed to broadcast NCCL unique ids")

    owned_stages = range(generator_stages) if rank == source_rank else (rank - 1,)
    bootstraps = []
    for generator_stage in owned_stages:
        unique_id = unique_ids[generator_stage]
        assert unique_id is not None
        bootstraps.append(
            M2nPPGroupBootstrap(
                group_id=f"stage-0-to-{generator_stage}",
                key=(0, generator_stage),
                unique_id=unique_id,
                source_size=1,
                destination_size=1,
                comm_rank=0 if rank == source_rank else 1,
            )
        )

    executor = NcclM2nExecutor.create(local_rank, bootstraps, max_cta=8)
    runtime = executor._runtime
    pp_groups = runtime.pp_groups

    initial_value = 0.0
    local_tensor = _tensor(value=initial_value, device=local_rank)
    local_param = ReshardParam(
        name="shared.weight",
        global_shape=tuple(local_tensor.shape),
        shard_dim=0,
        local_tensor=local_tensor,
        local_shard_index=0,
    )

    recorded_order: list[int] = []
    recorded_source_ptrs: list[int] = []
    intervals: dict[tuple[int, int], tuple[torch.cuda.Event, torch.cuda.Event]] = {}
    original_reshard = runtime.handle.reshard
    original_poll_completion = runtime._poll_pp_groups_completion
    current_version = 0
    stream_handles: dict[int, int] = {}
    if rank == source_rank:
        comm_to_stage = {
            id(pp_group.communicator): pp_group.key[1] for pp_group in pp_groups
        }
        stream_handles = {
            pp_group.key[1]: int(pp_group.stream.cuda_stream) for pp_group in pp_groups
        }
        origin = torch.cuda.Event(enable_timing=True)
        origin.record()
        origin.synchronize()

        def recording_reshard(comm, src, dst, *, stream):
            generator_stage = comm_to_stage[id(comm)]
            key = (current_version, generator_stage)
            if key not in intervals:
                intervals[key] = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                intervals[key][0].record(stream)
            recorded_order.append(generator_stage)
            recorded_source_ptrs.append(int(src.buffer.data_ptr()))
            return original_reshard(comm, src, dst, stream=stream)

        def recording_poll_completion(pp_groups, *, operation, deadline):
            if operation == "model-version staging":
                for pp_group in sorted(pp_groups, key=lambda group: group.key):
                    key = (current_version, pp_group.key[1])
                    intervals[key][1].record(pp_group.stream)
            return original_poll_completion(
                pp_groups,
                operation=operation,
                deadline=deadline,
            )

        runtime.handle.reshard = recording_reshard
        runtime._poll_pp_groups_completion = recording_poll_completion

    failed = False
    version_values = (1.0, 2.0)
    try:
        for current_version, value in enumerate(version_values):
            previous_value = (
                initial_value
                if current_version == 0
                else version_values[current_version - 1]
            )
            if rank == source_rank:
                local_tensor.fill_(value)
                update = executor.stage(
                    {
                        (0, 1): [local_param],
                        (0, 0): [local_param],
                    }
                )
                source_unchanged = bool(torch.all(local_tensor == value))
                failed |= not source_unchanged
                print(
                    f"source version={current_version} unchanged={source_unchanged}",
                    flush=True,
                )
            else:
                generator_stage = rank - 1
                update = executor.stage({(0, generator_stage): [local_param]})
                invisible = bool(torch.all(local_tensor == previous_value))
                failed |= not invisible
                print(
                    f"destination={generator_stage} version={current_version} "
                    f"stage_invisible={invisible}",
                    flush=True,
                )

            executor.apply(update)
            executor.release(update)

            if rank != source_rank:
                generator_stage = rank - 1
                exact = bool(torch.all(local_tensor == value))
                failed |= not exact
                print(
                    f"destination={generator_stage} version={current_version} "
                    f"exact={exact}",
                    flush=True,
                )
    except BaseException:
        if dist.is_initialized():
            dist.destroy_process_group()
        raise
    finally:
        runtime.handle.reshard = original_reshard
        runtime._poll_pp_groups_completion = original_poll_completion

    if rank == source_rank:
        expected_order = [0, 1] * len(version_values)
        expected_ptr = int(local_tensor.data_ptr())
        pointers_shared = recorded_source_ptrs == [expected_ptr] * len(expected_order)
        order_canonical = recorded_order == expected_order
        streams_distinct = len(set(stream_handles.values())) == generator_stages
        failed |= not pointers_shared
        failed |= not order_canonical
        failed |= not streams_distinct

        overlap_by_version = {}
        for version in range(len(version_values)):
            timestamps = {
                stage: (
                    origin.elapsed_time(intervals[(version, stage)][0]),
                    origin.elapsed_time(intervals[(version, stage)][1]),
                )
                for stage in range(generator_stages)
            }
            overlap = max(start for start, _ in timestamps.values()) < min(
                end for _, end in timestamps.values()
            )
            overlap_by_version[version] = (timestamps, overlap)
            failed |= not overlap

        print(
            f"source order={recorded_order} canonical={order_canonical} "
            f"streams={stream_handles} streams_distinct={streams_distinct} "
            f"source_ptr={expected_ptr} pointers_shared={pointers_shared} "
            f"stream_intervals_overlap={overlap_by_version}",
            flush=True,
        )

    try:
        status = torch.tensor(int(failed))
        dist.all_reduce(status, op=dist.ReduceOp.MAX)
        dist.barrier()
        executor.close()
        dist.barrier()
        return int(status.item())
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
