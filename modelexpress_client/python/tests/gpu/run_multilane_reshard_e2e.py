# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Three-GPU PP2->PP1 NCCL M2N PP-group order and overlap validation.

Ranks 0 and 1 are trainer stages. Rank 2 is one generator stage and owns both
PP-pair communicators. Generator passes updates in reverse insertion order;
runtime submits groups as ``(0, 0)``, then ``(1, 0)``. CUDA events verify the
two independently streamed M2N operations overlap on the generator GPU.
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


def _tensor(stage: int, *, destination: bool, device: int) -> torch.Tensor:
    size_mib = int(os.environ.get("M2N_OVERLAP_MIB", "256"))
    elements = (
        size_mib * 1024 * 1024 // torch.empty((), dtype=torch.float32).element_size()
    )
    value = 0.0 if destination else float(stage + 1)
    return torch.full((elements,), value, dtype=torch.float32, device=f"cuda:{device}")


def main() -> int:
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    trainer_pp = 2
    generator_rank = trainer_pp
    if world != trainer_pp + 1:
        raise SystemExit(f"expected three ranks, got {world}")

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="gloo")

    unique_ids: list[bytes | None]
    if rank == generator_rank:
        unique_ids = [bytes(nccl.get_unique_id().as_bytes) for _ in range(trainer_pp)]
    else:
        unique_ids = [None] * trainer_pp
    dist.broadcast_object_list(unique_ids, src=generator_rank)
    assert all(unique_id is not None for unique_id in unique_ids)

    owned_stages = range(trainer_pp) if rank == generator_rank else (rank,)
    params: dict[int, list[ReshardParam]] = {}
    bootstraps = []
    for stage in owned_stages:
        unique_id = unique_ids[stage]
        assert unique_id is not None
        bootstraps.append(
            M2nPPGroupBootstrap(
                group_id=f"stage-{stage}-to-0",
                key=(stage, 0),
                unique_id=unique_id,
                source_size=1,
                destination_size=1,
                comm_rank=1 if rank == generator_rank else 0,
            )
        )

    executor = NcclM2nExecutor.create(
        local_rank,
        bootstraps,
        max_cta=8,
    )
    # Test-only instrumentation below observes canonical native submission and
    # stream overlap; production callers use only the public executor surface.
    runtime = executor._runtime
    pp_groups = runtime.pp_groups
    for stage in owned_stages:
        tensor = _tensor(
            stage,
            destination=rank == generator_rank,
            device=local_rank,
        )
        params[stage] = [
            ReshardParam(
                name=f"stage_{stage}.weight",
                global_shape=tuple(tensor.shape),
                shard_dim=0,
                local_tensor=tensor,
                local_shard_index=0,
            )
        ]

    intervals: dict[int, tuple[torch.cuda.Event, torch.cuda.Event]] = {}
    recorded_order: list[int] = []
    original_reshard = runtime.handle.reshard
    original_poll_completion = runtime._poll_pp_groups_completion
    if rank == generator_rank:
        comm_to_stage = {
            id(pp_group.communicator): pp_group.key[0] for pp_group in pp_groups
        }
        intervals = {
            stage: (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            for stage in range(trainer_pp)
        }
        origin = torch.cuda.Event(enable_timing=True)
        origin.record()
        origin.synchronize()

        started = set()
        ended = set()

        def recording_reshard(comm, src, dst, *, stream):
            stage = comm_to_stage[id(comm)]
            if stage not in started:
                recorded_order.append(stage)
                intervals[stage][0].record(stream)
                started.add(stage)
            return original_reshard(comm, src, dst, stream=stream)

        def recording_poll_completion(pp_groups, *, operation, deadline):
            if operation == "model-version staging":
                for pp_group in sorted(pp_groups, key=lambda group: group.key):
                    stage = pp_group.key[0]
                    if stage not in ended:
                        intervals[stage][1].record(pp_group.stream)
                        ended.add(stage)
            return original_poll_completion(
                pp_groups,
                operation=operation,
                deadline=deadline,
            )

        runtime.handle.reshard = recording_reshard
        runtime._poll_pp_groups_completion = recording_poll_completion

    try:
        if rank == generator_rank:
            update = executor.stage(
                {
                    (1, 0): params[1],
                    (0, 0): params[0],
                }
            )
        else:
            update = executor.stage({(rank, 0): params[rank]})
        executor.apply(update)
        executor.release(update)
    finally:
        runtime.handle.reshard = original_reshard
        runtime._poll_pp_groups_completion = original_poll_completion

    failed = False
    if rank == generator_rank:
        for stage in range(trainer_pp):
            expected = float(stage + 1)
            failed |= not bool(torch.all(params[stage][0].local_tensor == expected))

        timestamps = {
            stage: (
                origin.elapsed_time(intervals[stage][0]),
                origin.elapsed_time(intervals[stage][1]),
            )
            for stage in range(trainer_pp)
        }
        overlap = max(start for start, _ in timestamps.values()) < min(
            end for _, end in timestamps.values()
        )
        failed |= recorded_order != [0, 1]
        failed |= not overlap
        print(
            f"generator order={recorded_order} intervals_ms={timestamps} "
            f"overlap={overlap}",
            flush=True,
        )

    status = torch.tensor(int(failed))
    dist.all_reduce(status, op=dist.ReduceOp.MAX)
    dist.barrier()
    executor.close()
    dist.barrier()
    dist.destroy_process_group()
    return int(status.item())


if __name__ == "__main__":
    raise SystemExit(main())
