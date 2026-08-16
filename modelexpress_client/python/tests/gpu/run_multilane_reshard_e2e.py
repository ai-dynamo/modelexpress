# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Three-GPU PP2->PP1 NCCL M2N lane-ordering and overlap validation.

Ranks 0 and 1 are trainer stages. Rank 2 is one generator stage and owns both
PP-pair communicator lanes. The generator passes updates in reverse order; the
runtime must submit lanes as ``(0, 0)``, then ``(1, 0)``. CUDA events verify the
two independently streamed M2N operations overlap on the generator GPU.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

from modelexpress.refit.reshard.transport.nccl_m2n.executor import (
    NcclM2nExecutor,
    ReshardParam,
)
from modelexpress.refit.reshard.transport.nccl_m2n.runtime import (
    _M2nLaneSpec,
    _M2nRuntime,
)


def _tensor(stage: int, *, destination: bool, device: int) -> torch.Tensor:
    size_mib = int(os.environ.get("M2N_OVERLAP_MIB", "256"))
    elements = size_mib * 1024 * 1024 // torch.empty((), dtype=torch.float32).element_size()
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
    runtime = _M2nRuntime(local_rank, max_cta=8)

    unique_ids: list[bytes | None]
    if rank == generator_rank:
        unique_ids = [runtime.new_unique_id_bytes() for _ in range(trainer_pp)]
    else:
        unique_ids = [None] * trainer_pp
    dist.broadcast_object_list(unique_ids, src=generator_rank)
    assert all(unique_id is not None for unique_id in unique_ids)

    owned_stages = range(trainer_pp) if rank == generator_rank else (rank,)
    executors: dict[int, NcclM2nExecutor] = {}
    params: dict[int, list[ReshardParam]] = {}
    lanes = {}
    for stage in owned_stages:
        unique_id = unique_ids[stage]
        assert unique_id is not None
        lane = runtime.create_lane(
            _M2nLaneSpec(
                lane_id=f"stage-{stage}-to-0",
                key=(stage, 0),
                unique_id=unique_id,
                nranks=2,
                comm_rank=1 if rank == generator_rank else 0,
                device_id=local_rank,
            )
        )
        tensor = _tensor(
            stage,
            destination=rank == generator_rank,
            device=local_rank,
        )
        lanes[stage] = lane
        executors[stage] = NcclM2nExecutor(runtime, lane, tp_src=1, tp_dst=1)
        params[stage] = [
            ReshardParam(
                name=f"stage_{stage}.weight",
                global_shape=tuple(tensor.shape),
                shard_dim=0,
                local_tensor=tensor,
            )
        ]

    intervals: dict[int, tuple[torch.cuda.Event, torch.cuda.Event]] = {}
    recorded_order: list[int] = []
    original_reshard = runtime._m2n.reshard
    if rank == generator_rank:
        comm_to_stage = {
            id(lanes[stage].communicator): stage for stage in range(trainer_pp)
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

        def recording_reshard(**kwargs):
            stage = comm_to_stage[id(kwargs["comm"])]
            recorded_order.append(stage)
            start, end = intervals[stage]
            start.record(kwargs["stream"])
            result = original_reshard(**kwargs)
            end.record(kwargs["stream"])
            return result

        runtime._m2n.reshard = recording_reshard

    try:
        if rank == generator_rank:
            NcclM2nExecutor.execute_batch(
                [
                    (executors[1], params[1]),
                    (executors[0], params[0]),
                ]
            )
        else:
            executors[rank].execute(params[rank])
    finally:
        runtime._m2n.reshard = original_reshard

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
    for executor in executors.values():
        executor.teardown()
    dist.barrier()
    runtime.close()
    dist.barrier()
    dist.destroy_process_group()
    return int(status.item())


if __name__ == "__main__":
    raise SystemExit(main())
