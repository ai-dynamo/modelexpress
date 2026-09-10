# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end NCCL M2N reshard across two disjoint meshes on real GPUs.

Every other test of this path stops at the boundary of the code in this
repository: the plan is built, the digest agrees, the arguments marshal. None
of them establishes that the call moves bytes, because none of them links
NCCL. This one runs four ranks on four devices, issues one co-called reshard
through the same backend entry point production uses, and compares what lands
on the destination ranks against the tensor the source ranks started from.

The two mismatch tests are not decoration. A comparison that cannot fail
proves nothing, so one of them skips the reshard and one perturbs the expected
value, and both must report a mismatch for the passing case to mean anything.
"""

from __future__ import annotations

import ctypes
import multiprocessing as mp
import os

import pytest
import torch

pytestmark = pytest.mark.gpu

RANKS = 4
SRC_RANKS = (0, 1)
DST_RANKS = (2, 3)
GLOBAL_SHAPE = (8, 16)

#: The reshard entry points were added in this NCCL release. An older library
#: fails inside the native call rather than at import, so the floor is checked
#: against the library actually mapped into the process.
MIN_NCCL = (2, 30, 7)


def _loaded_nccl_version() -> tuple[int, int, int] | None:
    """Version of the libnccl this process actually resolves, or None.

    nccl4py's own ``get_version()`` reports the library it would load by path,
    which is not necessarily the one that wins: a CUDA image ships its own
    libnccl, and whichever is mapped first is the one the reshard runs
    against. Asking the loaded library directly is the only reading that
    tracks the failure.
    """
    try:
        lib = ctypes.CDLL("libnccl.so.2")
        raw = ctypes.c_int()
        if lib.ncclGetVersion(ctypes.byref(raw)) != 0:
            return None
    except OSError:
        return None
    value = raw.value
    return (value // 10000, (value // 100) % 100, value % 100)


def _requirements() -> str | None:
    if torch.cuda.device_count() < RANKS:
        return f"needs {RANKS} CUDA devices"
    try:
        import nccl.m2n  # noqa: F401
    except Exception as error:  # noqa: BLE001 - any import failure is a skip
        return f"nccl.m2n unavailable ({error})"
    found = _loaded_nccl_version()
    if found is None:
        return "libnccl.so.2 did not resolve"
    if found < MIN_NCCL:
        want = ".".join(str(p) for p in MIN_NCCL)
        have = ".".join(str(p) for p in found)
        return (
            f"the libnccl this process loads is {have}, and reshard needs {want}; "
            "preload the one nccl-extensions installed if the image ships an "
            "older library"
        )
    return None


def _golden() -> torch.Tensor:
    total = GLOBAL_SHAPE[0] * GLOBAL_SHAPE[1]
    return torch.arange(total, dtype=torch.float32).reshape(GLOBAL_SHAPE)


def _rank_main(rank: int, unique_id: bytes, mode: str, results) -> None:
    """One rank of the cohort. Reports (rank, matched) for destination ranks."""
    from modelexpress_rl.collective import backend as mx_backend
    from modelexpress_rl.collective.comm import CommunicatorCache, LaneKey
    from modelexpress_rl.collective.types import MeshSpec, ParamPlan, Placement

    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    stream = torch.cuda.Stream(device=device)

    lane = CommunicatorCache().create(
        LaneKey(group_id="gpu-reshard-test", epoch=1, lane_id=0),
        rank=rank,
        world_size=RANKS,
        unique_id=unique_id,
        device=device,
        stream=stream,
        timeout_s=120.0,
    )

    plan = ParamPlan(
        name="probe.weight",
        global_shape=GLOBAL_SHAPE,
        dtype="float32",
        partition_id=0,
        src_mesh=MeshSpec(shape=(len(SRC_RANKS),), rank_offset=SRC_RANKS[0]),
        src_placements=(Placement.shard(0),),
        dst_mesh=MeshSpec(shape=(len(DST_RANKS),), rank_offset=DST_RANKS[0]),
        dst_placements=(Placement.shard(1),),
    )

    reference = _golden()
    src = dst = None
    if rank in SRC_RANKS:
        rows = GLOBAL_SHAPE[0] // len(SRC_RANKS)
        index = SRC_RANKS.index(rank)
        src = reference[index * rows : (index + 1) * rows].contiguous().to(device)
    else:
        cols = GLOBAL_SHAPE[1] // len(DST_RANKS)
        dst = torch.full((GLOBAL_SHAPE[0], cols), -1.0, dtype=torch.float32, device=device)

    if mode != "skip-reshard":
        with torch.cuda.stream(stream):
            mx_backend._reshard(comm=lane, entry=plan, src=src, dst=dst)
    lane.synchronize()
    torch.cuda.synchronize()

    if rank in DST_RANKS:
        cols = GLOBAL_SHAPE[1] // len(DST_RANKS)
        index = DST_RANKS.index(rank)
        want = reference[:, index * cols : (index + 1) * cols].to(device)
        if mode == "perturb-expectation":
            want = want + 1.0
        results.put((rank, bool(torch.equal(dst, want))))


def _run_cohort(mode: str) -> dict[int, bool]:
    from nccl.core import utils

    unique_id = utils.get_unique_id()
    raw = getattr(unique_id, "as_bytes", None)
    raw = bytes(raw() if callable(raw) else raw)

    context = mp.get_context("spawn")
    results = context.Queue()
    workers = [
        context.Process(target=_rank_main, args=(rank, raw, mode, results))
        for rank in range(RANKS)
    ]
    for worker in workers:
        worker.start()
    collected: dict[int, bool] = {}
    try:
        for _ in DST_RANKS:
            rank, matched = results.get(timeout=300)
            collected[rank] = matched
    finally:
        for worker in workers:
            worker.join(timeout=60)
            if worker.is_alive():
                worker.terminate()
    exits = [worker.exitcode for worker in workers]
    assert all(code == 0 for code in exits), f"rank exit codes {exits}"
    return collected


@pytest.mark.skipif(_requirements() is not None, reason=_requirements() or "")
def test_reshard_delivers_the_source_tensor_to_the_destination_mesh() -> None:
    """A 2x2 reshard lands byte-exact on both destination ranks."""
    assert _run_cohort("real") == {rank: True for rank in DST_RANKS}


@pytest.mark.skipif(_requirements() is not None, reason=_requirements() or "")
def test_a_cohort_that_never_reshards_does_not_match() -> None:
    """Control: the comparison above is answering the reshard, not the fill."""
    assert _run_cohort("skip-reshard") == {rank: False for rank in DST_RANKS}


@pytest.mark.skipif(_requirements() is not None, reason=_requirements() or "")
def test_a_wrong_expectation_does_not_match() -> None:
    """Control: the comparison discriminates values, not merely shapes."""
    assert _run_cohort("perturb-expectation") == {rank: False for rank in DST_RANKS}
