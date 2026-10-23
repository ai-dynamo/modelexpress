# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A layer group of several reshards on a communicator without GIN.

With ``MX_NCCL_REFIT_GROUPED_SUBMISSION=1``, ``_submit_layer_group`` records a
layer group's reshards inside one ``nccl.m2n.group()``. A group of two or more calls in one bucket takes M2N's
fused staging path, and that path creates its device communicator with GIN
requirements even when every peer is NVLink-reachable. A communicator whose
ranks have no GIN backend (a pod with no RDMA devices, or ``NCCL_GIN_ENABLE=0``)
fails the group end on every rank::

    ncclM2nGroupEnd: entry 1 of 2 failed: NCCL operation m2nWaitCommReady(comm)
    failed: invalid argument

with NCCL warning ``User requested GIN but not all ranks in the communicator
support GIN``. The same calls issued one at a time take the LSA PACK path and
move the bytes. The synthetic bench and the failure test ran where GIN worked
or with single-entry groups, so neither reached this.

The test forces GIN off with ``NCCL_GIN_ENABLE=0`` so it reproduces on any
host, NICs or not. The ungrouped control runs the same entries in the same
environment and must pass, which shows the environment is otherwise healthy
and the failure belongs to the group.
"""

from __future__ import annotations

import multiprocessing as mp
import os

import pytest
import torch

from modelexpress_rl.collective.backend import MIN_NCCL, loaded_nccl_version

pytestmark = pytest.mark.gpu

RANKS = 4
SRC_RANKS = (0, 1)
DST_RANKS = (2, 3)
# (name, global shape, destination shard dim): two topologies, three calls,
# shaped like one MoE layer's grouped expert projections.
ENTRIES = (
    ("layers.0.experts.down_proj.weight", (4, 16, 8), 2),
    ("layers.0.experts.gate_proj.weight", (4, 8, 16), 1),
    ("layers.0.experts.up_proj.weight", (4, 8, 16), 1),
)


def _requirements() -> str | None:
    if torch.cuda.device_count() < RANKS:
        return f"needs {RANKS} CUDA devices"
    try:
        import nccl.m2n  # noqa: F401
    except Exception as error:  # noqa: BLE001 - any import failure is a skip
        return f"nccl.m2n unavailable ({error})"
    found = loaded_nccl_version()
    if found is None or found < MIN_NCCL:
        return "the loaded libnccl is missing or older than reshard needs"
    return None


def _golden(index: int, shape: tuple[int, ...]) -> torch.Tensor:
    total = 1
    for extent in shape:
        total *= extent
    return (torch.arange(total, dtype=torch.float32) + 1000.0 * index).reshape(shape)


def _rank_main(rank: int, unique_id: bytes, grouped: bool, results) -> None:
    try:
        from modelexpress_rl.collective import backend as mx_backend
        from modelexpress_rl.collective.comm import CommunicatorCache, LaneKey
        from modelexpress_rl.collective.spi import LocalParamSpec
        from modelexpress_rl.collective.types import (
            MeshSpec,
            ParamPlan,
            Placement,
            ReshardPlan,
        )

        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
        cache = CommunicatorCache()
        lane = cache.create(
            LaneKey(group_id="gin-group-test", epoch=1, lane_id=0),
            rank=rank,
            world_size=RANKS,
            unique_id=unique_id,
            device=device,
            stream=torch.cuda.Stream(device=device),
            timeout_s=120.0,
        )
        plans, specs, expected = [], {}, {}
        for index, (name, shape, dst_dim) in enumerate(ENTRIES):
            plan = ParamPlan(
                name=name,
                global_shape=shape,
                dtype="float32",
                partition_id=0,
                src_mesh=MeshSpec(shape=(len(SRC_RANKS),), rank_offset=SRC_RANKS[0]),
                src_placements=(Placement.shard(0),),
                dst_mesh=MeshSpec(shape=(len(DST_RANKS),), rank_offset=DST_RANKS[0]),
                dst_placements=(Placement.shard(dst_dim),),
            )
            plans.append(plan)
            golden = _golden(index, shape)
            if rank in SRC_RANKS:
                part = golden.chunk(len(SRC_RANKS), dim=0)[SRC_RANKS.index(rank)]
                specs[name] = LocalParamSpec(base=part.contiguous().to(device))
            else:
                want = golden.chunk(len(DST_RANKS), dim=dst_dim)[DST_RANKS.index(rank)]
                expected[name] = want.contiguous()
                specs[name] = LocalParamSpec(
                    base=torch.full_like(want, -1.0).to(device)
                )

        is_source = rank in SRC_RANKS
        half_type = mx_backend.NcclM2nSender if is_source else mx_backend.NcclM2nReceiver
        half = half_type(
            plan=ReshardPlan(bulk=plans),
            specs=specs,
            group_id="gin-group-test",
            epoch=1,
            cache=cache,
        )
        half._ensure_m2n_handle()
        src = (lambda ctx: ctx.buf) if is_source else (lambda ctx: None)
        dst = (lambda ctx: None) if is_source else (lambda ctx: ctx.buf)
        ordered = sorted(plans, key=lambda entry: entry.canonical())
        if grouped:
            half._submit_layer_group(ordered, src=src, dst=dst)
        else:
            # The pre-215b74a6 submission: one layer group entry at a time.
            for entry in ordered:
                half._submit_layer_group([entry], src=src, dst=dst)
        lane.synchronize(timeout_s=120.0)
        torch.cuda.synchronize()
        outcome: object = True
        if not is_source:
            outcome = all(
                torch.equal(specs[name].base.cpu(), want)
                for name, want in expected.items()
            )
        results.put((rank, outcome))
    except Exception as error:  # noqa: BLE001 - the failure is the result
        results.put((rank, f"{type(error).__name__}: {error}"))
    # Flush the queue before a hard exit, which skips the feeder thread.
    results.close()
    results.join_thread()
    os._exit(0)


def _run_cohort(grouped: bool) -> dict[int, object]:
    from nccl.core import utils

    value = utils.get_unique_id()
    raw = getattr(value, "as_bytes", None)
    raw = bytes(raw() if callable(raw) else raw)
    context = mp.get_context("spawn")
    results = context.Queue()
    workers = [
        context.Process(target=_rank_main, args=(rank, raw, grouped, results))
        for rank in range(RANKS)
    ]
    for worker in workers:
        worker.start()
    collected: dict[int, object] = {}
    try:
        for _ in range(RANKS):
            rank, outcome = results.get(timeout=240)
            collected[rank] = outcome
    finally:
        for worker in workers:
            worker.join(timeout=30)
            if worker.is_alive():
                worker.kill()
    return collected


@pytest.fixture
def gin_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    # Spawned ranks inherit the parent's environment before NCCL initializes.
    monkeypatch.setenv("NCCL_GIN_ENABLE", "0")


@pytest.mark.skipif(_requirements() is not None, reason=_requirements() or "")
def test_ungrouped_reshards_move_bytes_without_gin(gin_disabled) -> None:
    """Control: the same calls, one per submission, work with GIN off."""
    assert _run_cohort(grouped=False) == {rank: True for rank in range(RANKS)}


@pytest.mark.skipif(_requirements() is not None, reason=_requirements() or "")
@pytest.mark.xfail(
    strict=True,
    reason=(
        "nccl-extensions 0.1.0: a fused m2n.group() requests GIN for its device "
        "communicator and fails the group end on a communicator without GIN; "
        "remove this mark once the group path works there"
    ),
)
def test_a_grouped_layer_moves_bytes_without_gin(
    gin_disabled, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A layer group of several reshards must work where GIN is unavailable."""
    # Grouped submission is opt-in; force it so this still pins the GIN failure.
    monkeypatch.setenv("MX_NCCL_REFIT_GROUPED_SUBMISSION", "1")
    assert _run_cohort(grouped=True) == {rank: True for rank in range(RANKS)}
