# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A refit that fails mid-round, then re-forms and moves weights again.

The unit suite proves the failure path's order - abort, report, reset - on
recording stand-ins. None of that establishes what happens on devices: that a
peer left with a collective outstanding is actually released, that the
operation MX holds ends FAILED with the cause attached, that the group comes
back at the next epoch, and that the re-formed group then moves bytes exactly.
This test runs that sequence on two GPUs against a live ModelExpress server.

One trainer (rank 0) and one generator (rank 1), two layer groups, generator
pipeline depth 1. Round A moves layer group 0, then the trainer's ``pre`` hook
for layer group 1 raises. The trainer fails its round at once: the group is
discarded unissued, its handle destroyed, its lanes aborted, the operation
reported FAILED (which moves the group to the next epoch). The generator has
layer group 1's receives enqueued with no sender, so it is released by its
transfer deadline and takes the same failure path. Both then re-run
``compute_plan`` - the production recovery - and round B moves both layer
groups at the new epoch, verified byte-exact against values the generator
computes for itself, with a control showing the comparison discriminates.

Needs two CUDA devices, nccl.m2n on a new enough libnccl, and a ModelExpress
server with the Redis collective backend at ``MX_ENDPOINT``.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import traceback
import uuid

import pytest
import torch

from modelexpress_rl.collective.backend import MIN_NCCL, loaded_nccl_version

pytestmark = pytest.mark.gpu

RANKS = 2
TRAINER, GENERATOR = 0, 1
ROWS, COLS = 64, 32
MODULUS = 251
LAYERS = ("layers.0.weight", "layers.1.weight")
V_FAIL, V_OK = "v-fail", "v-ok"

#: The generator's transfer deadline: what releases it from layer group 1.
PEER_DEADLINE_S = float(os.environ.get("MX_GPU_FAILURE_DEADLINE_S", "30"))
#: Allowed on top of that deadline for the abort itself. A native M2N wait that
#: Python cannot interrupt shows up here as a failure, which is the point.
RELEASE_SLACK_S = float(os.environ.get("MX_GPU_FAILURE_RELEASE_SLACK_S", "60"))
RESULT_TIMEOUT_S = float(os.environ.get("MX_GPU_FAILURE_RESULT_TIMEOUT_S", "900"))


def _requirements() -> str | None:
    if torch.cuda.device_count() < RANKS:
        return f"needs {RANKS} CUDA devices"
    try:
        import nccl.m2n  # noqa: F401
    except Exception as error:  # noqa: BLE001 - any import failure is a skip
        return f"nccl.m2n unavailable ({error})"
    found = loaded_nccl_version()
    if found is None:
        return "libnccl.so.2 did not resolve"
    if found < MIN_NCCL:
        want = ".".join(str(p) for p in MIN_NCCL)
        have = ".".join(str(p) for p in found)
        return f"the libnccl this process loads is {have}, and reshard needs {want}"
    if not os.environ.get("MX_ENDPOINT"):
        return (
            "needs a live ModelExpress server with the Redis collective backend; "
            "set MX_ENDPOINT"
        )
    return None


def _golden(version: str, layer: int, device: torch.device) -> torch.Tensor:
    """The global tensor for one layer at one version, from index arithmetic.

    Both sides compute it without exchanging anything, and integers below
    MODULUS are exact in float32, so equality is a byte-exact comparison.
    """
    seed = 1 if version == V_FAIL else 2
    rows = torch.arange(ROWS, device=device, dtype=torch.int64).unsqueeze(1)
    cols = torch.arange(COLS, device=device, dtype=torch.int64).unsqueeze(0)
    values = (rows * COLS + cols + seed * 7919 + layer * 131) % MODULUS
    return values.to(torch.float32)


class InjectedFailure(RuntimeError):
    pass


class _Engine:
    """Publisher and Loader over pre-allocated tensors, with a failure switch."""

    def __init__(self, plan, specs, *, fill=None):
        self._plan = plan
        self._specs = specs
        self._fill = fill
        self.fail_layer: str | None = None
        self.installed: list[int] = []

    def capture(self):
        return self._plan

    def parameter_names(self):
        return self._plan.parameter_names()

    def local_params(self):
        return self._specs

    def start_new_round(self, version):
        if self._fill is not None:
            self._fill(version)

    def install(self, layer_group_id):
        self.installed.append(layer_group_id)

    def finish(self):
        pass

    def cleanup(self):
        pass


def _plan():
    from modelexpress_rl.collective.types import MeshSpec, ParamPlan, Placement, ReshardPlan

    return ReshardPlan(
        bulk=[
            ParamPlan(
                name=name,
                global_shape=(ROWS, COLS),
                dtype="float32",
                partition_id=0,
                src_mesh=MeshSpec(shape=(1,), rank_offset=TRAINER),
                src_placements=(Placement.shard(0),),
                dst_mesh=MeshSpec(shape=(1,), rank_offset=GENERATOR),
                dst_placements=(Placement.shard(1),),
            )
            for name in LAYERS
        ],
        misc=[],
        source_partition_count=1,
    )


def _rank_main(rank: int, endpoint: str, run_id: str, results) -> None:
    os.environ["MX_NCCL_REFIT_TRANSFER_TIMEOUT_S"] = str(PEER_DEADLINE_S)
    os.environ["MX_NCCL_REFIT_PIPELINE_DEPTH"] = "1"
    import time

    import grpc

    from modelexpress_rl import refit_collective_pb2 as pb
    from modelexpress_rl.collective import (
        LocalParamSpec,
        RefitClientGenerator,
        RefitClientTrainer,
        RefitCtx,
    )
    from modelexpress_rl.collective.rendezvous import CollectiveRendezvous

    report: dict = {"rank": rank}
    client = None
    try:
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
        plan = _plan()
        is_trainer = rank == TRAINER
        bufs = {
            name: torch.full((ROWS, COLS), -1.0, dtype=torch.float32, device=device)
            for name in LAYERS
        }
        engine: _Engine

        def fill(version):
            for index, name in enumerate(LAYERS):
                bufs[name].copy_(_golden(version, index, device))
            # The lane stream is not the fill's stream.
            torch.cuda.synchronize()

        def pre_for(name):
            def pre(base):
                if engine.fail_layer == name:
                    raise InjectedFailure(f"injected while staging {name}")
                return RefitCtx(buf=base)

            return pre

        if is_trainer:
            specs = {
                name: LocalParamSpec(base=bufs[name], pre=pre_for(name)) for name in LAYERS
            }
            engine = _Engine(plan, specs, fill=fill)
        else:
            specs = {name: LocalParamSpec(base=bufs[name]) for name in LAYERS}
            engine = _Engine(plan, specs)

        channel = grpc.insecure_channel(endpoint)
        grpc.channel_ready_future(channel).result(timeout=120)
        rendezvous = CollectiveRendezvous(channel)
        common = dict(
            rendezvous=rendezvous,
            model_name=f"gpu-reform-{run_id}",
            trainer_slots=["t0"],
            generator_slots=["g0"],
            source_partition_count=1,
            device=device,
            streams=[torch.cuda.Stream(device=device)],
        )
        if is_trainer:
            client = RefitClientTrainer(
                slot_id="t0", worker_id=f"t0-{run_id}", index_in_role=0, **common
            )
            client.setup_layer_groups([[LAYERS[0]], [LAYERS[1]]])
            client.initialize(engine, source_partition=0)
        else:
            client = RefitClientGenerator(
                slot_id="g0", worker_id=f"g0-{run_id}", index_in_role=0, **common
            )
            client.setup_layer_groups([[LAYERS[0]], [LAYERS[1]]])
            client.initialize(engine)

        # Round A: layer group 0 moves, then the trainer fails mid-round.
        membership = client.compute_plan()
        report["epoch_a"] = membership.epoch
        op_a = client.create_operation(V_FAIL, key_prefix=run_id)
        client.start_weight_update(V_FAIL, op_a)
        if is_trainer:
            client.publish_weights(V_FAIL, 0)
            engine.fail_layer = LAYERS[1]
            started = time.monotonic()
            try:
                client.publish_weights(V_FAIL, 1)
                report["round_a"] = "completed"
            except InjectedFailure:
                report["round_a"] = "InjectedFailure"
            report["released_s"] = time.monotonic() - started
            engine.fail_layer = None
        else:
            client.update_weights(V_FAIL, 0)
            torch.cuda.synchronize()
            report["layer0_after_a_is_v_fail"] = bool(
                torch.equal(bufs[LAYERS[0]], _golden(V_FAIL, 0, device))
            )
            started = time.monotonic()
            try:
                client.update_weights(V_FAIL, 1)
                client.finish_weight_update(V_FAIL)
                report["round_a"] = "completed"
            except Exception as error:  # noqa: BLE001 - what releases it is the question
                report["round_a"] = type(error).__name__
                report["round_a_error"] = repr(error)
            report["released_s"] = time.monotonic() - started

        half = client._half
        report["round_reset"] = client._operation_id is None and client._version is None
        report["handle_released"] = half is None or half.m2n_handle is None
        failed = rendezvous.get_transfer(op_a)
        report["op_a_state"] = pb.CollectiveTransferState.Name(failed.state)
        report["op_a_message"] = failed.failure_message

        # Re-form at the next epoch, then move every layer group exactly.
        membership = client.compute_plan()
        report["epoch_b"] = membership.epoch
        op_b = client.create_operation(V_OK, key_prefix=run_id)
        client.start_weight_update(V_OK, op_b)
        for layer_group in (0, 1):
            if is_trainer:
                client.publish_weights(V_OK, layer_group)
            else:
                client.update_weights(V_OK, layer_group)
        client.finish_weight_update(V_OK)
        torch.cuda.synchronize()
        done = rendezvous.get_transfer(op_b)
        report["op_b_state"] = pb.CollectiveTransferState.Name(done.state)
        if not is_trainer:
            report["exact"] = [
                bool(torch.equal(bufs[name], _golden(V_OK, index, device)))
                for index, name in enumerate(LAYERS)
            ]
            # Control: the comparison answers which version landed.
            report["matches_wrong_version"] = [
                bool(torch.equal(bufs[name], _golden(V_FAIL, index, device)))
                for index, name in enumerate(LAYERS)
            ]
    except BaseException:  # noqa: BLE001 - every failure reaches the parent
        report["error"] = traceback.format_exc()
    finally:
        results.put(report)
        if client is not None:
            try:
                client.cleanup()
            except Exception:  # noqa: BLE001 - the report already went out
                pass


def _run_cohort() -> dict[int, dict]:
    endpoint = os.environ["MX_ENDPOINT"]
    run_id = uuid.uuid4().hex[:12]
    context = mp.get_context("spawn")
    results = context.Queue()
    workers = [
        context.Process(target=_rank_main, args=(rank, endpoint, run_id, results))
        for rank in range(RANKS)
    ]
    for worker in workers:
        worker.start()
    collected: dict[int, dict] = {}
    try:
        for _ in range(RANKS):
            report = results.get(timeout=RESULT_TIMEOUT_S)
            collected[report["rank"]] = report
    finally:
        for worker in workers:
            worker.join(timeout=120)
            if worker.is_alive():
                worker.terminate()
    return collected


@pytest.mark.skipif(_requirements() is not None, reason=_requirements() or "")
def test_a_failed_round_releases_peers_reports_and_the_group_re_forms() -> None:
    reports = _run_cohort()
    for rank in range(RANKS):
        assert "error" not in reports[rank], f"rank {rank}:\n{reports[rank].get('error')}"
    trainer, generator = reports[TRAINER], reports[GENERATOR]

    # Round A moved layer group 0 before anything failed.
    assert generator["layer0_after_a_is_v_fail"]

    # The failing rank failed at once; the peer was released, not left hanging.
    assert trainer["round_a"] == "InjectedFailure"
    assert generator["round_a"] != "completed", generator
    assert generator["released_s"] < PEER_DEADLINE_S + RELEASE_SLACK_S, generator

    for report in (trainer, generator):
        # One failure path on both sides: round state reset, handle released.
        assert report["round_reset"], report
        assert report["handle_released"], report
        # MX holds the operation as FAILED, with the trainer's cause on it.
        assert report["op_a_state"] == "COLLECTIVE_TRANSFER_STATE_FAILED", report
        assert "InjectedFailure" in report["op_a_message"], report
        # The failure moved the group on by exactly one epoch, and it re-formed.
        assert report["epoch_b"] == report["epoch_a"] + 1, report
        assert report["op_b_state"] == "COLLECTIVE_TRANSFER_STATE_COMPLETE", report

    # Round B landed byte-exact, and the comparison can tell versions apart.
    assert generator["exact"] == [True, True]
    assert generator["matches_wrong_version"] == [False, False]
