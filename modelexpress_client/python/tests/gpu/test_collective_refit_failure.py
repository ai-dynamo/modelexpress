# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A refit that fails mid-round, wedges a generator, then re-forms in a fresh one.

The unit suite proves the failure path's order - abort, report, reset - and the
bounded-abort path on recording stand-ins. None of that establishes what
happens on devices: that a generator whose ``ncclCommAbort`` cannot interrupt a
running M2N kernel does not hang the process forever, that the operation MX
holds ends FAILED with the cause attached, that the server bumps the epoch, and
that a brand-new generator process then re-forms the group and moves bytes
exactly. This test runs that whole sequence on two GPUs against a live
ModelExpress server.

Three processes:

* **trainer** (rank 0), persistent across both rounds. Round A moves layer
  group 0, then its ``pre`` hook for layer group 1 raises. It fails the round at
  once, the operation is reported FAILED (which moves the group to the next
  epoch), and it re-runs ``compute_plan`` and waits for a generator at the new
  epoch.
* **generator A** (rank 1), round A only. It has layer group 1's receives
  enqueued with no sender, so it is released by its transfer deadline and takes
  the failure path. On a libnccl below v2.33 the abort cannot interrupt the M2N
  kernel still on the stream, so it raises :class:`CommunicatorWedgedError`.
  This process implements the runner contract: report the failure, then exit
  the OS process with :data:`EXIT_WEDGED`. Nothing in-process can recover a
  wedged communicator; the epoch bump plus a fresh process is the recovery.
* **generator B** (rank 1), launched after A exits. It re-forms at epoch+1 and
  round B moves both layer groups, verified byte-exact against values it
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

from modelexpress_rl.collective import CommunicatorWedgedError
from modelexpress_rl.collective.backend import MIN_NCCL, loaded_nccl_version

pytestmark = pytest.mark.gpu

TRAINER, GENERATOR = 0, 1
ROWS, COLS = 64, 32
MODULUS = 251
LAYERS = ("layers.0.weight", "layers.1.weight")
V_FAIL, V_OK = "v-fail", "v-ok"

#: A libnccl at or above this version can interrupt an M2N kernel in
#: ncclCommAbort, so the bounded abort returns rather than wedging. Below it the
#: abort cannot return and the generator must exit the process.
NCCL_ABORTABLE = (2, 33, 0)

#: Exit code generator A uses when its abort wedges - the runner contract.
EXIT_WEDGED = 42

#: The generator's transfer deadline: what releases it from layer group 1.
PEER_DEADLINE_S = float(os.environ.get("MX_GPU_FAILURE_DEADLINE_S", "30"))
#: How long the bounded abort waits before declaring the communicator wedged.
ABORT_TIMEOUT_S = float(os.environ.get("MX_GPU_FAILURE_ABORT_TIMEOUT_S", "10"))
#: Bound group (re-)formation so a lost peer cannot stall the whole test.
GROUP_TIMEOUT_S = float(os.environ.get("MX_GPU_FAILURE_GROUP_TIMEOUT_S", "180"))
RESULT_TIMEOUT_S = float(os.environ.get("MX_GPU_FAILURE_RESULT_TIMEOUT_S", "480"))


def _requirements() -> str | None:
    if torch.cuda.device_count() < 2:
        return "needs 2 CUDA devices"
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


def _apply_env() -> None:
    os.environ["MX_NCCL_REFIT_TRANSFER_TIMEOUT_S"] = str(PEER_DEADLINE_S)
    os.environ["MX_NCCL_REFIT_ABORT_TIMEOUT_S"] = str(ABORT_TIMEOUT_S)
    os.environ["MX_NCCL_REFIT_GROUP_TIMEOUT_S"] = str(GROUP_TIMEOUT_S)
    os.environ["MX_NCCL_REFIT_COMM_INIT_TIMEOUT_S"] = str(GROUP_TIMEOUT_S)
    os.environ["MX_NCCL_REFIT_PIPELINE_DEPTH"] = "1"


def _make_client(rank: int, endpoint: str, run_id: str):
    """Build a client for one role, its buffers, device and engine."""
    import grpc

    from modelexpress_rl.collective import (
        LocalParamSpec,
        RefitClientGenerator,
        RefitClientTrainer,
        RefitCtx,
    )
    from modelexpress_rl.collective.rendezvous import CollectiveRendezvous

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
    else:
        # Both generator processes reclaim the SAME slot and worker id: the
        # orchestrator restarts a failed generator with its own identity, and
        # that is what lets the fresh process re-form at the next epoch.
        client = RefitClientGenerator(
            slot_id="g0", worker_id=f"g0-{run_id}", index_in_role=0, **common
        )
    client.setup_layer_groups([[LAYERS[0]], [LAYERS[1]]])
    if is_trainer:
        client.initialize(engine, source_partition=0)
    else:
        client.initialize(engine)
    return client, bufs, device, engine, rendezvous


def _report_op(rendezvous, report: dict, key: str, op) -> None:
    from modelexpress_rl import refit_collective_pb2 as pb

    state = rendezvous.get_transfer(op)
    report[f"{key}_state"] = pb.CollectiveTransferState.Name(state.state)
    report[f"{key}_message"] = state.failure_message


def _flush_and_exit(results, report: dict, code: int) -> None:
    """Deliver the report, then exit the OS process hard - the runner contract.

    ``os._exit`` skips the queue's atexit flush, so the feeder is drained
    explicitly first; otherwise the parent would see the wedged generator exit
    with no report at all.
    """
    results.put(report)
    results.close()
    results.join_thread()
    os._exit(code)


def _trainer_main(endpoint: str, run_id: str, results) -> None:
    _apply_env()
    report: dict = {"role": "trainer"}
    client = None
    try:
        client, bufs, device, engine, rendezvous = _make_client(TRAINER, endpoint, run_id)

        # Round A: layer group 0 moves, then the trainer fails mid-round.
        membership = client.compute_plan()
        report["epoch_a"] = membership.epoch
        op_a = client.create_operation(V_FAIL, key_prefix=run_id)
        client.start_weight_update(V_FAIL, op_a)
        client.publish_weights(V_FAIL, 0)
        engine.fail_layer = LAYERS[1]
        try:
            client.publish_weights(V_FAIL, 1)
            report["round_a"] = "completed"
        except InjectedFailure:
            report["round_a"] = "InjectedFailure"
        except Exception as error:  # noqa: BLE001 - record anything unexpected
            report["round_a"] = type(error).__name__
            report["round_a_error"] = repr(error)
        engine.fail_layer = None
        report["round_reset"] = client._operation_id is None and client._version is None
        report["handle_released"] = client._half is None or client._half.m2n_handle is None
        _report_op(rendezvous, report, "op_a", op_a)

        # Re-form at the next epoch and wait for the fresh generator.
        membership = client.compute_plan()
        report["epoch_b"] = membership.epoch
        op_b = client.create_operation(V_OK, key_prefix=run_id)
        client.start_weight_update(V_OK, op_b)
        for layer_group in (0, 1):
            client.publish_weights(V_OK, layer_group)
        client.finish_weight_update(V_OK)
        torch.cuda.synchronize()
        _report_op(rendezvous, report, "op_b", op_b)
    except BaseException:  # noqa: BLE001 - every failure reaches the parent
        report["error"] = traceback.format_exc()
    finally:
        results.put(report)
        if client is not None:
            try:
                client.cleanup()
            except Exception:  # noqa: BLE001 - the report already went out
                pass


def _generator_a_main(endpoint: str, run_id: str, results) -> None:
    _apply_env()
    report: dict = {"role": "gen_a"}
    client = None
    try:
        client, bufs, device, engine, rendezvous = _make_client(GENERATOR, endpoint, run_id)

        membership = client.compute_plan()
        report["epoch_a"] = membership.epoch
        op_a = client.create_operation(V_FAIL, key_prefix=run_id)
        client.start_weight_update(V_FAIL, op_a)
        client.update_weights(V_FAIL, 0)
        torch.cuda.synchronize()
        report["layer0_after_a_is_v_fail"] = bool(
            torch.equal(bufs[LAYERS[0]], _golden(V_FAIL, 0, device))
        )
        try:
            client.update_weights(V_FAIL, 1)
            client.finish_weight_update(V_FAIL)
            report["round_a"] = "completed"
        except CommunicatorWedgedError as wedged:
            # The runner contract: the communicator cannot be torn down in this
            # process. The round was already reported FAILED inside the client,
            # so peers re-form without us; now exit the process hard.
            report["round_a"] = "CommunicatorWedgedError"
            report["wedged_epoch"] = wedged.epoch
            report["wedged_version"] = wedged.version
            _report_op(rendezvous, report, "op_a", op_a)
            _flush_and_exit(results, report, EXIT_WEDGED)
        except Exception as error:  # noqa: BLE001 - what releases it is the question
            report["round_a"] = type(error).__name__
            report["round_a_error"] = repr(error)
        _report_op(rendezvous, report, "op_a", op_a)
    except BaseException:  # noqa: BLE001 - every failure reaches the parent
        report["error"] = traceback.format_exc()
    finally:
        results.put(report)
        if client is not None:
            try:
                client.cleanup()
            except Exception:  # noqa: BLE001 - the report already went out
                pass


def _generator_b_main(endpoint: str, run_id: str, results) -> None:
    _apply_env()
    report: dict = {"role": "gen_b"}
    client = None
    try:
        client, bufs, device, engine, rendezvous = _make_client(GENERATOR, endpoint, run_id)

        membership = client.compute_plan()
        report["epoch_b"] = membership.epoch
        op_b = client.create_operation(V_OK, key_prefix=run_id)
        client.start_weight_update(V_OK, op_b)
        for layer_group in (0, 1):
            client.update_weights(V_OK, layer_group)
        client.finish_weight_update(V_OK)
        torch.cuda.synchronize()
        _report_op(rendezvous, report, "op_b", op_b)
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


def _run_cohort() -> tuple[dict[str, dict], int | None]:
    endpoint = os.environ["MX_ENDPOINT"]
    run_id = uuid.uuid4().hex[:12]
    context = mp.get_context("spawn")
    results = context.Queue()

    trainer = context.Process(target=_trainer_main, args=(endpoint, run_id, results))
    gen_a = context.Process(target=_generator_a_main, args=(endpoint, run_id, results))
    trainer.start()
    gen_a.start()

    # Generator A must finish (wedge and exit, or fail cleanly) before the fresh
    # generator reclaims its slot at the new epoch.
    gen_a.join(timeout=RESULT_TIMEOUT_S)
    gen_a_exit = gen_a.exitcode
    if gen_a.is_alive():
        gen_a.terminate()

    gen_b = context.Process(target=_generator_b_main, args=(endpoint, run_id, results))
    gen_b.start()

    collected: dict[str, dict] = {}
    try:
        for _ in range(3):
            report = results.get(timeout=RESULT_TIMEOUT_S)
            collected[report["role"]] = report
    finally:
        for worker in (trainer, gen_b):
            worker.join(timeout=120)
            if worker.is_alive():
                worker.terminate()
    return collected, gen_a_exit


@pytest.mark.skipif(_requirements() is not None, reason=_requirements() or "")
def test_a_wedged_generator_exits_and_a_fresh_one_re_forms() -> None:
    reports, gen_a_exit = _run_cohort()
    for role in ("trainer", "gen_a", "gen_b"):
        assert role in reports, f"missing report for {role}: {reports}"
        assert "error" not in reports[role], f"{role}:\n{reports[role].get('error')}"
    trainer, gen_a, gen_b = reports["trainer"], reports["gen_a"], reports["gen_b"]

    # Round A moved layer group 0 before anything failed.
    assert gen_a["layer0_after_a_is_v_fail"], gen_a

    # The trainer failed at once on the injected hook; its round reset and its
    # handle was released on the one failure path.
    assert trainer["round_a"] == "InjectedFailure", trainer
    assert trainer["round_reset"], trainer
    assert trainer["handle_released"], trainer

    # MX holds round A's operation FAILED, with the trainer's cause attached.
    assert trainer["op_a_state"] == "COLLECTIVE_TRANSFER_STATE_FAILED", trainer
    assert "InjectedFailure" in trainer["op_a_message"], trainer

    found = loaded_nccl_version()
    if found is not None and found < NCCL_ABORTABLE:
        # On this libnccl the abort cannot interrupt the M2N kernel: the
        # generator wedges and the process exits fatally, rather than hanging.
        assert gen_a["round_a"] == "CommunicatorWedgedError", gen_a
        assert gen_a_exit == EXIT_WEDGED, (gen_a_exit, gen_a)
    else:
        # A new-enough libnccl can abort in time; the generator fails the round
        # cleanly and the process exits normally. Either way it did not hang.
        assert gen_a["round_a"] != "completed", gen_a
        assert gen_a_exit in (0, EXIT_WEDGED), (gen_a_exit, gen_a)

    # The failure moved the group on by exactly one epoch, and a fresh
    # generator process re-formed there and moved every layer group byte-exact.
    assert trainer["epoch_b"] == trainer["epoch_a"] + 1, trainer
    assert gen_b["epoch_b"] == trainer["epoch_b"], (trainer, gen_b)
    assert trainer["op_b_state"] == "COLLECTIVE_TRANSFER_STATE_COMPLETE", trainer
    assert gen_b["op_b_state"] == "COLLECTIVE_TRANSFER_STATE_COMPLETE", gen_b
    assert gen_b["exact"] == [True, True], gen_b
    assert gen_b["matches_wrong_version"] == [False, False], gen_b
