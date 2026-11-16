# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The MILES protocol's trainer coordination over four real CPU Gloo ranks.

A DP2 x TP2 trainer runs the real protocol core end to end on a gathered
stream: run-id agreement, the manifest all-gather, the per-rank publisher,
rank-0 generator fan-out and transfer-id broadcast, and the error fan-outs.
Only the edges are faked: the mx-server channel, the trainer session (the
NCCL client), and the SGLang fan-out futures. Every rank runs in its own
spawned process under a wall deadline.

Adapted from 0eab6fbd's test_collective_miles_multirank.py: the TP-local
(trainer-local shard) scenarios are dropped because MX-3a sources are always
whole tensors.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import time
import traceback
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

WORLD = 4
TP = 2
DP = WORLD // TP
DEADLINE_S = 120.0

_SHAPES = {
    "model.embed_tokens.weight": (64, 8),
    "model.layers.0.self_attn.q_proj.weight": (16, 8),
    "model.layers.0.self_attn.o_proj.weight": (8, 16),
    "model.layers.0.mlp.down_proj.weight": (8, 12),
    "model.layers.0.input_layernorm.weight": (8,),
    "model.norm.weight": (8,),
}


def _global(name, version):
    seed = int(hashlib.sha256(f"{name}:{version}".encode()).hexdigest()[:8], 16)
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(_SHAPES[name], generator=generator).to(torch.bfloat16)


class _Group:
    def __init__(self, size, rank):
        self.size = size
        self.rank = rank


class _AsCuda:
    def __init__(self, tensor):
        self._tensor = tensor
        self.shape = tensor.shape
        self.dtype = tensor.dtype
        self.device = "cuda:0"

    def data_ptr(self):
        return self._tensor.data_ptr()

    def is_contiguous(self):
        return self._tensor.is_contiguous()


def _install_fakes(miles_protocol, rank, scenario, record):
    """Fake the server channel and the NCCL session; keep everything else real."""
    import modelexpress_rl.collective.integrations._common as common
    import modelexpress_rl.collective.integrations.miles as miles

    # Torch's default group in this test is Gloo already.
    miles_protocol._gloo_group = lambda: None
    original_signature = common._tensor_signature

    def cpu_signature(name, tensor, **kwargs):
        return original_signature(name, _AsCuda(tensor), **kwargs)

    common._tensor_signature = cpu_signature
    miles._tensor_signature = cpu_signature

    class Channel:
        def close(self):
            record["events"].append("channel-close")

    class Rendezvous:
        def __init__(self, channel):
            pass

        def close(self):
            record["events"].append("rendezvous-close")

    miles_protocol.grpc.insecure_channel = lambda endpoint: Channel()
    miles_protocol.auth.with_auth = lambda channel: channel
    miles_protocol._await_endpoint_ready = lambda *args, **kwargs: None
    miles_protocol.CollectiveRendezvous = Rendezvous

    class Session:
        def __init__(self, kwargs):
            publisher = kwargs["publisher"]
            record["slot_id"] = kwargs["slot_id"]
            record["index_in_role"] = kwargs["index_in_role"]
            record["trainer_slots"] = list(kwargs["topology"].trainer_slots)
            record["local_shapes"] = {
                name: list(spec.base.shape)
                for name, spec in publisher.local_params().items()
            }
            plan = publisher.capture()
            record["plan"] = [entry.canonical() for entry in plan.bulk]
            if scenario == "reconnect":
                from modelexpress_rl.collective.integrations.manifest import (
                    manifest_to_wire,
                )

                record.setdefault("sessions", []).append({
                    "manifest": manifest_to_wire(plan, kwargs["topology"]),
                    "worker_id": kwargs["worker_id"],
                    "slot_id": kwargs["slot_id"],
                    "index_in_role": kwargs["index_in_role"],
                    "layer_groups": kwargs["layer_groups"],
                    "buffers": {
                        name: spec.base.data_ptr()
                        for name, spec in publisher.local_params().items()
                    },
                })
            self._publisher = publisher

        def prepare(self):
            record["events"].append("prepare")

        def create_transfer(self, *, version):
            assert rank == 0, "only rank 0 creates the transfer"
            record["events"].append(("create", version))
            return f"operation-{version}"

        def begin_round(self, *, version, operation_id):
            assert operation_id == f"operation-{version}"
            record["events"].append(("begin", version))
            failed_version = "2" if scenario == "reuse_begin_fails" else "1"
            if (
                scenario in ("begin_fails", "reuse_begin_fails")
                and rank == 1
                and version == failed_version
            ):
                raise RuntimeError("synthetic bind-transfer failure on rank 1")

        def publish_group(self, *, version, layer_group_id):
            failed_version = "2" if scenario == "reuse_publish_fails" else "1"
            if (
                scenario in ("publish_fails", "reuse_publish_fails")
                and rank == 0
                and version == failed_version
            ):
                raise torch.OutOfMemoryError("synthetic publish allocation failure on rank 0")
            if scenario == "publish_rank3_fails" and rank == 3:
                raise RuntimeError("synthetic publish failure on rank 3")
            specs = self._publisher.local_params()
            record["published"][version] = {
                name: hashlib.sha256(
                    spec.base.contiguous().view(torch.uint8).numpy().tobytes()
                ).hexdigest()
                for name, spec in specs.items()
            }

        def finish_round(self, *, version, operation_id):
            record["events"].append(("finish", version, operation_id))
            if scenario == "finish_fails" and rank == 3:
                raise RuntimeError("synthetic finish failure on rank 3")

        def report_failure(self, operation_id, error):
            record["events"].append(("report", operation_id))

        def close(self):
            record["events"].append("session-close")

    class SessionFactory:
        @staticmethod
        def create(**kwargs):
            if scenario == "setup_fails" and rank == 1:
                raise RuntimeError("synthetic session setup failure on rank 1")
            return Session(kwargs)

    miles_protocol.MilesTrainerSession = SessionFactory


def _run(rank, scenario, record, world, tp):
    from modelexpress_rl.collective.integrations import miles_protocol

    if scenario == "stacked_reconnect":
        os.environ["MX_NCCL_REFIT_STACK_BYTES"] = str(1 << 20)
        scenario = "reconnect"
    _install_fakes(miles_protocol, rank, scenario, record)
    dp_rank, tp_rank = divmod(rank, tp)
    dp = world // tp
    if scenario == "world_preflight_fails" and rank == 1:
        dp, dp_rank = 1, 0
    names = list(_SHAPES)
    if scenario == "diverged_manifest" and rank == 3:
        names.remove("model.norm.weight")
    if scenario == "stacked":
        os.environ["MX_NCCL_REFIT_STACK_BYTES"] = str(1 << 20)
    if scenario.startswith("stack_budget_"):
        # Four equal 16-byte norms distinguish a pair of two-member stacks
        # from one four-member stack, as well as off/too-small budgets.
        _SHAPES.update(
            {f"model.layers.{index}.input_layernorm.weight": (8,) for index in (1, 2)}
        )
        names = list(_SHAPES)
        contrasting_budget = int(scenario.removeprefix("stack_budget_"))
        os.environ["MX_NCCL_REFIT_STACK_BYTES"] = str(
            contrasting_budget if rank == 3 else 64
        )

        def unexpected_allocation(*args, **kwargs):
            record["stack_allocation_attempted"] = True
            raise AssertionError("stack allocation preceded stacking agreement")

        miles_protocol._stack_trainer_buffers = unexpected_allocation
    protocol = miles_protocol.MilesCollectiveProtocolCore(SimpleNamespace())
    protocol.connect(
        [SimpleNamespace(name="original")],
        [2],
        [0],
        SimpleNamespace(
            pp=_Group(1, 0),
            tp=_Group(tp, tp_rank),
            ep=_Group(1, 0),
            etp=_Group(1, 0),
            cp=_Group(1, 0),
            intra_dp=_Group(dp, dp_rank),
            indep_dp=_Group(1, 0),
        ),
        SimpleNamespace(gather_pp=False, gather_tp=True, gather_ep=True),
        "target",
    )
    record["lane_rank"] = protocol._lane_rank

    def generator_futures(action, **kwargs):
        record["generator_actions"].append([action, kwargs.get("operation_id")])

        def cancel():
            record["events"].append(f"cancel-{action}")
            return True

        if scenario == "reconnect":
            return protocol._public_engine_calls(action, **kwargs)
        return [SimpleNamespace(cancel=cancel)]

    def call_engine(engine, endpoint, payload):
        record.setdefault("engine_payloads", []).append({
            "engine": engine.name,
            "endpoint": endpoint,
            "payload": payload,
        })
        return SimpleNamespace(cancel=lambda: True)

    if scenario == "reconnect":
        protocol._call_engine = call_engine
        protocol._destroy_on_engine = lambda engine, payload: call_engine(
            engine, "destroy_weights_update_group", payload
        )
    protocol._generator_futures = generator_futures
    protocol._wait_generator_futures = lambda futures, *, timeout_s=None: record[
        "events"
    ].append("wait-generators")

    collective_iterator = scenario in (
        "begin_fails",
        "reuse_begin_fails",
        "clone_fails",
        "copy_fails",
        "validation_fails",
        "shape_fails",
        "reconnect",
        "publish_fails",
        "reuse_publish_fails",
        "world_preflight_fails",
        "reuse_world_preflight_fails",
        "mesh_preflight_fails",
        "reuse_topology_preflight_fails",
    )
    versions = (1, 2, 3, 4) if scenario == "reconnect" else (1, 2)
    for version in versions:
        if rank == 1 and version == 2 and scenario in (
            "reuse_world_preflight_fails", "reuse_topology_preflight_fails",
        ):
            state = protocol._parallel_state
            if scenario == "reuse_world_preflight_fails":
                state.intra_dp = _Group(1, 0)
            protocol.connect(
                [SimpleNamespace(name="replacement")],
                [1] if scenario == "reuse_topology_preflight_fails" else [2],
                [0],
                state,
                SimpleNamespace(gather_pp=False, gather_tp=True, gather_ep=True),
                "target",
            )
        if scenario == "reconnect" and version == 3:
            frozen_plan, frozen_topology = protocol._plan, protocol._topology
            frozen_buffers = dict(protocol._tensors)
            protocol.connect(
                [SimpleNamespace(name="replacement")],
                [2],
                [0],
                protocol._parallel_state,
                SimpleNamespace(gather_pp=False, gather_tp=True, gather_ep=True),
                "target",
            )
            assert protocol._plan is frozen_plan
            assert protocol._topology is frozen_topology
            assert all(
                protocol._tensors[name] is tensor
                for name, tensor in frozen_buffers.items()
            )
        # A gathered stream: every rank holds every tensor whole.
        tensors = [(name, _global(name, version)) for name in names]
        if rank == 1 and scenario == "validation_fails":
            tensors[0] = (names[0], tensors[0][1].float())
        if rank == 1 and scenario == "shape_fails" and version == 2:
            tensors[0] = (names[0], torch.zeros((1,), dtype=torch.bfloat16))

        def iter_buckets(*, materialize):
            assert materialize is True
            record.setdefault("iterator_calls", []).append(version)
            for index, tensor in enumerate(tensors):
                if collective_iterator:
                    # Model MILES' TP gather BEFORE each yielded bucket.
                    value = torch.tensor([1])
                    torch.distributed.all_reduce(value)
                    assert value.item() == world
                record.setdefault("iterator_steps", []).append([version, index])
                yield [tensor]

        def fail_staging(tensor, *args, **kwargs):
            record["staging_calls"] = record.get("staging_calls", 0) + 1
            if record["staging_calls"] == 2:
                raise torch.OutOfMemoryError(f"synthetic {method} OOM on rank 1")
            return original(tensor, *args, **kwargs)

        method = "copy_" if scenario == "copy_fails" else "clone"
        original = getattr(torch.Tensor, method)
        fault = rank == 1 and (
            (scenario == "clone_fails" and version == 1)
            or (scenario == "copy_fails" and version == 2)
        )
        old_tensors = protocol._tensors
        try:
            if fault:
                with patch.object(torch.Tensor, method, fail_staging):
                    protocol.begin_sync(version, iter_buckets)
            elif rank == 1 and scenario == "mesh_preflight_fails":
                with patch.object(
                    protocol, "_destination_mesh",
                    side_effect=ValueError("synthetic destination mesh probe failure"),
                ):
                    protocol.begin_sync(version, iter_buckets)
            else:
                protocol.begin_sync(version, iter_buckets)
        except BaseException:
            record["round_version_after_failure"] = protocol._round_version
            record["buffers_unchanged"] = protocol._tensors is old_tensors
            raise
        record["run_id"] = protocol._run_id
        record["stack_views"] = {
            stack.name: [
                protocol._tensors[member].untyped_storage().data_ptr()
                == protocol._stack_storage[stack.name].untyped_storage().data_ptr()
                and protocol._tensors[member].data_ptr()
                == protocol._stack_storage[stack.name].data_ptr()
                + index * protocol._tensors[member].numel() * 2
                for index, member in enumerate(stack.members)
            ]
            for stack in protocol._stacks
        }
        record["abi"] = protocol._topology.m2n_abi_version
        # The updater drives a separate materialized iterator for sending.
        for index, bucket in enumerate(iter_buckets(materialize=True)):
            if scenario == "diverged_bucket" and rank == 3 and index == 0:
                # One rank's bucket stream alone diverges from the frozen plan.
                protocol.send_bucket([("model.forged.weight", bucket[0][1])])
                continue
            try:
                protocol.send_bucket(bucket)
            except BaseException:
                record["closed"] = protocol._closed
                record["round_begun"] = protocol._round_begun
                raise
        # Model the updater's post-stream barrier, before finalize. A peer
        # must never reach it while another rank is still failing publication.
        record.setdefault("post_stream_barriers", []).append(version)
        torch.distributed.barrier()
        protocol.finalize(version)
        record["rounds"] += 1
        if scenario == "reconnect":
            record.setdefault("connection_rounds", []).append({
                "generation": protocol._connection_generation,
                "worker_id": protocol._slot_worker_id,
                "frozen_model": protocol._topology.model_name,
            })
    if scenario == "reconnect":
        protocol.close()
        assert not protocol._close_pending
        assert not protocol._retired_engine_groups


def _rank_main(rank, init_file, scenario, result_dir, world, tp, timeout_s):
    import torch.distributed as dist

    record = {
        "rank": rank,
        "events": [],
        "generator_actions": [],
        "published": {},
        "rounds": 0,
    }
    try:
        dist.init_process_group(
            "gloo",
            init_method=f"file://{init_file}",
            rank=rank,
            world_size=world,
            timeout=timedelta(seconds=timeout_s),
        )
        started = time.monotonic()
        try:
            _run(rank, scenario, record, world, tp)
        except BaseException as error:
            record["failure_elapsed_s"] = time.monotonic() - started
            record["error"] = str(error)
            record["error_type"] = type(error).__name__
            record["cause"] = repr(error.__cause__)
            record["traceback"] = traceback.format_exc()
        finally:
            record["elapsed_s"] = time.monotonic() - started
            dist.destroy_process_group()
    except BaseException as error:
        record["fatal"] = traceback.format_exc() or repr(error)
    with open(os.path.join(result_dir, f"rank-{rank}.json"), "w") as handle:
        json.dump(record, handle, default=str)


def _launch(tmp_path, scenario, *, world=WORLD, tp=TP, timeout_s=DEADLINE_S / 2):
    os.environ.setdefault("MX_SERVER_ADDRESS", "mx:50051")
    context = multiprocessing.get_context("spawn")
    init_file = tmp_path / "gloo-init"
    processes = [
        context.Process(
            target=_rank_main,
            args=(rank, str(init_file), scenario, str(tmp_path), world, tp, timeout_s),
            daemon=True,
        )
        for rank in range(world)
    ]
    deadline = time.monotonic() + DEADLINE_S
    for process in processes:
        process.start()
    try:
        for process in processes:
            process.join(timeout=max(0.0, deadline - time.monotonic()))
        hung = [rank for rank, process in enumerate(processes) if process.is_alive()]
        assert not hung, f"ranks {hung} missed the {DEADLINE_S:.0f}s wall deadline"
    finally:
        for process in processes:
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
    records = []
    for rank in range(world):
        with open(tmp_path / f"rank-{rank}.json") as handle:
            record = json.load(handle)
        assert "fatal" not in record, record["fatal"]
        records.append(record)
    return records


@pytest.fixture(autouse=True)
def _server_address(monkeypatch):
    monkeypatch.setenv("MX_SERVER_ADDRESS", "mx:50051")
    monkeypatch.delenv("MX_MILES_DST_LAYOUT", raising=False)
    monkeypatch.delenv("MX_NCCL_REFIT_STACK_BYTES", raising=False)


@pytest.mark.parametrize("contrasting_budget", [0, 16, 32])
def test_mixed_stack_budgets_fail_uniformly_before_allocation(tmp_path, contrasting_budget):
    records = _launch(tmp_path, f"stack_budget_{contrasting_budget}")

    assert len({record["error"] for record in records}) == 1
    for record in records:
        assert record["error_type"] == "ValueError", record.get("traceback")
        assert "same MX_NCCL_REFIT_STACK_BYTES" in record["error"]
        assert f"[64, 64, 64, {contrasting_budget}]" in record["error"]
        assert record["elapsed_s"] < DEADLINE_S / 4
        assert not record.get("stack_allocation_attempted", False)
        assert record["events"] == []
        assert record["generator_actions"] == []
        assert record["rounds"] == 0


def test_dp2_tp2_without_layouts_publishes_whole_tensors_from_every_rank(tmp_path):
    records = _launch(tmp_path, "gathered")

    for record in records:
        assert "error" not in record, record.get("traceback")
        assert record["rounds"] == 2
    # Run-id agreement and identical plans on every rank.
    assert len({record["run_id"] for record in records}) == 1
    run_id = records[0]["run_id"]
    assert len(set(json.dumps(record["plan"]) for record in records)) == 1
    for entry in map(json.loads, records[0]["plan"]):
        assert entry[4] == "2x2@0"
        assert entry[5] == ["R", "R"]
        assert entry[6] == "2@4"
        assert entry[7] == ["R"]
    for record in records:
        rank = record["rank"]
        # Lane rank is the (DP, TP) row-major coordinate.
        assert record["lane_rank"] == rank
        assert record["index_in_role"] == rank
        assert record["slot_id"] == f"{run_id}:trainer-{rank}"
        assert record["trainer_slots"] == [
            f"{run_id}:trainer-{lane}" for lane in range(WORLD)
        ]
        for name, shape in _SHAPES.items():
            assert record["local_shapes"][name] == list(shape)
        # Each round publishes this rank's whole copy of that round's weights.
        for version in ("1", "2"):
            expected = {
                name: hashlib.sha256(
                    _global(name, int(version)).view(torch.uint8).numpy().tobytes()
                ).hexdigest()
                for name in _SHAPES
            }
            assert record["published"][version] == expected
        # Every rank finishes under rank 0's transfer id.
        finishes = [event for event in record["events"] if event[0] == "finish"]
        assert finishes == [
            ["finish", "1", "operation-1"],
            ["finish", "2", "operation-2"],
        ]
    # Rank 0 alone drives the generator fan-out.
    assert records[0]["generator_actions"] == [
        ["prepare", None],
        ["run_round", "operation-1"],
        ["run_round", "operation-2"],
    ]
    assert all(record["generator_actions"] == [] for record in records[1:])
    assert [event for event in records[0]["events"] if event[0] == "create"] == [
        ["create", "1"],
        ["create", "2"],
    ]


@pytest.mark.parametrize("stacked", [False, True])
def test_reconnect_uses_one_fresh_session_topology_on_every_rank(tmp_path, stacked):
    from modelexpress_rl.collective.integrations._common import _derive_wire_plan
    from modelexpress_rl.collective.integrations.manifest import (
        manifest_from_wire,
        manifest_to_wire,
        plan_to_wire,
    )

    records = _launch(tmp_path, "stacked_reconnect" if stacked else "reconnect")

    for record in records:
        assert "error" not in record, record.get("traceback")
    reference_manifests = [session["manifest"] for session in records[0]["sessions"]]
    assert len(reference_manifests) == 2
    first, replacement = reference_manifests
    assert first["topology"]["model_name"] == "miles-model"
    assert replacement["topology"]["model_name"] == "miles-model-connection-1"
    assert first["plan"] == replacement["plan"]
    for field in ("trainer_slots", "generator_slots", "m2n_abi_version"):
        assert first["topology"][field] == replacement["topology"][field]
    expected_abi = "miles-sglang-bf16-replicated-v1" + ("+stack1" if stacked else "")
    assert first["topology"]["m2n_abi_version"] == expected_abi
    calls = records[0]["engine_payloads"]
    prepares = [call for call in calls if call["endpoint"] == "init_weights_update_group"]
    per_tensor, _topology = manifest_from_wire(prepares[0]["payload"]["receiver_init_payload"])
    wire_plan, stacks = _derive_wire_plan(per_tensor)
    assert bool(stacks) is stacked
    assert plan_to_wire(wire_plan) == first["plan"]
    members = {name for stack in stacks for name in stack.members}
    for record in records:
        assert record["rounds"] == 4
        sessions = record["sessions"]
        assert [session["manifest"] for session in sessions] == reference_manifests
        assert sessions[0]["buffers"] == sessions[1]["buffers"]
        assert sessions[0]["worker_id"] != sessions[1]["worker_id"]
        assert sessions[0]["slot_id"] == sessions[1]["slot_id"]
        assert sessions[0]["slot_id"] == first["topology"]["trainer_slots"][record["rank"]]
        assert all(session["index_in_role"] == record["rank"] for session in sessions)
        expected_groups = (
            [[entry.name] for entry in wire_plan.bulk]
            if stacked else [[entry.name for entry in wire_plan.bulk]]
        )
        assert all(session["layer_groups"] == expected_groups for session in sessions)
        rounds = record["connection_rounds"]
        assert [item["generation"] for item in rounds] == [0, 0, 1, 1]
        assert [item["worker_id"] for item in rounds] == [
            sessions[0]["worker_id"], sessions[0]["worker_id"],
            sessions[1]["worker_id"], sessions[1]["worker_id"],
        ]
        assert all(item["frozen_model"] == "miles-model" for item in rounds)
        assert record["events"].count("prepare") == 2
        assert record["events"].count("session-close") == 2
        for version in range(1, 5):
            tensors = {
                name: _global(name, version) for name in _SHAPES if name not in members
            }
            tensors.update({
                stack.name: torch.stack([_global(name, version) for name in stack.members])
                for stack in stacks
            })
            assert record["published"][str(version)] == {
                name: hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest()
                for name, tensor in tensors.items()
            }
        assert all(all(views) for views in record["stack_views"].values())
    assert [call["engine"] for call in prepares] == ["original", "replacement"]
    for call, expected in zip(prepares, reference_manifests, strict=True):
        receiver_plan, receiver_topology = manifest_from_wire(
            call["payload"]["receiver_init_payload"]
        )
        receiver_wire, _stacks = _derive_wire_plan(receiver_plan)
        assert manifest_to_wire(receiver_wire, receiver_topology) == expected
    assert prepares[0]["payload"]["group_name"] != prepares[1]["payload"]["group_name"]
    destroys = [call for call in calls if call["endpoint"] == "destroy_weights_update_group"]
    assert [(call["engine"], call["payload"]["group_name"]) for call in destroys] == [
        (call["engine"], call["payload"]["group_name"]) for call in prepares
    ]
    assert all(record.get("engine_payloads", []) == [] for record in records[1:])


def test_a_diverged_manifest_fails_the_plan_on_every_rank(tmp_path):
    records = _launch(tmp_path, "diverged_manifest")

    for record in records:
        assert record["error_type"] == "ValueError", record.get("traceback")
        assert "ranks [3] differ from rank 0" in record["error"]
        assert record["generator_actions"] == []


def test_a_session_setup_failure_on_one_rank_stops_every_rank(tmp_path):
    records = _launch(tmp_path, "setup_fails")

    for record in records:
        assert record["error_type"] == "RuntimeError", record.get("traceback")
        assert "local session setup failed" in record["error"]
        assert "rank 1:" in record["error"]
        assert record["rounds"] == 0
    # The failure fans out before rank 0 asks any generator to prepare; the
    # terminal close still tells the engines to drop any session.
    assert records[0]["generator_actions"] == [["close", None]]
    assert all(record["generator_actions"] == [] for record in records[1:])


def test_a_diverged_bucket_stream_fails_send_bucket_on_every_rank(tmp_path):
    # _launch's wall deadline is the no-hang guard; below, the failure vote.
    records = _launch(tmp_path, "diverged_bucket")

    for record in records:
        assert record["error_type"] == "RuntimeError", record.get("traceback")
        assert "bucket stream diverged" in record["error"]
        assert "rank 3:" in record["error"]
        assert "outside the frozen plan" in record["error"]
        assert record["rounds"] == 0
        # The vote runs before any setup collective: no session was prepared
        # and no round began on any rank.
        assert "prepare" not in record["events"]
        assert not any(
            isinstance(event, list) and event[0] in ("create", "begin")
            for event in record["events"]
        )
    # Rank 0's terminal close still destroys the (never-joined) engine group.
    assert records[0]["generator_actions"] == [["close", None]]
    assert all(record["generator_actions"] == [] for record in records[1:])


@pytest.mark.parametrize(
    "scenario,version", [("begin_fails", 1), ("reuse_begin_fails", 2)]
)
def test_a_round_start_failure_stops_every_rank_before_next_bucket(
    tmp_path, scenario, version
):
    records = _launch(tmp_path, scenario)

    for record in records:
        assert record["error_type"] == "RuntimeError", record.get("traceback")
        assert "round start failed" in record["error"]
        assert "rank 1: RuntimeError('synthetic bind-transfer failure" in record["error"]
        assert record["failure_elapsed_s"] < DEADLINE_S / 4
        assert record["rounds"] == version - 1
        assert record["closed"] is True
        assert record["round_begun"] is False
        assert "session-close" in record["events"]
        assert "rendezvous-close" in record["events"]
        assert "channel-close" in record["events"]
        assert ["report", f"operation-{version}"] in record["events"]
        assert str(version) not in record["published"]
        # Preparation drained all six valid BF16 buckets; sending stopped at
        # the first bucket, before the next TP iterator collective.
        assert record["iterator_steps"].count([version, 0]) == 2
        for index in range(1, len(_SHAPES)):
            assert record["iterator_steps"].count([version, index]) == 1
    assert "synthetic bind-transfer failure" in records[1]["cause"]
    assert "cancel-run_round" in records[0]["events"]
    assert records[0]["generator_actions"][-1] == ["close", None]
    assert all(record["generator_actions"] == [] for record in records[1:])


@pytest.mark.parametrize(
    "scenario,version,message",
    [
        ("clone_fails", 1, "synthetic clone OOM on rank 1"),
        ("copy_fails", 2, "synthetic copy_ OOM on rank 1"),
        ("validation_fails", 1, "only BF16 base weights are supported"),
        ("shape_fails", 2, "MILES tensor names or canonical shapes changed"),
    ],
)
def test_preparation_failure_drains_the_collective_iterator_on_every_rank(
    tmp_path, scenario, version, message
):
    records = _launch(tmp_path, scenario)

    for record in records:
        assert record["error_type"] == "RuntimeError", record.get("traceback")
        assert "begin_sync preparation failed: rank 1:" in record["error"]
        assert message in record["error"]
        assert record["failure_elapsed_s"] < DEADLINE_S / 4
        assert record["rounds"] == version - 1
        assert record["round_version_after_failure"] is None
        assert record["buffers_unchanged"] is True
        assert str(version) not in record["published"]
        # Exactly the original materializing iterator was exhausted, even on
        # the failed rank: no restart and no switch to materialize=False.
        assert record["iterator_calls"].count(version) == 1
        assert [step for step in record["iterator_steps"] if step[0] == version] == [
            [version, index] for index in range(len(_SHAPES))
        ]
    assert message in records[1]["cause"]
    if scenario in ("clone_fails", "copy_fails"):
        # One successful stage, then OOM; no more clone/copy allocations on
        # that rank while the rest of the iterator is drained.
        assert records[1]["staging_calls"] == 2


@pytest.mark.parametrize(
    "scenario,version", [("publish_fails", 1), ("reuse_publish_fails", 2)]
)
def test_publish_failure_stops_every_rank_before_updater_post_stream_barrier(
    tmp_path, scenario, version
):
    records = _launch(tmp_path, scenario, world=2, tp=1, timeout_s=5)

    for record in records:
        assert record["error_type"] == "RuntimeError", record.get("traceback")
        assert "publish failed: rank 0: OutOfMemoryError" in record["error"]
        assert "synthetic publish allocation failure on rank 0" in record["error"]
        assert record["failure_elapsed_s"] < 3
        assert record["rounds"] == version - 1
        assert record.get("post_stream_barriers", []) == list(range(1, version))
        assert record["closed"] is True
        assert "session-close" in record["events"]
        assert "rendezvous-close" in record["events"]
        assert "channel-close" in record["events"]
        assert ["report", f"operation-{version}"] in record["events"]
    assert "synthetic publish allocation failure on rank 0" in records[0]["cause"]
    assert "cancel-run_round" in records[0]["events"]
    assert records[0]["generator_actions"][-1] == ["close", None]
    assert records[1]["generator_actions"] == []


@pytest.mark.parametrize(
    "scenario,version,message",
    [
        ("world_preflight_fails", 1, "must be exactly DP x TP (1 x 1)"),
        ("reuse_world_preflight_fails", 2, "must be exactly DP x TP (1 x 1)"),
        ("mesh_preflight_fails", 1, "synthetic destination mesh probe failure"),
        ("reuse_topology_preflight_fails", 2, "topology changed"),
    ],
)
def test_contract_preflight_failure_reaches_every_rank_before_contract_collectives(
    tmp_path, scenario, version, message
):
    records = _launch(tmp_path, scenario, world=2, tp=1, timeout_s=5)

    for record in records:
        assert record["error_type"] == "RuntimeError", record.get("traceback")
        assert "contract preflight failed: rank 1:" in record["error"]
        assert message in record["error"]
        assert record["failure_elapsed_s"] < 3
        assert record["rounds"] == version - 1
        assert record["round_version_after_failure"] is None
        assert record["buffers_unchanged"] is True
        assert record.get("post_stream_barriers", []) == list(range(1, version))
        assert [step for step in record["iterator_steps"] if step[0] == version] == [
            [version, index] for index in range(len(_SHAPES))
        ]
        assert not any(
            action == ["run_round", f"operation-{version}"]
            for action in record["generator_actions"]
        )
    assert message in records[1]["cause"]
def test_a_publish_failure_fails_send_bucket_on_every_rank(tmp_path):
    # _launch's wall deadline is the no-hang guard; below, the failure vote.
    records = _launch(tmp_path, "publish_rank3_fails")

    for record in records:
        assert record["error_type"] == "RuntimeError", record.get("traceback")
        assert "publish failed" in record["error"]
        assert "rank 3:" in record["error"]
        assert record["rounds"] == 0
        # The round began, but the publish group never finished on rank 3 and
        # the vote stops every rank at the same bucket: nobody finalizes.
        assert not any(
            isinstance(event, list) and event[0] == "finish"
            for event in record["events"]
        )
    # Rank 0 submitted the round, then the terminal close retires the
    # futures and destroys the engine group.
    assert records[0]["generator_actions"] == [
        ["prepare", None],
        ["run_round", "operation-1"],
        ["close", None],
    ]
    assert all(record["generator_actions"] == [] for record in records[1:])


def test_a_finish_failure_on_one_rank_fails_finalize_on_every_rank(tmp_path):
    records = _launch(tmp_path, "finish_fails")

    for record in records:
        assert record["error_type"] == "RuntimeError", record.get("traceback")
        assert "round failed" in record["error"]
        assert "rank 3:" in record["error"]
        assert record["rounds"] == 0
        assert "session-close" in record["events"]
        assert ["report", "operation-1"] in record["events"]
    # Rank 0 retires the run_round futures instead of waiting on them, then
    # closes the engines.
    assert records[0]["generator_actions"] == [
        ["prepare", None],
        ["run_round", "operation-1"],
        ["close", None],
    ]
    assert "cancel-run_round" in records[0]["events"]
    assert all(record["generator_actions"] == [] for record in records[1:])


def test_dp2_tp2_stacked_ranks_agree_and_publish_the_stacks(tmp_path):
    records = _launch(tmp_path, "stacked")

    for record in records:
        assert "error" not in record, record.get("traceback")
        assert record["rounds"] == 2
        assert record["abi"] == "miles-sglang-bf16-replicated-v1+stack1"
    # One wire plan on every rank: the two norms are one stacked entry.
    assert len({json.dumps(record["plan"]) for record in records}) == 1
    plan = [json.loads(entry) for entry in records[0]["plan"]]
    stacked = [entry for entry in plan if entry[0].startswith("m2n-stack1/")]
    assert len(plan) == len(_SHAPES) - 1
    assert len(stacked) == 1
    assert stacked[0][1] == [2, 8]
    assert stacked[0][4] == "2x2@0"
    assert stacked[0][5] == ["R", "R"]
    assert stacked[0][6] == "2@4"
    assert stacked[0][7] == ["R"]
    assert stacked[0][8] == "m2n-stack1-0"
    members = ("model.layers.0.input_layernorm.weight", "model.norm.weight")
    for record in records:
        (virtual,) = record["stack_views"]
        assert virtual == stacked[0][0]
        assert record["stack_views"][virtual] == [True, True]
        assert record["local_shapes"][virtual] == [2, 8]
        assert set(record["local_shapes"]) == {
            name for name in _SHAPES if name not in members
        } | {virtual}
        for version in ("1", "2"):
            stack = torch.stack([_global(name, int(version)) for name in members])
            published = record["published"][version]
            assert (
                published[virtual]
                == hashlib.sha256(stack.view(torch.uint8).numpy().tobytes()).hexdigest()
            )
            for name in _SHAPES:
                if name in members:
                    continue
                assert (
                    published[name]
                    == hashlib.sha256(
                        _global(name, int(version)).view(torch.uint8).numpy().tobytes()
                    ).hexdigest()
                )
    assert records[0]["generator_actions"] == [
        ["prepare", None],
        ["run_round", "operation-1"],
        ["run_round", "operation-2"],
    ]
