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
            self._publisher = publisher

        def prepare(self):
            record["events"].append("prepare")

        def create_transfer(self, *, version):
            assert rank == 0, "only rank 0 creates the transfer"
            record["events"].append(("create", version))
            return f"operation-{version}"

        def begin_round(self, *, version):
            record["events"].append(("begin", version))

        def publish_group(self, *, version, layer_group_id):
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


def _run(rank, scenario, record):
    from modelexpress_rl.collective.integrations import miles_protocol

    _install_fakes(miles_protocol, rank, scenario, record)
    dp_rank, tp_rank = divmod(rank, TP)
    names = list(_SHAPES)
    if scenario == "diverged_manifest" and rank == 3:
        names.remove("model.norm.weight")
    protocol = miles_protocol.MilesCollectiveProtocolCore(SimpleNamespace())
    protocol.connect(
        [object()],
        [2],
        [0],
        SimpleNamespace(
            pp=_Group(1, 0),
            tp=_Group(TP, tp_rank),
            ep=_Group(1, 0),
            etp=_Group(1, 0),
            cp=_Group(1, 0),
            intra_dp=_Group(DP, dp_rank),
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

        return [SimpleNamespace(cancel=cancel)]

    protocol._generator_futures = generator_futures
    protocol._wait_generator_futures = lambda futures: record["events"].append(
        "wait-generators"
    )

    for version in (1, 2):
        # A gathered stream: every rank holds every tensor whole.
        tensors = [(name, _global(name, version)) for name in names]
        protocol.begin_sync(version, lambda *, materialize, t=tensors: iter([t]))
        record["run_id"] = protocol._run_id
        for index, (name, tensor) in enumerate(tensors):
            if scenario == "diverged_bucket" and rank == 3 and index == 0:
                # One rank's bucket stream alone diverges from the frozen plan.
                protocol.send_bucket([("model.forged.weight", tensor)])
                continue
            protocol.send_bucket([(name, tensor)])
        protocol.finalize(version)
        record["rounds"] += 1


def _rank_main(rank, init_file, scenario, result_dir):
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
            world_size=WORLD,
            timeout=timedelta(seconds=DEADLINE_S / 2),
        )
        try:
            _run(rank, scenario, record)
        except BaseException as error:
            record["error"] = str(error)
            record["error_type"] = type(error).__name__
            record["cause"] = repr(error.__cause__)
            record["traceback"] = traceback.format_exc()
        finally:
            dist.destroy_process_group()
    except BaseException as error:
        record["fatal"] = traceback.format_exc() or repr(error)
    with open(os.path.join(result_dir, f"rank-{rank}.json"), "w") as handle:
        json.dump(record, handle, default=str)


def _launch(tmp_path, scenario):
    os.environ.setdefault("MX_SERVER_ADDRESS", "mx:50051")
    context = multiprocessing.get_context("spawn")
    init_file = tmp_path / "gloo-init"
    processes = [
        context.Process(
            target=_rank_main,
            args=(rank, str(init_file), scenario, str(tmp_path)),
            daemon=True,
        )
        for rank in range(WORLD)
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
    for rank in range(WORLD):
        with open(tmp_path / f"rank-{rank}.json") as handle:
            record = json.load(handle)
        assert "fatal" not in record, record["fatal"]
        records.append(record)
    return records


@pytest.fixture(autouse=True)
def _server_address(monkeypatch):
    monkeypatch.setenv("MX_SERVER_ADDRESS", "mx:50051")
    monkeypatch.delenv("MX_MILES_DST_LAYOUT", raising=False)


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
