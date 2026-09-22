# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

import modelexpress_rl.collective.integrations.sglang as sglang_integration
from modelexpress.refit.reshard.verify import tensor_digest
from modelexpress_rl.collective import MeshSpec, ParamPlan, Placement, ReshardPlan
from modelexpress_rl.collective.integrations._common import _exact_tensor_sha256
from modelexpress_rl.collective.integrations.miles import CollectiveTopology
from modelexpress_rl.collective.integrations.sglang import (
    SglangGeneratorSession,
    SglangLoader,
    SglangParameterBinding,
)


class FakeTensor:
    def __init__(
        self,
        shape,
        *,
        dtype="torch.bfloat16",
        address=0x1000,
        contiguous=True,
        digest="01" * 32,
    ):
        self.shape = shape
        self.dtype = dtype
        self.device = "cuda:0"
        self.address = address
        self.contiguous = contiguous
        self.digest = digest

    def data_ptr(self):
        return self.address

    def is_contiguous(self):
        return self.contiguous


@pytest.fixture(autouse=True)
def _digest_fake_tensors(monkeypatch):
    monkeypatch.setattr(
        sglang_integration,
        "_exact_tensor_sha256",
        lambda tensor: tensor.digest,
    )


def _entry(name):
    return ParamPlan(
        name=name,
        global_shape=(8, 4),
        dtype="bfloat16",
        partition_id=0,
        src_mesh=MeshSpec(shape=(2,), rank_offset=0),
        src_placements=(Placement.shard(0),),
        dst_mesh=MeshSpec(shape=(2,), rank_offset=2),
        dst_placements=(Placement.shard(1),),
    )


def _plan():
    return ReshardPlan(
        bulk=[_entry("model.a"), _entry("model.b")],
        source_partition_count=1,
    )


def test_sglang_loader_exposes_direct_and_explicit_staged_receive_buffers():
    direct = FakeTensor((8, 2), address=0x1000)
    live = FakeTensor((8, 2), address=0x2000)
    scratch = FakeTensor((8, 2), address=0x3000)
    installed = []
    loader = SglangLoader(
        plan=_plan(),
        bindings={
            "model.a": SglangParameterBinding(live=direct),
            "model.b": SglangParameterBinding(
                live=live,
                receive=scratch,
                install=lambda receive, destination: installed.append(
                    (receive, destination)
                ),
            ),
        },
        layer_groups=(("model.a", "model.b"),),
    )

    specs = loader.local_params()
    loader.expect_round(
        version="version-1",
        operation_id="operation-1",
        tensor_digests=None,
    )
    loader.start_new_round("version-1")
    loader.install(0)

    assert specs["model.a"].base is direct
    assert specs["model.b"].base is scratch
    assert installed == [(scratch, live)]


def test_sglang_loader_rejects_cpu_and_cross_device_staging_storage():
    cpu = FakeTensor((8, 2))
    cpu.device = "cpu"
    with pytest.raises(ValueError, match="indexed CUDA device"):
        SglangLoader(
            plan=_plan(),
            bindings={
                "model.a": SglangParameterBinding(live=cpu),
                "model.b": SglangParameterBinding(live=FakeTensor((8, 2))),
            },
        )

    unindexed = FakeTensor((8, 2))
    unindexed.device = "cuda"
    with pytest.raises(ValueError, match="indexed CUDA device"):
        SglangLoader(
            plan=_plan(),
            bindings={
                "model.a": SglangParameterBinding(live=unindexed),
                "model.b": SglangParameterBinding(live=FakeTensor((8, 2))),
            },
        )

    receive = FakeTensor((8, 2), address=0x3000)
    receive.device = "cuda:1"
    with pytest.raises(ValueError, match="live and receive storage"):
        SglangLoader(
            plan=_plan(),
            bindings={
                "model.a": SglangParameterBinding(
                    live=FakeTensor((8, 2)),
                    receive=receive,
                    install=lambda *_: None,
                ),
                "model.b": SglangParameterBinding(live=FakeTensor((8, 2))),
            },
        )


def test_sglang_loader_validates_the_whole_group_before_any_install_callback_runs():
    first_calls = []
    first_live = FakeTensor((8, 2), address=0x1000)
    second_live = FakeTensor((8, 2), address=0x2000)
    loader = SglangLoader(
        plan=_plan(),
        bindings={
            "model.a": SglangParameterBinding(
                live=first_live,
                receive=FakeTensor((8, 2), address=0x3000),
                install=lambda *_: first_calls.append("installed"),
            ),
            "model.b": SglangParameterBinding(
                live=second_live,
                receive=FakeTensor((8, 2), address=0x4000),
                install=lambda *_: None,
            ),
        },
        layer_groups=(("model.a", "model.b"),),
    )
    loader.expect_round(
        version="version-1",
        operation_id="operation-1",
        tensor_digests=None,
    )
    loader.start_new_round("version-1")
    second_live.address = 0x5000

    with pytest.raises(RuntimeError, match="model.b.*address"):
        loader.install(0)

    assert first_calls == []


def test_sglang_loader_poison_is_permanent_after_a_possibly_mutating_failure():
    writes = []

    def install_first(*_):
        writes.append("model.a")

    def fail_second(*_):
        raise RuntimeError("native install failed")

    loader = SglangLoader(
        plan=_plan(),
        bindings={
            "model.a": SglangParameterBinding(
                live=FakeTensor((8, 2), address=0x1000),
                receive=FakeTensor((8, 2), address=0x3000),
                install=install_first,
            ),
            "model.b": SglangParameterBinding(
                live=FakeTensor((8, 2), address=0x2000),
                receive=FakeTensor((8, 2), address=0x4000),
                install=fail_second,
            ),
        },
        layer_groups=(("model.a", "model.b"),),
    )
    loader.expect_round(
        version="version-1",
        operation_id="operation-1",
        tensor_digests=None,
    )
    loader.start_new_round("version-1")

    with pytest.raises(RuntimeError, match="native install failed"):
        loader.install(0)

    assert writes == ["model.a"]
    assert loader.poisoned
    with pytest.raises(RuntimeError, match="poisoned"):
        loader.start_new_round("version-2")


def test_sglang_loader_verifies_each_wire_tensor_before_staged_install():
    installs = []
    loader = SglangLoader(
        plan=_plan(),
        bindings={
            "model.a": SglangParameterBinding(
                live=FakeTensor((8, 2), address=0x1000),
                receive=FakeTensor(
                    (8, 2),
                    address=0x3000,
                    digest="01" * 32,
                ),
                install=lambda *_: installs.append("model.a"),
            ),
            "model.b": SglangParameterBinding(
                live=FakeTensor((8, 2), address=0x2000),
                receive=FakeTensor(
                    (8, 2),
                    address=0x4000,
                    digest="ff" * 32,
                ),
                install=lambda *_: installs.append("model.b"),
            ),
        },
        layer_groups=(("model.a", "model.b"),),
        verify_tensor_equality=True,
    )
    loader.expect_round(
        version="version-1",
        operation_id="operation-1",
        tensor_digests=(
            ("model.a", "01" * 32),
            ("model.b", "02" * 32),
        ),
    )
    loader.start_new_round("version-1")

    with pytest.raises(RuntimeError, match="model.b.*digest mismatch"):
        loader.install(0)

    assert installs == []
    assert loader.poisoned
    with pytest.raises(RuntimeError, match="poisoned"):
        loader.start_new_round("version-2")


def test_sglang_loader_rejects_missing_or_extra_round_digests():
    loader = _direct_loader(verify_tensor_equality=True)

    with pytest.raises(ValueError, match="exactly match the frozen plan"):
        loader.expect_round(
            version="version-1",
            operation_id="operation-1",
            tensor_digests=(("model.a", "01" * 32),),
        )
    assert loader.poisoned

    loader = _direct_loader(verify_tensor_equality=True)
    with pytest.raises(ValueError, match="exactly match the frozen plan"):
        loader.expect_round(
            version="version-1",
            operation_id="operation-1",
            tensor_digests=(
                ("model.a", "01" * 32),
                ("model.b", "02" * 32),
                ("model.c", "03" * 32),
            ),
        )
    assert loader.poisoned


def test_sglang_loader_logs_receipt_only_after_all_wire_tensors_verify(
    caplog,
):
    loader = SglangLoader(
        plan=_plan(),
        bindings={
            "model.a": SglangParameterBinding(
                live=FakeTensor((8, 2), address=0x1000, digest="01" * 32)
            ),
            "model.b": SglangParameterBinding(
                live=FakeTensor((8, 2), address=0x2000, digest="02" * 32)
            ),
        },
        layer_groups=(("model.a",), ("model.b",)),
        verify_tensor_equality=True,
    )
    loader.expect_round(
        version="version-1",
        operation_id="operation-1",
        tensor_digests=(
            ("model.a", "01" * 32),
            ("model.b", "02" * 32),
        ),
    )
    loader.start_new_round("version-1")
    loader.install(0)

    assert "tensor equality receipt" not in caplog.text

    with caplog.at_level("INFO"):
        loader.install(1)
        loader.finish()

    assert "MILES tensor equality receipt" in caplog.text
    assert "version=version-1" in caplog.text
    assert "operation_id=operation-1" in caplog.text
    assert "tensor_count=2" in caplog.text


def test_exact_tensor_sha256_breaks_the_legacy_4k_row_swap_collision():
    original = torch.arange(2048, dtype=torch.bfloat16)
    swapped = original.clone()
    swapped.view(-1, 2)[[10, 11]] = swapped.view(-1, 2)[[11, 10]]

    assert tensor_digest(original) == tensor_digest(swapped)
    assert _exact_tensor_sha256(original, chunk_bytes=257) != _exact_tensor_sha256(
        swapped,
        chunk_bytes=257,
    )


class FakeRendezvous:
    def __init__(self, *, report_error=None):
        self.reported = []
        self.closed = 0
        self.report_error = report_error

    def report(self, **kwargs):
        self.reported.append(kwargs)
        if self.report_error is not None:
            raise self.report_error

    def close(self):
        self.closed += 1


class FakeGeneratorClient:
    def __init__(self, *, update_error=None, finish_error=None, shared_events=None):
        self.membership = SimpleNamespace(group_id="group-1", epoch=9)
        self.events = []
        self.update_error = update_error
        self.finish_error = finish_error
        self.shared_events = shared_events

    def initialize(self, loader):
        self.events.append(("initialize", loader))

    def setup_layer_groups(self, groups):
        self.events.append(("groups", groups))

    def compute_plan(self):
        self.events.append(("compute",))
        return self.membership

    def start_weight_update(self, version):
        self.events.append(("start", version))
        if self.shared_events is not None:
            self.shared_events.append("start")

    def update_weights(self, version, layer_group_id):
        self.events.append(("update", version, layer_group_id))
        if self.shared_events is not None:
            self.shared_events.append(f"update-{layer_group_id}")
        if self.update_error is not None:
            raise self.update_error

    def finish_weight_update(self, version, operation_id=None):
        self.events.append(("finish", version, operation_id))
        if self.shared_events is not None:
            self.shared_events.append("finish")
        if self.finish_error is not None:
            raise self.finish_error

    def cleanup(self):
        self.events.append(("cleanup",))


def _direct_loader(*, verify_tensor_equality=False):
    return SglangLoader(
        plan=_plan(),
        bindings={
            "model.a": SglangParameterBinding(live=FakeTensor((8, 2), address=0x1000)),
            "model.b": SglangParameterBinding(live=FakeTensor((8, 2), address=0x2000)),
        },
        layer_groups=(("model.a",), ("model.b",)),
        verify_tensor_equality=verify_tensor_equality,
    )


def _topology():
    return CollectiveTopology(
        model_name="qwen",
        trainer_slots=("t0", "t1"),
        generator_slots=("g0", "g1"),
        source_partition_count=1,
        m2n_abi_version="abi-1",
    )


def test_generator_session_runs_the_collective_only_inside_the_engine_safe_point():
    events = []

    @contextmanager
    def safe_point():
        events.append("safe-point-enter")
        try:
            yield
        finally:
            events.append("safe-point-exit")

    client = FakeGeneratorClient(shared_events=events)
    rendezvous = FakeRendezvous()
    loader = _direct_loader()
    session = SglangGeneratorSession(
        client=client,
        rendezvous=rendezvous,
        loader=loader,
        worker_id="generator-worker-0",
        safe_point=safe_point,
        layer_groups=(("model.a",), ("model.b",)),
    )
    session.prepare()

    session.run_round(
        version="version-1",
        operation_id="operation-1",
        tensor_digests=None,
    )

    assert events == [
        "safe-point-enter",
        "start",
        "update-0",
        "update-1",
        "finish",
        "safe-point-exit",
    ]
    assert [event[0] for event in client.events] == [
        "initialize",
        "groups",
        "compute",
        "start",
        "update",
        "update",
        "finish",
    ]
    assert client.events[-1] == ("finish", "version-1", "operation-1")
    assert rendezvous.reported == []


def test_generator_session_factory_passes_the_frozen_topology_and_abi(monkeypatch):
    captured = {}
    lane_stream = object()

    class Client(FakeGeneratorClient):
        def __init__(self, **kwargs):
            captured.update(kwargs)
            super().__init__()

    monkeypatch.setattr(sglang_integration, "RefitClientGenerator", Client)
    monkeypatch.setattr(
        sglang_integration,
        "_collective_streams",
        lambda streams, *, device: [lane_stream],
    )
    rendezvous = FakeRendezvous()

    @contextmanager
    def safe_point():
        yield

    session = SglangGeneratorSession.create(
        rendezvous=rendezvous,
        topology=_topology(),
        loader=_direct_loader(),
        slot_id="g0",
        worker_id="generator-worker-0",
        index_in_role=0,
        safe_point=safe_point,
    )
    session.prepare()

    assert captured["rendezvous"] is rendezvous
    assert captured["trainer_slots"] == ["t0", "t1"]
    assert captured["generator_slots"] == ["g0", "g1"]
    assert captured["source_partition_count"] == 1
    assert captured["m2n_abi_version"] == "abi-1"
    assert captured["receiver_protocol"] == _topology().receiver_protocol
    assert captured["streams"] == [lane_stream]


def test_generator_session_factory_rejects_destination_mesh_outside_topology():
    topology = CollectiveTopology(
        model_name="qwen",
        trainer_slots=("t0", "t1"),
        generator_slots=("g0",),
        source_partition_count=1,
        m2n_abi_version="abi-1",
    )

    with pytest.raises(ValueError, match="dst_mesh ranks"):
        SglangGeneratorSession.create(
            rendezvous=FakeRendezvous(),
            topology=topology,
            loader=_direct_loader(),
            slot_id="g0",
            worker_id="generator-worker-0",
            index_in_role=0,
            safe_point=lambda: None,
        )


def test_generator_session_factory_rejects_a_device_mismatched_with_storage():
    with pytest.raises(ValueError, match="client device cuda:1.*storage device cuda:0"):
        SglangGeneratorSession.create(
            rendezvous=FakeRendezvous(),
            topology=_topology(),
            loader=_direct_loader(),
            slot_id="g0",
            worker_id="generator-worker-0",
            index_in_role=0,
            safe_point=lambda: None,
            device="cuda:1",
        )


def test_generator_session_factory_resolves_bare_cuda_to_the_current_device(
    monkeypatch,
):
    captured = {}

    class Client(FakeGeneratorClient):
        def __init__(self, **kwargs):
            captured.update(kwargs)
            super().__init__()

    monkeypatch.setattr(sglang_integration, "RefitClientGenerator", Client)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)

    SglangGeneratorSession.create(
        rendezvous=FakeRendezvous(),
        topology=_topology(),
        loader=_direct_loader(),
        slot_id="g0",
        worker_id="generator-worker-0",
        index_in_role=0,
        safe_point=lambda: None,
        device=torch.device("cuda"),
    )

    assert captured["device"] == "cuda:0"


def test_generator_session_factory_rejects_bare_cuda_on_a_different_current_device(
    monkeypatch,
):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)

    with pytest.raises(ValueError, match="client device cuda:1.*storage device cuda:0"):
        SglangGeneratorSession.create(
            rendezvous=FakeRendezvous(),
            topology=_topology(),
            loader=_direct_loader(),
            slot_id="g0",
            worker_id="generator-worker-0",
            index_in_role=0,
            safe_point=lambda: None,
            device=torch.device("cuda"),
        )


def test_generator_session_poison_closes_and_preserves_the_original_failure():
    original = RuntimeError("collective update failed after receive began")
    client = FakeGeneratorClient(update_error=original)
    rendezvous = FakeRendezvous(report_error=RuntimeError("report failed"))
    loader = _direct_loader()

    @contextmanager
    def safe_point():
        yield

    session = SglangGeneratorSession(
        client=client,
        rendezvous=rendezvous,
        loader=loader,
        worker_id="generator-worker-0",
        safe_point=safe_point,
        layer_groups=(("model.a",), ("model.b",)),
    )
    session.prepare()

    with pytest.raises(RuntimeError, match="collective update failed") as caught:
        session.run_round(
            version="version-1",
            operation_id="operation-1",
            tensor_digests=None,
        )

    assert caught.value is original
    assert loader.poisoned
    assert client.events[-1] == ("cleanup",)
    assert rendezvous.reported[0]["succeeded"] is False
    assert rendezvous.reported[0]["worker_id"] == "generator-worker-0"
    assert rendezvous.closed == 1


def test_generator_session_falls_back_to_an_idempotent_finish_failure_report():
    original = RuntimeError("finish reported this failure")
    client = FakeGeneratorClient(finish_error=original)
    rendezvous = FakeRendezvous()
    loader = _direct_loader()

    @contextmanager
    def safe_point():
        yield

    session = SglangGeneratorSession(
        client=client,
        rendezvous=rendezvous,
        loader=loader,
        worker_id="generator-worker-0",
        safe_point=safe_point,
        layer_groups=(("model.a",), ("model.b",)),
    )
    session.prepare()

    with pytest.raises(RuntimeError, match="finish reported") as caught:
        session.run_round(
            version="version-1",
            operation_id="operation-1",
            tensor_digests=None,
        )

    assert caught.value is original
    assert loader.poisoned
    assert len(rendezvous.reported) == 1
    assert rendezvous.reported[0]["succeeded"] is False
    assert "finish reported" in rendezvous.reported[0]["message"]
    assert rendezvous.closed == 1
