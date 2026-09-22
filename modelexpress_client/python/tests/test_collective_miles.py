# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from types import SimpleNamespace

import pytest
import torch

import modelexpress_rl.collective.integrations.miles as miles_integration
from modelexpress_rl.collective import MeshSpec, ParamPlan, Placement, ReshardPlan
from modelexpress_rl.collective.integrations.miles import (
    CollectiveTopology,
    MilesPublisher,
    MilesTrainerSession,
    MilesTransferCoordinator,
)


class FakeTensor:
    def __init__(
        self,
        shape,
        *,
        dtype="torch.bfloat16",
        address=0x1000,
        contiguous=True,
    ):
        self.shape = shape
        self.dtype = dtype
        self.device = "cuda:0"
        self.address = address
        self.contiguous = contiguous

    def data_ptr(self):
        return self.address

    def is_contiguous(self):
        return self.contiguous


def _entry(name, *, partition, src_shard=0, dst_shard=1):
    return ParamPlan(
        name=name,
        global_shape=(8, 4),
        dtype="bfloat16",
        partition_id=partition,
        src_mesh=MeshSpec(shape=(2,), rank_offset=0),
        src_placements=(Placement.shard(src_shard),),
        dst_mesh=MeshSpec(shape=(2,), rank_offset=2),
        dst_placements=(Placement.shard(dst_shard),),
        group_key=f"layer-{partition}",
    )


def _plan():
    return ReshardPlan(
        bulk=[
            _entry("model.layers.0.mlp.gate_proj.weight", partition=0),
            _entry("model.layers.1.mlp.down_proj.weight", partition=1, src_shard=1),
        ],
        source_partition_count=2,
    )


def _topology():
    return CollectiveTopology(
        model_name="qwen",
        trainer_slots=(
            "trainer-pp0-tp0",
            "trainer-pp0-tp1",
            "trainer-pp1-tp0",
            "trainer-pp1-tp1",
        ),
        generator_slots=("generator-tp0", "generator-tp1"),
        source_partition_count=2,
        m2n_abi_version="nccl-m2n-2.30.7",
    )


def test_topology_requires_an_explicit_non_empty_abi_and_partitioned_slot_order():
    with pytest.raises(ValueError, match="m2n_abi_version"):
        CollectiveTopology(
            model_name="qwen",
            trainer_slots=("t0",),
            generator_slots=("g0",),
            source_partition_count=1,
            m2n_abi_version=" ",
        )

    lanes = _topology().lanes()

    assert lanes[0].trainer_slots == ("trainer-pp0-tp0", "trainer-pp0-tp1")
    assert lanes[1].trainer_slots == ("trainer-pp1-tp0", "trainer-pp1-tp1")
    assert lanes[2].kind == "BROADCAST"


def test_topology_copies_mutable_slot_inputs():
    trainers = ["t0"]
    generators = ["g0"]
    topology = CollectiveTopology(
        model_name="qwen",
        trainer_slots=trainers,
        generator_slots=generators,
        source_partition_count=1,
        m2n_abi_version="abi-1",
    )
    trainers.append("t1")
    generators.append("g1")

    assert topology.trainer_slots == ("t0",)
    assert topology.generator_slots == ("g0",)


def test_miles_publisher_maps_explicit_aliases_to_the_partition_local_buffers():
    plan = _plan()
    tensor = FakeTensor((4, 4))
    publisher = MilesPublisher(
        plan=plan,
        source_partition=0,
        tensors={"decoder.layers.0.mlp.gate": tensor},
        aliases={"decoder.layers.0.mlp.gate": "model.layers.0.mlp.gate_proj.weight"},
    )
    plan.bulk.clear()

    captured = publisher.capture()

    assert captured.parameter_names() == [
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.1.mlp.down_proj.weight",
    ]
    assert publisher.parameter_names() == captured.parameter_names()
    specs = publisher.local_params()
    assert list(specs) == ["model.layers.0.mlp.gate_proj.weight"]
    assert specs["model.layers.0.mlp.gate_proj.weight"].base is tensor


def test_miles_publisher_derives_wire_order_from_the_plan_not_mapping_order():
    plan = ReshardPlan(
        bulk=[
            _entry("model.a", partition=0),
            _entry("model.b", partition=0),
        ],
        source_partition_count=1,
    )
    first = FakeTensor((4, 4))
    second = FakeTensor((4, 4), address=0x2000)
    publisher = MilesPublisher(
        plan=plan,
        source_partition=0,
        tensors={"native-b": second, "native-a": first},
        aliases={"native-b": "model.b", "native-a": "model.a"},
    )

    specs = publisher.local_params()

    assert list(specs) == ["model.a", "model.b"]
    assert specs["model.a"].base is first
    assert specs["model.b"].base is second


def test_miles_publisher_rejects_incomplete_alias_coverage_and_non_bulk_plans():
    with pytest.raises(ValueError, match="exactly cover partition 0"):
        MilesPublisher(
            plan=_plan(),
            source_partition=0,
            tensors={"wrong": FakeTensor((4, 4))},
            aliases={"wrong": "model.layers.1.mlp.down_proj.weight"},
        )

    plan = _plan()
    plan.misc.append(
        SimpleNamespace(
            name="model.norm.weight",
            global_shape=(4,),
            dtype="bfloat16",
        )
    )
    with pytest.raises(ValueError, match="all-bulk"):
        MilesPublisher(
            plan=plan,
            source_partition=0,
            tensors={"native": FakeTensor((4, 4))},
            aliases={"native": "model.layers.0.mlp.gate_proj.weight"},
        )


def test_miles_publisher_rejects_cpu_and_mixed_device_storage():
    cpu = FakeTensor((4, 4))
    cpu.device = "cpu"
    with pytest.raises(ValueError, match="indexed CUDA device"):
        MilesPublisher(
            plan=_plan(),
            source_partition=0,
            tensors={"native": cpu},
            aliases={"native": "model.layers.0.mlp.gate_proj.weight"},
        )

    unindexed = FakeTensor((4, 4))
    unindexed.device = "cuda"
    with pytest.raises(ValueError, match="indexed CUDA device"):
        MilesPublisher(
            plan=_plan(),
            source_partition=0,
            tensors={"native": unindexed},
            aliases={"native": "model.layers.0.mlp.gate_proj.weight"},
        )

    plan = ReshardPlan(
        bulk=[
            _entry("model.a", partition=0),
            _entry("model.b", partition=0),
        ],
        source_partition_count=1,
    )
    first = FakeTensor((4, 4))
    second = FakeTensor((4, 4), address=0x2000)
    second.device = "cuda:1"
    with pytest.raises(ValueError, match="share one CUDA device"):
        MilesPublisher(
            plan=plan,
            source_partition=0,
            tensors={"a": first, "b": second},
            aliases={"a": "model.a", "b": "model.b"},
        )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda tensor: setattr(tensor, "address", 0x2000), "address"),
        (lambda tensor: setattr(tensor, "shape", (2, 8)), "shape"),
        (lambda tensor: setattr(tensor, "dtype", "torch.float32"), "dtype"),
        (lambda tensor: setattr(tensor, "contiguous", False), "contiguous"),
    ],
)
def test_miles_publisher_rejects_storage_drift_before_a_new_round(change, message):
    tensor = FakeTensor((4, 4))
    publisher = MilesPublisher(
        plan=_plan(),
        source_partition=0,
        tensors={"native": tensor},
        aliases={"native": "model.layers.0.mlp.gate_proj.weight"},
    )
    change(tensor)

    with pytest.raises(RuntimeError, match=message):
        publisher.start_new_round("version-2")


class FakeRendezvous:
    def __init__(self, *, report_error=None):
        self.created = []
        self.reported = []
        self.got = []
        self.deleted = []
        self.closed = 0
        self.report_error = report_error

    def create_transfer(self, **kwargs):
        self.created.append(kwargs)
        return SimpleNamespace(operation_id="operation-1")

    def get_transfer(self, operation_id):
        self.got.append(operation_id)
        return SimpleNamespace(operation_id=operation_id)

    def delete_transfer(self, operation_id):
        self.deleted.append(operation_id)
        return SimpleNamespace(operation_id=operation_id)

    def report(self, **kwargs):
        self.reported.append(kwargs)
        if self.report_error is not None:
            raise self.report_error

    def close(self):
        self.closed += 1


def test_transfer_coordinator_uses_the_same_frozen_topology_for_create_get_delete():
    rendezvous = FakeRendezvous()
    coordinator = MilesTransferCoordinator(rendezvous, _topology())

    transfer = coordinator.create("version-7", idempotency_key="update-7")
    fetched = coordinator.get(transfer.operation_id)
    deleted = coordinator.delete(transfer.operation_id)

    sent = rendezvous.created[0]
    assert sent["model_name"] == "qwen"
    assert sent["trainer_slots"] == list(_topology().trainer_slots)
    assert sent["generator_slots"] == list(_topology().generator_slots)
    assert sent["lanes"] == _topology().lanes()
    assert sent["version_id"] == "version-7"
    assert sent["idempotency_key"] == "update-7"
    assert fetched.operation_id == deleted.operation_id == "operation-1"


class FakeTrainerClient:
    def __init__(self, *, publish_error=None, finish_error=None):
        self.membership = SimpleNamespace(group_id="group-1", epoch=4)
        self.events = []
        self.publish_error = publish_error
        self.finish_error = finish_error

    def initialize(self, publisher, *, source_partition):
        self.events.append(("initialize", publisher, source_partition))

    def setup_layer_groups(self, groups):
        self.events.append(("groups", groups))

    def compute_plan(self):
        self.events.append(("compute",))
        return self.membership

    def start_weight_update(self, version):
        self.events.append(("start", version))

    def publish_weights(self, version, layer_group_id):
        self.events.append(("publish", version, layer_group_id))
        if self.publish_error is not None:
            raise self.publish_error

    def finish_weight_update(self, version, operation_id=None):
        self.events.append(("finish", version, operation_id))
        if self.finish_error is not None:
            raise self.finish_error

    def cleanup(self):
        self.events.append(("cleanup",))


def _publisher():
    return MilesPublisher(
        plan=_plan(),
        source_partition=0,
        tensors={"native": FakeTensor((4, 4))},
        aliases={"native": "model.layers.0.mlp.gate_proj.weight"},
    )


def test_trainer_session_runs_every_group_and_reports_only_after_finish():
    client = FakeTrainerClient()
    rendezvous = FakeRendezvous()
    publisher = _publisher()
    session = MilesTrainerSession(
        client=client,
        rendezvous=rendezvous,
        publisher=publisher,
        source_partition=0,
        worker_id="trainer-worker-0",
        layer_groups=(
            ("model.layers.0.mlp.gate_proj.weight",),
            ("model.layers.1.mlp.down_proj.weight",),
        ),
    )
    session.prepare()

    session.run_round(version="version-1", operation_id="operation-1")

    assert [event[0] for event in client.events] == [
        "initialize",
        "groups",
        "compute",
        "start",
        "publish",
        "publish",
        "finish",
    ]
    assert client.events[-1] == ("finish", "version-1", "operation-1")
    assert rendezvous.reported == []


def test_trainer_session_factory_passes_the_frozen_topology_and_abi(monkeypatch):
    captured = {}
    lane_stream = object()

    class Client(FakeTrainerClient):
        def __init__(self, **kwargs):
            captured.update(kwargs)
            super().__init__()

    monkeypatch.setattr(miles_integration, "RefitClientTrainer", Client)
    monkeypatch.setattr(
        miles_integration,
        "_collective_streams",
        lambda streams, *, device: [lane_stream],
    )
    rendezvous = FakeRendezvous()

    session = MilesTrainerSession.create(
        rendezvous=rendezvous,
        topology=_topology(),
        publisher=_publisher(),
        source_partition=0,
        slot_id="trainer-pp0-tp0",
        worker_id="trainer-worker-0",
        index_in_role=0,
    )
    session.prepare()

    assert captured["rendezvous"] is rendezvous
    assert captured["trainer_slots"] == list(_topology().trainer_slots)
    assert captured["generator_slots"] == list(_topology().generator_slots)
    assert captured["source_partition_count"] == 2
    assert captured["m2n_abi_version"] == "nccl-m2n-2.30.7"
    assert captured["receiver_protocol"] == _topology().receiver_protocol
    assert captured["device"] == "cuda:0"
    assert captured["streams"] == [lane_stream]


def test_default_collective_streams_honor_the_shared_stream_count(monkeypatch):
    created = []

    class DeviceContext:
        def __enter__(self):
            return None

        def __exit__(self, *_):
            return None

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device", lambda device: DeviceContext())
    monkeypatch.setenv("MX_NCCL_REFIT_NUM_STREAMS", "3")
    monkeypatch.setattr(
        torch.cuda,
        "Stream",
        lambda *, device: created.append(device) or f"stream-{len(created)}",
    )

    streams = miles_integration._collective_streams(None, device="cuda:0")

    assert streams == ["stream-1", "stream-2", "stream-3"]
    assert created == ["cuda:0", "cuda:0", "cuda:0"]


def test_trainer_session_factory_rejects_plan_meshes_outside_topology():
    plan = _plan()
    plan.bulk[0] = ParamPlan(
        name=plan.bulk[0].name,
        global_shape=plan.bulk[0].global_shape,
        dtype=plan.bulk[0].dtype,
        partition_id=plan.bulk[0].partition_id,
        src_mesh=MeshSpec(shape=(2,), rank_offset=1),
        src_placements=plan.bulk[0].src_placements,
        dst_mesh=plan.bulk[0].dst_mesh,
        dst_placements=plan.bulk[0].dst_placements,
    )
    publisher = MilesPublisher(
        plan=plan,
        source_partition=0,
        tensors={"native": FakeTensor((4, 4))},
        aliases={"native": "model.layers.0.mlp.gate_proj.weight"},
    )

    with pytest.raises(ValueError, match="src_mesh ranks"):
        MilesTrainerSession.create(
            rendezvous=FakeRendezvous(),
            topology=_topology(),
            publisher=publisher,
            source_partition=0,
            slot_id="trainer-pp0-tp0",
            worker_id="trainer-worker-0",
            index_in_role=0,
        )


def test_trainer_session_factory_rejects_a_device_mismatched_with_storage():
    with pytest.raises(ValueError, match="client device cuda:1.*storage device cuda:0"):
        MilesTrainerSession.create(
            rendezvous=FakeRendezvous(),
            topology=_topology(),
            publisher=_publisher(),
            source_partition=0,
            slot_id="trainer-pp0-tp0",
            worker_id="trainer-worker-0",
            index_in_role=0,
            device="cuda:1",
        )


def test_trainer_session_factory_resolves_bare_cuda_to_the_current_device(monkeypatch):
    captured = {}

    class Client(FakeTrainerClient):
        def __init__(self, **kwargs):
            captured.update(kwargs)
            super().__init__()

    monkeypatch.setattr(miles_integration, "RefitClientTrainer", Client)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)

    MilesTrainerSession.create(
        rendezvous=FakeRendezvous(),
        topology=_topology(),
        publisher=_publisher(),
        source_partition=0,
        slot_id="trainer-pp0-tp0",
        worker_id="trainer-worker-0",
        index_in_role=0,
        device=torch.device("cuda"),
    )

    assert captured["device"] == "cuda:0"


def test_trainer_session_orders_the_producer_stream_before_lane_streams(monkeypatch):
    events = []

    class Stream:
        def __init__(self, name):
            self.name = name

        def wait_event(self, event):
            events.append(("wait", self.name, event))

    class Event:
        def record(self, stream):
            events.append(("record", stream.name, self))

    class Cuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def current_stream(*, device):
            assert device == "cuda:0"
            return Stream("producer")

        @staticmethod
        def Event():
            return Event()

        @staticmethod
        def ExternalStream(handle):
            return Stream(f"external-{handle}")

    Cuda.Stream = Stream

    class DeviceContext:
        def __enter__(self):
            return None

        def __exit__(self, *_):
            return None

    Cuda.device = staticmethod(lambda _device: DeviceContext())
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=Cuda))
    client = FakeTrainerClient()
    lane_streams = [Stream("lane-0"), Stream("lane-1")]
    session = MilesTrainerSession(
        client=client,
        rendezvous=FakeRendezvous(),
        publisher=_publisher(),
        source_partition=0,
        worker_id="trainer-worker-0",
        streams=lane_streams,
    )
    session.prepare()

    session.run_round(version="version-1", operation_id="operation-1")

    assert events[0][:2] == ("record", "producer")
    assert [event[:2] for event in events[1:]] == [
        ("wait", "lane-0"),
        ("wait", "lane-1"),
    ]
    assert events[1][2] is events[0][2]
    assert events[2][2] is events[0][2]


def test_trainer_session_orders_the_producer_before_the_default_lane_stream(
    monkeypatch,
):
    events = []

    class Stream:
        def __init__(self, name):
            self.name = name

        def wait_event(self, event):
            events.append(("wait", self.name, event))

    class Event:
        def record(self, stream):
            events.append(("record", stream.name, self))

    class Cuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def current_stream(*, device):
            assert device == "cuda:0"
            return Stream("producer")

        @staticmethod
        def default_stream(*, device):
            assert device == "cuda:0"
            return Stream("default")

        @staticmethod
        def Event():
            return Event()

    Cuda.Stream = Stream

    class DeviceContext:
        def __enter__(self):
            return None

        def __exit__(self, *_):
            return None

    Cuda.device = staticmethod(lambda _device: DeviceContext())
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=Cuda))
    client = FakeTrainerClient()
    session = MilesTrainerSession(
        client=client,
        rendezvous=FakeRendezvous(),
        publisher=_publisher(),
        source_partition=0,
        worker_id="trainer-worker-0",
    )
    session.prepare()

    session.run_round(version="version-1", operation_id="operation-1")

    assert events[0][:2] == ("record", "producer")
    assert events[1][:2] == ("wait", "default")
    assert events[1][2] is events[0][2]


def test_trainer_session_preserves_the_round_error_while_aborting_reporting_and_closing():
    original = RuntimeError("collective publish failed")
    client = FakeTrainerClient(publish_error=original)
    rendezvous = FakeRendezvous(report_error=RuntimeError("report failed"))
    session = MilesTrainerSession(
        client=client,
        rendezvous=rendezvous,
        publisher=_publisher(),
        source_partition=0,
        worker_id="trainer-worker-0",
        layer_groups=(
            ("model.layers.0.mlp.gate_proj.weight",),
            ("model.layers.1.mlp.down_proj.weight",),
        ),
    )
    session.prepare()

    with pytest.raises(RuntimeError, match="collective publish failed") as caught:
        session.run_round(version="version-1", operation_id="operation-1")

    assert caught.value is original
    assert client.events[-1] == ("cleanup",)
    assert rendezvous.reported[0]["succeeded"] is False
    assert "collective publish failed" in rendezvous.reported[0]["message"]
    assert rendezvous.closed == 1


def test_trainer_session_falls_back_to_an_idempotent_finish_failure_report():
    original = RuntimeError("finish reported this failure")
    client = FakeTrainerClient(finish_error=original)
    rendezvous = FakeRendezvous()
    session = MilesTrainerSession(
        client=client,
        rendezvous=rendezvous,
        publisher=_publisher(),
        source_partition=0,
        worker_id="trainer-worker-0",
        layer_groups=(
            ("model.layers.0.mlp.gate_proj.weight",),
            ("model.layers.1.mlp.down_proj.weight",),
        ),
    )
    session.prepare()

    with pytest.raises(RuntimeError, match="finish reported") as caught:
        session.run_round(version="version-1", operation_id="operation-1")

    assert caught.value is original
    assert len(rendezvous.reported) == 1
    assert rendezvous.reported[0]["succeeded"] is False
    assert "finish reported" in rendezvous.reported[0]["message"]
    assert rendezvous.closed == 1
