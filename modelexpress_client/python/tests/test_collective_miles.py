# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from types import SimpleNamespace

import pytest
import torch

import modelexpress_rl.collective.integrations.miles as miles_integration
from modelexpress_rl.collective import (
    MeshSpec,
    ParamPlan,
    Placement,
    ReshardPlan,
)
from modelexpress_rl.collective.integrations._common import (
    REPLICATED_DESTINATION_ABI,
    SHARDED_DESTINATION_ABI,
)
from modelexpress_rl.collective.integrations.miles import (
    CollectiveTopology,
    MilesPublisher,
    MilesTrainerSession,
)
from modelexpress_rl.collective.rendezvous import LaneDeclaration


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


GATE = "model.layers.0.mlp.gate_proj.weight"
DOWN = "model.layers.1.mlp.down_proj.weight"


def _entry(name, *, src_placement=None, dst_placement=None):
    # One lane: two trainer ranks (a TP2 or DP2 source), then two generators.
    return ParamPlan(
        name=name,
        global_shape=(4, 4),
        dtype="bfloat16",
        partition_id=0,
        src_mesh=MeshSpec(shape=(2,), rank_offset=0),
        src_placements=(src_placement or Placement.replicate(),),
        dst_mesh=MeshSpec(shape=(2,), rank_offset=2),
        dst_placements=(dst_placement or Placement.replicate(),),
        group_key="publish-group-0",
    )


def _plan():
    return ReshardPlan(
        bulk=[_entry(GATE), _entry(DOWN)],
        source_partition_count=1,
    )


def _tensors():
    return {GATE: FakeTensor((4, 4)), DOWN: FakeTensor((4, 4), address=0x2000)}


def _topology():
    return CollectiveTopology(
        model_name="qwen",
        trainer_slots=("trainer-0", "trainer-1"),
        generator_slots=("generator-tp0", "generator-tp1"),
        source_partition_count=1,
        m2n_abi_version="nccl-m2n-2.30.7",
    )


def _lane_declarations(topology):
    """Mirror the lane set the trainer client declares for this topology.

    ``RefitClientTrainer`` declares its lanes internally; tests that drive
    ``rendezvous.join`` directly rebuild the same declaration here.
    """
    per_lane = len(topology.trainer_slots) // topology.source_partition_count
    lanes = [
        LaneDeclaration(
            partition,
            "RESHARD",
            tuple(
                topology.trainer_slots[
                    partition * per_lane : (partition + 1) * per_lane
                ]
            ),
            tuple(topology.generator_slots),
        )
        for partition in range(topology.source_partition_count)
    ]
    lanes.append(
        LaneDeclaration(
            topology.source_partition_count,
            "BROADCAST",
            tuple(topology.trainer_slots),
            tuple(topology.generator_slots),
        )
    )
    return lanes


def test_topology_requires_an_explicit_non_empty_abi_and_partitioned_slot_order():
    with pytest.raises(ValueError, match="m2n_abi_version"):
        CollectiveTopology(
            model_name="qwen",
            trainer_slots=("t0",),
            generator_slots=("g0",),
            source_partition_count=1,
            m2n_abi_version=" ",
        )

    lanes = _lane_declarations(_topology())

    # One reshard lane carries every trainer rank, then the broadcast lane.
    assert lanes[0].trainer_slots == ("trainer-0", "trainer-1")
    assert lanes[0].kind == "RESHARD"
    assert lanes[1].kind == "BROADCAST"
    assert len(lanes) == 2


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


def test_miles_publisher_maps_canonical_names_to_the_rank_local_buffers():
    plan = _plan()
    tensors = _tensors()
    publisher = MilesPublisher(plan=plan, source_partition=0, tensors=tensors)
    plan.bulk.clear()

    captured = publisher.capture()

    assert captured.parameter_names() == [GATE, DOWN]
    assert publisher.parameter_names() == captured.parameter_names()
    specs = publisher.local_params()
    assert list(specs) == [GATE, DOWN]
    assert specs[GATE].base is tensors[GATE]
    assert specs[DOWN].base is tensors[DOWN]


def test_miles_publisher_derives_wire_order_from_the_plan_not_dict_order():
    plan = ReshardPlan(
        bulk=[_entry("model.a"), _entry("model.b")],
        source_partition_count=1,
    )
    first = FakeTensor((4, 4))
    second = FakeTensor((4, 4), address=0x2000)
    publisher = MilesPublisher(
        plan=plan,
        source_partition=0,
        tensors={"model.b": second, "model.a": first},
    )

    specs = publisher.local_params()

    assert list(specs) == ["model.a", "model.b"]
    assert specs["model.a"].base is first
    assert specs["model.b"].base is second


def _one_entry_plan(**kwargs):
    defaults = {
        "name": "model.a",
        "global_shape": (4, 4),
        "dtype": "bfloat16",
        "partition_id": 0,
        "src_mesh": MeshSpec(shape=(2,), rank_offset=0),
        "src_placements": (Placement.replicate(),),
        "dst_mesh": MeshSpec(shape=(2,), rank_offset=2),
        "dst_placements": (Placement.replicate(),),
    }
    defaults.update(kwargs)
    return ReshardPlan(bulk=[ParamPlan(**defaults)], source_partition_count=1)


@pytest.mark.parametrize(
    ("src_mesh", "rank"),
    [
        (MeshSpec((2,)), 1),
        (MeshSpec((2, 2)), 3),
    ],
)
def test_miles_publisher_validates_this_ranks_slice_of_a_replicated_source(
    src_mesh, rank
):
    plan = _one_entry_plan(
        src_mesh=src_mesh,
        src_placements=(Placement.replicate(),) * len(src_mesh.shape),
        dst_mesh=MeshSpec((2,), rank_offset=src_mesh.size),
    )
    tensor = FakeTensor((4, 4))

    publisher = MilesPublisher(
        plan=plan, source_partition=0, tensors={"model.a": tensor}, source_rank=rank
    )

    assert publisher.local_params()["model.a"].base is tensor
    with pytest.raises(ValueError, match="expected local shape"):
        MilesPublisher(
            plan=plan,
            source_partition=0,
            tensors={"model.a": FakeTensor((2, 4))},
            source_rank=rank,
        )
    with pytest.raises(ValueError, match="source_rank must be in"):
        MilesPublisher(
            plan=plan,
            source_partition=0,
            tensors={"model.a": tensor},
            source_rank=src_mesh.size,
        )


def test_miles_publisher_rejects_sharded_sources():
    for src_mesh, placements in (
        (MeshSpec((2,)), (Placement.shard(0),)),
        (MeshSpec((2, 2)), (Placement.replicate(), Placement.shard(1))),
    ):
        plan = _one_entry_plan(
            src_mesh=src_mesh,
            src_placements=placements,
            dst_mesh=MeshSpec((2,), rank_offset=src_mesh.size),
        )
        with pytest.raises(
            ValueError, match="gathered \\(replicated\\) trainer source"
        ):
            MilesPublisher(
                plan=plan,
                source_partition=0,
                tensors={"model.a": FakeTensor((4, 4))},
            )


def test_miles_publisher_rejects_meshes_outside_the_one_lane_geometry():
    with pytest.raises(ValueError, match="innermost axis"):
        MilesPublisher(
            plan=_one_entry_plan(
                dst_mesh=MeshSpec((2, 2), rank_offset=2),
                dst_placements=(Placement.shard(0), Placement.replicate()),
            ),
            source_partition=0,
            tensors={"model.a": FakeTensor((4, 4))},
        )
    with pytest.raises(ValueError, match="start at lane rank 0"):
        MilesPublisher(
            plan=_one_entry_plan(
                src_mesh=MeshSpec((1,), rank_offset=1),
                dst_mesh=MeshSpec((2,), rank_offset=2),
            ),
            source_partition=0,
            tensors={"model.a": FakeTensor((4, 4))},
        )
    with pytest.raises(ValueError, match="right after the trainer ranks"):
        MilesPublisher(
            plan=_one_entry_plan(dst_mesh=MeshSpec((2,), rank_offset=1)),
            source_partition=0,
            tensors={"model.a": FakeTensor((4, 4))},
        )
    with pytest.raises(ValueError, match="one reshard lane"):
        MilesPublisher(
            plan=ReshardPlan(
                bulk=[_entry("model.a")],
                source_partition_count=2,
            ),
            source_partition=0,
            tensors={"model.a": FakeTensor((4, 4))},
        )
    mixed = ReshardPlan(
        bulk=[
            _entry("model.a"),
            ParamPlan(
                name="model.b",
                global_shape=(4, 4),
                dtype="bfloat16",
                partition_id=0,
                src_mesh=MeshSpec((2, 1)),
                src_placements=(Placement.replicate(), Placement.replicate()),
                dst_mesh=MeshSpec((2,), rank_offset=2),
                dst_placements=(Placement.replicate(),),
            ),
        ],
        source_partition_count=1,
    )
    with pytest.raises(ValueError, match="share one source mesh"):
        MilesPublisher(
            plan=mixed,
            source_partition=0,
            tensors={"model.a": FakeTensor((4, 4)), "model.b": FakeTensor((4, 4))},
        )


def test_frozen_plan_requires_the_sharded_abi_for_sharded_destinations():
    plan = _one_entry_plan(dst_placements=(Placement.shard(0),))
    publisher = MilesPublisher(
        plan=plan, source_partition=0, tensors={"model.a": FakeTensor((4, 4))}
    )
    topology = CollectiveTopology(
        model_name="qwen",
        trainer_slots=("trainer-0", "trainer-1"),
        generator_slots=("generator-0", "generator-1"),
        source_partition_count=1,
        m2n_abi_version=REPLICATED_DESTINATION_ABI,
    )

    with pytest.raises(ValueError, match="sharded destinations requires"):
        publisher.validate_topology(topology)
    publisher.validate_topology(
        CollectiveTopology(
            model_name="qwen",
            trainer_slots=topology.trainer_slots,
            generator_slots=topology.generator_slots,
            source_partition_count=1,
            m2n_abi_version=SHARDED_DESTINATION_ABI,
        )
    )


def test_miles_publisher_rejects_incomplete_tensor_coverage_and_non_bulk_plans():
    with pytest.raises(ValueError, match="exactly cover partition 0"):
        MilesPublisher(
            plan=_plan(),
            source_partition=0,
            tensors={GATE: FakeTensor((4, 4)), "wrong": FakeTensor((4, 4))},
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
        MilesPublisher(plan=plan, source_partition=0, tensors=_tensors())


def test_miles_publisher_rejects_cpu_and_mixed_device_storage():
    cpu = _tensors()
    cpu[GATE].device = "cpu"
    with pytest.raises(ValueError, match="indexed CUDA device"):
        MilesPublisher(plan=_plan(), source_partition=0, tensors=cpu)

    unindexed = _tensors()
    unindexed[GATE].device = "cuda"
    with pytest.raises(ValueError, match="indexed CUDA device"):
        MilesPublisher(plan=_plan(), source_partition=0, tensors=unindexed)

    plan = ReshardPlan(
        bulk=[_entry("model.a"), _entry("model.b")],
        source_partition_count=1,
    )
    first = FakeTensor((4, 4))
    second = FakeTensor((4, 4), address=0x2000)
    second.device = "cuda:1"
    with pytest.raises(ValueError, match="share one CUDA device"):
        MilesPublisher(
            plan=plan,
            source_partition=0,
            tensors={"model.a": first, "model.b": second},
        )


def test_mixed_device_error_lists_devices_in_numeric_order():
    first = FakeTensor((4, 4))
    first.device = "cuda:2"
    second = FakeTensor((4, 4), address=0x2000)
    second.device = "cuda:10"
    plan = ReshardPlan(
        bulk=[
            _entry("model.a"),
            _entry("model.b"),
        ],
        source_partition_count=1,
    )

    with pytest.raises(ValueError, match=r"cuda:2', 'cuda:10"):
        MilesPublisher(
            plan=plan,
            source_partition=0,
            tensors={"model.a": first, "model.b": second},
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
    tensors = _tensors()
    publisher = MilesPublisher(plan=_plan(), source_partition=0, tensors=tensors)
    change(tensors[GATE])

    with pytest.raises(RuntimeError, match=message):
        publisher.start_new_round("version-2")


class FakeRendezvous:
    def __init__(self):
        self.reported = []
        self.closed = 0

    def report(self, **kwargs):
        self.reported.append(kwargs)

    def close(self):
        self.closed += 1


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
    return MilesPublisher(plan=_plan(), source_partition=0, tensors=_tensors())


def _session(client, rendezvous=None, **kwargs):
    return MilesTrainerSession(
        client=client,
        rendezvous=FakeRendezvous() if rendezvous is None else rendezvous,
        publisher=_publisher(),
        source_partition=0,
        device="cuda:0",
        **kwargs,
    )


def test_trainer_session_runs_every_group_in_order_and_reports_nothing():
    client = FakeTrainerClient()
    rendezvous = FakeRendezvous()
    session = _session(
        client,
        rendezvous,
        layer_groups=(
            (GATE,),
            (DOWN,),
        ),
    )
    session.prepare()

    session.begin_round(version="version-1")
    session.publish_group(version="version-1", layer_group_id=0)
    session.publish_group(version="version-1", layer_group_id=1)
    session.finish_round(version="version-1")

    assert [event[0] for event in client.events] == [
        "initialize",
        "groups",
        "compute",
        "start",
        "publish",
        "publish",
        "finish",
    ]
    assert client.events[-1] == ("finish", "version-1", None)
    assert rendezvous.reported == []


def test_trainer_session_forwards_server_operation_to_finish():
    client = FakeTrainerClient()
    session = _session(client)
    session.prepare()
    session.begin_round(version="2")
    session.finish_round(version="2", operation_id="server-issued-id")
    assert client.events[-1] == ("finish", "2", "server-issued-id")


def test_trainer_session_rejects_calls_outside_an_open_round():
    client = FakeTrainerClient()
    session = _session(client)
    session.prepare()

    with pytest.raises(RuntimeError, match="begin_round must run before"):
        session.publish_group(version="version-1", layer_group_id=0)
    with pytest.raises(RuntimeError, match="begin_round must run before"):
        session.finish_round(version="version-1")

    session.begin_round(version="version-1")
    with pytest.raises(RuntimeError, match="already in flight"):
        session.begin_round(version="version-1")
    with pytest.raises(ValueError, match="round in flight"):
        session.finish_round(version="version-2")
    with pytest.raises(ValueError, match="declared publish groups"):
        session.publish_group(version="version-1", layer_group_id=7)
    session.finish_round(version="version-1")


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
        slot_id="trainer-0",
        worker_id="trainer-worker-0",
        index_in_role=0,
    )
    session.prepare()

    assert captured["rendezvous"] is rendezvous
    assert captured["trainer_slots"] == list(_topology().trainer_slots)
    assert captured["generator_slots"] == list(_topology().generator_slots)
    assert captured["source_partition_count"] == 1
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
        src_mesh=plan.bulk[0].src_mesh,
        src_placements=plan.bulk[0].src_placements,
        dst_mesh=MeshSpec(shape=(3,), rank_offset=2),
        dst_placements=plan.bulk[0].dst_placements,
    )
    plan.bulk[1] = ParamPlan(
        name=plan.bulk[1].name,
        global_shape=plan.bulk[1].global_shape,
        dtype=plan.bulk[1].dtype,
        partition_id=plan.bulk[1].partition_id,
        src_mesh=plan.bulk[1].src_mesh,
        src_placements=plan.bulk[1].src_placements,
        dst_mesh=MeshSpec(shape=(3,), rank_offset=2),
        dst_placements=plan.bulk[1].dst_placements,
    )
    publisher = MilesPublisher(plan=plan, source_partition=0, tensors=_tensors())

    with pytest.raises(ValueError, match="one generator slot per"):
        MilesTrainerSession.create(
            rendezvous=FakeRendezvous(),
            topology=_topology(),
            publisher=publisher,
            source_partition=0,
            slot_id="trainer-0",
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
            slot_id="trainer-0",
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
        slot_id="trainer-0",
        worker_id="trainer-worker-0",
        index_in_role=0,
        device=torch.device("cuda"),
    )

    assert captured["device"] == "cuda:0"


def _install_fake_cuda(monkeypatch, events):
    """Fake torch.cuda that records producer/lane stream ordering events."""

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
    return Stream


def test_trainer_session_orders_the_producer_stream_before_lane_streams(monkeypatch):
    events = []
    Stream = _install_fake_cuda(monkeypatch, events)
    client = FakeTrainerClient()
    lane_streams = [Stream("lane-0"), Stream("lane-1")]
    session = _session(client, streams=lane_streams)
    session.prepare()

    session.begin_round(version="version-1")

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
    _install_fake_cuda(monkeypatch, events)
    client = FakeTrainerClient()
    session = _session(client, streams=[None])
    session.prepare()

    session.begin_round(version="version-1")

    assert events[0][:2] == ("record", "producer")
    assert events[1][:2] == ("wait", "default")
    assert events[1][2] is events[0][2]


def test_trainer_session_rejects_layer_groups_that_reorder_the_plan():
    with pytest.raises(ValueError, match="pure reordering"):
        _session(
            FakeTrainerClient(),
            layer_groups=(
                (DOWN,),
                (GATE,),
            ),
        )


def test_trainer_session_preserves_the_round_error_while_aborting_and_closing():
    original = RuntimeError("collective publish failed")
    client = FakeTrainerClient(publish_error=original)
    rendezvous = FakeRendezvous()
    session = _session(
        client,
        rendezvous,
        layer_groups=(
            (GATE,),
            (DOWN,),
        ),
    )
    session.prepare()

    session.begin_round(version="version-1")
    with pytest.raises(RuntimeError, match="collective publish failed") as caught:
        session.publish_group(version="version-1", layer_group_id=0)

    assert caught.value is original
    assert client.events[-1] == ("cleanup",)
    assert rendezvous.reported == []
    assert rendezvous.closed == 1

    with pytest.raises(RuntimeError, match="session is closed"):
        session.begin_round(version="version-2")


def test_trainer_session_preserves_the_prepare_error_while_closing():
    original = RuntimeError("collective initialize failed")

    class Client(FakeTrainerClient):
        def initialize(self, publisher, *, source_partition):
            raise original

    client = Client()
    rendezvous = FakeRendezvous()
    session = _session(client, rendezvous)

    with pytest.raises(RuntimeError, match="collective initialize failed") as caught:
        session.prepare()

    assert caught.value is original
    assert client.events[-1] == ("cleanup",)
    assert rendezvous.reported == []
    assert rendezvous.closed == 1

    with pytest.raises(RuntimeError, match="session is closed"):
        session.prepare()


def test_trainer_session_closes_after_a_finish_failure():
    original = RuntimeError("finish reported this failure")
    client = FakeTrainerClient(finish_error=original)
    rendezvous = FakeRendezvous()
    session = _session(
        client,
        rendezvous,
        layer_groups=(
            (GATE,),
            (DOWN,),
        ),
    )
    session.prepare()

    session.begin_round(version="version-1")
    session.publish_group(version="version-1", layer_group_id=0)
    session.publish_group(version="version-1", layer_group_id=1)
    with pytest.raises(RuntimeError, match="finish reported") as caught:
        session.finish_round(version="version-1")

    assert caught.value is original
    assert client.events[-1] == ("cleanup",)
    assert rendezvous.reported == []
    assert rendezvous.closed == 1


def test_trainer_session_close_is_one_shot_even_when_teardown_raises():
    calls = []

    class Client(FakeTrainerClient):
        def cleanup(self):
            calls.append("cleanup")
            raise RuntimeError("cleanup failed")

    class Rendezvous(FakeRendezvous):
        def close(self):
            calls.append("rendezvous")
            raise RuntimeError("rendezvous close failed")

    session = _session(Client(), Rendezvous())

    with pytest.raises(RuntimeError, match="rendezvous close failed"):
        session.close()

    assert session._closed is True
    assert calls == ["cleanup", "rendezvous"]

    session.close()

    assert calls == ["cleanup", "rendezvous"]
