# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

import modelexpress_rl.collective.integrations.miles as miles_integration
from modelexpress_rl import refit_collective_pb2 as collective_pb
from modelexpress_rl.collective import (
    CollectiveRendezvous,
    MeshSpec,
    ParamPlan,
    Placement,
    RendezvousError,
    ReshardPlan,
    Role,
)
from modelexpress_rl.collective.integrations.miles import (
    CollectiveTopology,
    MilesPublisher,
    MilesTrainerSession,
    MilesTransferCoordinator,
)
from modelexpress_rl.collective.integrations.miles_topology import (
    MilesTensorSpec,
    MilesTrainerTopology,
    build_miles_reshard_plan,
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


def _reordered_miles_plan():
    world_rank_by_coordinate = {
        (0, 0): 30,
        (0, 1): 10,
        (1, 0): 40,
        (1, 1): 20,
    }
    topologies = [
        MilesTrainerTopology(
            world_rank=world_rank_by_coordinate[pp_rank, ep_rank],
            pp_rank=pp_rank,
            pp_size=2,
            tp_rank=0,
            tp_size=1,
            cp_rank=0,
            cp_size=1,
            dense_dp_rank=ep_rank,
            dense_dp_size=2,
            ep_rank=ep_rank,
            ep_size=2,
            etp_rank=0,
            etp_size=1,
            expert_dp_rank=0,
            expert_dp_size=1,
            independent_dp_rank=0,
            independent_dp_size=1,
        )
        for pp_rank in range(2)
        for ep_rank in range(2)
    ]
    tensor_specs = []
    update_units = []
    for pp_rank in range(2):
        fc1_names = tuple(
            f"native.layers.{pp_rank}.experts.ep{ep_rank}.linear_fc1.weight"
            for ep_rank in range(2)
        )
        fc2_names = tuple(
            f"native.layers.{pp_rank}.experts.ep{ep_rank}.linear_fc2.weight"
            for ep_rank in range(2)
        )
        update_units.extend((fc1_names, fc2_names))
        for ep_rank in range(2):
            world_rank = world_rank_by_coordinate[pp_rank, ep_rank]
            for canonical_suffix, source_name in (
                ("gate_proj.weight", fc1_names[ep_rank]),
                ("up_proj.weight", fc1_names[ep_rank]),
                ("down_proj.weight", fc2_names[ep_rank]),
            ):
                tensor_specs.append(
                    MilesTensorSpec(
                        world_rank=world_rank,
                        source_name=source_name,
                        canonical_name=(
                            f"model.layers.{pp_rank}.mlp.experts.{canonical_suffix}"
                        ),
                        family="routed_expert",
                        global_shape=(8, 4),
                    )
                )
    return build_miles_reshard_plan(
        topologies,
        tensor_specs,
        update_units=update_units,
        rollout_engine_sizes=(2,),
        pp_wave_size=1,
    )


class _MembershipRoundTripStub:
    def __init__(self, *, corrupt_slot=None):
        self.requests = []
        self.corrupt_slot = corrupt_slot

    def JoinCollectiveGroup(self, request, timeout):
        request = collective_pb.JoinCollectiveGroupRequest.FromString(
            request.SerializeToString()
        )
        self.requests.append(request)
        assignments = []
        is_bootstrap_leader = False
        for lane in request.spec.lanes:
            slots = tuple(lane.trainer_slots) + tuple(lane.generator_slots)
            if request.slot_id not in slots:
                continue
            rank_in_lane = slots.index(request.slot_id)
            if (
                request.slot_id == self.corrupt_slot
                and lane.kind == collective_pb.LANE_KIND_RESHARD
            ):
                rank_in_lane += 1
            assignments.append(
                collective_pb.LaneAssignment(
                    lane_id=lane.lane_id,
                    kind=lane.kind,
                    rank_in_lane=rank_in_lane,
                    world_size=len(slots),
                )
            )
            is_bootstrap_leader |= rank_in_lane == 0
        response = collective_pb.CollectiveGroupMembership(
            group_id="group-1",
            epoch=1,
            assignments=assignments,
            state=collective_pb.COLLECTIVE_GROUP_STATE_READY,
            is_bootstrap_leader=is_bootstrap_leader,
        )
        return collective_pb.CollectiveGroupMembership.FromString(
            response.SerializeToString()
        )


def _membership_rendezvous(*, corrupt_slot=None):
    rendezvous = CollectiveRendezvous.__new__(CollectiveRendezvous)
    rendezvous._stub = _MembershipRoundTripStub(corrupt_slot=corrupt_slot)
    rendezvous._ensure_worker_registration = lambda *args, **kwargs: None
    rendezvous._bounded_rpc_timeout = lambda *args, **kwargs: 1.0
    return rendezvous


def _replace_first_route(miles_plan, **changes):
    return replace(
        miles_plan,
        routes=(
            replace(miles_plan.routes[0], **changes),
            *miles_plan.routes[1:],
        ),
    )


def _bool_route_owner_plan(*, bool_route_owner, bool_named_owner):
    miles_plan = _reordered_miles_plan()
    routes = []
    for route in miles_plan.routes:
        routes.append(
            replace(
                route,
                source_world_ranks=tuple(
                    (
                        True
                        if bool_route_owner and world_rank == 30
                        else 1
                        if world_rank == 30
                        else world_rank
                    )
                    for world_rank in route.source_world_ranks
                ),
                source_names_by_world=tuple(
                    (
                        (
                            True
                            if bool_named_owner and world_rank == 30
                            else 1
                            if world_rank == 30
                            else world_rank
                        ),
                        source_names,
                    )
                    for world_rank, source_names in route.source_names_by_world
                ),
            )
        )
    return replace(
        miles_plan,
        trainer_lanes=((1, 10), (40, 20)),
        routes=tuple(routes),
    )


def test_miles_topology_projects_reordered_world_ranks_through_membership():
    miles_plan = _reordered_miles_plan()
    slots_by_world_rank = {
        10: "trainer-actor-b",
        20: "trainer-actor-d",
        30: "trainer-actor-a",
        40: "trainer-actor-c",
    }
    topology = CollectiveTopology.from_miles_reshard_plan(
        model_name="deepseek-v4",
        miles_plan=miles_plan,
        trainer_slots_by_world_rank=slots_by_world_rank,
        generator_slots=("generator-1", "generator-0"),
        m2n_abi_version="miles-native-v1",
    )

    assert miles_plan.trainer_lanes == ((30, 10), (40, 20))
    assert topology.trainer_slots == (
        "trainer-actor-a",
        "trainer-actor-b",
        "trainer-actor-c",
        "trainer-actor-d",
    )
    assert topology.generator_slots == ("generator-1", "generator-0")
    assert [lane.trainer_slots for lane in topology.lanes()] == [
        ("trainer-actor-a", "trainer-actor-b"),
        ("trainer-actor-c", "trainer-actor-d"),
        topology.trainer_slots,
    ]

    rendezvous = _membership_rendezvous()
    memberships = {}
    for world_rank, slot_id in slots_by_world_rank.items():
        memberships[world_rank] = rendezvous.join(
            model_name=topology.model_name,
            trainer_slots=list(topology.trainer_slots),
            generator_slots=list(topology.generator_slots),
            lanes=topology.lanes(),
            slot_id=slot_id,
            worker_id=f"worker-{world_rank}",
            role=Role.TRAINER,
            index_in_role=topology.trainer_slots.index(slot_id),
            plan_digest="plan-digest",
        )

    requests_by_slot = {
        request.slot_id: request for request in rendezvous._stub.requests
    }
    assert {
        slot_id: requests_by_slot[slot_id].index_in_role
        for slot_id in topology.trainer_slots
    } == {
        slot_id: actor_ordinal
        for actor_ordinal, slot_id in enumerate(topology.trainer_slots)
    }
    for partition_id, lane_world_ranks in enumerate(miles_plan.trainer_lanes):
        for expected_rank, world_rank in enumerate(lane_world_ranks):
            assert (
                memberships[world_rank].lane(partition_id).rank_in_lane == expected_rank
            )
            assert (
                memberships[world_rank].broadcast_lane.rank_in_lane
                == partition_id * len(lane_world_ranks) + expected_rank
            )
    entries_by_name = {entry.name: entry for entry in miles_plan.plan.bulk}
    for route in miles_plan.routes:
        source_ranks = entries_by_name[route.canonical_name].src_mesh.ranks()
        assert tuple(
            memberships[world_rank].lane(route.partition_id).rank_in_lane
            for world_rank in route.source_world_ranks
        ) == tuple(source_ranks)


def test_miles_topology_rejects_a_server_rank_outside_the_projection():
    topology = CollectiveTopology.from_miles_reshard_plan(
        model_name="deepseek-v4",
        miles_plan=_reordered_miles_plan(),
        trainer_slots_by_world_rank={
            10: "trainer-actor-b",
            20: "trainer-actor-d",
            30: "trainer-actor-a",
            40: "trainer-actor-c",
        },
        generator_slots=("generator-0",),
        m2n_abi_version="miles-native-v1",
    )
    rendezvous = _membership_rendezvous(corrupt_slot="trainer-actor-b")

    with pytest.raises(RendezvousError, match="disagree"):
        rendezvous.join(
            model_name=topology.model_name,
            trainer_slots=list(topology.trainer_slots),
            generator_slots=list(topology.generator_slots),
            lanes=topology.lanes(),
            slot_id="trainer-actor-b",
            worker_id="worker-10",
            role=Role.TRAINER,
            index_in_role=topology.trainer_slots.index("trainer-actor-b"),
            plan_digest="plan-digest",
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda plan: replace(
                plan,
                trainer_lanes=tuple(
                    tuple(reversed(lane)) for lane in plan.trainer_lanes
                ),
            ),
            "source owners do not match source mesh ranks",
        ),
        (
            lambda plan: replace(plan, trainer_lanes=((30, 10), (40, 30))),
            "duplicate world ranks",
        ),
        (
            lambda plan: replace(plan, trainer_lanes=((30, 10, 40, 20),)),
            "source partitions",
        ),
        (
            lambda plan: replace(plan, trainer_lanes=((30, 10), (40,))),
            "same number of trainers",
        ),
        (
            lambda plan: replace(plan, trainer_lanes=((True, 10), (40, 20))),
            "world ranks must be integers",
        ),
        (
            lambda plan: replace(plan, routes=plan.routes[:-1]),
            "exactly one source route",
        ),
        (
            lambda plan: replace(plan, routes=(*plan.routes, plan.routes[0])),
            "exactly one source route",
        ),
        (
            lambda plan: _replace_first_route(plan, partition_id=1),
            "route partition",
        ),
        (
            lambda plan: _replace_first_route(
                plan,
                source_world_ranks=tuple(reversed(plan.routes[0].source_world_ranks)),
                source_names_by_world=tuple(
                    reversed(plan.routes[0].source_names_by_world)
                ),
            ),
            "source owners do not match source mesh ranks",
        ),
        (
            lambda plan: _replace_first_route(
                plan,
                source_names_by_world=tuple(
                    reversed(plan.routes[0].source_names_by_world)
                ),
            ),
            "source_names_by_world owners",
        ),
        (
            lambda _plan: _bool_route_owner_plan(
                bool_route_owner=True,
                bool_named_owner=False,
            ),
            "route source world ranks must be integers",
        ),
        (
            lambda _plan: _bool_route_owner_plan(
                bool_route_owner=False,
                bool_named_owner=True,
            ),
            "source_names_by_world owners must be integers",
        ),
    ],
)
def test_miles_topology_rejects_invalid_authoritative_trainer_lanes(
    mutation,
    message,
):
    with pytest.raises(ValueError, match=message):
        CollectiveTopology.from_miles_reshard_plan(
            model_name="deepseek-v4",
            miles_plan=mutation(_reordered_miles_plan()),
            trainer_slots_by_world_rank={
                10: "trainer-b",
                20: "trainer-d",
                30: "trainer-a",
                40: "trainer-c",
            },
            generator_slots=("generator-0",),
            m2n_abi_version="miles-native-v1",
        )


@pytest.mark.parametrize(
    ("slots_by_world_rank", "message"),
    [
        (
            {10: "trainer-b", 30: "trainer-a", 40: "trainer-c"},
            "missing world ranks",
        ),
        (
            {
                10: "trainer-b",
                20: "trainer-d",
                30: "trainer-a",
                40: "trainer-c",
                50: "trainer-e",
            },
            "unexpected world ranks",
        ),
        (
            {
                10: "trainer-a",
                20: "trainer-d",
                30: "trainer-a",
                40: "trainer-c",
            },
            "duplicate trainer slots",
        ),
        (
            {
                "10": "trainer-b",
                20: "trainer-d",
                30: "trainer-a",
                40: "trainer-c",
            },
            "world ranks must be integers",
        ),
    ],
)
def test_miles_topology_rejects_incomplete_or_ambiguous_world_rank_slots(
    slots_by_world_rank,
    message,
):
    with pytest.raises(ValueError, match=message):
        CollectiveTopology.from_miles_reshard_plan(
            model_name="deepseek-v4",
            miles_plan=_reordered_miles_plan(),
            trainer_slots_by_world_rank=slots_by_world_rank,
            generator_slots=("generator-0",),
            m2n_abi_version="miles-native-v1",
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
