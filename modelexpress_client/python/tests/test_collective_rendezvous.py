# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rendezvous behaviour for the NCCL M2N collective path.

Driven against a fake stub rather than a live server. The behaviour worth
pinning here is what the client does when things go wrong, because every one of
these cases otherwise presents as a hung collective: a group that never becomes
READY, an epoch that moves underneath a waiter, and a bootstrap identifier of
the wrong size.
"""

import grpc
import pytest
from modelexpress_rl import refit_collective_pb2 as pb
from modelexpress_rl.collective import rendezvous as rz
from modelexpress_rl.collective.rendezvous import (
    BootstrapFenceTimeoutError,
    CollectiveRendezvous,
    EpochChangedError,
    GroupNotReadyError,
    Membership,
    RendezvousError,
)
from modelexpress_rl.collective.types import Role


class FakeStub:
    """Stands in for RefitCollectiveServiceStub."""

    def __init__(self, groups=None, membership=None, get_error=None):
        self._groups = list(groups or [])
        self._membership = membership
        self._get_error = get_error
        self.joined = []
        self.registered = []
        self.published = []
        self.reported = []
        self.created = []
        self.fetched = []
        self.deleted = []
        self.events = []
        self.get_calls = 0
        self.get_timeouts = []
        self.join_timeouts = []
        self.register_timeouts = []
        self.publish_timeouts = []
        self.reached_fences = []
        self.fence_timeouts = []
        self.fence_responses = []
        self.aborted_bootstraps = []
        self.abort_timeouts = []

    def CreateCollectiveTransfer(self, request, timeout=None):  # noqa: N802
        self.created.append(request)
        return pb.CollectiveTransfer(
            operation_id="op1",
            version_id=request.version_id,
            model_name=request.spec.model_name,
            idempotency_key=request.idempotency_key,
        )

    def GetCollectiveTransfer(self, request, timeout=None):  # noqa: N802
        self.fetched.append(request)
        return pb.CollectiveTransfer(operation_id=request.operation_id)

    def DeleteCollectiveTransfer(self, request, timeout=None):  # noqa: N802
        self.deleted.append(request)
        return pb.CollectiveTransfer(operation_id=request.operation_id)

    def JoinCollectiveGroup(self, request, timeout=None):  # noqa: N802 - gRPC naming
        self.events.append("join")
        self.joined.append(request)
        self.join_timeouts.append(timeout)
        return self._membership

    def RegisterWorker(self, request, timeout=None):  # noqa: N802
        self.events.append("register")
        self.registered.append(request)
        self.register_timeouts.append(timeout)
        return request.worker

    def GetCollectiveGroup(self, request, timeout=None):  # noqa: N802
        self.get_calls += 1
        self.get_timeouts.append(timeout)
        if self._get_error is not None:
            raise self._get_error
        index = min(self.get_calls - 1, len(self._groups) - 1)
        return self._groups[index]

    def PublishGroupBootstrap(self, request, timeout=None):  # noqa: N802
        self.published.append(request)
        self.publish_timeouts.append(timeout)
        return pb.CollectiveGroup()

    def ReachCollectiveBootstrapFence(self, request, timeout=None):  # noqa: N802
        self.reached_fences.append(request)
        self.fence_timeouts.append(timeout)
        if self.fence_responses:
            return self.fence_responses.pop(0)
        return pb.CollectiveBootstrapFence(
            group_id=request.group_id,
            epoch=request.epoch,
            lane_id=request.lane_id,
            released=True,
            phase=request.phase,
        )

    def AbortCollectiveBootstrap(self, request, timeout=None):  # noqa: N802
        self.aborted_bootstraps.append(request)
        self.abort_timeouts.append(timeout)
        return pb.CollectiveGroup(group_id=request.group_id, epoch=request.epoch + 1)

    def ReportCollectiveTransfer(self, request, timeout=None):  # noqa: N802
        self.reported.append(request)
        return pb.CollectiveTransfer(operation_id=request.operation_id)


class TestBootstrapFence:
    def test_arrival_is_retried_idempotently_until_released(self, monkeypatch):
        stub = FakeStub()
        stub.fence_responses = [
            pb.CollectiveBootstrapFence(
                group_id="g1",
                epoch=3,
                lane_id=7,
                missing_slots=["g0", "t1"],
                phase=pb.BOOTSTRAP_FENCE_PHASE_PRE_BARRIER,
            ),
            pb.CollectiveBootstrapFence(
                group_id="g1",
                epoch=3,
                lane_id=7,
                released=True,
                phase=pb.BOOTSTRAP_FENCE_PHASE_PRE_BARRIER,
            ),
        ]
        monkeypatch.setattr(rz.time, "sleep", lambda _duration: None)

        make_rendezvous(stub).await_bootstrap_fence(
            group_id="g1",
            epoch=3,
            lane_id=7,
            slot_id="t0",
            worker_id="w0",
            timeout_s=5.0,
            poll_interval_s=0.01,
        )

        assert len(stub.reached_fences) == 2
        assert stub.reached_fences[0] == stub.reached_fences[1]
        assert stub.reached_fences[0].slot_id == "t0"
        assert stub.reached_fences[0].phase == pb.BOOTSTRAP_FENCE_PHASE_PRE_BARRIER

    def test_timeout_preserves_sorted_missing_slots(self, monkeypatch):
        stub = FakeStub()
        stub.fence_responses = [
            pb.CollectiveBootstrapFence(
                group_id="g1",
                epoch=3,
                lane_id=7,
                missing_slots=["g0", "t1"],
                phase=pb.BOOTSTRAP_FENCE_PHASE_PRE_BARRIER,
            )
        ]
        moments = iter((10.0, 10.0, 10.1, 11.1))
        monkeypatch.setattr(rz.time, "monotonic", lambda: next(moments))
        monkeypatch.setattr(rz.time, "sleep", lambda _duration: None)

        with pytest.raises(BootstrapFenceTimeoutError) as caught:
            make_rendezvous(stub).await_bootstrap_fence(
                group_id="g1",
                epoch=3,
                lane_id=7,
                slot_id="t0",
                worker_id="w0",
                timeout_s=1.0,
                poll_interval_s=0.01,
            )

        assert caught.value.missing == ["g0", "t1"]

    def test_abort_names_the_exact_admitted_generation(self):
        stub = FakeStub()
        group = make_rendezvous(stub).abort_bootstrap(
            group_id="g1",
            epoch=3,
            slot_id="t0",
            worker_id="w0",
            message="communicator timeout",
            timeout_s=2.0,
        )

        assert group.epoch == 4
        assert stub.aborted_bootstraps == [
            pb.AbortCollectiveBootstrapRequest(
                group_id="g1",
                epoch=3,
                slot_id="t0",
                worker_id="w0",
                message="communicator timeout",
            )
        ]


def lanes_for(trainer_slots, generator_slots, *, lane_count=1):
    """Declare `lane_count` reshard lanes over the trainers, plus a broadcast.

    How the trainers are split is the TEST's choice, exactly as it is the
    caller's in production. MX is never told what the split means.
    """
    per_lane = max(-(-len(trainer_slots) // max(lane_count, 1)), 1)
    lanes = [
        rz.LaneDeclaration(
            index,
            "RESHARD",
            tuple(trainer_slots[index * per_lane : (index + 1) * per_lane]),
            tuple(generator_slots),
        )
        for index in range(lane_count)
    ]
    lanes.append(
        rz.LaneDeclaration(
            lane_count, "BROADCAST", tuple(trainer_slots), tuple(generator_slots)
        )
    )
    return lanes


def make_rendezvous(stub, *, start_thread=False):
    client = CollectiveRendezvous.__new__(CollectiveRendezvous)
    client._stub = stub
    client._registration_stub = stub
    client._rpc_timeout_s = 5.0
    client._registration_ttl_s = 90
    client._registration_lock = rz.threading.Lock()
    client._registration_stop = rz.threading.Event()
    client._registration_thread = None
    client._registration = None
    client._closed = False
    if not start_thread:
        client._start_registration_renewal = lambda: None
    return client


def membership(*, epoch=1, assignments=(), leader=False):
    result = pb.CollectiveGroupMembership(
        group_id="g1", epoch=epoch, is_bootstrap_leader=leader
    )
    for lane_id, kind, rank, world_size in assignments:
        assignment = result.assignments.add()
        assignment.lane_id = lane_id
        assignment.kind = kind
        assignment.rank_in_lane = rank
        assignment.world_size = world_size
    return result


def group(
    *, epoch=1, state=pb.COLLECTIVE_GROUP_STATE_FORMING, admitted=(), lanes_ready=True
):
    g = pb.CollectiveGroup(
        group_id="g1",
        epoch=epoch,
        state=state,
        expected_trainer_slots=["t0", "t1"],
        expected_generator_slots=["g0"],
    )
    lane = g.lanes.add()
    lane.lane_id = 0
    lane.kind = pb.LANE_KIND_RESHARD
    lane.bootstrap_epoch = epoch if lanes_ready else 0
    broadcast = g.lanes.add()
    broadcast.lane_id = 1
    broadcast.kind = pb.LANE_KIND_BROADCAST
    broadcast.bootstrap_epoch = epoch if lanes_ready else 0
    for slot in admitted:
        role = (
            pb.COLLECTIVE_ROLE_TRAINER
            if slot.startswith("t")
            else pb.COLLECTIVE_ROLE_GENERATOR
        )
        p = broadcast.participants.add()
        p.slot_id = slot
        p.role = role
    return g


class TestJoin:
    def test_registration_and_join_share_one_explicit_deadline(self, monkeypatch):
        stub = FakeStub(
            membership=membership(
                assignments=[
                    (0, pb.LANE_KIND_RESHARD, 0, 2),
                    (1, pb.LANE_KIND_BROADCAST, 0, 2),
                ],
                leader=True,
            )
        )
        moments = iter((10.0, 10.0, 11.0, 12.0))
        monkeypatch.setattr(rz.time, "monotonic", lambda: next(moments))

        make_rendezvous(stub).join(
            model_name="m",
            trainer_slots=["t0"],
            generator_slots=["g0"],
            lanes=lanes_for(["t0"], ["g0"]),
            slot_id="t0",
            worker_id="w0",
            role=Role.TRAINER,
            index_in_role=0,
            plan_digest="d",
            timeout_s=3.0,
        )

        assert stub.register_timeouts == [2.0]
        assert stub.join_timeouts == [1.0]

    def test_registration_that_consumes_the_budget_prevents_join(self, monkeypatch):
        stub = FakeStub()
        moments = iter((10.0, 10.0, 10.0, 14.0))
        monkeypatch.setattr(rz.time, "monotonic", lambda: next(moments))

        with pytest.raises(TimeoutError, match="group join exhausted"):
            make_rendezvous(stub).join(
                model_name="m",
                trainer_slots=["t0"],
                generator_slots=["g0"],
                lanes=lanes_for(["t0"], ["g0"]),
                slot_id="t0",
                worker_id="w0",
                role=Role.TRAINER,
                index_in_role=0,
                plan_digest="d",
                timeout_s=3.0,
            )

        assert len(stub.registered) == 1
        assert stub.joined == []

    def test_a_trainer_declares_its_partition_and_takes_the_assigned_rank(self):
        response = pb.CollectiveGroupMembership(
            group_id="g1", epoch=3, is_bootstrap_leader=True
        )
        a = response.assignments.add()
        a.lane_id, a.kind, a.rank_in_lane, a.world_size = 0, pb.LANE_KIND_RESHARD, 0, 3
        b = response.assignments.add()
        b.lane_id, b.kind, b.rank_in_lane, b.world_size = (
            1,
            pb.LANE_KIND_BROADCAST,
            0,
            3,
        )

        stub = FakeStub(membership=response)
        result = make_rendezvous(stub).join(
            model_name="m",
            trainer_slots=["t0", "t1"],
            generator_slots=["g0"],
            lanes=lanes_for(["t0", "t1"], ["g0"]),
            slot_id="t0",
            worker_id="w0",
            role=Role.TRAINER,
            index_in_role=0,
            plan_digest="d",
        )

        assert isinstance(result, Membership)
        assert result.epoch == 3
        assert result.is_bootstrap_leader
        assert result.lane(0).rank_in_lane == 0
        assert result.broadcast_lane.lane_id == 1
        assert len(result.reshard_lanes) == 1
        assert stub.events[:2] == ["register", "join"]
        registered = stub.registered[0]
        assert registered.worker.role == rz.refit_pb2.WORKER_ROLE_TRAINER
        assert registered.ttl_seconds == 90
        sent = stub.joined[0]
        assert [lane.lane_id for lane in sent.spec.lanes] == [0, 1]
        assert sent.role == pb.COLLECTIVE_ROLE_TRAINER

    def test_a_generator_sends_no_partition(self):
        stub = FakeStub(
            membership=membership(
                assignments=[
                    (0, pb.LANE_KIND_RESHARD, 1, 2),
                    (1, pb.LANE_KIND_BROADCAST, 1, 2),
                ]
            )
        )
        make_rendezvous(stub).join(
            model_name="m",
            trainer_slots=["t0"],
            generator_slots=["g0"],
            lanes=lanes_for(["t0"], ["g0"]),
            slot_id="g0",
            worker_id="w1",
            role=Role.GENERATOR,
            index_in_role=0,
            plan_digest="d",
        )
        # A generator is placed by the lanes that declared its slot, so the
        # request has nowhere to carry a partition and no need of one.
        assert "source_partition" not in (
            pb.JoinCollectiveGroupRequest.DESCRIPTOR.fields_by_name
        )
        assert stub.registered[0].worker.role == rz.refit_pb2.WORKER_ROLE_GENERATOR

    def test_a_plan_endpoint_is_advertised_with_the_matching_digest(self):
        stub = FakeStub(
            membership=membership(
                assignments=[
                    (0, pb.LANE_KIND_RESHARD, 0, 2),
                    (1, pb.LANE_KIND_BROADCAST, 0, 2),
                ],
                leader=True,
            )
        )
        make_rendezvous(stub).join(
            model_name="m",
            trainer_slots=["t0"],
            generator_slots=["g0"],
            lanes=lanes_for(["t0"], ["g0"]),
            slot_id="t0",
            worker_id="w0",
            role=Role.TRAINER,
            index_in_role=0,
            plan_digest="abc",
            plan_endpoint="host:1234",
        )
        source = stub.joined[0].plan_source
        assert source.endpoint == "host:1234"
        # Generators verify the fetched plan against the digest MX advertises,
        # so the two must not be allowed to drift apart at the source.
        assert source.digest == "abc"
        # Advertising a plan endpoint does not put one on the worker
        # registration: registration carries identity and liveness only, and
        # the wire has no field to put an endpoint in.
        assert (
            "endpoint" not in rz.refit_pb2.WorkerRegistration.DESCRIPTOR.fields_by_name
        )

    def test_a_server_rank_disagreement_is_rejected_before_communicator_init(self):
        stub = FakeStub(
            membership=membership(
                assignments=[
                    (0, pb.LANE_KIND_RESHARD, 0, 2),
                    # Generator 0 must be rank 1 on the broadcast lane too.
                    (1, pb.LANE_KIND_BROADCAST, 0, 2),
                ]
            )
        )
        with pytest.raises(RendezvousError, match="rank mirror"):
            make_rendezvous(stub).join(
                model_name="m",
                trainer_slots=["t0"],
                generator_slots=["g0"],
                lanes=lanes_for(["t0"], ["g0"]),
                slot_id="g0",
                worker_id="w1",
                role=Role.GENERATOR,
                index_in_role=0,
                plan_digest="d",
            )

    def test_only_the_first_trainer_can_advertise_the_plan(self):
        stub = FakeStub()
        with pytest.raises(ValueError, match="only trainer index 0"):
            make_rendezvous(stub).join(
                model_name="m",
                trainer_slots=["t0", "t1"],
                generator_slots=["g0"],
                lanes=lanes_for(["t0", "t1"], ["g0"]),
                slot_id="t1",
                worker_id="w1",
                role=Role.TRAINER,
                index_in_role=1,
                plan_digest="d",
                plan_endpoint="host:1234",
            )
        assert stub.joined == []
        assert stub.registered == []

    def test_role_ordinal_is_canonicalized_from_slot_identity(self):
        stub = FakeStub(
            membership=membership(
                assignments=[
                    (0, pb.LANE_KIND_RESHARD, 1, 3),
                    (1, pb.LANE_KIND_BROADCAST, 1, 3),
                ]
            )
        )
        make_rendezvous(stub).join(
            model_name="m",
            trainer_slots=["t0", "t1"],
            generator_slots=["g0"],
            lanes=lanes_for(["t0", "t1"], ["g0"]),
            slot_id="t1",
            worker_id="w1",
            role=Role.TRAINER,
            index_in_role=0,
            plan_digest="d",
        )
        assert stub.joined[0].index_in_role == 1

    def test_equivalent_membership_orders_send_the_same_slot_ordinal(self):
        lanes = lanes_for(["t0", "t1"], ["g0"])
        first_stub = FakeStub(
            membership=membership(
                assignments=[
                    (0, pb.LANE_KIND_RESHARD, 0, 3),
                    (1, pb.LANE_KIND_BROADCAST, 0, 3),
                ],
                leader=True,
            )
        )
        second_stub = FakeStub(
            membership=membership(
                assignments=[
                    (0, pb.LANE_KIND_RESHARD, 0, 3),
                    (1, pb.LANE_KIND_BROADCAST, 0, 3),
                ],
                leader=True,
            )
        )

        make_rendezvous(first_stub).join(
            model_name="m",
            trainer_slots=["t0", "t1"],
            generator_slots=["g0"],
            lanes=lanes,
            slot_id="t0",
            worker_id="w0",
            role=Role.TRAINER,
            index_in_role=0,
            plan_digest="d",
        )
        make_rendezvous(second_stub).join(
            model_name="m",
            trainer_slots=["t1", "t0"],
            generator_slots=["g0"],
            lanes=lanes,
            slot_id="t0",
            worker_id="w0",
            role=Role.TRAINER,
            index_in_role=1,
            plan_digest="d",
        )

        first = first_stub.joined[0]
        second = second_stub.joined[0]
        assert first.index_in_role == second.index_in_role == 0
        assert (
            first.spec.expected_trainer_slots
            == second.spec.expected_trainer_slots
            == ["t0", "t1"]
        )

    @pytest.mark.parametrize(
        ("trainer_slots", "generator_slots", "lanes"),
        [
            (
                ["t,0"],
                ["g0"],
                lanes_for(["t,0"], ["g0"]),
            ),
            (
                ["t0"],
                ["g0"],
                [
                    rz.LaneDeclaration(
                        0,
                        "RESHARD",
                        ("t0",),
                        ("g,0",),
                    )
                ],
            ),
        ],
    )
    def test_comma_in_a_slot_is_rejected_before_the_rpc(
        self, trainer_slots, generator_slots, lanes
    ):
        stub = FakeStub(membership=membership())
        with pytest.raises(ValueError, match="Redis record delimiters"):
            make_rendezvous(stub).join(
                model_name="m",
                trainer_slots=trainer_slots,
                generator_slots=generator_slots,
                lanes=lanes,
                slot_id=trainer_slots[0],
                worker_id="w0",
                role=Role.TRAINER,
                index_in_role=0,
                plan_digest="d",
            )
        assert stub.events == []

    def test_one_rendezvous_cannot_register_two_worker_identities(self):
        stub = FakeStub(
            membership=membership(
                assignments=[
                    (0, pb.LANE_KIND_RESHARD, 1, 2),
                    (1, pb.LANE_KIND_BROADCAST, 1, 2),
                ]
            )
        )
        client = make_rendezvous(stub)
        kwargs = dict(
            model_name="m",
            trainer_slots=["t0"],
            generator_slots=["g0"],
            lanes=lanes_for(["t0"], ["g0"]),
            slot_id="g0",
            worker_id="w1",
            role=Role.GENERATOR,
            index_in_role=0,
            plan_digest="d",
        )
        client.join(**kwargs)
        client.join(**kwargs)
        assert [r.worker.worker_id for r in stub.registered] == ["w1", "w1"]

        with pytest.raises(RendezvousError, match="more than one worker identity"):
            client.join(**{**kwargs, "worker_id": "w2"})
        assert len(stub.joined) == 2

    def test_close_stops_the_registration_renewal_thread(self, monkeypatch):
        threads = []

        class FakeThread:
            def __init__(self, *, target, name, daemon):
                self.target = target
                self.name = name
                self.daemon = daemon
                self.started = False
                self.joined = False
                threads.append(self)

            def start(self):
                self.started = True

            def join(self):
                self.joined = True

        monkeypatch.setattr(rz.threading, "Thread", FakeThread)
        stub = FakeStub(
            membership=membership(
                assignments=[
                    (0, pb.LANE_KIND_RESHARD, 1, 2),
                    (1, pb.LANE_KIND_BROADCAST, 1, 2),
                ]
            )
        )
        client = make_rendezvous(stub, start_thread=True)
        client.join(
            model_name="m",
            trainer_slots=["t0"],
            generator_slots=["g0"],
            lanes=lanes_for(["t0"], ["g0"]),
            slot_id="g0",
            worker_id="w1",
            role=Role.GENERATOR,
            index_in_role=0,
            plan_digest="d",
        )
        assert len(threads) == 1
        assert threads[0].started and threads[0].daemon

        client.close()
        assert client._registration_stop.is_set()
        assert threads[0].joined


class TestTransferControl:
    def test_create_uses_the_exact_group_spec_and_idempotency_key(self):
        stub = FakeStub()
        lanes = lanes_for(["t0", "t1"], ["g0"], lane_count=2)

        result = make_rendezvous(stub).create_transfer(
            model_name="model/name",
            trainer_slots=["t0", "t1"],
            generator_slots=["g0"],
            lanes=lanes,
            version_id="version-7",
            idempotency_key="training-step-7",
        )

        assert result.operation_id == "op1"
        assert len(stub.created) == 1
        request = stub.created[0]
        assert request.version_id == "version-7"
        assert request.idempotency_key == "training-step-7"
        assert request.spec == pb.CollectiveGroupSpec(
            model_name="model/name",
            expected_trainer_slots=["t0", "t1"],
            expected_generator_slots=["g0"],
            lanes=[
                pb.LaneSpec(
                    lane_id=lane.lane_id,
                    kind=rz._KIND_TO_PROTO[lane.kind],
                    trainer_slots=lane.trainer_slots,
                    generator_slots=lane.generator_slots,
                )
                for lane in lanes
            ],
        )

    def test_get_and_delete_use_the_operation_identifier(self):
        stub = FakeStub()
        client = make_rendezvous(stub)

        fetched = client.get_transfer("op1")
        deleted = client.delete_transfer("op1")

        assert fetched.operation_id == "op1"
        assert deleted.operation_id == "op1"
        assert stub.fetched == [pb.GetCollectiveTransferRequest(operation_id="op1")]
        assert stub.deleted == [pb.DeleteCollectiveTransferRequest(operation_id="op1")]

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("model_name", ""),
            ("model_name", " "),
            ("version_id", ""),
            ("version_id", " "),
            ("idempotency_key", ""),
            ("idempotency_key", " "),
        ],
    )
    def test_create_rejects_blank_identifiers_before_the_rpc(self, field, value):
        stub = FakeStub()
        kwargs = {
            "model_name": "model/name",
            "trainer_slots": ["t0"],
            "generator_slots": ["g0"],
            "lanes": lanes_for(["t0"], ["g0"]),
            "version_id": "version-7",
            "idempotency_key": "training-step-7",
        }
        kwargs[field] = value

        with pytest.raises(ValueError, match=field):
            make_rendezvous(stub).create_transfer(**kwargs)

        assert stub.created == []

    @pytest.mark.parametrize("method", ["get_transfer", "delete_transfer"])
    @pytest.mark.parametrize("operation_id", ["", " "])
    def test_operation_calls_reject_blank_identifiers_before_the_rpc(
        self, method, operation_id
    ):
        stub = FakeStub()
        with pytest.raises(ValueError, match="operation_id"):
            getattr(make_rendezvous(stub), method)(operation_id)
        assert stub.fetched == []
        assert stub.deleted == []


class TestAwaitReady:
    def test_it_returns_once_the_group_is_ready(self):
        stub = FakeStub(
            groups=[
                group(state=pb.COLLECTIVE_GROUP_STATE_FORMING),
                group(state=pb.COLLECTIVE_GROUP_STATE_READY),
            ]
        )
        result = make_rendezvous(stub).await_ready(
            group_id="g1", epoch=1, timeout_s=5, poll_interval_s=0.001
        )
        assert result.state == pb.COLLECTIVE_GROUP_STATE_READY
        assert stub.get_calls == 2

    def test_an_epoch_move_is_not_retryable(self):
        # The caller's plan and communicator are stale, so waiting longer would
        # never help; it has to rebuild.
        stub = FakeStub(groups=[group(epoch=4, state=pb.COLLECTIVE_GROUP_STATE_READY)])
        with pytest.raises(EpochChangedError) as caught:
            make_rendezvous(stub).await_ready(
                group_id="g1", epoch=1, timeout_s=5, poll_interval_s=0.001
            )
        assert caught.value.expected == 1
        assert caught.value.actual == 4

    def test_the_timeout_names_the_slots_that_never_joined(self):
        # "The collective hung" is not actionable; "t1 never joined" is.
        stub = FakeStub(groups=[group(admitted=["t0", "g0"])])
        with pytest.raises(GroupNotReadyError) as caught:
            make_rendezvous(stub).await_ready(
                group_id="g1", epoch=1, timeout_s=0.05, poll_interval_s=0.001
            )
        assert caught.value.missing == ["trainer slot t1"]
        assert "t1" in str(caught.value)

    def test_a_fully_admitted_group_blames_the_unbootstrapped_lane(self):
        stub = FakeStub(groups=[group(admitted=["t0", "t1", "g0"], lanes_ready=False)])
        with pytest.raises(GroupNotReadyError) as caught:
            make_rendezvous(stub).await_ready(
                group_id="g1", epoch=1, timeout_s=0.05, poll_interval_s=0.001
            )
        assert caught.value.missing == ["lane 0 bootstrap", "lane 1 bootstrap"]

    def test_each_poll_rpc_is_bounded_by_the_remaining_group_deadline(self):
        stub = FakeStub(groups=[group(state=pb.COLLECTIVE_GROUP_STATE_READY)])
        make_rendezvous(stub).await_ready(
            group_id="g1", epoch=1, timeout_s=0.05, poll_interval_s=0.001
        )
        assert 0 < stub.get_timeouts[0] <= 0.05

    @pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
    def test_invalid_direct_deadlines_are_rejected(self, value):
        with pytest.raises(ValueError, match="positive finite"):
            make_rendezvous(FakeStub()).await_ready(
                group_id="g1", epoch=1, timeout_s=value, poll_interval_s=0.01
            )

    def test_a_releasing_group_fails_without_polling_until_timeout(self):
        stub = FakeStub(groups=[group(state=pb.COLLECTIVE_GROUP_STATE_RELEASING)])
        with pytest.raises(RendezvousError, match="releasing"):
            make_rendezvous(stub).await_ready(
                group_id="g1", epoch=1, timeout_s=5, poll_interval_s=0.001
            )
        assert stub.get_calls == 1

    def test_an_rpc_deadline_at_the_group_deadline_becomes_not_ready(self, monkeypatch):
        class DeadlineExceeded(grpc.RpcError):
            def code(self):
                return grpc.StatusCode.DEADLINE_EXCEEDED

        clock = iter([0.0, 0.0, 1.0])
        monkeypatch.setattr(rz.time, "monotonic", lambda: next(clock))
        stub = FakeStub(get_error=DeadlineExceeded())
        with pytest.raises(GroupNotReadyError):
            make_rendezvous(stub).await_ready(
                group_id="g1", epoch=1, timeout_s=0.5, poll_interval_s=0.01
            )

    def test_missing_slots_are_role_qualified_when_names_overlap(self):
        g = pb.CollectiveGroup(
            group_id="g1",
            epoch=1,
            state=pb.COLLECTIVE_GROUP_STATE_FORMING,
            expected_trainer_slots=["rank0"],
            expected_generator_slots=["rank0"],
        )
        lane = g.lanes.add()
        lane.kind = pb.LANE_KIND_BROADCAST
        participant = lane.participants.add()
        participant.slot_id = "rank0"
        participant.role = pb.COLLECTIVE_ROLE_TRAINER
        assert rz._missing_slots(g) == ["generator slot rank0"]


class _FlakyStub(FakeStub):
    """Fails the first `failures` polls with `error`, then behaves."""

    def __init__(self, *, error, failures, groups):
        super().__init__(groups=groups)
        self._error = error
        self._failures = failures

    def GetCollectiveGroup(self, request, timeout=None):  # noqa: N802
        self.get_calls += 1
        self.get_timeouts.append(timeout)
        if self.get_calls <= self._failures:
            raise self._error
        return self._groups[min(self.get_calls - 1, len(self._groups) - 1)]


def _rpc_error(status):
    class _Error(grpc.RpcError):
        def code(self):
            return status

    return _Error()


class TestAwaitReadyRetries:
    """A poll failure is not a verdict on whether the group will form.

    Each poll is bounded by rpc_timeout_s, which is much shorter than the
    group timeout, so failing the whole rendezvous on one slow or restarted
    control-plane call throws away most of the deadline the caller asked for.
    """

    def test_a_slow_poll_does_not_end_a_wait_that_has_time_left(self):
        stub = _FlakyStub(
            error=_rpc_error(grpc.StatusCode.DEADLINE_EXCEEDED),
            failures=2,
            groups=[group(state=pb.COLLECTIVE_GROUP_STATE_READY)],
        )
        result = make_rendezvous(stub).await_ready(
            group_id="g1", epoch=1, timeout_s=5, poll_interval_s=0.001
        )
        assert result.state == pb.COLLECTIVE_GROUP_STATE_READY
        assert stub.get_calls == 3

    def test_a_control_plane_restart_does_not_end_it_either(self):
        stub = _FlakyStub(
            error=_rpc_error(grpc.StatusCode.UNAVAILABLE),
            failures=1,
            groups=[group(state=pb.COLLECTIVE_GROUP_STATE_READY)],
        )
        result = make_rendezvous(stub).await_ready(
            group_id="g1", epoch=1, timeout_s=5, poll_interval_s=0.001
        )
        assert result.state == pb.COLLECTIVE_GROUP_STATE_READY

    def test_a_non_retryable_code_still_fails_at_once(self):
        # The retry must not swallow a real refusal and sit there until the
        # group deadline reporting missing slots instead of the cause.
        stub = FakeStub(get_error=_rpc_error(grpc.StatusCode.PERMISSION_DENIED))
        with pytest.raises(grpc.RpcError):
            make_rendezvous(stub).await_ready(
                group_id="g1", epoch=1, timeout_s=5, poll_interval_s=0.001
            )
        assert stub.get_calls == 1

    def test_a_retryable_code_past_the_deadline_names_the_missing_slots(self):
        stub = FakeStub(get_error=_rpc_error(grpc.StatusCode.UNAVAILABLE))
        with pytest.raises(GroupNotReadyError):
            make_rendezvous(stub).await_ready(
                group_id="g1", epoch=1, timeout_s=0.05, poll_interval_s=0.001
            )


class TestRegistrationRenewal:
    def test_the_loop_outlives_a_failure_that_is_not_an_rpc_error(self):
        # grpcio raises ValueError on a closed channel in several versions.
        # Letting that escape ends the thread while _registration_thread stays
        # set, so nothing restarts it and the lease expires with no error
        # naming the cause.
        client = make_rendezvous(FakeStub())
        client._registration = object()
        client._registration_ttl_s = 0.003
        attempts = []

        def register(_registration):
            attempts.append(len(attempts))
            if len(attempts) == 1:
                raise ValueError("channel closed")
            if len(attempts) >= 3:
                client._registration_stop.set()

        client._register_worker = register
        client._renew_worker_registration()

        assert len(attempts) >= 3, (
            "the renewal loop stopped at the first non-RpcError failure"
        )


class _RejectingPublishStub(FakeStub):
    """A stub whose PublishGroupBootstrap always fails the precondition."""

    def PublishGroupBootstrap(self, request, timeout=None):  # noqa: N802
        class FailedPrecondition(grpc.RpcError):
            def code(self):
                return grpc.StatusCode.FAILED_PRECONDITION

        raise FailedPrecondition()


class TestPublishBootstrap:
    def test_an_explicit_deadline_bounds_the_publish_rpc(self, monkeypatch):
        stub = FakeStub()
        moments = iter((10.0, 11.0))
        monkeypatch.setattr(rz.time, "monotonic", lambda: next(moments))

        make_rendezvous(stub).publish_bootstrap(
            group_id="g1",
            epoch=2,
            lane_id=0,
            worker_id="w0",
            nccl_unique_id=b"\x01" * rz.NCCL_UNIQUE_ID_BYTES,
            timeout_s=3.0,
        )

        assert stub.publish_timeouts == [2.0]

    def test_a_correctly_sized_identifier_is_published_with_its_epoch(self):
        stub = FakeStub()
        make_rendezvous(stub).publish_bootstrap(
            group_id="g1",
            epoch=2,
            lane_id=0,
            worker_id="w0",
            nccl_unique_id=b"\x01" * rz.NCCL_UNIQUE_ID_BYTES,
        )
        assert stub.published[0].epoch == 2
        assert len(stub.published[0].nccl_unique_id) == rz.NCCL_UNIQUE_ID_BYTES

    def test_a_stale_epoch_rejection_names_the_epoch_it_moved_to(self):
        # "moved from epoch 2 to -1" tells the caller nothing, so the epoch is
        # read back from the group MX rejected the publication against.
        stub = _RejectingPublishStub(groups=[group(epoch=5)])
        with pytest.raises(EpochChangedError) as caught:
            make_rendezvous(stub).publish_bootstrap(
                group_id="g1",
                epoch=2,
                lane_id=0,
                worker_id="w0",
                nccl_unique_id=b"\x01" * rz.NCCL_UNIQUE_ID_BYTES,
            )
        assert caught.value.actual == 5

    def test_a_rejection_that_is_not_an_epoch_move_keeps_the_server_error(self):
        # MX also rejects a publisher that is not the lane's live rank 0. That
        # is not a stale epoch, and telling the caller to rebuild would hide the
        # reason it was actually turned away.
        stub = _RejectingPublishStub(groups=[group(epoch=2)])
        with pytest.raises(grpc.RpcError):
            make_rendezvous(stub).publish_bootstrap(
                group_id="g1",
                epoch=2,
                lane_id=0,
                worker_id="w0",
                nccl_unique_id=b"\x01" * rz.NCCL_UNIQUE_ID_BYTES,
            )

    @pytest.mark.parametrize("size", [0, 1, 127, 129])
    def test_a_wrong_sized_identifier_is_rejected_before_the_rpc(self, size):
        # A truncated identifier surfaces at Communicator.init as every rank of
        # the lane blocking, so it is caught here instead.
        stub = FakeStub()
        with pytest.raises(ValueError, match="must be 128 bytes"):
            make_rendezvous(stub).publish_bootstrap(
                group_id="g1",
                epoch=1,
                lane_id=0,
                worker_id="w0",
                nccl_unique_id=b"\x01" * size,
            )
        assert stub.published == []


class TestReport:
    def test_a_successful_report_needs_no_message(self):
        stub = FakeStub()
        make_rendezvous(stub).report(
            operation_id="op", group_id="g1", epoch=1, worker_id="w0", succeeded=True
        )
        assert stub.reported[0].succeeded

    def test_a_failed_report_must_explain_itself(self):
        stub = FakeStub()
        with pytest.raises(ValueError, match="must carry a message"):
            make_rendezvous(stub).report(
                operation_id="op",
                group_id="g1",
                epoch=1,
                worker_id="w0",
                succeeded=False,
            )
        assert stub.reported == []

    def test_a_failure_carries_the_epoch_it_was_admitted_against(self):
        stub = FakeStub()
        make_rendezvous(stub).report(
            operation_id="op",
            group_id="g1",
            epoch=7,
            worker_id="w0",
            succeeded=False,
            message="nccl abort",
        )
        assert stub.reported[0].epoch == 7
        assert stub.reported[0].message == "nccl abort"


class TestMembershipLookup:
    def test_asking_for_a_lane_this_worker_is_not_in_is_an_error(self):
        m = Membership(group_id="g1", epoch=1, lanes=(), is_bootstrap_leader=False)
        with pytest.raises(KeyError):
            m.lane(0)
        with pytest.raises(KeyError):
            _ = m.broadcast_lane
