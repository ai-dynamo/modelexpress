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
        self.events = []
        self.get_calls = 0
        self.get_timeouts = []

    def JoinCollectiveGroup(self, request, timeout=None):  # noqa: N802 - gRPC naming
        self.events.append("join")
        self.joined.append(request)
        return self._membership

    def RegisterWorker(self, request, timeout=None):  # noqa: N802
        self.events.append("register")
        self.registered.append(request)
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
        return pb.CollectiveGroup()

    def ReportCollectiveTransfer(self, request, timeout=None):  # noqa: N802
        self.reported.append(request)
        return pb.CollectiveTransfer(operation_id=request.operation_id)


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


def group(*, epoch=1, state=pb.COLLECTIVE_GROUP_STATE_FORMING, admitted=(), lanes_ready=True):
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
        role = pb.COLLECTIVE_ROLE_TRAINER if slot.startswith("t") else pb.COLLECTIVE_ROLE_GENERATOR
        p = broadcast.participants.add()
        p.slot_id = slot
        p.role = role
    return g


class TestJoin:
    def test_a_trainer_declares_its_partition_and_takes_the_assigned_rank(self):
        response = pb.CollectiveGroupMembership(group_id="g1", epoch=3, is_bootstrap_leader=True)
        a = response.assignments.add()
        a.lane_id, a.kind, a.rank_in_lane, a.world_size = 0, pb.LANE_KIND_RESHARD, 0, 3
        b = response.assignments.add()
        b.lane_id, b.kind, b.rank_in_lane, b.world_size = 1, pb.LANE_KIND_BROADCAST, 0, 3

        stub = FakeStub(membership=response)
        result = make_rendezvous(stub).join(
            model_name="m",
            trainer_slots=["t0", "t1"],
            generator_slots=["g0"],
            source_partition_count=1,
            slot_id="t0",
            worker_id="w0",
            role=Role.TRAINER,
            index_in_role=0,
            plan_digest="d",
            source_partition=0,
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
        assert registered.worker.endpoint == "collective://w0"
        assert registered.ttl_seconds == 90
        sent = stub.joined[0]
        assert sent.source_partition == 0
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
            source_partition_count=1,
            slot_id="g0",
            worker_id="w1",
            role=Role.GENERATOR,
            index_in_role=0,
            plan_digest="d",
        )
        assert not stub.joined[0].HasField("source_partition")
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
            source_partition_count=1,
            slot_id="t0",
            worker_id="w0",
            role=Role.TRAINER,
            index_in_role=0,
            plan_digest="abc",
            source_partition=0,
            plan_endpoint="host:1234",
        )
        source = stub.joined[0].plan_source
        assert source.endpoint == "host:1234"
        # Generators verify the fetched plan against the digest MX advertises,
        # so the two must not be allowed to drift apart at the source.
        assert source.digest == "abc"
        assert stub.registered[0].worker.endpoint == "host:1234"

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
                source_partition_count=1,
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
                source_partition_count=1,
                slot_id="t1",
                worker_id="w1",
                role=Role.TRAINER,
                index_in_role=1,
                plan_digest="d",
                source_partition=0,
                plan_endpoint="host:1234",
            )
        assert stub.joined == []
        assert stub.registered == []

    def test_an_explicit_registration_endpoint_is_immutable_across_joins(self):
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
            source_partition_count=1,
            slot_id="g0",
            worker_id="w1",
            worker_endpoint="actor://generator-0",
            role=Role.GENERATOR,
            index_in_role=0,
            plan_digest="d",
        )
        client.join(**kwargs)
        client.join(**kwargs)
        assert [r.worker.endpoint for r in stub.registered] == [
            "actor://generator-0",
            "actor://generator-0",
        ]

        with pytest.raises(RendezvousError, match="more than one worker identity"):
            client.join(**{**kwargs, "worker_endpoint": "actor://other"})
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
            source_partition_count=1,
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


class _RejectingPublishStub(FakeStub):
    """A stub whose PublishGroupBootstrap always fails the precondition."""

    def PublishGroupBootstrap(self, request, timeout=None):  # noqa: N802
        class FailedPrecondition(grpc.RpcError):
            def code(self):
                return grpc.StatusCode.FAILED_PRECONDITION

        raise FailedPrecondition()


class TestPublishBootstrap:
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
                group_id="g1", epoch=1, lane_id=0, worker_id="w0", nccl_unique_id=b"\x01" * size
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
                operation_id="op", group_id="g1", epoch=1, worker_id="w0", succeeded=False
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
