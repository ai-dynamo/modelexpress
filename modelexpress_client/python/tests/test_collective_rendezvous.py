# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rendezvous behaviour for the NCCL M2N collective path.

Driven against a fake stub rather than a live server. The behaviour worth
pinning here is what the client does when things go wrong, because every one of
these cases otherwise presents as a hung collective: a group that never becomes
READY, an epoch that moves underneath a waiter, and a bootstrap identifier of
the wrong size.
"""

import pytest
from modelexpress_rl import refit_collective_pb2 as pb
from modelexpress_rl.collective import rendezvous as rz
from modelexpress_rl.collective.rendezvous import (
    CollectiveRendezvous,
    EpochChangedError,
    GroupNotReadyError,
    Membership,
)
from modelexpress_rl.collective.types import Role


class FakeStub:
    """Stands in for RefitCollectiveServiceStub."""

    def __init__(self, groups=None, membership=None):
        self._groups = list(groups or [])
        self._membership = membership
        self.joined = []
        self.published = []
        self.reported = []
        self.get_calls = 0

    def JoinCollectiveGroup(self, request, timeout=None):  # noqa: N802 - gRPC naming
        self.joined.append(request)
        return self._membership

    def GetCollectiveGroup(self, request, timeout=None):  # noqa: N802
        self.get_calls += 1
        index = min(self.get_calls - 1, len(self._groups) - 1)
        return self._groups[index]

    def PublishGroupBootstrap(self, request, timeout=None):  # noqa: N802
        self.published.append(request)
        return pb.CollectiveGroup()

    def ReportCollectiveTransfer(self, request, timeout=None):  # noqa: N802
        self.reported.append(request)
        return pb.CollectiveTransfer(operation_id=request.operation_id)


def make_rendezvous(stub):
    client = CollectiveRendezvous.__new__(CollectiveRendezvous)
    client._stub = stub
    client._rpc_timeout_s = 5.0
    return client


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
        p = broadcast.participants.add()
        p.slot_id = slot
    return g


class TestJoin:
    def test_a_trainer_declares_its_partition_and_takes_the_assigned_rank(self):
        membership = pb.CollectiveGroupMembership(group_id="g1", epoch=3, is_bootstrap_leader=True)
        a = membership.assignments.add()
        a.lane_id, a.kind, a.rank_in_lane, a.world_size = 0, pb.LANE_KIND_RESHARD, 0, 6
        b = membership.assignments.add()
        b.lane_id, b.kind, b.rank_in_lane, b.world_size = 1, pb.LANE_KIND_BROADCAST, 0, 6

        stub = FakeStub(membership=membership)
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
        sent = stub.joined[0]
        assert sent.source_partition == 0
        assert sent.role == pb.COLLECTIVE_ROLE_TRAINER

    def test_a_generator_sends_no_partition(self):
        stub = FakeStub(membership=pb.CollectiveGroupMembership(group_id="g1", epoch=1))
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

    def test_a_plan_endpoint_is_advertised_with_the_matching_digest(self):
        stub = FakeStub(membership=pb.CollectiveGroupMembership(group_id="g1", epoch=1))
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
        assert caught.value.missing == ["t1"]
        assert "t1" in str(caught.value)

    def test_a_fully_admitted_group_blames_the_unbootstrapped_lane(self):
        stub = FakeStub(groups=[group(admitted=["t0", "t1", "g0"], lanes_ready=False)])
        with pytest.raises(GroupNotReadyError) as caught:
            make_rendezvous(stub).await_ready(
                group_id="g1", epoch=1, timeout_s=0.05, poll_interval_s=0.001
            )
        assert caught.value.missing == ["lane 0 bootstrap", "lane 1 bootstrap"]


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
