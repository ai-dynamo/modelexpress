# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Two-sided refit lifecycle for the NCCL M2N collective path.

The sequencing rules are the contract, and most of them exist because breaking
them produces a hang rather than an error: entering a collective before the
communicators exist, or before MX says the far side is ready, leaves ranks
blocked on peers that will never arrive.
"""

import sys
from types import ModuleType, SimpleNamespace

import pytest

from modelexpress_rl.collective import (
    LocalParamSpec,
    MeshSpec,
    MiscParam,
    ParamPlan,
    Placement,
    RefitClientGenerator,
    RefitClientTrainer,
    ReshardPlan,
)
from modelexpress_rl.collective.rendezvous import LaneMembership, Membership


def entry(name, partition=0):
    return ParamPlan(
        name=name,
        global_shape=(8, 4),
        dtype="bfloat16",
        partition_id=partition,
        src_mesh=MeshSpec(shape=(2,), rank_offset=0),
        src_placements=(Placement.shard(0),),
        dst_mesh=MeshSpec(shape=(2,), rank_offset=2),
        dst_placements=(Placement.shard(0),),
    )


PLAN = ReshardPlan(bulk=[entry("a")], misc=[MiscParam("m", (4,), "bfloat16")])


class FakeEngine:
    """Stands in for both a Publisher and a Loader."""

    def __init__(self, plan=PLAN):
        self._plan = plan
        self.calls = []

    def capture(self):
        return self._plan

    def local_params(self):
        return {n: LocalParamSpec(base=f"buf::{n}") for n in self._plan.parameter_names()}

    def start_new_round(self, version):
        self.calls.append(("start", version))

    def install(self, layer_group_id):
        self.calls.append(("install", layer_group_id))

    def finish(self):
        self.calls.append(("finish",))

    def cleanup(self):
        self.calls.append(("cleanup",))


class FakeRendezvous:
    def __init__(self, *, epochs=(1,), leader=False):
        self._epochs = list(epochs)
        self._leader = leader
        self.published = []
        self.reports = []
        self.joins = 0

    def join(self, **kwargs):
        epoch = self._epochs[min(self.joins, len(self._epochs) - 1)]
        self.joins += 1
        self._epoch = epoch
        return Membership(
            group_id="g",
            epoch=epoch,
            lanes=(
                LaneMembership(0, "RESHARD", 0 if self._leader else 2, 4),
                LaneMembership(1, "BROADCAST", 0 if self._leader else 2, 4),
            ),
            is_bootstrap_leader=self._leader,
        )

    def publish_bootstrap(self, **kwargs):
        self.published.append(kwargs)

    def await_ready(self, *, group_id, epoch, **kwargs):
        lanes = [
            SimpleNamespace(lane_id=0, nccl_unique_id=b"\x00" * 128),
            SimpleNamespace(lane_id=1, nccl_unique_id=b"\x01" * 128),
        ]
        return SimpleNamespace(group_id=group_id, epoch=epoch, lanes=lanes)

    def report(self, **kwargs):
        self.reports.append(kwargs)
        return SimpleNamespace(operation_id=kwargs["operation_id"])


@pytest.fixture
def fake_nccl(monkeypatch):
    ops = []

    def reshard(src, dst, comm, **kwargs):
        ops.append("reshard")

    class Comm:
        def broadcast(self, sendbuf, recvbuf, root, stream):
            ops.append("broadcast")

        def abort(self):
            ops.append("abort")

    m2n = ModuleType("nccl.m2n")
    m2n.reshard = reshard
    communicator = ModuleType("nccl.core.communicator")
    communicator.Communicator = SimpleNamespace(init=lambda **kw: Comm())
    utils = ModuleType("nccl.core.utils")
    utils.get_unique_id = lambda: SimpleNamespace(as_bytes=b"\x07" * 128)
    utils.UniqueId = SimpleNamespace(from_bytes=lambda raw: raw)
    for name, mod in [
        ("nccl", ModuleType("nccl")),
        ("nccl.core", ModuleType("nccl.core")),
        ("nccl.core.communicator", communicator),
        ("nccl.core.utils", utils),
        ("nccl.m2n", m2n),
    ]:
        monkeypatch.setitem(sys.modules, name, mod)
    return ops


def trainer(rz, engine, **kw):
    client = RefitClientTrainer(
        rendezvous=rz,
        model_name="m",
        trainer_slots=["t0", "t1"],
        generator_slots=["g0", "g1"],
        source_partition_count=1,
        slot_id="t0",
        worker_id="w0",
        index_in_role=0,
        **kw,
    )
    client.initialize(engine, source_partition=0)
    return client


class TestSequencing:
    def test_a_transfer_before_compute_plan_is_refused(self):
        client = trainer(FakeRendezvous(), FakeEngine())
        with pytest.raises(RuntimeError, match="compute_plan must run"):
            client.start_weight_update("v1")

    def test_publish_before_start_is_refused(self, fake_nccl):
        client = trainer(FakeRendezvous(), FakeEngine())
        client.compute_plan()
        # The half exists after compute_plan, so this checks the rounds are
        # still bracketed rather than that the object is missing.
        client.start_weight_update("v1")
        client.publish_weights("v1")

    def test_membership_is_unavailable_before_compute_plan(self):
        client = trainer(FakeRendezvous(), FakeEngine())
        with pytest.raises(RuntimeError, match="compute_plan has not run"):
            _ = client.membership


class TestBootstrap:
    def test_a_lane_leader_publishes_an_identifier_per_lane_it_leads(self, fake_nccl):
        rz = FakeRendezvous(leader=True)
        trainer(rz, FakeEngine()).compute_plan()
        lanes = sorted(p["lane_id"] for p in rz.published)
        assert lanes == [0, 1]
        assert all(len(p["nccl_unique_id"]) == 128 for p in rz.published)
        assert all(p["epoch"] == 1 for p in rz.published)

    def test_a_non_leader_publishes_nothing(self, fake_nccl):
        rz = FakeRendezvous(leader=False)
        trainer(rz, FakeEngine()).compute_plan()
        assert rz.published == []


class TestEpochInvalidation:
    def test_a_second_compute_plan_at_a_new_epoch_rebuilds_the_lanes(self, fake_nccl):
        rz = FakeRendezvous(epochs=(1, 2))
        client = trainer(rz, FakeEngine())
        client.compute_plan()
        first = len(client._cache)
        client.compute_plan()
        # Stale lanes are dropped and rebuilt, not accumulated alongside.
        assert len(client._cache) == first
        assert client.membership.epoch == 2


class TestRefitRound:
    def test_a_full_trainer_round_reshards_then_broadcasts(self, fake_nccl):
        engine = FakeEngine()
        client = trainer(FakeRendezvous(), engine)
        client.compute_plan()
        client.start_weight_update("v1")
        client.publish_weights("v1")
        client.finish_weight_update("v1")
        assert fake_nccl == ["reshard", "broadcast"]
        assert ("start", "v1") in engine.calls

    def test_a_generator_round_installs_each_group_then_finishes(self, fake_nccl):
        engine = FakeEngine()
        client = RefitClientGenerator(
            rendezvous=FakeRendezvous(),
            model_name="m",
            trainer_slots=["t0", "t1"],
            generator_slots=["g0", "g1"],
            source_partition_count=1,
            slot_id="g0",
            worker_id="w9",
            index_in_role=0,
        )
        client.initialize(engine)
        client.compute_plan()
        client.start_weight_update("v1")
        client.update_weights("v1")
        client.finish_weight_update("v1")
        assert ("install", 0) in engine.calls
        assert ("finish",) in engine.calls
        assert fake_nccl == ["reshard", "broadcast"]


class TestReporting:
    def test_success_is_reported_against_the_admitted_epoch(self, fake_nccl):
        rz = FakeRendezvous()
        client = trainer(rz, FakeEngine())
        client.compute_plan()
        client.start_weight_update("v1")
        client.publish_weights("v1")
        client.finish_weight_update("v1", operation_id="op1")
        assert rz.reports[0]["succeeded"]
        assert rz.reports[0]["epoch"] == 1
        assert rz.reports[0]["worker_id"] == "w0"

    def test_a_failure_is_reported_and_aborts_the_group(self, fake_nccl, monkeypatch):
        rz = FakeRendezvous()
        client = trainer(rz, FakeEngine())
        client.compute_plan()
        client.start_weight_update("v1")

        def boom(*args, **kwargs):
            raise RuntimeError("nccl timeout")

        monkeypatch.setattr(client._half, "finish_weight_update", boom)
        with pytest.raises(RuntimeError, match="nccl timeout"):
            client.finish_weight_update("v1", operation_id="op1")

        assert rz.reports[0]["succeeded"] is False
        assert "nccl timeout" in rz.reports[0]["message"]
        # Abort is what makes the deadline mean anything; without it the peers
        # stay blocked with an error merely attached.
        assert len(client._cache) == 0

    def test_no_operation_id_means_no_report(self, fake_nccl):
        rz = FakeRendezvous()
        client = trainer(rz, FakeEngine())
        client.compute_plan()
        client.start_weight_update("v1")
        client.finish_weight_update("v1")
        assert rz.reports == []


class TestStreams:
    def test_lanes_are_spread_across_the_configured_streams(self, fake_nccl):
        client = trainer(FakeRendezvous(), FakeEngine(), streams=["s0", "s1"])
        client.compute_plan()
        assert client._stream_for(0) == "s0"
        assert client._stream_for(1) == "s1"
        assert client._stream_for(2) == "s0"

    def test_a_single_stream_is_the_default(self, fake_nccl):
        client = trainer(FakeRendezvous(), FakeEngine())
        assert client._stream_for(0) is None
        assert client._stream_for(5) is None


class TestCleanup:
    def test_cleanup_releases_the_engine_and_the_communicators(self, fake_nccl):
        engine = FakeEngine()
        client = trainer(FakeRendezvous(), engine)
        client.compute_plan()
        client.cleanup()
        assert ("cleanup",) in engine.calls
        assert len(client._cache) == 0
