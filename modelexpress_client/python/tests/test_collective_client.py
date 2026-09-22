# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Two-sided refit lifecycle for the NCCL M2N collective path.

The sequencing rules are the contract, and most of them exist because breaking
them produces a hang rather than an error: entering a collective before the
communicators exist, or before MX says the far side is ready, leaves ranks
blocked on peers that will never arrive.
"""

import sys
import threading
from contextlib import contextmanager, nullcontext
from types import ModuleType, SimpleNamespace

import pytest

import modelexpress_rl.collective.client as collective_client
from modelexpress_rl.collective import (
    CommunicatorCache,
    LaneKey,
    LocalParamSpec,
    MeshSpec,
    MiscParam,
    ParamPlan,
    Placement,
    RefitClientGenerator,
    RefitClientTrainer,
    ReshardPlan,
    build_partitioned_lanes,
)
from modelexpress_rl.collective.comm import new_unique_id
from modelexpress_rl.collective.rendezvous import (
    EpochChangedError,
    GroupNotReadyError,
    LaneMembership,
    Membership,
)


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

    def parameter_names(self):
        return self._plan.parameter_names()

    def local_params(self):
        return {
            n: LocalParamSpec(base=f"buf::{n}") for n in self._plan.parameter_names()
        }

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
        self.fences = []
        self.bootstrap_aborts = []
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

    def await_bootstrap_fence(self, **kwargs):
        self.fences.append(kwargs)

    def abort_bootstrap(self, **kwargs):
        self.bootstrap_aborts.append(kwargs)
        raise EpochChangedError(
            kwargs["group_id"], kwargs["epoch"], kwargs["epoch"] + 1
        )

    def report(self, **kwargs):
        self.reports.append(kwargs)
        return SimpleNamespace(operation_id=kwargs["operation_id"])


class FakeRendezvousPP2(FakeRendezvous):
    def join(self, **kwargs):
        self.joins += 1
        # Four trainers over two reshard lanes, so the client must declare
        # lanes 0 and 1 plus a broadcast lane, and t2 leads lane 1.
        lanes = kwargs["lanes"]
        assert [(lane.lane_id, lane.kind) for lane in lanes] == [
            (0, "RESHARD"),
            (1, "RESHARD"),
            (2, "BROADCAST"),
        ]
        assert lanes[0].trainer_slots == ("t0", "t1")
        assert lanes[1].trainer_slots == ("t2", "t3")
        assert lanes[2].trainer_slots == ("t0", "t1", "t2", "t3")
        return Membership(
            group_id="g",
            epoch=1,
            lanes=(
                LaneMembership(1, "RESHARD", 0, 4),
                LaneMembership(2, "BROADCAST", 2, 6),
            ),
            is_bootstrap_leader=True,
        )

    def await_ready(self, *, group_id, epoch, **kwargs):
        lanes = [
            SimpleNamespace(lane_id=i, nccl_unique_id=bytes([i]) * 128)
            for i in range(3)
        ]
        return SimpleNamespace(group_id=group_id, epoch=epoch, lanes=lanes)


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

        def get_async_error(self):
            return Result.Success

        def get_last_error(self):
            return ""

    class NCCLConfig:
        def __init__(self, *, blocking=None):
            self.blocking = blocking

    class Result:
        Success = 0
        InProgress = 7

    m2n = ModuleType("nccl.m2n")
    m2n.reshard = reshard
    communicator = ModuleType("nccl.core.communicator")
    communicator.NCCLConfig = NCCLConfig
    communicator.Communicator = SimpleNamespace(init=lambda **kw: Comm())
    utils = ModuleType("nccl.core.utils")
    utils.get_unique_id = lambda: SimpleNamespace(as_bytes=b"\x07" * 128)
    utils.UniqueId = SimpleNamespace(from_bytes=lambda raw: raw)
    bindings = ModuleType("nccl.bindings.nccl")
    bindings.Result = Result
    for name, mod in [
        ("nccl", ModuleType("nccl")),
        ("nccl.bindings", ModuleType("nccl.bindings")),
        ("nccl.bindings.nccl", bindings),
        ("nccl.core", ModuleType("nccl.core")),
        ("nccl.core.communicator", communicator),
        ("nccl.core.utils", utils),
        ("nccl.m2n", m2n),
    ]:
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setattr(
        collective_client,
        "_bootstrap_barrier",
        lambda lane, device, **kwargs: None,
    )
    monkeypatch.setattr(
        "modelexpress_rl.collective.comm.LaneCommunicator.synchronize",
        lambda self, timeout_s=None: None,
    )
    return ops


def _raises(error):
    def boom(*args, **kwargs):
        raise error

    return boom


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
        with pytest.raises(RuntimeError, match="start_weight_update must run"):
            client.publish_weights("v1")

    def test_a_call_naming_another_version_is_refused(self, fake_nccl):
        # version was accepted and ignored, so publishing "v2" into the round
        # "v1" opened moved v1's tensors under v2's label on every rank.
        client = trainer(FakeRendezvous(), FakeEngine())
        client.compute_plan()
        client.start_weight_update("v1")
        with pytest.raises(ValueError, match="round in flight is 'v1'"):
            client.publish_weights("v2")
        with pytest.raises(ValueError, match="round in flight is 'v1'"):
            client.finish_weight_update("v2")

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

    def test_every_step_fences_before_barrier_and_completes_after_the_final_barrier(
        self, fake_nccl
    ):
        rendezvous = FakeRendezvous()

        trainer(rendezvous, FakeEngine()).compute_plan()

        assert [(fence["lane_id"], fence["phase"]) for fence in rendezvous.fences] == [
            (1, "PRE_BARRIER"),
            (0, "PRE_BARRIER"),
            (0, "COMPLETE"),
        ]

    def test_fenced_bootstrap_forwards_the_framework_barrier_allocator(
        self, fake_nccl, monkeypatch
    ):
        allocator = object()
        seen = []
        client = trainer(FakeRendezvous(), FakeEngine(), barrier_alloc=allocator)

        def barrier(lane, device, *, timeout_s=None, alloc=None):
            seen.append((lane.rank, device, timeout_s, alloc))

        monkeypatch.setattr(collective_client, "_bootstrap_barrier", barrier)

        client.compute_plan()

        assert len(seen) == 2
        assert all(call[3] is allocator for call in seen)

    def test_pp2_nonmember_settles_broadcast_before_the_next_barrier(
        self, fake_nccl, monkeypatch
    ):
        plan = ReshardPlan(
            bulk=[entry("a", partition=0), entry("b", partition=1)],
            misc=[MiscParam("m", (4,), "bfloat16")],
            source_partition_count=2,
        )
        engine = FakeEngine(plan)
        client = RefitClientTrainer(
            rendezvous=FakeRendezvousPP2(),
            model_name="m",
            trainer_slots=["t0", "t1", "t2", "t3"],
            generator_slots=["g0", "g1"],
            source_partition_count=2,
            slot_id="t2",
            worker_id="w2",
            index_in_role=2,
        )
        client.initialize(engine, source_partition=1)

        events = []
        state = {"broadcast_polls": 0}
        original_create = client._cache.create
        original_settle_group = client._cache.settle_group

        def create(key, **kwargs):
            events.append(("create", key.lane_id))
            return original_create(key, **kwargs)

        def settle_group(group_id, epoch, **kwargs):
            events.append(("settle", group_id, epoch))
            return original_settle_group(group_id, epoch, **kwargs)

        def barrier(lane, device, *, timeout_s=None, alloc=None):
            barrier_count = sum(event[0] == "barrier" for event in events)
            if barrier_count:
                assert state["broadcast_polls"] >= 2
            events.append(("barrier", lane.rank))
            if barrier_count == 0:
                # Model the observed rank-1 state before B1. NCCL guarantees
                # this same communicator must be polled before its next use.
                statuses = iter(
                    [
                        sys.modules["nccl.bindings.nccl"].Result.InProgress,
                        sys.modules["nccl.bindings.nccl"].Result.Success,
                    ]
                )

                def get_async_error():
                    state["broadcast_polls"] += 1
                    return next(
                        statuses,
                        sys.modules["nccl.bindings.nccl"].Result.Success,
                    )

                lane.handle.get_async_error = get_async_error

        monkeypatch.setattr(client._cache, "create", create)
        monkeypatch.setattr(client._cache, "settle_group", settle_group)
        monkeypatch.setattr(collective_client, "_bootstrap_barrier", barrier)

        client.compute_plan()

        # This stage-1 trainer is not in lane 0, but it still waits at lane 0's
        # full-group barrier before it is allowed to initialize lane 1. It
        # re-polls the observed ncclInProgress state on lane 2 before B1.
        assert events == [
            ("create", 2),
            ("settle", "g", 1),
            ("barrier", 2),
            ("settle", "g", 1),
            ("barrier", 2),
            ("create", 1),
            ("settle", "g", 1),
            ("barrier", 2),
        ]

    def test_pp2_fast_nonmember_waits_for_slow_lane_member_before_barrier(
        self, fake_nccl
    ):
        class ConcurrentRendezvous:
            def __init__(self):
                self.condition = threading.Condition()
                self.arrivals = set()
                self.fast_waiting = threading.Event()
                self.slow_created = threading.Event()
                self.fast_released_after_slow = False

            def join(self, *, slot_id, **_kwargs):
                reshard = 0 if slot_id == "t0" else 1
                return Membership(
                    group_id="g",
                    epoch=1,
                    lanes=(
                        LaneMembership(reshard, "RESHARD", 0, 1),
                        LaneMembership(2, "BROADCAST", 0 if slot_id == "t0" else 1, 2),
                    ),
                    is_bootstrap_leader=True,
                )

            def publish_bootstrap(self, **_kwargs):
                return None

            def await_ready(self, **_kwargs):
                return SimpleNamespace(
                    lanes=[
                        SimpleNamespace(lane_id=i, nccl_unique_id=bytes([i]) * 128)
                        for i in range(3)
                    ]
                )

            def await_bootstrap_fence(self, *, slot_id, lane_id, phase, **_kwargs):
                with self.condition:
                    self.arrivals.add((slot_id, lane_id, phase))
                    if slot_id == "t2" and lane_id == 0 and phase == "PRE_BARRIER":
                        self.fast_waiting.set()
                    self.condition.notify_all()
                    assert self.condition.wait_for(
                        lambda: (
                            {
                                ("t0", lane_id, phase),
                                ("t2", lane_id, phase),
                            }
                            <= self.arrivals
                        ),
                        timeout=2.0,
                    )
                    if slot_id == "t2" and lane_id == 0 and phase == "PRE_BARRIER":
                        self.fast_released_after_slow = self.slow_created.is_set()

            def abort_bootstrap(self, **_kwargs):
                raise AssertionError("concurrent bootstrap should not abort")

        plan = ReshardPlan(
            bulk=[entry("a", partition=0), entry("b", partition=1)],
            misc=[MiscParam("m", (4,), "bfloat16")],
            source_partition_count=2,
        )
        rendezvous = ConcurrentRendezvous()

        def make_client(slot_id, partition):
            client = RefitClientTrainer(
                rendezvous=rendezvous,
                model_name="m",
                trainer_slots=["t0", "t1", "t2", "t3"],
                generator_slots=["g0", "g1"],
                source_partition_count=2,
                slot_id=slot_id,
                worker_id=f"w-{slot_id}",
                index_in_role=partition * 2,
            )
            client.initialize(FakeEngine(plan), source_partition=partition)
            return client

        slow = make_client("t0", 0)
        fast = make_client("t2", 1)
        original_create = slow._cache.create

        def delayed_create(key, **kwargs):
            if key.lane_id == 0:
                assert rendezvous.fast_waiting.wait(timeout=2.0)
                lane = original_create(key, **kwargs)
                rendezvous.slow_created.set()
                return lane
            return original_create(key, **kwargs)

        slow._cache.create = delayed_create
        errors = []

        def run(client):
            try:
                client.compute_plan()
            except BaseException as error:  # noqa: BLE001 - surfaced in main test thread
                errors.append(error)

        threads = [
            threading.Thread(target=run, args=(fast,)),
            threading.Thread(target=run, args=(slow,)),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5.0)

        assert all(not thread.is_alive() for thread in threads)
        assert errors == []
        assert rendezvous.fast_released_after_slow


class TestCommunicatorBootstrap:
    def test_init_is_nonblocking_bounded_and_device_scoped(
        self, fake_nccl, monkeypatch
    ):
        import torch

        seen = {}
        communicator = sys.modules["nccl.core.communicator"]
        original_init = communicator.Communicator.init

        def init(**kwargs):
            seen["config"] = kwargs["config"]
            return original_init(**kwargs)

        @contextmanager
        def device_context(device):
            seen["device"] = device
            yield

        communicator.Communicator.init = init
        monkeypatch.setattr(torch.cuda, "device", device_context)
        cache = CommunicatorCache()
        cache.create(
            LaneKey("g", 1, 0),
            rank=0,
            world_size=2,
            unique_id=b"x" * 128,
            device=3,
            stream=None,
            timeout_s=0.1,
        )

        assert seen["config"].blocking is False
        assert seen["device"] == 3

    def test_creating_a_lane_does_not_poll_an_unrelated_lane(self, fake_nccl):
        communicator = sys.modules["nccl.core.communicator"]

        class Ready:
            def __init__(self):
                self.polls = 0

            def get_async_error(self):
                self.polls += 1
                return sys.modules["nccl.bindings.nccl"].Result.Success

        first = Ready()
        second = Ready()
        communicators = iter([first, second])
        communicator.Communicator.init = lambda **kwargs: next(communicators)
        cache = CommunicatorCache()

        cache.create(
            LaneKey("g", 1, 0),
            rank=0,
            world_size=2,
            unique_id=b"a" * 128,
            device=None,
            stream=None,
            timeout_s=0.1,
        )
        cache.create(
            LaneKey("g", 1, 1),
            rank=0,
            world_size=2,
            unique_id=b"b" * 128,
            device=None,
            stream=None,
            timeout_s=0.1,
        )

        assert first.polls == 1
        assert second.polls == 1

    def test_a_stalled_nonblocking_init_is_aborted_at_the_deadline(
        self, fake_nccl, monkeypatch
    ):
        bindings = sys.modules["nccl.bindings.nccl"]
        communicator = sys.modules["nccl.core.communicator"]

        class Stuck:
            def __init__(self):
                self.aborted = False

            def get_async_error(self):
                return bindings.Result.InProgress

            def abort(self):
                self.aborted = True

        stuck = Stuck()
        communicator.Communicator.init = lambda **kwargs: stuck
        cache = CommunicatorCache()
        with pytest.raises(TimeoutError, match="did not complete"):
            cache.create(
                LaneKey("g", 1, 0),
                rank=0,
                world_size=2,
                unique_id=b"x" * 128,
                device=None,
                stream=None,
                timeout_s=0.001,
            )
        assert stuck.aborted

    def test_forced_nccl_comm_id_is_rejected_before_minting(
        self, fake_nccl, monkeypatch
    ):
        monkeypatch.setenv("NCCL_COMM_ID", "mxray-gen:1234")
        with pytest.raises(RuntimeError, match="incompatible with MX-brokered"):
            new_unique_id()
        with pytest.raises(RuntimeError, match="incompatible with MX-brokered"):
            CommunicatorCache().create(
                LaneKey("g", 1, 0),
                rank=1,
                world_size=2,
                unique_id=b"x" * 128,
                device=None,
                stream=None,
                timeout_s=0.1,
            )

    def test_blocking_override_cannot_silently_disable_the_timeout(
        self, fake_nccl, monkeypatch
    ):
        monkeypatch.setenv("NCCL_COMM_BLOCKING", "1")
        cache = CommunicatorCache()
        with pytest.raises(RuntimeError, match="defeat.*TIMEOUT"):
            cache.create(
                LaneKey("g", 1, 0),
                rank=0,
                world_size=2,
                unique_id=b"x" * 128,
                device=None,
                stream=None,
                timeout_s=0.1,
            )

    def test_settle_group_only_waits_for_live_lanes_in_the_current_epoch(
        self, fake_nccl
    ):
        bindings = sys.modules["nccl.bindings.nccl"]

        class Reinitializing:
            def __init__(self):
                self.polls = 0

            def get_async_error(self):
                self.polls += 1
                if self.polls == 1:
                    return bindings.Result.InProgress
                return bindings.Result.Success

        current = Reinitializing()
        stale = Reinitializing()
        other = Reinitializing()
        aborted = Reinitializing()
        cache = CommunicatorCache()
        cache._lanes[LaneKey("g", 2, 2)] = collective_client.LaneCommunicator(
            current, rank=0, world_size=2, stream=None
        )
        cache._lanes[LaneKey("g", 1, 2)] = collective_client.LaneCommunicator(
            stale, rank=0, world_size=2, stream=None
        )
        cache._lanes[LaneKey("other", 2, 2)] = collective_client.LaneCommunicator(
            other, rank=0, world_size=2, stream=None
        )
        aborted_lane = collective_client.LaneCommunicator(
            aborted, rank=0, world_size=2, stream=None
        )
        aborted_lane.abort()
        cache._lanes[LaneKey("g", 2, 1)] = aborted_lane

        assert cache.settle_group("g", 2, timeout_s=0.1) == 1
        assert current.polls == 2
        assert stale.polls == 0
        assert other.polls == 0
        assert aborted.polls == 0

    def test_settle_group_reports_the_lane_that_exceeded_its_deadline(self, fake_nccl):
        bindings = sys.modules["nccl.bindings.nccl"]

        class Stuck:
            def get_async_error(self):
                return bindings.Result.InProgress

        cache = CommunicatorCache()
        cache._lanes[LaneKey("g", 3, 2)] = collective_client.LaneCommunicator(
            Stuck(), rank=0, world_size=2, stream=None
        )

        with pytest.raises(
            RuntimeError,
            match=r"lane 2 of group g at epoch 3.*did not complete",
        ):
            cache.settle_group("g", 3, timeout_s=0.001)


class TestBootstrapBarrier:
    """The barrier between lane initializations runs after READY.

    Everything past READY carries a deadline. This one is the odd case: it is
    not inside Communicator.init, so COMM_INIT_TIMEOUT does not reach it on
    its own, and the transfer deadline does not arm until bootstrap is done.
    """

    def _lane(self, waits):
        class FakeHandle:
            def broadcast(self, **kwargs):
                pass

        return SimpleNamespace(
            handle=FakeHandle(),
            stream=None,
            synchronize=lambda timeout_s=None: waits.append(timeout_s),
        )

    def test_the_barrier_wait_is_bounded(self, monkeypatch):
        waits = []
        torch = ModuleType("torch")
        torch.uint8 = "uint8"
        torch.zeros = lambda *a, **kw: object()
        torch.cuda = SimpleNamespace(device=lambda d: nullcontext())
        monkeypatch.setitem(sys.modules, "torch", torch)

        collective_client._bootstrap_barrier(self._lane(waits), None)

        assert waits and waits[0] is not None, (
            "an unbounded barrier blocks forever if a peer dies after READY"
        )
        assert waits[0] > 0

    def test_an_explicit_deadline_wins(self, monkeypatch):
        waits = []
        torch = ModuleType("torch")
        torch.uint8 = "uint8"
        torch.zeros = lambda *a, **kw: object()
        torch.cuda = SimpleNamespace(device=lambda d: nullcontext())
        monkeypatch.setitem(sys.modules, "torch", torch)

        collective_client._bootstrap_barrier(self._lane(waits), None, timeout_s=1.5)

        assert waits == [1.5]


class TestLaneDeclaration:
    @pytest.mark.parametrize("source_partition_count", [0, -1, True, 1.5])
    def test_partition_count_must_be_a_positive_integer(self, source_partition_count):
        with pytest.raises(ValueError, match="must be a positive integer"):
            build_partitioned_lanes(
                ["t0", "t1"],
                ["g0"],
                source_partition_count,
            )

    def test_a_trainer_count_the_partition_count_does_not_divide_is_refused(self):
        # Floor division would silently leave the last trainers off every
        # reshard lane. Their parameters never move and the peers expecting
        # their ranks block, which is the failure mode with no error in it.
        with pytest.raises(ValueError, match="must divide the trainer slot count"):
            build_partitioned_lanes(
                ["t0", "t1", "t2", "t3", "t4"],
                ["g0"],
                2,
            )

    def test_more_partitions_than_trainers_is_refused(self):
        with pytest.raises(ValueError, match="must divide the trainer slot count"):
            build_partitioned_lanes(["t0", "t1"], ["g0"], 4)

    def test_a_split_that_divides_is_unchanged(self):
        lanes = build_partitioned_lanes(
            ["t0", "t1", "t2", "t3"],
            ["g0"],
            2,
        )
        assert [lane.trainer_slots for lane in lanes] == [
            ("t0", "t1"),
            ("t2", "t3"),
            ("t0", "t1", "t2", "t3"),
        ]
        assert [(lane.lane_id, lane.kind) for lane in lanes] == [
            (0, "RESHARD"),
            (1, "RESHARD"),
            (2, "BROADCAST"),
        ]
        assert all(lane.generator_slots == ("g0",) for lane in lanes)


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

    def test_successive_restart_epoch_moves_rejoin_with_one_worker_id(self, fake_nccl):
        class RestartingRendezvous(FakeRendezvous):
            def __init__(self):
                super().__init__(epochs=(2, 3, 4), leader=True)
                self.worker_ids = []
                self.join_timeouts = []
                self.await_timeouts = []
                self.publish_timeouts = []

            def join(self, **kwargs):
                self.worker_ids.append(kwargs["worker_id"])
                self.join_timeouts.append(kwargs["timeout_s"])
                return super().join(**kwargs)

            def publish_bootstrap(self, **kwargs):
                self.publish_timeouts.append(kwargs["timeout_s"])
                return super().publish_bootstrap(**kwargs)

            def await_ready(self, *, group_id, epoch, timeout_s=None, **kwargs):
                self.await_timeouts.append(timeout_s)
                if epoch < 4:
                    raise EpochChangedError(group_id, epoch, epoch + 1)
                return super().await_ready(group_id=group_id, epoch=epoch, **kwargs)

        rz = RestartingRendezvous()
        client = trainer(rz, FakeEngine())

        client.compute_plan()

        assert client.membership.epoch == 4
        assert rz.worker_ids == ["w0", "w0", "w0"]
        assert [item["epoch"] for item in rz.published] == [2, 2, 3, 3, 4, 4]
        assert len(rz.await_timeouts) == 3
        budgets = rz.join_timeouts + rz.publish_timeouts + rz.await_timeouts
        assert all(
            0 < timeout <= collective_client.envs.MX_NCCL_REFIT_GROUP_TIMEOUT_S
            for timeout in budgets
        )
        assert rz.join_timeouts == sorted(rz.join_timeouts, reverse=True)
        assert rz.publish_timeouts == sorted(rz.publish_timeouts, reverse=True)
        assert rz.await_timeouts == sorted(rz.await_timeouts, reverse=True)

    def test_restart_epoch_churn_is_bounded_by_the_cohort_size(self, fake_nccl):
        class ChurningRendezvous(FakeRendezvous):
            def __init__(self):
                super().__init__(epochs=range(1, 20))
                self.worker_ids = []

            def join(self, **kwargs):
                self.worker_ids.append(kwargs["worker_id"])
                return super().join(**kwargs)

            def await_ready(self, *, group_id, epoch, **kwargs):
                raise EpochChangedError(group_id, epoch, epoch + 1)

        rz = ChurningRendezvous()
        client = trainer(rz, FakeEngine())

        with pytest.raises(EpochChangedError, match="moved from epoch 5 to 6"):
            client.compute_plan()

        # Two trainers plus two generators permit four replacement-driven
        # epoch moves after the initial attempt, and no unbounded retry.
        assert rz.joins == 5
        assert rz.worker_ids == ["w0"] * 5
        assert len(client._cache) == 0
        with pytest.raises(RuntimeError, match="compute_plan has not run"):
            _ = client.membership

    def test_expired_budget_does_not_start_bootstrap_publication(
        self, fake_nccl, monkeypatch
    ):
        rz = FakeRendezvous(leader=True)
        monkeypatch.setenv("MX_NCCL_REFIT_GROUP_TIMEOUT_S", "1")
        moments = iter((0.0, 0.1, 1.1))
        monkeypatch.setattr(collective_client.time, "monotonic", lambda: next(moments))

        with pytest.raises(GroupNotReadyError, match="did not reach READY"):
            trainer(rz, FakeEngine()).compute_plan()

        assert rz.published == []

    def test_expired_fence_failure_aborts_epoch_before_a_later_retry(
        self, fake_nccl, monkeypatch
    ):
        class DeadlineRendezvous(FakeRendezvous):
            def __init__(self):
                super().__init__(leader=False)
                self.epoch = 1
                self.fail_fence = True
                self.arrivals = set()
                self.abort_timeouts = []

            def join(self, **kwargs):
                self._epochs = [self.epoch]
                return super().join(**kwargs)

            def await_bootstrap_fence(self, **kwargs):
                self.arrivals.add(
                    (kwargs["epoch"], kwargs["lane_id"], kwargs["slot_id"])
                )
                if self.fail_fence:
                    self.fail_fence = False
                    collective_client.time.sleep(0.06)
                    raise TimeoutError("bootstrap fence deadline expired")

            def abort_bootstrap(self, **kwargs):
                self.abort_timeouts.append(kwargs["timeout_s"])
                assert kwargs["epoch"] == self.epoch
                self.epoch += 1
                self.arrivals.clear()
                return SimpleNamespace(epoch=self.epoch)

        rendezvous = DeadlineRendezvous()
        monkeypatch.setenv("MX_NCCL_REFIT_GROUP_TIMEOUT_S", "0.05")
        client = trainer(rendezvous, FakeEngine())

        with pytest.raises(TimeoutError, match="fence deadline"):
            client.compute_plan()

        assert rendezvous.abort_timeouts == [
            collective_client._BOOTSTRAP_ABORT_TIMEOUT_S
        ]
        assert rendezvous.arrivals == set()
        assert client.compute_plan().epoch == 2
        assert all(epoch == 2 for epoch, _lane, _slot in rendezvous.arrivals)

    def test_unconfirmed_bootstrap_abort_poisons_the_client(self, fake_nccl):
        class UnreachableAbortRendezvous(FakeRendezvous):
            def await_bootstrap_fence(self, **_kwargs):
                raise TimeoutError("bootstrap fence failed")

            def abort_bootstrap(self, **_kwargs):
                raise RuntimeError("control plane unreachable")

        rendezvous = UnreachableAbortRendezvous()
        client = trainer(rendezvous, FakeEngine())

        with pytest.raises(TimeoutError, match="fence failed"):
            client.compute_plan()
        joins = rendezvous.joins
        with pytest.raises(RuntimeError, match="cannot bootstrap again"):
            client.compute_plan()
        assert rendezvous.joins == joins

    @pytest.mark.parametrize(
        "shutdown",
        [
            KeyboardInterrupt("stop"),
            SystemExit("stop"),
            GeneratorExit("stop"),
        ],
    )
    def test_shutdown_base_exceptions_abort_once_without_retry(
        self, fake_nccl, shutdown
    ):
        class ShutdownRendezvous(FakeRendezvous):
            def await_bootstrap_fence(self, **_kwargs):
                raise shutdown

        rendezvous = ShutdownRendezvous()
        client = trainer(rendezvous, FakeEngine())

        with pytest.raises(type(shutdown)):
            client.compute_plan()

        assert rendezvous.joins == 1
        assert len(rendezvous.bootstrap_aborts) == 1


class TestPlanGates:
    def test_coverage_evidence_is_mandatory(self):
        class NoInventory:
            def capture(self):
                return PLAN

        client = RefitClientTrainer(
            rendezvous=FakeRendezvous(),
            model_name="m",
            trainer_slots=["t0", "t1"],
            generator_slots=["g0", "g1"],
            source_partition_count=1,
            slot_id="t0",
            worker_id="w0",
            index_in_role=0,
        )
        with pytest.raises(ValueError, match="coverage cannot be optional"):
            client.initialize(NoInventory(), source_partition=0)

    def test_plan_partition_count_must_match_the_join_spec(self):
        plan = ReshardPlan(
            bulk=[entry("a")],
            source_partition_count=2,
        )
        client = RefitClientTrainer(
            rendezvous=FakeRendezvous(),
            model_name="m",
            trainer_slots=["t0", "t1"],
            generator_slots=["g0", "g1"],
            source_partition_count=1,
            slot_id="t0",
            worker_id="w0",
            index_in_role=0,
        )
        with pytest.raises(ValueError, match="does not match the group spec"):
            client.initialize(FakeEngine(plan), source_partition=0)

    def test_missing_local_storage_fails_before_the_worker_joins(self, fake_nccl):
        class MissingLocal(FakeEngine):
            def local_params(self):
                return {"a": LocalParamSpec(base="buf::a")}

        rz = FakeRendezvous()
        client = trainer(rz, MissingLocal())
        with pytest.raises(KeyError, match="no local storage"):
            client.compute_plan()
        assert rz.joins == 0


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

    def test_a_trainer_aborts_even_when_reporting_the_failure_fails(
        self, fake_nccl, monkeypatch
    ):
        """Reporting is a network call and the abort is not.

        Ordering them report-then-abort means one unreachable control plane
        leaves every peer parked in NCCL waiting for a rank that has already
        given up. The abort has to happen first and unconditionally, and the
        caller has to keep seeing the failure that actually happened rather
        than the RPC that failed while describing it.
        """
        rz = FakeRendezvous()
        monkeypatch.setattr(
            rz, "report", _raises(RuntimeError("control plane unreachable"))
        )
        client = trainer(rz, FakeEngine())
        client.compute_plan()
        client.start_weight_update("v1")
        assert len(client._cache) > 0, (
            "control: the cache must be populated, or the post-abort "
            "assertion below passes without anything having been aborted"
        )

        monkeypatch.setattr(
            client._half, "finish_weight_update", _raises(RuntimeError("nccl timeout"))
        )
        with pytest.raises(RuntimeError, match="nccl timeout"):
            client.finish_weight_update("v1", operation_id="op1")

        assert len(client._cache) == 0

    def test_a_generator_aborts_even_when_reporting_the_failure_fails(
        self, fake_nccl, monkeypatch
    ):
        rz = FakeRendezvous()
        monkeypatch.setattr(
            rz, "report", _raises(RuntimeError("control plane unreachable"))
        )
        client = RefitClientGenerator(
            rendezvous=rz,
            model_name="m",
            trainer_slots=["t0", "t1"],
            generator_slots=["g0", "g1"],
            source_partition_count=1,
            slot_id="g0",
            worker_id="w9",
            index_in_role=0,
        )
        client.initialize(FakeEngine())
        client.compute_plan()
        client.start_weight_update("v1")
        assert len(client._cache) > 0, (
            "control: the cache must be populated, or the post-abort "
            "assertion below passes without anything having been aborted"
        )

        monkeypatch.setattr(
            client._half, "finish_weight_update", _raises(RuntimeError("nccl timeout"))
        )
        with pytest.raises(RuntimeError, match="nccl timeout"):
            client.finish_weight_update("v1", operation_id="op1")

        assert len(client._cache) == 0


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

    def test_cleanup_is_safe_before_initialize(self):
        client = RefitClientTrainer(
            rendezvous=FakeRendezvous(),
            model_name="m",
            trainer_slots=["t0"],
            generator_slots=["g0"],
            source_partition_count=1,
            slot_id="t0",
            worker_id="w0",
            index_in_role=0,
        )
        client.cleanup()
