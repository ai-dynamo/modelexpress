# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Two-sided refit lifecycle for the NCCL M2N collective path.

The sequencing rules are the contract, and most of them exist because breaking
them produces a hang rather than an error: entering a collective before the
communicators exist, or before MX says the far side is ready, leaves ranks
blocked on peers that will never arrive.
"""

import sys
from contextlib import contextmanager, nullcontext
from types import ModuleType, SimpleNamespace

import pytest

import modelexpress_rl.collective.backend as collective_backend
import modelexpress_rl.collective.client as collective_client
import modelexpress_rl.collective.envs as envs
from modelexpress_rl.collective import (
    CommunicatorCache,
    CommunicatorWedgedError,
    LaneCommunicator,
    LaneKey,
    LocalParamSpec,
    MeshSpec,
    MiscParam,
    ParamPlan,
    Placement,
    RefitClientGenerator,
    RefitClientTrainer,
    RendezvousError,
    ReshardPlan,
)
from modelexpress_rl import refit_collective_pb2 as pb
from modelexpress_rl.collective.comm import new_unique_id
from modelexpress_rl.collective.rendezvous import (
    EpochChangedError,
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
        self.reports = []
        self.joins = 0
        self.freshness_reads = []
        self.stale = None
        self.binds = []
        self.bind_error = None

    def join(self, **kwargs):
        epoch = self._epochs[min(self.joins, len(self._epochs) - 1)]
        self.joins += 1
        self._holder = (kwargs["slot_id"], kwargs["worker_id"])
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

    def _participants(self):
        slot_id, worker_id = self._holder
        return [SimpleNamespace(slot_id=slot_id, worker_id=worker_id)]

    def await_ready(self, *, group_id, epoch, **kwargs):
        lanes = [
            SimpleNamespace(
                lane_id=0,
                nccl_unique_id=b"\x00" * 128,
                participants=self._participants(),
            ),
            SimpleNamespace(
                lane_id=1,
                nccl_unique_id=b"\x01" * 128,
                participants=self._participants(),
            ),
        ]
        return SimpleNamespace(group_id=group_id, epoch=epoch, lanes=lanes)

    def check_current(self, *, group_id, epoch):
        self.freshness_reads.append((group_id, epoch))
        if self.stale is not None:
            raise self.stale

    def report(self, **kwargs):
        self.reports.append(kwargs)
        return SimpleNamespace(operation_id=kwargs["operation_id"])

    def bind_transfer(self, membership, operation_id, version):
        self.binds.append((membership.group_id, membership.epoch, operation_id, version))
        if self.bind_error is not None:
            raise self.bind_error
        return SimpleNamespace(
            operation_id=operation_id,
            group_id=membership.group_id,
            epoch=membership.epoch,
            version_id=version,
        )


class FakeRendezvousPP2(FakeRendezvous):
    def join(self, **kwargs):
        self.joins += 1
        self._holder = (kwargs["slot_id"], kwargs["worker_id"])
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
            SimpleNamespace(
                lane_id=i,
                nccl_unique_id=bytes([i]) * 128,
                participants=self._participants(),
            )
            for i in range(3)
        ]
        return SimpleNamespace(group_id=group_id, epoch=epoch, lanes=lanes)


@pytest.fixture
def fake_nccl(monkeypatch):
    # The fake communicators below have no split(), so these tests run the
    # non-blocking data plane; tests of the blocking one set the flag to "1".
    monkeypatch.setenv("MX_NCCL_REFIT_BLOCKING_DATA_PLANE", "0")
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
    install_fake_m2n_handles(m2n, ops)
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
    # Same reason as the backend tests: the stub stands in for nccl.m2n, so the
    # host's libnccl must not decide whether these CPU-only tests pass.
    monkeypatch.setattr(
        collective_backend, "loaded_nccl_version", lambda: collective_backend.MIN_NCCL
    )
    monkeypatch.setattr(
        collective_client, "_bootstrap_barrier", lambda lane, device: None
    )
    monkeypatch.setattr(
        "modelexpress_rl.collective.comm.LaneCommunicator.synchronize",
        lambda self, timeout_s=None: None,
    )
    return ops


def install_fake_m2n_handles(m2n, timeline=None):
    """Give a stub ``nccl.m2n`` the explicit-handle API, logging its lifecycle.

    ``timeline``, when given, also receives ``"m2n_destroy"`` so a test can
    order the handle's destruction against communicator aborts.
    """
    m2n.events = []

    class Handle:
        def __init__(self, serial):
            self.serial = serial
            self.destroyed = False

        def destroy(self):
            self.destroyed = True
            m2n.events.append(("destroy", self.serial))
            if timeline is not None:
                timeline.append("m2n_destroy")

    class Config:
        pass

    def init(config=None):
        handle = Handle(len([e for e in m2n.events if e[0] == "init"]))
        m2n.events.append(("init", handle.serial))
        return handle

    m2n.depth = 0

    @contextmanager
    def group():
        m2n.events.append(("group_start",))
        m2n.depth += 1
        try:
            yield
        except BaseException:
            m2n.events.append(("group_abort",))
            raise
        else:
            m2n.events.append(("group_end",))
        finally:
            m2n.depth -= 1

    m2n.Config = Config
    m2n.init = init
    m2n.group = group
    return m2n


def handle_events(m2n):
    """The handle lifecycle and communicator aborts, without group bookkeeping."""
    return [event for event in m2n.events if event[0] in ("init", "destroy", "comm_abort")]


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
        worker_id=kw.pop("worker_id", "w0"),
        index_in_role=0,
        **kw,
    )
    client.initialize(engine, source_partition=0)
    return client


class TestSupersededSlot:
    def test_a_displaced_generation_refuses_to_enter_the_collective(self, fake_nccl):
        """A same-slot replacement in a clean FORMING epoch does not bump the
        epoch, so the displaced worker still sees READY. It must notice that
        its slot now belongs to someone else before building communicators."""

        class Displaced(FakeRendezvous):
            def _participants(self):
                slot_id, _ = self._holder
                return [SimpleNamespace(slot_id=slot_id, worker_id="w0-replacement")]

        client = trainer(Displaced(), FakeEngine())
        with pytest.raises(RendezvousError, match="superseded"):
            client.compute_plan()


class TestSequencing:
    def test_a_transfer_before_compute_plan_is_refused(self):
        client = trainer(FakeRendezvous(), FakeEngine())
        with pytest.raises(RuntimeError, match="compute_plan must run"):
            client.start_weight_update("v1", "op1")

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
        client.start_weight_update("v1", "op1")
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

    def test_a_failed_bootstrap_closes_the_rendezvous_so_the_epoch_is_not_reused(
        self, fake_nccl, monkeypatch
    ):
        # The local communicators of this epoch were just aborted. A renewed
        # registration would keep the group READY at the same epoch with the
        # same bootstrap ids, so a same-process retry would re-enter them.
        # Bootstrap failure is fatal for the worker: the lease stops renewing,
        # its TTL moves the epoch, and a restarted worker joins the new one.
        class ClosingRendezvous(FakeRendezvous):
            closed = False

            def close(self):
                self.closed = True

            def join(self, **kwargs):
                # As CollectiveRendezvous: a closed rendezvous refuses to join.
                if self.closed:
                    raise RendezvousError("the collective rendezvous is closed")
                return super().join(**kwargs)

        rz = ClosingRendezvous()
        client = trainer(rz, FakeEngine())
        monkeypatch.setattr(
            collective_client,
            "_bootstrap_barrier",
            _raises(RuntimeError("barrier failed")),
        )
        with pytest.raises(RuntimeError, match="barrier failed"):
            client.compute_plan()
        assert rz.closed, "the lease kept renewing after the bootstrap failed"
        assert len(client._cache) == 0

        monkeypatch.setattr(
            collective_client, "_bootstrap_barrier", lambda lane, device: None
        )
        with pytest.raises(RendezvousError, match="closed"):
            client.compute_plan()
        assert rz.joins == 1, "the failed epoch was joined again"
        assert len(client._cache) == 0

    def test_every_worker_barriers_between_global_pp_lane_initializations(
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
        original_create = client._cache.create

        def create(key, **kwargs):
            events.append(("create", key.lane_id))
            return original_create(key, **kwargs)

        monkeypatch.setattr(client._cache, "create", create)
        monkeypatch.setattr(
            collective_client,
            "_bootstrap_barrier",
            lambda lane, device: events.append(("barrier", lane.rank)),
        )

        client.compute_plan()

        # This stage-1 trainer is not in lane 0, but it still waits at lane 0's
        # full-group barrier before it is allowed to initialize lane 1.
        assert events == [
            ("create", 2),
            ("barrier", 2),
            ("barrier", 2),
            ("create", 1),
            ("barrier", 2),
        ]

    def test_every_worker_settles_local_lanes_before_each_bootstrap_barrier(
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
        original_create = client._cache.create
        original_settle = client._cache.settle_group

        def create(key, **kwargs):
            events.append(("create", key.lane_id))
            return original_create(key, **kwargs)

        def settle(group_id, epoch, **kwargs):
            events.append(("settle", group_id, epoch))
            return original_settle(group_id, epoch, **kwargs)

        monkeypatch.setattr(client._cache, "create", create)
        monkeypatch.setattr(client._cache, "settle_group", settle)
        monkeypatch.setattr(
            collective_client,
            "_bootstrap_barrier",
            lambda lane, device, **kwargs: events.append(("barrier", lane.rank)),
        )

        client.compute_plan()

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

    def test_blocking_data_plane_is_split_off_the_bounded_comm(
        self, fake_nccl, monkeypatch
    ):
        monkeypatch.setenv("MX_NCCL_REFIT_BLOCKING_DATA_PLANE", "1")
        communicator = sys.modules["nccl.core.communicator"]
        made = {}

        class Child:
            is_valid = True

            def abort(self):
                made["child_aborted"] = True

        class Parent:
            def get_async_error(self):
                return 0

            def split(self, *, color, key, config):
                made["split"] = (color, key, config.blocking)
                return Child()

            def abort(self):
                made["parent_aborted"] = True

        def init(**kwargs):
            made["init_blocking"] = kwargs["config"].blocking
            return Parent()

        communicator.Communicator.init = init
        cache = CommunicatorCache()
        lane = cache.create(
            LaneKey("g", 1, 0),
            rank=1,
            world_size=2,
            unique_id=b"x" * 128,
            device=None,
            stream=None,
            timeout_s=0.1,
        )
        assert made["init_blocking"] is False
        assert made["split"] == (0, 1, True)
        assert isinstance(lane.handle, Child)
        lane.abort()
        assert made == {**made, "child_aborted": True, "parent_aborted": True}

    def test_with_the_flag_off_the_data_plane_is_the_nonblocking_comm(
        self, fake_nccl, monkeypatch
    ):
        monkeypatch.setenv("MX_NCCL_REFIT_BLOCKING_DATA_PLANE", "0")
        communicator = sys.modules["nccl.core.communicator"]

        class Parent:
            def get_async_error(self):
                return 0

            def split(self, **kwargs):
                raise AssertionError("no split without the flag")

            def abort(self):
                pass

        parent = Parent()
        communicator.Communicator.init = lambda **kwargs: parent
        lane = CommunicatorCache().create(
            LaneKey("g", 1, 0),
            rank=0,
            world_size=2,
            unique_id=b"x" * 128,
            device=None,
            stream=None,
            timeout_s=0.1,
        )
        assert lane.handle is parent

    def test_the_data_plane_mode_is_frozen_when_the_cache_is_built(
        self, fake_nccl, monkeypatch
    ):
        # The client hashes the mode it froze, so an env flip afterwards must
        # not move this worker onto a split the digest did not promise.
        monkeypatch.setenv("MX_NCCL_REFIT_BLOCKING_DATA_PLANE", "0")
        cache = CommunicatorCache()
        monkeypatch.setenv("MX_NCCL_REFIT_BLOCKING_DATA_PLANE", "1")
        communicator = sys.modules["nccl.core.communicator"]

        class Parent:
            def get_async_error(self):
                return 0

            def split(self, **kwargs):
                raise AssertionError("the frozen mode does not split")

            def abort(self):
                pass

        parent = Parent()
        communicator.Communicator.init = lambda **kwargs: parent
        lane = cache.create(
            LaneKey("g", 1, 0),
            rank=0,
            world_size=2,
            unique_id=b"x" * 128,
            device=None,
            stream=None,
            timeout_s=0.1,
        )
        assert lane.handle is parent

    def test_a_split_whose_peer_never_joins_aborts_the_parent_at_the_deadline(
        self, fake_nccl, monkeypatch
    ):
        monkeypatch.setenv("MX_NCCL_REFIT_BLOCKING_DATA_PLANE", "1")
        bindings = sys.modules["nccl.bindings.nccl"]
        communicator = sys.modules["nccl.core.communicator"]

        class Parent:
            split_called = False
            aborted = False

            def get_async_error(self):
                return bindings.Result.InProgress if self.split_called else 0

            def split(self, **kwargs):
                self.split_called = True
                return SimpleNamespace(is_valid=False, abort=lambda: None)

            def abort(self):
                self.aborted = True

        parent = Parent()
        communicator.Communicator.init = lambda **kwargs: parent
        with pytest.raises(TimeoutError, match="did not complete"):
            CommunicatorCache().create(
                LaneKey("g", 1, 0),
                rank=0,
                world_size=2,
                unique_id=b"x" * 128,
                device=None,
                stream=None,
                timeout_s=0.01,
            )
        assert parent.aborted

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

    def test_settle_group_polls_every_local_lane_including_the_fresh_one(
        self, fake_nccl
    ):
        bindings = sys.modules["nccl.bindings.nccl"]
        polls = []

        class Flapping:
            def __init__(self, name, in_progress_polls):
                self.name = name
                self.remaining = in_progress_polls

            def get_async_error(self):
                polls.append(self.name)
                if self.remaining:
                    self.remaining -= 1
                    return bindings.Result.InProgress
                return bindings.Result.Success

            def get_last_error(self):
                return ""

        cache = CommunicatorCache()
        cache._lanes[LaneKey("g", 1, 0)] = LaneCommunicator(
            Flapping("lane-0", 0), rank=0, world_size=3, stream=None
        )
        cache._lanes[LaneKey("g", 1, 2)] = LaneCommunicator(
            Flapping("lane-2", 2), rank=0, world_size=4, stream=None
        )
        # A lane from another epoch is not this barrier's concern.
        cache._lanes[LaneKey("g", 0, 1)] = LaneCommunicator(
            Flapping("stale", 0), rank=0, world_size=3, stream=None
        )

        settled = cache.settle_group("g", 1, timeout_s=5.0)

        assert settled == 2
        assert polls == ["lane-0", "lane-2", "lane-2", "lane-2"]

    def test_settle_group_raises_a_clean_error_naming_the_lane_on_async_failure(
        self, fake_nccl
    ):
        class Broken:
            def get_async_error(self):
                return 3

            def get_last_error(self):
                return "ncclInternalError"

        cache = CommunicatorCache()
        cache._lanes[LaneKey("g", 1, 2)] = LaneCommunicator(
            Broken(), rank=1, world_size=4, stream=None
        )

        with pytest.raises(RuntimeError, match="lane 2 of group g at epoch 1"):
            cache.settle_group("g", 1, timeout_s=5.0)

    def test_settle_group_times_out_naming_the_lane(self, fake_nccl):
        bindings = sys.modules["nccl.bindings.nccl"]

        class Stuck:
            def get_async_error(self):
                return bindings.Result.InProgress

            def get_last_error(self):
                return ""

        cache = CommunicatorCache()
        cache._lanes[LaneKey("g", 1, 1)] = LaneCommunicator(
            Stuck(), rank=0, world_size=3, stream=None
        )

        with pytest.raises(
            TimeoutError,
            match="lane 1 of group g at epoch 1 did not settle",
        ):
            cache.settle_group("g", 1, timeout_s=0.05)

    def test_settle_group_re_polls_a_lane_still_initializing_after_create(
        self, fake_nccl
    ):
        # create()'s readiness check can pass while the communicator later
        # reports ncclInProgress again; settle_group must re-poll it.
        bindings = sys.modules["nccl.bindings.nccl"]
        communicator = sys.modules["nccl.core.communicator"]
        polls = []

        class Heals:
            def get_async_error(self):
                polls.append(1)
                if len(polls) == 2:
                    return bindings.Result.InProgress
                return bindings.Result.Success

            def get_last_error(self):
                return ""

            def abort(self):
                pass

        heals = Heals()
        communicator.Communicator.init = lambda **kw: heals

        cache = CommunicatorCache()
        cache.create(
            LaneKey("g", 1, 0),
            rank=0,
            world_size=2,
            unique_id=b"x" * 128,
            device=None,
            stream=None,
            timeout_s=5.0,
        )
        assert len(polls) == 1

        settled = cache.settle_group("g", 1, timeout_s=5.0)

        assert settled == 1
        assert len(polls) == 3

    def test_settle_group_timeout_for_a_lane_stuck_after_create_names_the_lane(
        self, fake_nccl
    ):
        bindings = sys.modules["nccl.bindings.nccl"]
        communicator = sys.modules["nccl.core.communicator"]
        polls = []

        class StuckAfterCreate:
            def get_async_error(self):
                polls.append(1)
                if len(polls) == 1:
                    return bindings.Result.Success
                return bindings.Result.InProgress

            def get_last_error(self):
                return ""

            def abort(self):
                pass

        stuck = StuckAfterCreate()
        communicator.Communicator.init = lambda **kw: stuck

        cache = CommunicatorCache()
        cache.create(
            LaneKey("g", 1, 2),
            rank=0,
            world_size=3,
            unique_id=b"x" * 128,
            device=None,
            stream=None,
            timeout_s=5.0,
        )

        with pytest.raises(
            TimeoutError,
            match="lane 2 of group g at epoch 1 did not settle",
        ):
            cache.settle_group("g", 1, timeout_s=0.05)


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
    @pytest.mark.parametrize(
        ("trainer_count", "partitions"),
        [(5, 2), (2, 4)],
        ids=["count-not-divided", "more-partitions-than-trainers"],
    )
    def test_a_partition_count_that_does_not_divide_the_trainers_is_refused(
        self, trainer_count, partitions
    ):
        # Floor division would silently leave the last trainers off every
        # reshard lane. Their parameters never move and the peers expecting
        # their ranks block, which is the failure mode with no error in it.
        client = RefitClientTrainer(
            rendezvous=FakeRendezvous(),
            model_name="m",
            trainer_slots=[f"t{i}" for i in range(trainer_count)],
            generator_slots=["g0"],
            source_partition_count=partitions,
            slot_id="t0",
            worker_id="w0",
            index_in_role=0,
        )
        with pytest.raises(ValueError, match="must divide the trainer slot count"):
            client._declared_lanes()

    def test_a_split_that_divides_is_unchanged(self):
        client = RefitClientTrainer(
            rendezvous=FakeRendezvous(),
            model_name="m",
            trainer_slots=["t0", "t1", "t2", "t3"],
            generator_slots=["g0"],
            source_partition_count=2,
            slot_id="t0",
            worker_id="w0",
            index_in_role=0,
        )
        lanes = client._declared_lanes()
        assert [lane.trainer_slots for lane in lanes] == [
            ("t0", "t1"),
            ("t2", "t3"),
            ("t0", "t1", "t2", "t3"),
        ]


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
        client.start_weight_update("v1", "op1")
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
        client.start_weight_update("v1", "op1")
        client.update_weights("v1")
        client.finish_weight_update("v1")
        assert ("install", 0) in engine.calls
        assert ("finish",) in engine.calls
        assert fake_nccl == ["reshard", "broadcast"]


class TestReporting:
    def test_early_round_failure_uses_admitted_identity(self, fake_nccl):
        rz = FakeRendezvous()
        client = trainer(rz, FakeEngine())
        client.compute_plan()
        client.report_failure("server-op", RuntimeError("publish failed"))
        assert rz.reports == [
            {
                "operation_id": "server-op",
                "group_id": "g",
                "epoch": 1,
                "worker_id": "w0",
                "succeeded": False,
                "message": "RuntimeError('publish failed')",
            }
        ]

    def test_reporting_before_any_join_is_a_named_error(self, fake_nccl):
        # Without an admitted membership the report would raise an opaque
        # AttributeError on None or attach the failure to a group this
        # client never joined.
        client = trainer(FakeRendezvous(), FakeEngine())
        with pytest.raises(RendezvousError, match="compute_plan has not admitted"):
            client.report_failure("server-op", RuntimeError("publish failed"))
        assert client._rendezvous.reports == []

    def test_success_is_reported_against_the_admitted_epoch(self, fake_nccl):
        rz = FakeRendezvous()
        client = trainer(rz, FakeEngine())
        client.compute_plan()
        client.start_weight_update("v1", "op1")
        client.publish_weights("v1")
        client.finish_weight_update("v1")
        assert rz.reports[0]["succeeded"]
        assert rz.reports[0]["epoch"] == 1
        assert rz.reports[0]["worker_id"] == "w0"

    def test_a_failure_is_reported_and_aborts_the_group(self, fake_nccl, monkeypatch):
        rz = FakeRendezvous()
        client = trainer(rz, FakeEngine())
        client.compute_plan()
        client.start_weight_update("v1", "op1")

        def boom(*args, **kwargs):
            raise RuntimeError("nccl timeout")

        monkeypatch.setattr(client._half, "finish_weight_update", boom)
        with pytest.raises(RuntimeError, match="nccl timeout"):
            client.finish_weight_update("v1")

        assert rz.reports[0]["succeeded"] is False
        assert "nccl timeout" in rz.reports[0]["message"]
        # Abort is what makes the deadline mean anything; without it the peers
        # stay blocked with an error merely attached.
        assert len(client._cache) == 0

    def test_the_round_reports_against_the_operation_start_bound(self, fake_nccl):
        rz = FakeRendezvous()
        client = trainer(rz, FakeEngine())
        client.compute_plan()
        client.start_weight_update("v1", "op1")
        client.publish_weights("v1")
        client.finish_weight_update("v1")
        assert rz.binds == [("g", 1, "op1", "v1")]
        assert [report["operation_id"] for report in rz.reports] == ["op1"]

    def test_create_operation_replays_against_the_admitted_membership(
        self, fake_nccl
    ):
        rz = FakeRendezvous()
        created = []

        def create_transfer(membership, version, *, key_prefix):
            created.append((membership.group_id, membership.epoch, version, key_prefix))
            return SimpleNamespace(operation_id=f"op-{version}")

        rz.create_transfer = create_transfer
        client = trainer(rz, FakeEngine())
        client.compute_plan()
        operation_id = client.create_operation("v1", key_prefix="run")
        assert operation_id == "op-v1"
        assert created == [("g", 1, "v1", "run")]
        client.start_weight_update("v1", operation_id)
        assert rz.binds == [("g", 1, "op-v1", "v1")]

    def test_start_requires_an_operation(self, fake_nccl):
        client = trainer(FakeRendezvous(), FakeEngine())
        client.compute_plan()
        with pytest.raises(TypeError):
            client.start_weight_update("v1")

    def test_a_mismatched_operation_is_refused_before_any_nccl_work(self, fake_nccl):
        from modelexpress_rl.collective import TransferBindingError

        rz = FakeRendezvous()
        rz.bind_error = TransferBindingError("version_id is 'v0', expected 'v1'")
        engine = FakeEngine()
        client = trainer(rz, engine)
        client.compute_plan()
        with pytest.raises(TransferBindingError, match="version_id"):
            client.start_weight_update("v1", "op-for-v0")
        # Nothing was prepared, nothing was issued, nothing was reported, and
        # the round never opened.
        assert ("start", "v1") not in engine.calls
        assert fake_nccl == []
        assert rz.reports == []
        with pytest.raises(RuntimeError, match="start_weight_update must run"):
            client.publish_weights("v1")
        # The worker can still start the round once handed the right operation.
        rz.bind_error = None
        client.start_weight_update("v1", "op1")
        client.publish_weights("v1")
        client.finish_weight_update("v1")
        assert rz.reports[-1]["operation_id"] == "op1"

    def test_the_finish_alias_must_name_the_bound_operation(self, fake_nccl):
        rz = FakeRendezvous()
        client = trainer(rz, FakeEngine())
        client.compute_plan()
        client.start_weight_update("v1", "op1")
        client.publish_weights("v1")
        with pytest.raises(ValueError, match="bound to 'op1'"):
            client.finish_weight_update("v1", operation_id="op2")
        client.finish_weight_update("v1", operation_id="op1")
        assert [report["operation_id"] for report in rz.reports] == ["op1"]

    def test_a_second_round_cannot_open_over_one_in_flight(self, fake_nccl):
        rz = FakeRendezvous()
        client = trainer(rz, FakeEngine())
        client.compute_plan()
        client.start_weight_update("v1", "op1")
        with pytest.raises(RuntimeError, match="still in flight"):
            client.start_weight_update("v2", "op2")
        assert rz.binds == [("g", 1, "op1", "v1")]

    def test_a_generator_binds_and_reports_the_same_operation(self, fake_nccl):
        rz = FakeRendezvous()
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
        client.start_weight_update("v1", "op1")
        client.update_weights("v1")
        client.finish_weight_update("v1")
        assert rz.binds == [("g", 1, "op1", "v1")]
        assert rz.reports[0]["operation_id"] == "op1"
        assert rz.reports[0]["worker_id"] == "w9"

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
        client.start_weight_update("v1", "op1")
        assert len(client._cache) > 0, (
            "control: the cache must be populated, or the post-abort "
            "assertion below passes without anything having been aborted"
        )

        monkeypatch.setattr(
            client._half, "finish_weight_update", _raises(RuntimeError("nccl timeout"))
        )
        with pytest.raises(RuntimeError, match="nccl timeout"):
            client.finish_weight_update("v1")

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
        client.start_weight_update("v1", "op1")
        assert len(client._cache) > 0, (
            "control: the cache must be populated, or the post-abort "
            "assertion below passes without anything having been aborted"
        )

        monkeypatch.setattr(
            client._half, "finish_weight_update", _raises(RuntimeError("nccl timeout"))
        )
        with pytest.raises(RuntimeError, match="nccl timeout"):
            client.finish_weight_update("v1")

        assert len(client._cache) == 0


class TestPerGroupFailure:
    def test_a_failed_publish_aborts_the_lanes_and_closes_the_round(
        self, fake_nccl, monkeypatch
    ):
        client = trainer(FakeRendezvous(), FakeEngine())
        client.compute_plan()
        client.start_weight_update("v1", "op1")
        assert len(client._cache) > 0
        monkeypatch.setattr(
            collective_backend, "_reshard", _raises(RuntimeError("enqueue failed"))
        )
        with pytest.raises(RuntimeError, match="enqueue failed"):
            client.publish_weights("v1")
        assert len(client._cache) == 0
        with pytest.raises(RuntimeError, match="start_weight_update must run"):
            client.finish_weight_update("v1")

    def test_a_failed_install_aborts_the_lanes_and_closes_the_round(
        self, fake_nccl, monkeypatch
    ):
        # Depth 1 installs inside update_weights, which is the path under test.
        monkeypatch.setenv("MX_NCCL_REFIT_PIPELINE_DEPTH", "1")
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
        client.start_weight_update("v1", "op1")
        monkeypatch.setattr(engine, "install", _raises(RuntimeError("install failed")))
        with pytest.raises(RuntimeError, match="install failed"):
            client.update_weights("v1")
        assert len(client._cache) == 0
        with pytest.raises(RuntimeError, match="start_weight_update must run"):
            client.update_weights("v1")


def _generator(rz, engine):
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
    client.initialize(engine)
    client.compute_plan()
    return client


class TestOneFailurePath:
    """Every failure after binding aborts, reports the bound operation, and resets."""

    @staticmethod
    def _assert_failed_round(client, rz, message, *, method="publish_weights"):
        assert len(client._cache) == 0
        assert [r["operation_id"] for r in rz.reports] == ["op1"]
        assert rz.reports[0]["succeeded"] is False
        assert message in rz.reports[0]["message"]
        assert client._operation_id is None
        assert client._version is None
        with pytest.raises(RuntimeError, match="start_weight_update must run"):
            getattr(client, method)("v1")

    def test_a_failed_publish_reports_the_bound_operation(self, fake_nccl, monkeypatch):
        rz = FakeRendezvous()
        client = trainer(rz, FakeEngine())
        client.compute_plan()
        client.start_weight_update("v1", "op1")
        monkeypatch.setattr(
            collective_backend, "_reshard", _raises(RuntimeError("enqueue failed"))
        )
        with pytest.raises(RuntimeError, match="enqueue failed"):
            client.publish_weights("v1")
        self._assert_failed_round(client, rz, "enqueue failed")

    def test_a_failed_publisher_start_reports_the_bound_operation(
        self, fake_nccl, monkeypatch
    ):
        rz = FakeRendezvous()
        engine = FakeEngine()
        client = trainer(rz, engine)
        client.compute_plan()
        monkeypatch.setattr(
            engine, "start_new_round", _raises(RuntimeError("prepare failed"))
        )
        with pytest.raises(RuntimeError, match="prepare failed"):
            client.start_weight_update("v1", "op1")
        self._assert_failed_round(client, rz, "prepare failed")
        assert "reshard" not in fake_nccl and "broadcast" not in fake_nccl

    def test_a_failed_issue_reports_the_bound_operation(self, fake_nccl, monkeypatch):
        rz = FakeRendezvous()
        client = _generator(rz, FakeEngine())
        client.start_weight_update("v1", "op1")
        monkeypatch.setattr(
            collective_backend, "_reshard", _raises(RuntimeError("recv failed"))
        )
        with pytest.raises(RuntimeError, match="recv failed"):
            client.update_weights("v1")
        self._assert_failed_round(client, rz, "recv failed", method="update_weights")
        assert not client._unfinished

    def test_a_failed_install_reports_the_bound_operation(self, fake_nccl, monkeypatch):
        monkeypatch.setenv("MX_NCCL_REFIT_PIPELINE_DEPTH", "1")
        rz = FakeRendezvous()
        engine = FakeEngine()
        client = _generator(rz, engine)
        client.start_weight_update("v1", "op1")
        monkeypatch.setattr(engine, "install", _raises(RuntimeError("install failed")))
        with pytest.raises(RuntimeError, match="install failed"):
            client.update_weights("v1")
        self._assert_failed_round(
            client, rz, "install failed", method="update_weights"
        )

    def test_an_interrupt_mid_round_still_releases_and_reports(
        self, fake_nccl, monkeypatch
    ):
        rz = FakeRendezvous()
        client = trainer(rz, FakeEngine())
        client.compute_plan()
        client.start_weight_update("v1", "op1")
        monkeypatch.setattr(collective_backend, "_reshard", _raises(KeyboardInterrupt()))
        with pytest.raises(KeyboardInterrupt):
            client.publish_weights("v1")
        self._assert_failed_round(client, rz, "KeyboardInterrupt")

    def test_a_raising_abort_still_reports_and_resets(self, fake_nccl, monkeypatch):
        rz = FakeRendezvous()
        client = trainer(rz, FakeEngine())
        client.compute_plan()
        client.start_weight_update("v1", "op1")
        monkeypatch.setattr(
            collective_backend, "_reshard", _raises(RuntimeError("enqueue failed"))
        )
        monkeypatch.setattr(client._half, "abort", _raises(RuntimeError("abort broke")))
        with pytest.raises(RuntimeError, match="enqueue failed"):
            client.publish_weights("v1")
        assert rz.reports[0]["succeeded"] is False
        assert client._operation_id is None

    def test_a_wedged_abort_reports_failed_then_re_raises_as_fatal(
        self, fake_nccl, monkeypatch
    ):
        # A wedged communicator abort cannot be swallowed like a dirty teardown:
        # the round is still reported FAILED so peers re-form, but the fatal
        # CommunicatorWedgedError then surfaces so the runner can exit.
        from modelexpress_rl.collective import CommunicatorWedgedError

        rz = FakeRendezvous()
        client = trainer(rz, FakeEngine())
        client.compute_plan()
        client.start_weight_update("v1", "op1")

        monkeypatch.setattr(
            client._half, "finish_weight_update", _raises(TimeoutError("deadline"))
        )
        wedge = CommunicatorWedgedError("abort stuck", group_id="g", epoch=1, lane_id=1)
        monkeypatch.setattr(client._half, "abort", _raises(wedge))

        with pytest.raises(CommunicatorWedgedError) as caught:
            client.finish_weight_update("v1")
        assert caught.value is wedge
        # Reported FAILED before the fatal error surfaced, and the round reset.
        assert rz.reports[0]["succeeded"] is False
        assert client._operation_id is None and client._version is None

    def test_a_refused_binding_reports_nothing_and_keeps_the_lanes(self, fake_nccl):
        from modelexpress_rl.collective import TransferBindingError

        rz = FakeRendezvous()
        rz.bind_error = TransferBindingError("epoch is 0, expected 1")
        client = trainer(rz, FakeEngine())
        client.compute_plan()
        lanes = len(client._cache)
        with pytest.raises(TransferBindingError):
            client.start_weight_update("v1", "stale-op")
        # Nothing ran, so there is nothing to release and no operation to blame.
        assert rz.reports == []
        assert len(client._cache) == lanes

    def test_cleanup_mid_round_reports_the_bound_operation(self, fake_nccl):
        rz = FakeRendezvous()
        client = trainer(rz, FakeEngine())
        client.compute_plan()
        client.start_weight_update("v1", "op1")
        client.cleanup()
        assert [r["operation_id"] for r in rz.reports] == ["op1"]
        assert rz.reports[0]["succeeded"] is False
        assert "cleanup ran" in rz.reports[0]["message"]
        assert len(client._cache) == 0

    def test_cleanup_after_a_finished_round_reports_nothing_more(self, fake_nccl):
        rz = FakeRendezvous()
        client = trainer(rz, FakeEngine())
        client.compute_plan()
        client.start_weight_update("v1", "op1")
        client.publish_weights("v1")
        client.finish_weight_update("v1")
        client.cleanup()
        assert [r["succeeded"] for r in rz.reports] == [True]


THREE_GROUPS = ReshardPlan(bulk=[entry("a"), entry("b"), entry("c")])


def pipelined_generator(engine):
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
    client.setup_layer_groups([["a"], ["b"], ["c"]])
    client.initialize(engine)
    client.compute_plan()
    return client


def installs(engine):
    return [call[1] for call in engine.calls if call[0] == "install"]


class TestLayerGroupPipeline:
    def test_depth_two_installs_each_group_while_the_next_is_issued(
        self, fake_nccl, monkeypatch
    ):
        monkeypatch.setenv("MX_NCCL_REFIT_PIPELINE_DEPTH", "2")
        engine = FakeEngine(plan=THREE_GROUPS)
        client = pipelined_generator(engine)
        client.start_weight_update("v1", "op1")
        client.update_weights("v1", 0)
        assert installs(engine) == []
        client.update_weights("v1", 1)
        assert installs(engine) == [0]
        client.update_weights("v1", 2)
        assert installs(engine) == [0, 1]
        client.finish_weight_update("v1")
        assert installs(engine) == [0, 1, 2]
        # finish() refreshes derived state, so it must follow the last install.
        assert [call[0] for call in engine.calls][-2:] == ["install", "finish"]

    def test_depth_one_installs_before_update_weights_returns(
        self, fake_nccl, monkeypatch
    ):
        monkeypatch.setenv("MX_NCCL_REFIT_PIPELINE_DEPTH", "1")
        engine = FakeEngine(plan=THREE_GROUPS)
        client = pipelined_generator(engine)
        client.start_weight_update("v1", "op1")
        for group in range(3):
            client.update_weights("v1", group)
            assert installs(engine) == list(range(group + 1))
        client.finish_weight_update("v1")

    def test_a_deferred_install_that_fails_aborts_at_finish_and_reports(
        self, fake_nccl, monkeypatch
    ):
        monkeypatch.setenv("MX_NCCL_REFIT_PIPELINE_DEPTH", "2")
        engine = FakeEngine(plan=THREE_GROUPS)
        client = pipelined_generator(engine)
        rz = client._rendezvous
        client.start_weight_update("v1", "op1")
        for group in range(3):
            client.update_weights("v1", group)
        monkeypatch.setattr(engine, "install", _raises(RuntimeError("install failed")))
        with pytest.raises(RuntimeError, match="install failed"):
            client.finish_weight_update("v1")
        assert rz.reports[0]["succeeded"] is False
        assert len(client._cache) == 0
        assert not client._unfinished
        with pytest.raises(RuntimeError, match="start_weight_update must run"):
            client.update_weights("v1", 0)


class TestLayerGroupContract:
    def test_a_repeated_group_is_refused_and_aborts_the_round(self, fake_nccl):
        client = trainer(FakeRendezvous(), FakeEngine())
        client.compute_plan()
        client.start_weight_update("v1", "op1")
        client.publish_weights("v1")
        with pytest.raises(RuntimeError, match="already ran in this round"):
            client.publish_weights("v1")
        assert len(client._cache) == 0

    def test_a_finish_that_skipped_a_group_fails_and_reports(self, fake_nccl):
        rz = FakeRendezvous()
        client = trainer(rz, FakeEngine())
        client.compute_plan()
        client.start_weight_update("v1", "op1")
        with pytest.raises(RuntimeError, match="before layer group"):
            client.finish_weight_update("v1")
        assert rz.reports[0]["succeeded"] is False
        assert len(client._cache) == 0

    def test_the_grouping_is_part_of_the_admitted_digest(self):
        engine = FakeEngine()
        ungrouped = trainer(FakeRendezvous(), engine)
        grouped = RefitClientTrainer(
            rendezvous=FakeRendezvous(),
            model_name="m",
            trainer_slots=["t0", "t1"],
            generator_slots=["g0", "g1"],
            source_partition_count=1,
            slot_id="t0",
            worker_id="w0",
            index_in_role=0,
        )
        names = [entry.name for entry in engine.capture().bulk]
        grouped.setup_layer_groups([names])
        grouped.initialize(engine, source_partition=0)
        assert grouped._digest != ungrouped._digest

    def test_regrouping_after_compute_plan_is_refused(self, fake_nccl):
        client = trainer(FakeRendezvous(), FakeEngine())
        client.compute_plan()
        with pytest.raises(RuntimeError, match="before compute_plan"):
            client.setup_layer_groups(None)


class TestLayerGroupsBeforeJoin:
    """A grouping the captured plan cannot execute is refused before the
    worker joins, not after READY and communicator bootstrap."""

    @staticmethod
    def _unjoined_trainer(rz):
        client = RefitClientTrainer(
            rendezvous=rz,
            model_name="m",
            trainer_slots=["t0", "t1"],
            generator_slots=["g0", "g1"],
            source_partition_count=1,
            slot_id="t0",
            worker_id="w0",
            index_in_role=0,
        )
        client.initialize(FakeEngine(), source_partition=0)
        return client

    @staticmethod
    def _unjoined_generator(rz):
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
        return client

    @pytest.mark.parametrize("make", ["_unjoined_trainer", "_unjoined_generator"])
    @pytest.mark.parametrize(
        "groupings, error",
        [
            ([["no.such.param"]], KeyError),
            ([[]], ValueError),
        ],
    )
    def test_an_invalid_grouping_never_joins(self, fake_nccl, make, groupings, error):
        rz = FakeRendezvous()
        client = getattr(self, make)(rz)
        client.setup_layer_groups(groupings)
        with pytest.raises(error):
            client.compute_plan()
        assert rz.joins == 0
        assert len(client._cache) == 0

    @pytest.mark.parametrize("half", ["NcclM2nSender", "NcclM2nReceiver"])
    def test_a_wedged_abort_after_ready_is_kept_for_cleanup(
        self, fake_nccl, monkeypatch, half
    ):
        rz = FakeRendezvous()
        client = (
            self._unjoined_trainer(rz)
            if half == "NcclM2nSender"
            else self._unjoined_generator(rz)
        )
        monkeypatch.setattr(
            collective_client, half, _raises(RuntimeError("half setup failed"))
        )
        wedge = CommunicatorWedgedError("abort stuck", group_id="g", epoch=1)
        aborts = []

        def abort_group(group_id):
            # Like the real cache, a wedged abort still drops every lane, so
            # only the first call has anything left to wedge on.
            aborts.append(group_id)
            if len(aborts) == 1:
                raise wedge
            return 0

        monkeypatch.setattr(client._cache, "abort_group", abort_group)
        with pytest.raises(CommunicatorWedgedError):
            client.compute_plan()
        with pytest.raises(CommunicatorWedgedError) as raised:
            client.cleanup()
        assert raised.value is wedge


class TestSourcePartition:
    def _client(self, slot_id):
        return RefitClientTrainer(
            rendezvous=FakeRendezvous(),
            model_name="m",
            trainer_slots=["t0", "t1", "t2", "t3"],
            generator_slots=["g0"],
            source_partition_count=2,
            slot_id=slot_id,
            worker_id="w0",
            index_in_role=0,
        )

    @pytest.mark.parametrize(
        ("slot_id", "partition", "error"),
        [
            ("t0", 1, "does not match the reshard lane"),
            ("g0", 0, "not on any reshard lane"),
        ],
        ids=["partition-off-the-slots-lane", "slot-outside-the-trainer-slots"],
    )
    def test_a_slot_and_partition_that_disagree_are_refused_before_joining(
        self, slot_id, partition, error
    ):
        with pytest.raises(ValueError, match=error):
            self._client(slot_id).initialize(FakeEngine(), source_partition=partition)


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


class TestFreshnessAtStart:
    """A membership change between refits moves the epoch. start_weight_update
    reads it once, before the engine prepares anything."""

    def test_a_trainer_reads_the_group_before_preparing(self, fake_nccl):
        rz, engine = FakeRendezvous(epochs=(3,)), FakeEngine()
        client = trainer(rz, engine)
        client.compute_plan()
        client.start_weight_update("v1", "op1")
        assert rz.freshness_reads == [("g", 3)]

    def test_a_moved_group_is_refused_before_the_engine_prepares(self, fake_nccl):
        rz, engine = FakeRendezvous(), FakeEngine()
        client = trainer(rz, engine)
        client.compute_plan()
        rz.stale = EpochChangedError("g", 1, 2)
        with pytest.raises(EpochChangedError):
            client.start_weight_update("v1", "op1")
        assert ("start", "v1") not in engine.calls
        with pytest.raises(RuntimeError, match="start_weight_update must run"):
            client.publish_weights("v1")

    def test_a_generator_is_refused_too(self, fake_nccl):
        rz, engine = FakeRendezvous(), FakeEngine()
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
        client.initialize(engine)
        client.compute_plan()
        rz.stale = EpochChangedError("g", 1, 2)
        with pytest.raises(EpochChangedError):
            client.start_weight_update("v1", "op1")
        assert ("start", "v1") not in engine.calls


class TestM2nHandleLifecycle:
    """Each group epoch owns one explicit M2N handle, destroyed before its lanes."""

    @staticmethod
    def _m2n():
        return sys.modules["nccl.m2n"]

    def test_one_handle_serves_every_round_of_an_epoch(self, fake_nccl):
        client = trainer(FakeRendezvous(), FakeEngine())
        client.compute_plan()
        for version in ("v1", "v2"):
            client.start_weight_update(version, f"op-{version}")
            client.publish_weights(version)
            client.finish_weight_update(version)
        assert handle_events(self._m2n()) == [("init", 0)]

    def test_cleanup_destroys_the_handle_before_any_lane_is_aborted(self, fake_nccl):
        client = trainer(FakeRendezvous(), FakeEngine())
        client.compute_plan()
        client.start_weight_update("v1", "op1")
        client.publish_weights("v1")
        client.finish_weight_update("v1")
        fake_nccl.clear()
        client.cleanup()
        assert fake_nccl[0] == "m2n_destroy"
        assert fake_nccl.count("abort") == 2
        assert handle_events(self._m2n()) == [("init", 0), ("destroy", 0)]

    def test_an_epoch_move_destroys_the_old_handle_before_its_lanes(self, fake_nccl):
        client = trainer(FakeRendezvous(epochs=(1, 2)), FakeEngine())
        client.compute_plan()
        client.start_weight_update("v1", "op1")
        client.publish_weights("v1")
        client.finish_weight_update("v1")
        fake_nccl.clear()
        membership = client.compute_plan()
        assert membership.epoch == 2
        # The stale epoch-1 lanes are dropped only after the handle that cached
        # resources against them is gone.
        assert fake_nccl.index("m2n_destroy") < fake_nccl.index("abort")
        client.start_weight_update("v2", "op2")
        client.publish_weights("v2")
        client.finish_weight_update("v2")
        assert handle_events(self._m2n()) == [("init", 0), ("destroy", 0), ("init", 1)]

    def test_a_failed_round_destroys_the_handle_before_aborting_lanes(
        self, fake_nccl, monkeypatch
    ):
        rz = FakeRendezvous()
        client = trainer(rz, FakeEngine())
        client.compute_plan()
        client.start_weight_update("v1", "op1")
        monkeypatch.setattr(
            collective_backend, "_reshard", _raises(RuntimeError("enqueue failed"))
        )
        fake_nccl.clear()
        with pytest.raises(RuntimeError, match="enqueue failed"):
            client.publish_weights("v1")
        assert fake_nccl[0] == "m2n_destroy"
        assert "abort" in fake_nccl
        assert rz.reports[0]["succeeded"] is False

    def test_compute_plan_mid_round_fails_that_round(self, fake_nccl):
        rz = FakeRendezvous(epochs=(1, 2))
        client = trainer(rz, FakeEngine())
        client.compute_plan()
        client.start_weight_update("v1", "op1")
        client.compute_plan()
        assert rz.reports[0]["operation_id"] == "op1"
        assert rz.reports[0]["succeeded"] is False
        assert "compute_plan re-ran" in rz.reports[0]["message"]
        assert handle_events(self._m2n()) == [("init", 0), ("destroy", 0)]


class TestCleanup:
    def test_cleanup_releases_the_engine_and_the_communicators(self, fake_nccl):
        engine = FakeEngine()
        client = trainer(FakeRendezvous(), engine)
        client.compute_plan()
        client.cleanup()
        assert ("cleanup",) in engine.calls
        assert len(client._cache) == 0

    @staticmethod
    def _generator(engine):
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
        return client

    @pytest.mark.parametrize("role", ["trainer", "generator"])
    def test_cleanup_releases_lanes_before_the_engine_frees_buffers(
        self, fake_nccl, role
    ):
        # Mid-round, the handle and the lanes may still reference the engine's
        # buffers: they must be gone before the engine frees them.
        engine = FakeEngine()
        engine.cleanup = lambda: fake_nccl.append("engine_cleanup")
        if role == "trainer":
            client = trainer(FakeRendezvous(), engine)
        else:
            client = self._generator(engine)
        client.compute_plan()
        client.start_weight_update("v1", "op1")
        fake_nccl.clear()
        client.cleanup()
        assert fake_nccl[-1] == "engine_cleanup"
        assert fake_nccl.count("engine_cleanup") == 1
        assert fake_nccl.index("m2n_destroy") < fake_nccl.index("engine_cleanup")
        assert fake_nccl.count("abort") == 2

    @pytest.mark.parametrize("role", ["trainer", "generator"])
    def test_the_engine_is_cleaned_up_even_if_lane_teardown_raises(
        self, fake_nccl, role, monkeypatch
    ):
        engine = FakeEngine()
        if role == "trainer":
            client = trainer(FakeRendezvous(), engine)
        else:
            client = self._generator(engine)
        client.compute_plan()
        monkeypatch.setattr(
            client._cache, "abort_group", _raises(RuntimeError("abort failed"))
        )
        with pytest.raises(RuntimeError, match="abort failed"):
            client.cleanup()
        assert ("cleanup",) in engine.calls

    @pytest.mark.parametrize("where", ["cleanup", "round"])
    @pytest.mark.parametrize("role", ["trainer", "generator"])
    def test_a_wedged_abort_keeps_the_buffers_gpu_work_may_still_use(
        self, fake_nccl, role, where, monkeypatch
    ):
        # In-flight work does not quiesce (the handle is quarantined) and the
        # communicator abort wedges: the M2N kernel may still be reading or
        # writing the pending buffers, so neither the half's pending contexts
        # nor the engine's buffers may be freed before the process exits.
        import gc
        import threading
        import weakref

        import modelexpress_rl.collective.comm as collective_comm
        from modelexpress_rl.collective import CommunicatorWedgedError

        engine = FakeEngine()
        if role == "trainer":
            client = trainer(FakeRendezvous(), engine)
        else:
            client = self._generator(engine)
        client.compute_plan()
        client.start_weight_update("v1", "op1")
        if role == "trainer":
            client.publish_weights("v1")
        else:
            client.update_weights("v1")
        half = client._half
        pending = [weakref.ref(ctx) for ctx in half._pending_contexts]
        assert pending, "the round left nothing in flight to protect"

        def never_quiesces(self, timeout_s=None):
            raise TimeoutError("the M2N kernel is still running")

        monkeypatch.setattr(
            "modelexpress_rl.collective.comm.LaneCommunicator.synchronize",
            never_quiesces,
        )
        monkeypatch.setattr(collective_comm, "abort_timeout", lambda: 0.05)
        release = threading.Event()

        class Stuck:
            def abort(self):
                release.wait(timeout=5.0)

        for lane_comm in client._cache._lanes.values():
            lane_comm._comm = Stuck()
        fake_nccl.clear()
        try:
            with pytest.raises(CommunicatorWedgedError):
                if where == "cleanup":
                    client.cleanup()
                else:
                    monkeypatch.setattr(
                        half, "finish_weight_update", _raises(TimeoutError("late"))
                    )
                    client.finish_weight_update("v1")
            if where == "round":
                # A runner that cleans up on the way out must not free them
                # either: the wedge was already raised.
                with pytest.raises(CommunicatorWedgedError):
                    client.cleanup()
            del half
            gc.collect()
            assert "m2n_destroy" not in fake_nccl
            assert all(ref() is not None for ref in pending), (
                "pending buffers were dropped while GPU work may still use them"
            )
            assert ("cleanup",) not in engine.calls
        finally:
            release.set()

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


class TestM2nBuildInDigest:
    @staticmethod
    def _install_fake_metadata(monkeypatch, m2n_version):
        from importlib import metadata
        from pathlib import PurePosixPath

        dists = {
            "nccl4py": ("0.1.0", ["nccl/core/__init__.py"]),
            "nccl-extensions": (m2n_version, ["nccl/m2n/__init__.py"]),
        }

        def distribution(name):
            version, files = dists[name]
            return SimpleNamespace(
                metadata={"Name": name},
                version=version,
                files=[PurePosixPath(f) for f in files],
            )

        monkeypatch.setattr(
            metadata, "packages_distributions", lambda: {"nccl": list(dists)}
        )
        monkeypatch.setattr(metadata, "distribution", distribution)

    def test_the_build_is_read_from_the_distribution_that_ships_m2n(self, monkeypatch):
        self._install_fake_metadata(monkeypatch, "0.3.1")
        assert collective_backend.installed_m2n_build() == "nccl-extensions==0.3.1"

    def test_differing_installed_builds_split_the_digest(self, fake_nccl, monkeypatch):
        digests = []
        for version in ("0.3.1", "0.4.0"):
            self._install_fake_metadata(monkeypatch, version)
            digests.append(trainer(FakeRendezvous(), FakeEngine())._digest)
        assert digests[0] != digests[1]

    def test_an_explicit_version_wins_over_the_installed_build(self, fake_nccl, monkeypatch):
        digests = []
        for version in ("0.3.1", "0.4.0"):
            self._install_fake_metadata(monkeypatch, version)
            client = trainer(FakeRendezvous(), FakeEngine(), m2n_abi_version="pinned")
            digests.append(client._digest)
        assert digests[0] == digests[1]


class _ReportRpcError(Exception):
    def __init__(self, code):
        super().__init__(code.name)
        self._code = code

    def code(self):
        return self._code


class FlakyReportRendezvous(FakeRendezvous):
    """Fails the first ``failures`` success reports with ``code``."""

    def __init__(self, failures, code="UNAVAILABLE"):
        super().__init__()
        import grpc

        self.failures = failures
        self.code = getattr(grpc.StatusCode, code)

    def report(self, **kwargs):
        if kwargs["succeeded"] and self.failures > 0:
            self.failures -= 1
            self.reports.append({**kwargs, "lost": True})
            raise _ReportRpcError(self.code)
        return super().report(**kwargs)


class TerminalReportRendezvous(FakeRendezvous):
    """Answers a report with the operation in ``state``, as MX does once
    another participant already ended it."""

    def __init__(self, state, failure_message="", **kwargs):
        super().__init__(**kwargs)
        self._state = state
        self._failure_message = failure_message

    def report(self, **kwargs):
        super().report(**kwargs)
        return SimpleNamespace(
            operation_id=kwargs["operation_id"],
            state=self._state,
            failure_message=self._failure_message,
        )


class TestTerminalReportResponse:
    @staticmethod
    def _finish(rz):
        client = trainer(rz, FakeEngine())
        client.compute_plan()
        client.start_weight_update("v1", "op1")
        client.publish_weights("v1")
        return client

    @pytest.mark.parametrize(
        "state",
        [
            pb.COLLECTIVE_TRANSFER_STATE_FAILED,
            pb.COLLECTIVE_TRANSFER_STATE_ABORTED,
        ],
    )
    def test_an_already_ended_operation_is_surfaced(self, fake_nccl, state):
        rz = TerminalReportRendezvous(state, failure_message="g1: lane 0 aborted")
        client = self._finish(rz)
        name = pb.CollectiveTransferState.Name(state)
        with pytest.raises(RendezvousError, match=f"{name}.*g1: lane 0 aborted"):
            client.finish_weight_update("v1")
        # Terminal on the server, so there is nothing left to retry.
        assert client._operation_id is None
        client.start_weight_update("v2", "op2")

    @pytest.mark.parametrize(
        "state",
        [
            pb.COLLECTIVE_TRANSFER_STATE_RUNNING,
            pb.COLLECTIVE_TRANSFER_STATE_COMPLETE,
        ],
    )
    def test_running_or_complete_is_a_normal_finish(self, fake_nccl, state):
        client = self._finish(TerminalReportRendezvous(state))
        client.finish_weight_update("v1")
        assert client._operation_id is None


class TestSuccessReportDelivery:
    @staticmethod
    def _finish(rz, monkeypatch):
        monkeypatch.setattr(collective_client.time, "sleep", lambda _s: None)
        client = trainer(rz, FakeEngine())
        client.compute_plan()
        client.start_weight_update("v1", "op1")
        client.publish_weights("v1")
        return client

    def test_a_transient_failure_is_retried_and_then_released(self, fake_nccl, monkeypatch):
        rz = FlakyReportRendezvous(failures=1)
        client = self._finish(rz, monkeypatch)
        client.finish_weight_update("v1")
        assert [r.get("lost", False) for r in rz.reports] == [True, False]
        assert client._operation_id is None
        client.start_weight_update("v2", "op2")

    def test_an_undelivered_report_keeps_the_binding_for_retry(self, fake_nccl, monkeypatch):
        rz = FlakyReportRendezvous(failures=3)
        client = self._finish(rz, monkeypatch)
        with pytest.raises(_ReportRpcError):
            client.finish_weight_update("v1")
        assert len(rz.reports) == 3
        assert client._operation_id == "op1"
        with pytest.raises(RuntimeError, match="retry_report"):
            client.start_weight_update("v2", "op2")
        client.retry_report()
        assert rz.reports[-1]["operation_id"] == "op1"
        assert rz.reports[-1]["succeeded"]
        assert "lost" not in rz.reports[-1]
        assert client._operation_id is None
        client.start_weight_update("v2", "op2")

    def test_a_refusal_is_not_retried(self, fake_nccl, monkeypatch):
        rz = FlakyReportRendezvous(failures=1, code="FAILED_PRECONDITION")
        client = self._finish(rz, monkeypatch)
        with pytest.raises(_ReportRpcError):
            client.finish_weight_update("v1")
        assert len(rz.reports) == 1
        assert client._operation_id is None


class AgreementRendezvous(FakeRendezvous):
    """Holds the cohort FORMING while reported digests disagree.

    Mirrors the digest clause of ``evaluate_readiness`` in the server's
    common.lua: READY needs every reported digest to equal the group's.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.digests = {}
        self.state = "FORMING"

    def join(self, **kwargs):
        self.digests[kwargs["slot_id"]] = kwargs["plan_digest"]
        return super().join(**kwargs)

    def await_ready(self, *, group_id, epoch, **kwargs):
        if len(set(self.digests.values())) != 1:
            self.state = "FORMING"
            raise RendezvousError(
                f"collective group {group_id} stayed FORMING: plan digests disagree"
            )
        self.state = "READY"
        return super().await_ready(group_id=group_id, epoch=epoch, **kwargs)


class TestExecutionModeInDigest:
    """Both modes change what a rank does inside native collectives, so peers
    that disagree on either must never be admitted together."""

    @staticmethod
    def _cohort(monkeypatch, generator_env, trainer_env):
        rz = AgreementRendezvous()
        for name, value in generator_env.items():
            monkeypatch.setenv(name, value)
        _generator(rz, FakeEngine())
        for name, value in trainer_env.items():
            monkeypatch.setenv(name, value)
        client = trainer(rz, FakeEngine())
        return rz, client

    def test_a_blocking_data_plane_mismatch_stays_forming(self, fake_nccl, monkeypatch):
        rz, client = self._cohort(
            monkeypatch,
            {"MX_NCCL_REFIT_BLOCKING_DATA_PLANE": "0"},
            {"MX_NCCL_REFIT_BLOCKING_DATA_PLANE": "1"},
        )
        with pytest.raises(RendezvousError, match="FORMING"):
            client.compute_plan()
        assert rz.state == "FORMING"
        assert rz.digests["t0"] != rz.digests["g0"]

    def test_a_grouped_submission_mismatch_stays_forming(self, fake_nccl, monkeypatch):
        rz, client = self._cohort(
            monkeypatch,
            {"MX_NCCL_REFIT_GROUPED_SUBMISSION": "0"},
            {"MX_NCCL_REFIT_GROUPED_SUBMISSION": "1"},
        )
        with pytest.raises(RendezvousError, match="FORMING"):
            client.compute_plan()
        assert rz.state == "FORMING"

    def test_matching_modes_reach_ready(self, fake_nccl, monkeypatch):
        rz, client = self._cohort(
            monkeypatch,
            {"MX_NCCL_REFIT_GROUPED_SUBMISSION": "1"},
            {"MX_NCCL_REFIT_GROUPED_SUBMISSION": "1"},
        )
        client.compute_plan()
        assert rz.state == "READY"

    def test_the_hashed_mode_is_the_one_the_client_runs(self, fake_nccl, monkeypatch):
        # Frozen at construction: an env flip afterwards changes neither the
        # digest nor the data plane this client bootstraps.
        monkeypatch.setenv("MX_NCCL_REFIT_BLOCKING_DATA_PLANE", "0")
        client = trainer(FakeRendezvous(), FakeEngine())
        digest = client._digest
        monkeypatch.setenv("MX_NCCL_REFIT_BLOCKING_DATA_PLANE", "1")
        client.compute_plan()
        assert client._digest == digest


class TestBlockingDataPlaneDefault:
    def test_unset_means_blocking(self, monkeypatch):
        monkeypatch.delenv("MX_NCCL_REFIT_BLOCKING_DATA_PLANE", raising=False)
        assert envs.MX_NCCL_REFIT_BLOCKING_DATA_PLANE is True

    def test_zero_turns_it_off(self, monkeypatch):
        monkeypatch.setenv("MX_NCCL_REFIT_BLOCKING_DATA_PLANE", "0")
        assert envs.MX_NCCL_REFIT_BLOCKING_DATA_PLANE is False

    def test_other_flags_stay_off_when_unset(self, monkeypatch):
        monkeypatch.delenv("MX_NCCL_REFIT_GROUPED_SUBMISSION", raising=False)
        assert envs.MX_NCCL_REFIT_GROUPED_SUBMISSION is False
