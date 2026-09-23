# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Data-plane behaviour for the NCCL M2N collective path.

Driven against a recording stand-in for nccl4py, because what needs protecting
is the *order and shape* of the operations rather than the bytes NCCL moves.
Every property here corresponds to a failure that would otherwise present as a
hung communicator rather than an exception.
"""

import math
import sys
from types import ModuleType, SimpleNamespace

import pytest

from modelexpress_rl.collective import backend
from modelexpress_rl.collective import (
    CommunicatorCache,
    LaneCommunicator,
    LaneKey,
    LocalParamSpec,
    MeshSpec,
    MiscParam,
    NcclM2nReceiver,
    NcclM2nSender,
    ParamPlan,
    Placement,
    RefitCtx,
    ReshardPlan,
    resolve_specs,
)


class Recorder:
    """Records every wire op both halves issue, in order."""

    def __init__(self):
        self.ops = []


@pytest.fixture
def recorder(monkeypatch):
    rec = Recorder()
    monkeypatch.setattr(
        backend,
        "_allocate_fence_buffer",
        lambda lane: f"fence::{id(lane)}",
        raising=False,
    )

    def reshard(src, dst, comm, **kwargs):
        rec.ops.append(
            SimpleNamespace(
                kind="reshard",
                src=src,
                dst=dst,
                comm=comm,
                kwargs=kwargs,
                src_mesh=kwargs["src_mesh"],
                dst_mesh=kwargs["dst_mesh"],
            )
        )

    module = ModuleType("nccl.m2n")
    module.reshard = reshard
    parent = ModuleType("nccl")
    monkeypatch.setitem(sys.modules, "nccl", parent)
    monkeypatch.setitem(sys.modules, "nccl.m2n", module)
    return rec


class FakeStream:
    def __init__(self, rec, name):
        self._rec = rec
        self._name = name
        self.cuda_stream = hash(name) & 0xFFFF

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def synchronize(self):
        self._rec.ops.append(SimpleNamespace(kind="sync", stream=self._name))


class FakeComm:
    def __init__(self, rec, name):
        self._rec = rec
        self._name = name
        self.aborted = False

    def broadcast(self, sendbuf, recvbuf, root, stream):
        self._rec.ops.append(
            SimpleNamespace(kind="broadcast", buf=sendbuf, root=root, comm=self._name)
        )

    def abort(self):
        self.aborted = True


_DEFAULT_STREAM = object()


def lane(rec, name, *, rank=0, world=4, stream=_DEFAULT_STREAM, device=None):
    if stream is _DEFAULT_STREAM:
        stream = FakeStream(rec, f"stream::{name}")
    return LaneCommunicator(
        FakeComm(rec, name),
        rank=rank,
        world_size=world,
        stream=stream,
        device=device,
    )


def entry(
    name,
    partition=0,
    group_key=None,
    *,
    src_shape=(2,),
    src_rank_offset=0,
):
    return ParamPlan(
        name=name,
        global_shape=(8, 4),
        dtype="bfloat16",
        partition_id=partition,
        src_mesh=MeshSpec(shape=src_shape, rank_offset=src_rank_offset),
        src_placements=(Placement.shard(0),),
        dst_mesh=MeshSpec(shape=(2,), rank_offset=2),
        dst_placements=(Placement.shard(0),),
        group_key=group_key,
    )


def build(
    rec,
    *,
    plan,
    half_cls,
    partitions=1,
    source_partition=None,
    with_streams=True,
):
    cache = CommunicatorCache()
    for lane_id in range(partitions + 1):
        stream = _DEFAULT_STREAM if with_streams else None
        cache._lanes[LaneKey("g", 1, lane_id)] = lane(
            rec, f"lane{lane_id}", stream=stream
        )
    specs = {
        name: LocalParamSpec(base=f"buf::{name}") for name in plan.parameter_names()
    }
    kwargs = {
        "plan": plan,
        "specs": specs,
        "group_id": "g",
        "epoch": 1,
        "cache": cache,
    }
    if half_cls is NcclM2nSender:
        kwargs["source_partition"] = source_partition
    half = half_cls(**kwargs)
    return half, cache


class TestOpOrdering:
    def test_a_source_mesh_transition_fences_the_previous_batch(
        self, recorder, monkeypatch
    ):
        monkeypatch.setattr(backend, "require_nccl_m2n", lambda: None)
        monkeypatch.setattr(backend, "transfer_timeout", lambda: math.inf)
        plan = ReshardPlan(
            bulk=[
                entry("a", src_shape=(1,), src_rank_offset=0),
                entry("b", src_shape=(1,), src_rank_offset=0),
                entry("c", src_shape=(1,), src_rank_offset=1),
                entry("d", src_shape=(1,), src_rank_offset=1),
            ]
        )
        half, _ = build(recorder, plan=plan, half_cls=NcclM2nSender)

        half.start_weight_update("v1")
        half.publish_weights(0)

        assert [(op.kind, getattr(op, "src", None)) for op in recorder.ops] == [
            ("reshard", "buf::a"),
            ("reshard", "buf::b"),
            ("sync", None),
            ("broadcast", None),
            ("sync", None),
            ("reshard", "buf::c"),
            ("reshard", "buf::d"),
        ]

    @pytest.mark.parametrize(
        ("source_rank_in_lane", "owned"),
        [
            (0, {"a", "b"}),
            (1, {"c", "d"}),
        ],
    )
    def test_sparse_nonowners_take_the_same_source_mesh_fence(
        self, recorder, monkeypatch, source_rank_in_lane, owned
    ):
        monkeypatch.setattr(backend, "require_nccl_m2n", lambda: None)
        monkeypatch.setattr(backend, "transfer_timeout", lambda: math.inf)
        plan = ReshardPlan(
            bulk=[
                entry("a", src_shape=(1,), src_rank_offset=0),
                entry("b", src_shape=(1,), src_rank_offset=0),
                entry("c", src_shape=(1,), src_rank_offset=1),
                entry("d", src_shape=(1,), src_rank_offset=1),
            ]
        )
        cache = CommunicatorCache()
        cache._lanes[LaneKey("g", 1, 0)] = lane(
            recorder, "lane0", rank=source_rank_in_lane, world=4
        )
        cache._lanes[LaneKey("g", 1, 1)] = lane(
            recorder, "broadcast", rank=source_rank_in_lane, world=4
        )
        half = NcclM2nSender(
            plan=plan,
            specs={name: LocalParamSpec(base=f"buf::{name}") for name in owned},
            group_id="g",
            epoch=1,
            cache=cache,
            source_partition=0,
            source_rank_in_lane=source_rank_in_lane,
        )

        half.start_weight_update("v1")
        half.publish_weights(0)

        assert [op.kind for op in recorder.ops] == [
            "reshard",
            "reshard",
            "sync",
            "broadcast",
            "sync",
            "reshard",
            "reshard",
        ]
        assert [op.src is not None for op in recorder.ops if op.kind == "reshard"] == [
            name in owned for name in ("a", "b", "c", "d")
        ]

    def test_a_failed_source_mesh_fence_aborts_the_whole_group(
        self, recorder, monkeypatch
    ):
        monkeypatch.setattr(backend, "require_nccl_m2n", lambda: None)
        monkeypatch.setattr(backend, "transfer_timeout", lambda: math.inf)
        plan = ReshardPlan(
            bulk=[
                entry("a", src_shape=(1,), src_rank_offset=0),
                entry("b", src_shape=(1,), src_rank_offset=1),
            ]
        )
        half, cache = build(recorder, plan=plan, half_cls=NcclM2nSender)
        live = cache.get(LaneKey("g", 1, 0))
        synchronize = live.synchronize
        waits = 0

        def fail_fence(timeout_s=None):
            nonlocal waits
            waits += 1
            if waits == 2:
                raise RuntimeError("injected source-mesh fence failure")
            synchronize(timeout_s)

        live.synchronize = fail_fence

        half.start_weight_update("v1")
        with pytest.raises(RuntimeError, match="source-mesh fence failure"):
            half.publish_weights(0)

        assert [op.kind for op in recorder.ops] == [
            "reshard",
            "sync",
            "broadcast",
        ]
        assert all(live.aborted for live in cache._lanes.values())
        assert half._pending_contexts == []
        assert not half._active_lanes

    def test_a_lane_fence_reuses_one_custom_buffer(self, recorder, monkeypatch):
        monkeypatch.setattr(backend, "require_nccl_m2n", lambda: None)
        monkeypatch.setattr(backend, "transfer_timeout", lambda: math.inf)
        plan = ReshardPlan(
            bulk=[
                entry("a", src_shape=(1,), src_rank_offset=0),
                entry("b", src_shape=(1,), src_rank_offset=1),
                entry("c", src_shape=(1,), src_rank_offset=0),
            ]
        )
        cache = CommunicatorCache()
        live = lane(recorder, "lane0", device="cuda:7")
        cache._lanes[LaneKey("g", 1, 0)] = live
        cache._lanes[LaneKey("g", 1, 1)] = lane(recorder, "broadcast")
        barrier = object()
        allocations = []

        def allocate(device):
            allocations.append(device)
            return barrier

        half = NcclM2nSender(
            plan=plan,
            specs={
                name: LocalParamSpec(base=f"buf::{name}")
                for name in plan.parameter_names()
            },
            group_id="g",
            epoch=1,
            cache=cache,
            barrier_alloc=allocate,
        )

        half.start_weight_update("v1")
        half.publish_weights(0)

        assert allocations == ["cuda:7"]
        assert [op.buf for op in recorder.ops if op.kind == "broadcast"] == [
            barrier,
            barrier,
        ]
        assert half._fence_buffers == {id(live): barrier}

    def test_the_default_fence_allocator_is_framework_fallback(self, recorder):
        half, _ = build(
            recorder,
            plan=ReshardPlan(bulk=[entry("a")]),
            half_cls=NcclM2nSender,
        )
        assert half._barrier_alloc is backend._allocate_fence_buffer

    def test_receiver_source_mesh_transitions_fence_only_the_matching_lane(
        self, recorder, monkeypatch
    ):
        monkeypatch.setattr(backend, "require_nccl_m2n", lambda: None)
        monkeypatch.setattr(backend, "transfer_timeout", lambda: math.inf)
        plan = ReshardPlan(
            bulk=[
                entry("a0", partition=0, src_shape=(1,), src_rank_offset=0),
                entry("a1", partition=1, src_shape=(1,), src_rank_offset=0),
                entry("b0", partition=0, src_shape=(1,), src_rank_offset=1),
                entry("b1", partition=1, src_shape=(1,), src_rank_offset=1),
            ],
            source_partition_count=2,
        )
        half, cache = build(
            recorder,
            plan=plan,
            half_cls=NcclM2nReceiver,
            partitions=2,
        )
        half._specs = {
            name: LocalParamSpec(
                base=f"buf::{name}",
                pre=lambda base, name=name: RefitCtx(buf=base, extra={"name": name}),
            )
            for name in plan.parameter_names()
        }
        retained_at_sync = []
        for lane_id in (0, 1):
            live = cache.get(LaneKey("g", 1, lane_id))
            synchronize = live.synchronize

            def record_retained(
                timeout_s=None,
                *,
                lane_id=lane_id,
                synchronize=synchronize,
            ):
                retained_at_sync.append(
                    (
                        lane_id,
                        {ctx.extra["name"] for _, ctx in half._pending_contexts},
                    )
                )
                synchronize(timeout_s)

            live.synchronize = record_retained

        half.start_weight_update("v1")
        half.update_weights(0)

        assert [
            (
                op.kind,
                getattr(op, "stream", None),
                getattr(getattr(op, "comm", None), "_name", None),
            )
            for op in recorder.ops
        ] == [
            ("reshard", None, "lane0"),
            ("reshard", None, "lane1"),
            ("sync", "stream::lane0", None),
            ("broadcast", None, None),
            ("sync", "stream::lane0", None),
            ("reshard", None, "lane0"),
            ("sync", "stream::lane1", None),
            ("broadcast", None, None),
            ("sync", "stream::lane1", None),
            ("reshard", None, "lane1"),
            ("sync", "stream::lane0", None),
            ("sync", "stream::lane1", None),
        ]
        assert retained_at_sync[:4] == [
            (0, {"a0", "a1"}),
            (0, {"a1"}),
            (1, {"a1", "b0"}),
            (1, {"b0"}),
        ]

    def test_the_misc_broadcast_waits_for_every_layer_group(self, recorder):
        # The regression this guards: running the broadcast at the end of each
        # publish_weights call means entering the all-ranks communicator while
        # another group is still resharding, which deadlocks.
        plan = ReshardPlan(
            bulk=[entry("a", group_key="g0"), entry("b", group_key="g1")],
            misc=[MiscParam("m", (4,), "bfloat16")],
        )
        half, _ = build(recorder, plan=plan, half_cls=NcclM2nSender)
        half.setup_layer_groups([["a"], ["b"]])

        half.start_weight_update("v1")
        half.publish_weights(0)
        half.publish_weights(1)
        assert [op.kind for op in recorder.ops] == ["reshard", "reshard"]

        half.finish_weight_update(broadcast_lane_id=1)
        assert [op.kind for op in recorder.ops] == [
            "reshard",
            "reshard",
            "sync",
            "broadcast",
            "sync",
        ]

    def test_the_broadcast_runs_once_per_refit_not_once_per_call(self, recorder):
        plan = ReshardPlan(bulk=[entry("a")], misc=[MiscParam("m", (4,), "f")])
        half, _ = build(recorder, plan=plan, half_cls=NcclM2nSender)
        half.start_weight_update("v1")
        half.publish_weights(0)
        half.finish_weight_update(broadcast_lane_id=1)
        half.finish_weight_update(broadcast_lane_id=1)
        assert [op.kind for op in recorder.ops].count("broadcast") == 1

    def test_both_halves_issue_the_same_op_sequence(self, recorder, monkeypatch):
        # A collective requires identical sequences; a divergence hangs the
        # communicator rather than failing on the rank that is wrong.
        monkeypatch.setattr(backend, "require_nccl_m2n", lambda: None)
        monkeypatch.setattr(backend, "transfer_timeout", lambda: math.inf)
        plan = ReshardPlan(
            bulk=[entry("a"), entry("b", src_shape=(1,), src_rank_offset=1)],
            misc=[MiscParam("m", (4,), "f"), MiscParam("n", (4,), "f")],
        )

        sender, _ = build(recorder, plan=plan, half_cls=NcclM2nSender)
        sender.start_weight_update("v1")
        sender.publish_weights(0)
        sender.finish_weight_update(broadcast_lane_id=1)
        sent = [op.kind for op in recorder.ops]

        recorder.ops.clear()
        receiver, _ = build(recorder, plan=plan, half_cls=NcclM2nReceiver)
        receiver.start_weight_update("v1")
        receiver.update_weights(0)
        receiver.finish_weight_update(broadcast_lane_id=1)
        received = [op.kind for op in recorder.ops]

        assert sent == received
        assert sent == [
            "reshard",
            "sync",
            "broadcast",
            "sync",
            "reshard",
            "sync",
            "broadcast",
            "broadcast",
            "sync",
        ]

    def test_each_half_supplies_only_its_own_end(self, recorder):
        # Co-called: the trainer passes dst=None, the generator src=None, and
        # NCCL routes from the two meshes.
        plan = ReshardPlan(bulk=[entry("a")])

        sender, _ = build(recorder, plan=plan, half_cls=NcclM2nSender)
        sender.start_weight_update("v")
        sender.publish_weights(0)
        assert recorder.ops[0].src == "buf::a"
        assert recorder.ops[0].dst is None

        recorder.ops.clear()
        receiver, _ = build(recorder, plan=plan, half_cls=NcclM2nReceiver)
        receiver.start_weight_update("v")
        receiver.update_weights(0)
        assert recorder.ops[0].src is None
        assert recorder.ops[0].dst == "buf::a"

    def test_both_meshes_travel_with_every_transfer(self, recorder):
        plan = ReshardPlan(bulk=[entry("a")])
        cache = CommunicatorCache()
        cache._lanes[LaneKey("g", 1, 0)] = lane(recorder, "lane0", stream=None)
        cache._lanes[LaneKey("g", 1, 1)] = lane(recorder, "lane1", stream=None)
        sender = NcclM2nSender(
            plan=plan,
            specs={"a": LocalParamSpec(base="buf::a")},
            group_id="g",
            epoch=1,
            cache=cache,
        )
        sender.start_weight_update("v")
        sender.publish_weights(0)
        assert recorder.ops[0].src_mesh == [0, 1]
        assert recorder.ops[0].dst_mesh == [2, 3]

    def test_the_call_shape_matches_the_nemo_rl_contract(self, recorder):
        """Pins the binding against NeMo RL's xferdtensor call site.

        Every one of these was wrong in the first draft, and each would have
        failed at the first real transfer rather than at import: tensors and
        communicator are positional, meshes and placements are keyword,
        placements are DTensor objects rather than strings, and stream is
        omitted entirely when there is none.
        """
        from torch.distributed.tensor.placement_types import Shard

        plan = ReshardPlan(bulk=[entry("a")])
        sender, _ = build(
            recorder,
            plan=plan,
            half_cls=NcclM2nSender,
            with_streams=False,
        )
        sender.start_weight_update("v")
        sender.publish_weights(0)

        op = recorder.ops[0]
        assert set(op.kwargs) == {
            "src_mesh",
            "src_placements",
            "dst_mesh",
            "dst_placements",
        }
        assert isinstance(op.kwargs["src_placements"][0], Shard)
        assert op.kwargs["src_placements"][0].dim == 0

    def test_a_multi_axis_mesh_is_nested_not_flattened(self, recorder):
        """A flat list would describe a different topology entirely."""
        nested = ParamPlan(
            name="a",
            global_shape=(8, 4),
            dtype="bfloat16",
            partition_id=0,
            src_mesh=MeshSpec(shape=(2, 2), rank_offset=0),
            src_placements=(Placement.replicate(), Placement.shard(0)),
            dst_mesh=MeshSpec(shape=(2,), rank_offset=4),
            dst_placements=(Placement.shard(0),),
        )
        sender, _ = build(
            recorder, plan=ReshardPlan(bulk=[nested]), half_cls=NcclM2nSender
        )
        sender.start_weight_update("v")
        sender.publish_weights(0)
        assert recorder.ops[0].src_mesh == [[0, 1], [2, 3]]

    def test_a_stream_is_passed_as_a_raw_handle_when_present(self, recorder):
        plan = ReshardPlan(bulk=[entry("a")])
        cache = CommunicatorCache()
        cache._lanes[LaneKey("g", 1, 0)] = LaneCommunicator(
            FakeComm(recorder, "lane0"),
            rank=0,
            world_size=4,
            stream=SimpleNamespace(cuda_stream=99),
        )
        cache._lanes[LaneKey("g", 1, 1)] = lane(recorder, "lane1")
        specs = {"a": LocalParamSpec(base="buf::a")}
        half = NcclM2nSender(plan=plan, specs=specs, group_id="g", epoch=1, cache=cache)
        half.start_weight_update("v")
        half.publish_weights(0)
        assert recorder.ops[0].kwargs["stream"] == 99


class TestLaneRouting:
    def test_a_parameter_goes_to_its_partition_s_lane(self, recorder):
        plan = ReshardPlan(
            bulk=[entry("a", partition=0), entry("b", partition=1)],
            source_partition_count=2,
        )
        half, _ = build(recorder, plan=plan, half_cls=NcclM2nSender, partitions=2)
        half.start_weight_update("v")
        half.publish_weights(0)
        assert [op.comm._name for op in recorder.ops] == ["lane0", "lane1"]

    def test_a_pp_trainer_requires_and_uses_only_its_admitted_partition(self, recorder):
        plan = ReshardPlan(
            bulk=[entry("stage0", partition=0), entry("stage1", partition=1)],
            misc=[MiscParam("m", (4,), "f")],
            source_partition_count=2,
        )
        cache = CommunicatorCache()
        cache._lanes[LaneKey("g", 1, 1)] = lane(recorder, "lane1")
        cache._lanes[LaneKey("g", 1, 2)] = lane(recorder, "broadcast")
        half = NcclM2nSender(
            plan=plan,
            specs={
                "stage1": LocalParamSpec(base="buf::stage1"),
                "m": LocalParamSpec(base="buf::m"),
            },
            group_id="g",
            epoch=1,
            cache=cache,
            source_partition=1,
        )

        half.start_weight_update("v")
        half.publish_weights(0)

        assert [op.src for op in recorder.ops if op.kind == "reshard"] == [
            "buf::stage1"
        ]
        assert [op.comm._name for op in recorder.ops if op.kind == "reshard"] == [
            "lane1"
        ]

    def test_a_nonowner_enters_the_same_reshard_with_neither_endpoint(
        self, recorder, monkeypatch
    ):
        monkeypatch.setattr(backend, "require_nccl_m2n", lambda: None)
        plan = ReshardPlan(
            bulk=[
                entry("owner0", src_shape=(1,), src_rank_offset=0),
                entry("owner1", src_shape=(1,), src_rank_offset=1),
            ]
        )
        cache = CommunicatorCache()
        cache._lanes[LaneKey("g", 1, 0)] = lane(recorder, "lane0", rank=1, world=4)
        cache._lanes[LaneKey("g", 1, 1)] = lane(recorder, "broadcast", rank=1, world=4)
        half = NcclM2nSender(
            plan=plan,
            specs={"owner1": LocalParamSpec(base="buf::owner1")},
            group_id="g",
            epoch=1,
            cache=cache,
            source_partition=0,
            source_rank_in_lane=1,
        )

        half.start_weight_update("v")
        half.publish_weights(0)

        assert [(op.src, op.dst) for op in recorder.ops if op.kind == "reshard"] == [
            (None, None),
            ("buf::owner1", None),
        ]

    def test_a_source_owner_still_requires_local_storage(self, recorder):
        plan = ReshardPlan(bulk=[entry("owner0", src_shape=(1,), src_rank_offset=0)])
        with pytest.raises(KeyError, match="no local storage"):
            NcclM2nSender(
                plan=plan,
                specs={},
                group_id="g",
                epoch=1,
                cache=CommunicatorCache(),
                source_partition=0,
                source_rank_in_lane=0,
            )

    def test_a_nonowner_still_requires_misc_storage(self, recorder):
        plan = ReshardPlan(
            bulk=[entry("owner0", src_shape=(1,), src_rank_offset=0)],
            misc=[MiscParam("m", (4,), "bfloat16")],
        )
        with pytest.raises(KeyError, match="m"):
            NcclM2nSender(
                plan=plan,
                specs={},
                group_id="g",
                epoch=1,
                cache=CommunicatorCache(),
                source_partition=0,
                source_rank_in_lane=1,
            )

    def test_a_missing_communicator_is_a_clear_error_not_a_hang(self, recorder):
        plan = ReshardPlan(
            bulk=[entry("a", partition=3)],
            source_partition_count=4,
        )
        half, _ = build(recorder, plan=plan, half_cls=NcclM2nSender, partitions=1)
        half.start_weight_update("v")
        with pytest.raises(RuntimeError, match="no communicator"):
            half.publish_weights(0)


class TestHooks:
    def test_pre_and_post_bracket_the_wire_op(self, recorder):
        order = []
        plan = ReshardPlan(bulk=[entry("a")])
        cache = CommunicatorCache()
        cache._lanes[LaneKey("g", 1, 0)] = lane(recorder, "lane0")
        cache._lanes[LaneKey("g", 1, 1)] = lane(recorder, "lane1")

        def pre(base):
            order.append("pre")
            return RefitCtx(buf="staged", extra={"region": base})

        def post(ctx):
            order.append("post")
            assert ctx.buf == "staged"

        specs = {"a": LocalParamSpec(base="live", pre=pre, post=post)}
        half = NcclM2nSender(plan=plan, specs=specs, group_id="g", epoch=1, cache=cache)
        half.start_weight_update("v")
        half.publish_weights(0)

        assert order == ["pre", "post"]
        # The wire op must see the staged buffer, not the live parameter.
        assert recorder.ops[0].src == "staged"

    def test_a_spec_with_neither_base_nor_pre_is_rejected(self):
        with pytest.raises(ValueError, match="base tensor or a pre hook"):
            LocalParamSpec().enter()

    def test_staging_context_is_retained_until_the_lane_is_drained(self, recorder):
        plan = ReshardPlan(bulk=[entry("a")])
        half, _ = build(recorder, plan=plan, half_cls=NcclM2nSender)
        half.start_weight_update("v")
        half.publish_weights(0)
        assert len(half._pending_contexts) == 1
        half.finish_weight_update(broadcast_lane_id=1)
        assert half._pending_contexts == []


class TestLayerGroups:
    def test_uncovered_bulk_parameters_are_rejected(self, recorder):
        plan = ReshardPlan(bulk=[entry("a"), entry("b")])
        half, _ = build(recorder, plan=plan, half_cls=NcclM2nSender)
        with pytest.raises(ValueError, match="uncovered"):
            half.setup_layer_groups([["a"]])

    def test_a_parameter_in_two_groups_is_rejected(self, recorder):
        plan = ReshardPlan(
            bulk=[entry("a", group_key="g0"), entry("b", group_key="g1")]
        )
        half, _ = build(recorder, plan=plan, half_cls=NcclM2nSender)
        with pytest.raises(ValueError, match="more than one layer group"):
            half.setup_layer_groups([["a"], ["a", "b"]])

    def test_an_unknown_parameter_is_rejected(self, recorder):
        plan = ReshardPlan(
            bulk=[entry("a", group_key="g0"), entry("b", group_key="g1")]
        )
        half, _ = build(recorder, plan=plan, half_cls=NcclM2nSender)
        with pytest.raises(KeyError, match="not a bulk parameter"):
            half.setup_layer_groups([["a"], ["ghost"]])

    def test_group_key_does_not_define_layer_groups(self, recorder):
        plan = ReshardPlan(bulk=[entry("a", group_key="same-fused-buffer"), entry("b")])
        half, _ = build(recorder, plan=plan, half_cls=NcclM2nSender)
        assert half.layer_group_ids == [0]
        assert [entry.name for entry in half.entries(0)] == ["a", "b"]

        half.setup_layer_groups([["a"], ["b"]])
        assert [entry.name for entry in half.entries(0)] == ["a"]
        assert [entry.name for entry in half.entries(1)] == ["b"]

    def test_names_within_a_declared_group_execute_in_canonical_order(self, recorder):
        plan = ReshardPlan(bulk=[entry("z", group_key="g"), entry("a", group_key="g")])
        half, _ = build(recorder, plan=plan, half_cls=NcclM2nSender)
        half.setup_layer_groups([["z", "a"]])
        half.start_weight_update("v")
        half.publish_weights(0)
        assert [op.src for op in recorder.ops if op.kind == "reshard"] == [
            "buf::a",
            "buf::z",
        ]

    def test_the_default_is_one_group_holding_everything(self, recorder):
        plan = ReshardPlan(bulk=[entry("a"), entry("b")])
        half, _ = build(recorder, plan=plan, half_cls=NcclM2nSender)
        assert half.layer_group_ids == [0]
        assert len(half.entries(0)) == 2


class TestSpecResolution:
    def test_a_parameter_without_local_storage_fails_before_the_collective(self):
        # Detected here rather than mid-transfer: this rank would otherwise
        # skip an op its peers issue, hanging the lane instead of raising.
        plan = ReshardPlan(bulk=[entry("a")], misc=[MiscParam("m", (4,), "f")])
        with pytest.raises(KeyError, match="no local storage"):
            resolve_specs(plan, {"a": LocalParamSpec(base="x")})


class TestCommunicatorCache:
    def test_a_lane_is_reused_within_an_epoch(self, recorder):
        cache = CommunicatorCache()
        key = LaneKey("g", 1, 0)
        cache._lanes[key] = lane(recorder, "lane0")
        assert cache.get(key) is cache.get(key)

    def test_an_epoch_move_drops_the_stale_lanes(self, recorder):
        cache = CommunicatorCache()
        old_lanes = [lane(recorder, "old0"), lane(recorder, "old1")]
        cache._lanes[LaneKey("g", 1, 0)] = old_lanes[0]
        cache._lanes[LaneKey("g", 1, 1)] = old_lanes[1]
        cache._lanes[LaneKey("other", 1, 0)] = lane(recorder, "untouched")

        dropped = cache.invalidate_epoch("g", 2)

        assert dropped == 2
        assert cache.get(LaneKey("g", 1, 0)) is None
        assert all(entry_lane.aborted for entry_lane in old_lanes)
        # A different group's lanes are not collateral damage.
        assert cache.get(LaneKey("other", 1, 0)) is not None

    def test_an_aborted_lane_is_never_handed_out_again(self, recorder):
        cache = CommunicatorCache()
        key = LaneKey("g", 1, 0)
        entry_lane = lane(recorder, "lane0")
        cache._lanes[key] = entry_lane
        entry_lane.abort()
        assert cache.get(key) is None

    def test_using_an_aborted_communicator_raises(self, recorder):
        entry_lane = lane(recorder, "lane0")
        entry_lane.abort()
        with pytest.raises(RuntimeError, match="aborted"):
            _ = entry_lane.handle

    def test_abort_takes_down_every_lane_of_the_group(self, recorder):
        # A partially aborted group leaves peers blocked in the lanes that did
        # not time out, waiting on ranks that already gave up.
        cache = CommunicatorCache()
        lanes = [lane(recorder, f"lane{i}") for i in range(3)]
        for i, entry_lane in enumerate(lanes):
            cache._lanes[LaneKey("g", 1, i)] = entry_lane

        assert cache.abort_group("g") == 3
        assert all(entry_lane.aborted for entry_lane in lanes)
        assert len(cache) == 0

    def test_abort_marks_the_lane_dead_even_if_teardown_fails(self, recorder):
        class Stubborn:
            def abort(self):
                raise RuntimeError("nccl refused")

        entry_lane = LaneCommunicator(Stubborn(), rank=0, world_size=2, stream=None)
        entry_lane.abort()
        assert entry_lane.aborted


class TestTransferDeadline:
    """MX_NCCL_REFIT_TRANSFER_TIMEOUT_S bounds what happens after READY.

    The deadline exists because the group forming is not the same as the
    transfer completing: a reshard that never lands would otherwise hang the
    process with no owner and no attributable failure.
    """

    @pytest.fixture
    def clock(self, monkeypatch):
        """A hand-advanced monotonic clock for backend.py."""

        class Clock:
            def __init__(self):
                self.now = 1000.0

            def advance(self, seconds):
                self.now += seconds

        c = Clock()
        monkeypatch.setattr(backend.time, "monotonic", lambda: c.now)
        return c

    @pytest.fixture
    def short_deadline(self, monkeypatch):
        monkeypatch.setattr(backend, "transfer_timeout", lambda: 30.0)

    def test_a_transfer_that_overruns_raises_instead_of_hanging(
        self, recorder, clock, short_deadline
    ):
        plan = ReshardPlan(bulk=[entry("a")])
        half, _ = build(recorder, plan=plan, half_cls=NcclM2nSender)
        half.start_weight_update("v7")

        clock.advance(31.0)

        with pytest.raises(TimeoutError) as excinfo:
            half.publish_weights(0)
        message = str(excinfo.value)
        assert "v7" in message, "the failure must name the version that overran"
        assert "MX_NCCL_REFIT_TRANSFER_TIMEOUT_S" in message, (
            "the failure must name the knob that produced it"
        )

    def test_overrunning_aborts_the_group_rather_than_leaving_it_usable(
        self, recorder, clock, short_deadline
    ):
        plan = ReshardPlan(bulk=[entry("a")])
        half, cache = build(recorder, plan=plan, half_cls=NcclM2nSender)
        half.start_weight_update("v7")
        clock.advance(31.0)

        with pytest.raises(TimeoutError):
            half.publish_weights(0)

        # Peers that disagree about which collectives completed cannot be
        # recovered on the same communicator, so every lane must be gone.
        assert all(lane.aborted for lane in cache._lanes.values())

    def test_a_transfer_inside_its_deadline_is_untouched(
        self, recorder, clock, short_deadline
    ):
        """Positive control: the guard must not fire on a healthy transfer."""
        plan = ReshardPlan(bulk=[entry("a")], misc=[MiscParam("m", (4,), "f")])
        half, cache = build(recorder, plan=plan, half_cls=NcclM2nSender)
        half.start_weight_update("v7")
        clock.advance(1.0)
        half.publish_weights(0)
        half.finish_weight_update(1)

        assert [op.comm._name for op in recorder.ops if op.kind == "reshard"] == [
            "lane0"
        ]
        assert not any(lane.aborted for lane in cache._lanes.values())

    def test_the_deadline_is_rearmed_per_version_not_per_client(
        self, recorder, clock, short_deadline
    ):
        """A second version gets its own budget, not the leftovers of the first."""
        plan = ReshardPlan(bulk=[entry("a")])
        half, _ = build(recorder, plan=plan, half_cls=NcclM2nSender)

        half.start_weight_update("v1")
        clock.advance(29.0)
        half.publish_weights(0)

        half.start_weight_update("v2")
        clock.advance(29.0)
        half.publish_weights(0)

    def test_a_half_with_no_armed_transfer_is_not_bounded(self, recorder, clock):
        """Nothing that never called start_weight_update inherits a deadline."""
        plan = ReshardPlan(bulk=[entry("a")])
        half, _ = build(recorder, plan=plan, half_cls=NcclM2nSender)
        clock.advance(10_000.0)
        assert half._remaining() == math.inf

    def test_the_remaining_budget_reaches_the_lane_and_shrinks(
        self, recorder, clock, short_deadline
    ):
        """The bound is PLUMBED, not merely checked at the boundaries.

        Checking the clock either side of a blocking synchronize cannot bound
        it, so the drain has to hand the lane what is left of the budget.
        """
        seen = []
        plan = ReshardPlan(bulk=[entry("a")])
        half, cache = build(recorder, plan=plan, half_cls=NcclM2nReceiver)

        for key, live in list(cache._lanes.items()):

            def record(timeout_s=None, _live=live):
                seen.append(timeout_s)

            live.synchronize = record

        half.start_weight_update("v7")
        clock.advance(10.0)
        half.update_weights(0)

        assert seen, "the drain never reached a lane"
        assert all(t is not None for t in seen), (
            "an unbounded synchronize is the hang this deadline exists to stop"
        )
        assert all(t == pytest.approx(20.0) for t in seen), seen

    def test_a_lane_timeout_is_reported_against_the_version(
        self, recorder, clock, short_deadline
    ):
        """A TimeoutError from the lane becomes an attributable refit failure."""
        plan = ReshardPlan(bulk=[entry("a")])
        half, cache = build(recorder, plan=plan, half_cls=NcclM2nReceiver)

        for live in cache._lanes.values():

            def boom(timeout_s=None):
                raise TimeoutError("stream never drained")

            live.synchronize = boom

        half.start_weight_update("v7")
        with pytest.raises(TimeoutError) as excinfo:
            half.update_weights(0)
        assert "v7" in str(excinfo.value)
        assert all(lane.aborted for lane in cache._lanes.values())


class TestBoundedSynchronizeFallback:
    """The bound applies only where an event can be recorded.

    This is the branch that decides whether a deadline is enforced at all, so
    a bug here silently un-bounds every transfer while every other test stays
    green. The polling path itself needs a device and is exercised by the GPU
    tests, not here.
    """

    def test_a_stream_that_cannot_carry_an_event_falls_back_rather_than_lying(
        self, recorder
    ):
        """Must hold with OR without a device.

        FakeStream carries a ``cuda_stream`` attribute, so on a CPU box this
        returns False at the availability gate and on a GPU box it returns
        False because recording against a test double fails. Asserting it
        without pinning the reason is deliberate: the earlier version of this
        test passed only because the box had no CUDA, which is the same test
        passing for a reason that does not generalize.
        """
        live = lane(recorder, "lane0")
        assert live._synchronize_bounded(5.0) is False

    def test_the_fallback_accepts_a_stream_that_only_carries_a_raw_handle(
        self, recorder, monkeypatch
    ):
        """The unbounded path must take the same shapes the bounded one does.

        _synchronize_bounded and backend._stream_handle both read
        ``cuda_stream`` off the object. int() on the object itself raises
        TypeError, and this path is reachable whenever timeout_s is None.
        """
        import torch

        synced = []

        class FakeExternalStream:
            def __init__(self, handle):
                self.handle = handle

            def synchronize(self):
                synced.append(self.handle)

        monkeypatch.setattr(torch.cuda, "ExternalStream", FakeExternalStream)

        class HandleOnlyStream:
            """A raw-handle carrier with no synchronize of its own."""

            cuda_stream = 4242

        live = lane(recorder, "lane0", stream=HandleOnlyStream())
        live.synchronize()

        assert synced == [4242]

    def test_an_implicit_lane_waits_on_the_cuda_default_stream(
        self, recorder, monkeypatch
    ):
        import torch

        synced = []

        class DefaultStream:
            def synchronize(self):
                synced.append("default")

        monkeypatch.setattr(
            torch.cuda,
            "default_stream",
            lambda *, device: DefaultStream(),
        )

        live = lane(recorder, "lane0", stream=None)
        live.synchronize()

        assert synced == ["default"]

    def test_the_fallback_still_waits(self, recorder):
        live = lane(recorder, "lane0")
        live.synchronize(timeout_s=5.0)
        assert any(op.kind == "sync" for op in recorder.ops), (
            "falling back must still drain the stream, not skip the wait"
        )


class TestNcclVersionFloor:
    """``require_nccl_m2n`` has to check the library, not just the import.

    The reshard entry points resolve inside the native call, so a process that
    imports ``nccl.m2n`` against an older libnccl gets through every guard and
    then fails mid-collective with its peers already waiting on it.
    """

    @staticmethod
    def _importable(monkeypatch):
        module = ModuleType("nccl.m2n")
        module.reshard = lambda *args, **kwargs: None
        monkeypatch.setitem(sys.modules, "nccl", ModuleType("nccl"))
        monkeypatch.setitem(sys.modules, "nccl.m2n", module)

    def test_a_library_below_the_floor_is_refused(self, monkeypatch):
        self._importable(monkeypatch)
        monkeypatch.setattr(backend, "loaded_nccl_version", lambda: (2, 27, 5))
        with pytest.raises(backend.NcclUnavailableError) as caught:
            backend.require_nccl_m2n()
        assert "2.27.5" in str(caught.value)

    def test_the_floor_itself_passes(self, monkeypatch):
        self._importable(monkeypatch)
        monkeypatch.setattr(backend, "loaded_nccl_version", lambda: backend.MIN_NCCL)
        backend.require_nccl_m2n()

    def test_an_unreadable_probe_is_not_an_old_library(self, monkeypatch):
        """None and below-the-floor are different answers.

        None says the ctypes handle did not resolve, which is a fact about the
        probe. Collapsing it into a refusal would ground the data plane on
        every host whose loader layout this check cannot see through, and the
        import already succeeded there.
        """
        self._importable(monkeypatch)
        monkeypatch.setattr(backend, "loaded_nccl_version", lambda: None)
        backend.require_nccl_m2n()

    def test_the_version_probe_reads_the_packed_integer(self, monkeypatch):
        """22807 is 2.28.7, not 2.2.807 - the packing is easy to get wrong."""
        backend.loaded_nccl_version.cache_clear()

        class FakeLib:
            def ncclGetVersion(self, ref):
                ref._obj.value = 23007
                return 0

        monkeypatch.setattr(backend.ctypes, "CDLL", lambda name: FakeLib())
        try:
            assert backend.loaded_nccl_version() == (2, 30, 7)
        finally:
            backend.loaded_nccl_version.cache_clear()
