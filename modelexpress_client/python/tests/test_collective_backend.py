# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Data-plane behaviour for the NCCL M2N collective path.

Driven against a recording stand-in for nccl4py, because what needs protecting
is the *order and shape* of the operations rather than the bytes NCCL moves.
Every property here corresponds to a failure that would otherwise present as a
hung communicator rather than an exception.
"""

import sys
from types import ModuleType, SimpleNamespace

import pytest

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


def lane(rec, name, *, rank=0, world=4, stream=_DEFAULT_STREAM):
    if stream is _DEFAULT_STREAM:
        stream = FakeStream(rec, f"stream::{name}")
    return LaneCommunicator(
        FakeComm(rec, name), rank=rank, world_size=world, stream=stream
    )


def entry(name, partition=0, group_key=None):
    return ParamPlan(
        name=name,
        global_shape=(8, 4),
        dtype="bfloat16",
        partition_id=partition,
        src_mesh=MeshSpec(shape=(2,), rank_offset=0),
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

    def test_both_halves_issue_the_same_op_sequence(self, recorder):
        # A collective requires identical sequences; a divergence hangs the
        # communicator rather than failing on the rank that is wrong.
        plan = ReshardPlan(
            bulk=[entry("a"), entry("b")],
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
