# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Equal-geometry stacking: derivation, fail-closed rules and the wire arithmetic.

Tensors that share their plan geometry are sent as one stacked reshard whose
shard dims move up by one. Everything here is decided from plan facts, so the
tests build plans from shapes and placements only and never name a model.
"""

from __future__ import annotations

import hashlib
import sys
from collections import defaultdict
from dataclasses import dataclass, replace
from types import ModuleType

import pytest
import torch

from modelexpress_rl.collective import (
    CommunicatorCache,
    LaneCommunicator,
    LaneKey,
    LocalParamSpec,
    MeshSpec,
    NcclM2nReceiver,
    NcclM2nSender,
    ParamPlan,
    Placement,
    ReshardPlan,
    backend,
    envs,
)
from modelexpress_rl.collective.integrations._common import (
    REPLICATED_DESTINATION_ABI,
    SHARDED_DESTINATION_ABI,
    STACK_ABI_SUFFIX,
    STACK_GROUP_KEY_PREFIX,
    STACK_NAME_PREFIX,
    _check_stack_budget,
    _derive_wire_plan,
    _expected_index,
    _FrozenPlan,
    _stack_classes,
    _stack_name,
    _stack_trainer_buffers,
)
from modelexpress_rl.collective.integrations.miles import CollectiveTopology
from modelexpress_rl.collective.plan import plan_digest


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("NCCL_RESHARD_PACK_BUFFSIZES", raising=False)


def _mesh_pair(shape=(2,), dst_shape=(2,)):
    src = MeshSpec(shape)
    return src, MeshSpec(dst_shape, rank_offset=src.size)


def _entry(
    name,
    shape=(8, 16),
    *,
    src_dim=0,
    dst_dim=0,
    meshes=None,
    dtype="bfloat16",
    group_key="publish-group-0",
):
    src_mesh, dst_mesh = meshes or _mesh_pair()
    inner_src = Placement.replicate() if src_dim is None else Placement.shard(src_dim)
    inner_dst = Placement.replicate() if dst_dim is None else Placement.shard(dst_dim)
    return ParamPlan(
        name=name,
        global_shape=shape,
        dtype=dtype,
        partition_id=0,
        src_mesh=src_mesh,
        src_placements=(Placement.replicate(),) * (len(src_mesh.shape) - 1)
        + (inner_src,),
        dst_mesh=dst_mesh,
        dst_placements=(Placement.replicate(),) * (len(dst_mesh.shape) - 1)
        + (inner_dst,),
        group_key=group_key,
    )


def _canonical(entries):
    return sorted(entries, key=lambda entry: entry.canonical())


def _stacked_plan(entries, budget):
    """The per-tensor plan the trainer would send for ``budget``."""
    entries = _canonical(entries)
    keys = _stack_classes(entries, budget)
    return ReshardPlan(
        bulk=[
            replace(entry, group_key=keys[entry.name]) if entry.name in keys else entry
            for entry in entries
        ]
    )


def _four_equal(prefix="w", count=4, **kwargs):
    return [_entry(f"{prefix}{index}", **kwargs) for index in range(count)]


# 8x16 bf16 on TP2 is 64 elements = 128 bytes per member on both sides.
_AREA = 128


class TestDerivation:
    def test_a_plan_without_stack_keys_is_its_own_wire_plan(self):
        plan = ReshardPlan(bulk=_canonical(_four_equal()))
        wire, stacks = _derive_wire_plan(plan)
        assert wire is plan
        assert stacks == ()

    def test_equal_geometry_tensors_become_one_stacked_entry(self):
        plan = _stacked_plan(_four_equal(), 4 * _AREA)
        wire, stacks = _derive_wire_plan(plan)

        assert [stack.members for stack in stacks] == [("w0", "w1", "w2", "w3")]
        assert len(wire.bulk) == 1
        virtual = wire.bulk[0]
        assert virtual == stacks[0].entry
        assert virtual.name.startswith(STACK_NAME_PREFIX)
        assert virtual.global_shape == (4, 8, 16)
        # Every shard dim moves up by one; replicate stays.
        assert [p.canonical() for p in virtual.src_placements] == ["S1"]
        assert [p.canonical() for p in virtual.dst_placements] == ["S1"]
        assert virtual.group_key == f"{STACK_GROUP_KEY_PREFIX}0"
        assert stacks[0].area_bytes == 4 * _AREA

    def test_the_shift_applies_to_replicate_and_two_axis_meshes(self):
        meshes = (MeshSpec((2, 2)), MeshSpec((2, 2), rank_offset=4))
        entries = _four_equal(
            count=2, shape=(8,), src_dim=None, dst_dim=0, meshes=meshes
        )
        wire, _ = _derive_wire_plan(_stacked_plan(entries, 1 << 20))
        virtual = wire.bulk[0]
        assert virtual.global_shape == (2, 8)
        assert [p.canonical() for p in virtual.src_placements] == ["R", "R"]
        assert [p.canonical() for p in virtual.dst_placements] == ["R", "S1"]

    def test_the_budget_chunks_a_class_at_the_boundary(self):
        # Four members of 128 bytes: a 3-member stack fits 384 and the fourth
        # member is a lone leftover, which stays a plain entry.
        entries = _four_equal()
        plan = _stacked_plan(entries, 3 * _AREA)
        wire, stacks = _derive_wire_plan(plan)
        assert [stack.members for stack in stacks] == [("w0", "w1", "w2")]
        assert [
            entry.name for entry in wire.bulk if not entry.name.startswith("m2n")
        ] == ["w3"]
        # One byte short of two members: nothing stacks at all.
        none = _stacked_plan(entries, 2 * _AREA - 1)
        assert all(entry.group_key == "publish-group-0" for entry in none.bulk)
        # Exactly four members fit one stack; two budgets of two give two stacks.
        two = _derive_wire_plan(_stacked_plan(entries, 2 * _AREA))[1]
        assert [stack.members for stack in two] == [("w0", "w1"), ("w2", "w3")]

    def test_only_plan_facts_decide_membership(self):
        # Same classes under unrelated names stack identically; a different
        # shape, dtype, placement or mesh never joins a class.
        entries = [
            _entry("zeta"),
            _entry("alpha"),
            _entry("mid", shape=(16, 8), src_dim=1, dst_dim=1),
            _entry("other", shape=(16, 8), src_dim=1, dst_dim=1),
            _entry("f16", dtype="float16"),
            _entry("repl", src_dim=None, dst_dim=None),
            _entry("wide", meshes=_mesh_pair((4,), (4,))),
        ]
        stacks = _derive_wire_plan(_stacked_plan(entries, 1 << 20))[1]
        assert sorted(stack.members for stack in stacks) == [
            ("alpha", "zeta"),
            ("mid", "other"),
        ]

    def test_rank_three_tensors_and_singletons_stay_plain(self):
        entries = [
            _entry("a0", shape=(4, 4, 8)),
            _entry("a1", shape=(4, 4, 8)),
            _entry("lone", shape=(8, 4)),
            *_four_equal(count=2),
        ]
        wire, stacks = _derive_wire_plan(_stacked_plan(entries, 1 << 20))
        assert [stack.members for stack in stacks] == [("w0", "w1")]
        plain = sorted(
            entry.name for entry in wire.bulk if not entry.name.startswith("m2n")
        )
        assert plain == ["a0", "a1", "lone"]

    def test_stack_ids_follow_the_first_member_in_canonical_order(self):
        entries = [
            *_four_equal("b", 2),
            *_four_equal("a", 2, shape=(16, 8), src_dim=1, dst_dim=1),
        ]
        stacks = _derive_wire_plan(_stacked_plan(entries, 1 << 20))[1]
        assert [stack.members for stack in stacks] == [("a0", "a1"), ("b0", "b1")]
        assert [stack.index for stack in stacks] == [0, 1]

    def test_the_derivation_is_deterministic(self):
        entries = _four_equal() + _four_equal("x", 3, shape=(4,), src_dim=None)
        first = _derive_wire_plan(_stacked_plan(entries, 1 << 20))[0]
        second = _derive_wire_plan(_stacked_plan(list(reversed(entries)), 1 << 20))[0]
        assert [e.canonical() for e in first.bulk] == [
            e.canonical() for e in second.bulk
        ]
        assert first.bulk == _canonical(first.bulk)

    def test_the_virtual_name_hashes_the_member_names_in_order(self):
        assert _stack_name(0, ("a", "b")) != _stack_name(0, ("b", "a"))
        assert _stack_name(0, ("a", "b")) != _stack_name(0, ("a", "c"))
        assert _stack_name(0, ("a", "b")) != _stack_name(1, ("a", "b"))
        assert _stack_name(3, ("a", "b")).startswith(f"{STACK_NAME_PREFIX}0003/")

    def test_a_non_positive_budget_is_rejected(self):
        with pytest.raises(ValueError, match="positive"):
            _stack_classes(_four_equal(), 0)


class TestFailClosed:
    """F1, F2 and F5 of the design; F3 (digest and ABI) is in the next class."""

    @staticmethod
    def _keyed(*keys, entries=None):
        entries = entries or [_entry(f"w{i}") for i in range(len(keys))]
        return ReshardPlan(
            bulk=_canonical(
                [replace(e, group_key=k) for e, k in zip(entries, keys, strict=True)]
            )
        )

    @pytest.mark.parametrize(
        "key", ["m2n-stack1-", "m2n-stack1-x", "m2n-stack1-01", "m2n-stack1--1"]
    )
    def test_f1_a_malformed_key_is_rejected(self, key):
        with pytest.raises(ValueError, match="malformed stack group key"):
            _derive_wire_plan(self._keyed(key, key))

    def test_f1_ids_must_be_exactly_zero_to_k_minus_one(self):
        plan = self._keyed(
            "m2n-stack1-0", "m2n-stack1-0", "m2n-stack1-2", "m2n-stack1-2"
        )
        with pytest.raises(ValueError, match="exactly 0..K-1"):
            _derive_wire_plan(plan)
        with pytest.raises(ValueError, match="exactly 0..K-1"):
            _derive_wire_plan(self._keyed("m2n-stack1-1", "m2n-stack1-1"))

    def test_f1_a_stack_needs_two_members(self):
        with pytest.raises(ValueError, match="at least 2"):
            _derive_wire_plan(self._keyed("m2n-stack1-0", "publish-group-0"))

    @pytest.mark.parametrize(
        "other",
        [
            {"shape": (8, 32)},
            {"dst_dim": None},
            {"src_dim": 1},
            {"dtype": "float16"},
            {"meshes": _mesh_pair((4,), (4,))},
        ],
    )
    def test_f1_members_must_agree_on_every_class_key_field(self, other):
        entries = [_entry("w0"), _entry("w1", **other)]
        plan = self._keyed("m2n-stack1-0", "m2n-stack1-0", entries=entries)
        with pytest.raises(ValueError, match="does not share the geometry"):
            _derive_wire_plan(plan)

    def test_f1_a_member_of_rank_three_is_rejected(self):
        entries = [_entry(f"w{i}", shape=(4, 4, 8)) for i in range(2)]
        plan = self._keyed("m2n-stack1-0", "m2n-stack1-0", entries=entries)
        with pytest.raises(ValueError, match="cannot be stacked"):
            _derive_wire_plan(plan)

    @pytest.mark.parametrize("name", ["m2n-stack1/0000/abc", "m2n-stack", "m2n-stackx"])
    def test_f2_reserved_names_are_rejected_stacked_or_not(self, name):
        plan = ReshardPlan(bulk=[_entry(name), _entry("ok")])
        with pytest.raises(ValueError, match="reserved"):
            _derive_wire_plan(plan)

    def test_f2_a_real_tensor_cannot_impersonate_a_stack(self):
        stacked = _stacked_plan(_four_equal(src_dim=None), 1 << 20)
        wire, _ = _derive_wire_plan(stacked)
        fake = ReshardPlan(bulk=[replace(wire.bulk[0], group_key="x")])
        # A wire entry handed in as a per-tensor plan is a reserved name.
        with pytest.raises(ValueError, match="reserved"):
            _derive_wire_plan(fake)
        # And a frozen wire plan insists on the stack group key.
        with pytest.raises(ValueError, match="must carry"):
            _FrozenPlan(fake)

    def test_f5_a_stack_larger_than_the_pack_bucket_is_refused(self, monkeypatch):
        entries = _four_equal(count=2, shape=(64, 64), meshes=_mesh_pair((1,), (1,)))
        stacks = _derive_wire_plan(_stacked_plan(entries, 1 << 30))[1]
        assert stacks[0].area_bytes == 2 * 64 * 64 * 2
        _check_stack_budget(stacks)  # the implicit 2 GiB bucket holds it
        monkeypatch.setenv("NCCL_RESHARD_PACK_BUFFSIZES", "2048")
        with pytest.raises(ValueError, match="largest PACK staging bucket of 2048"):
            _check_stack_budget(stacks)
        # The largest listed bucket is what counts, not the first.
        monkeypatch.setenv("NCCL_RESHARD_PACK_BUFFSIZES", "2048,16K:2")
        _check_stack_budget(stacks)
        # An invalid value is ignored natively, so the 2 GiB default holds.
        monkeypatch.setenv("NCCL_RESHARD_PACK_BUFFSIZES", "nonsense")
        _check_stack_budget(stacks)


def _topology(src_mesh, dst_mesh, abi):
    return CollectiveTopology(
        model_name="m",
        trainer_slots=tuple(f"t{i}" for i in range(src_mesh.size)),
        generator_slots=tuple(f"g{i}" for i in range(dst_mesh.size)),
        source_partition_count=1,
        m2n_abi_version=abi,
    )


class TestDigestAndAbi:
    def test_trainer_and_receiver_derive_the_same_wire_plan_and_digest(self):
        entries = _four_equal() + _four_equal(
            "x", 3, shape=(8,), src_dim=None, dst_dim=None
        )
        plan = _stacked_plan(entries, 1 << 20)
        trainer, _ = _derive_wire_plan(plan)
        # The receiver gets the plan over the wire and derives it again.
        from modelexpress_rl.collective.integrations.manifest import (
            plan_from_wire,
            plan_to_wire,
        )

        receiver, _ = _derive_wire_plan(plan_from_wire(plan_to_wire(plan)))
        assert trainer.bulk == receiver.bulk
        abi = REPLICATED_DESTINATION_ABI + STACK_ABI_SUFFIX
        assert plan_digest(trainer, m2n_abi_version=abi) == plan_digest(
            receiver, m2n_abi_version=abi
        )

    def test_stacking_changes_the_digest(self):
        entries = _four_equal()
        unstacked = ReshardPlan(bulk=_canonical(entries))
        wire, _ = _derive_wire_plan(_stacked_plan(entries, 1 << 20))
        assert plan_digest(wire, m2n_abi_version="x") != plan_digest(
            unstacked, m2n_abi_version="x"
        )

    def test_a_stacked_plan_needs_the_stack_suffix_on_the_abi(self):
        src, dst = _mesh_pair()
        wire, _ = _derive_wire_plan(_stacked_plan(_four_equal(src_dim=None), 1 << 20))
        frozen = _FrozenPlan(wire)
        assert frozen.stacked
        frozen.validate_topology(
            _topology(src, dst, SHARDED_DESTINATION_ABI + STACK_ABI_SUFFIX)
        )
        for abi in (
            SHARDED_DESTINATION_ABI,
            REPLICATED_DESTINATION_ABI + STACK_ABI_SUFFIX,
        ):
            with pytest.raises(ValueError, match="requires the"):
                frozen.validate_topology(_topology(src, dst, abi))

    def test_a_stacking_trainer_against_a_non_stacking_receiver_fails_at_prepare(self):
        src, dst = _mesh_pair()
        trainer_abi = SHARDED_DESTINATION_ABI + STACK_ABI_SUFFIX
        # A receiver whose plan carries no stacks (an old trainer, or a
        # receiver built without the derivation) refuses the suffixed ABI.
        plain = _FrozenPlan(ReshardPlan(bulk=_canonical(_four_equal(src_dim=None))))
        assert not plain.stacked
        with pytest.raises(ValueError, match="disagree on stacking"):
            plain.validate_topology(_topology(src, dst, trainer_abi))
        # The reverse: a stacked receiver plan against an unsuffixed topology.
        wire, _ = _derive_wire_plan(_stacked_plan(_four_equal(src_dim=None), 1 << 20))
        with pytest.raises(ValueError, match="requires the"):
            _FrozenPlan(wire).validate_topology(
                _topology(src, dst, SHARDED_DESTINATION_ABI)
            )
        # An old sharded receiver compares the whole string and refuses it too.
        old = _FrozenPlan(ReshardPlan(bulk=[_entry("only", src_dim=None)]))
        with pytest.raises(ValueError, match="disagree on stacking"):
            old.validate_topology(_topology(src, dst, trainer_abi))

    def test_unstacked_plans_keep_their_existing_abi_rules(self):
        src, dst = _mesh_pair()
        plain = _FrozenPlan(ReshardPlan(bulk=_canonical(_four_equal(src_dim=None))))
        plain.validate_topology(_topology(src, dst, SHARDED_DESTINATION_ABI))
        with pytest.raises(ValueError, match="requires the"):
            plain.validate_topology(_topology(src, dst, "other"))
        replicated = _FrozenPlan(
            ReshardPlan(bulk=[_entry("r", src_dim=None, dst_dim=None)])
        )
        replicated.validate_topology(_topology(src, dst, "arbitrary-abi"))


# --- trainer buffers ---------------------------------------------------------


class TestTrainerBuffers:
    def test_members_become_views_into_one_stack_with_no_later_copy(self):
        _wire, stacks = _derive_wire_plan(_stacked_plan(_four_equal(), 1 << 20))
        tensors = {
            f"w{i}": torch.full((4, 16), float(i + 1), dtype=torch.bfloat16)
            for i in range(4)
        }
        before = {name: tensor.clone() for name, tensor in tensors.items()}
        storage = _stack_trainer_buffers(stacks, tensors)

        stack = storage[stacks[0].name]
        assert tuple(stack.shape) == (4, 4, 16)
        base = stack.untyped_storage().data_ptr()
        for index, name in enumerate(stacks[0].members):
            member = tensors[name]
            assert member.is_contiguous()
            assert member.untyped_storage().data_ptr() == base
            assert member.data_ptr() == stack.data_ptr() + index * member.numel() * 2
            assert torch.equal(member, before[name])
        # The per-round refill writes the member buffer, which is the stack.
        tensors["w2"].copy_(torch.zeros((4, 16), dtype=torch.bfloat16))
        assert torch.equal(stack[2], torch.zeros((4, 16), dtype=torch.bfloat16))

    def test_mismatched_members_are_refused(self):
        _wire, stacks = _derive_wire_plan(_stacked_plan(_four_equal(count=2), 1 << 20))
        tensors = {
            "w0": torch.zeros((4, 16), dtype=torch.bfloat16),
            "w1": torch.zeros((4, 8), dtype=torch.bfloat16),
        }
        with pytest.raises(ValueError, match="needs identical members"):
            _stack_trainer_buffers(stacks, tensors)


# --- the wire arithmetic and a fake reshard ---------------------------------


@dataclass(frozen=True)
class Geometry:
    label: str
    dp: int
    tp: int
    engines: int
    engine_tp: int
    sharded: bool

    def meshes(self):
        src = MeshSpec((self.tp,)) if self.dp == 1 else MeshSpec((self.dp, self.tp))
        if not self.sharded:
            return src, MeshSpec((self.engines * self.engine_tp,), rank_offset=src.size)
        if self.engines == 1:
            return src, MeshSpec((self.engine_tp,), rank_offset=src.size)
        return src, MeshSpec((self.engines, self.engine_tp), rank_offset=src.size)


GEOMETRIES = [
    Geometry("tp1-replicate", 1, 1, 1, 1, False),
    Geometry("tp2-replicate", 1, 2, 1, 2, False),
    Geometry("tp1-to-tp2", 1, 1, 1, 2, True),
    Geometry("tp2-to-tp2", 1, 2, 1, 2, True),
    Geometry("tp2-to-tp4", 1, 2, 1, 4, True),
    Geometry("tp4-to-tp2", 1, 4, 1, 2, True),
    Geometry("tp4-to-tp4", 1, 4, 1, 4, True),
    Geometry("dp2tp2-to-tp2", 2, 2, 1, 2, True),
    Geometry("dp2tp2-to-2xtp2", 2, 2, 2, 2, True),
    Geometry("dp2tp2-replicate", 2, 2, 2, 2, False),
]


def _manifest(geometry):
    """Classes of equal-geometry members, one rank-3 pair and a singleton."""
    src_mesh, dst_mesh = geometry.meshes()
    meshes = (src_mesh, dst_mesh)
    dst_inner = (lambda dim: dim) if geometry.sharded else (lambda _dim: None)
    members = []
    for prefix, count, shape, dim in (
        ("a", 4, (8, 16), 0),
        ("b", 3, (16, 8), 1),
        ("c", 5, (8,), None),
        ("e", 1, (4, 8), 0),
        ("t", 2, (4, 4, 8), 0),
    ):
        for index in range(count):
            members.append(
                _entry(
                    f"p.{prefix}{index}",
                    shape,
                    src_dim=dim,
                    dst_dim=None if dim is None else dst_inner(dim),
                    meshes=meshes,
                )
            )
    return _canonical(members)


def _global_shaped(entry):
    seed = int(hashlib.sha256(entry.name.encode()).hexdigest()[:8], 16)
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(entry.global_shape, generator=generator).to(torch.bfloat16)


@pytest.mark.parametrize("geometry", GEOMETRIES, ids=lambda g: g.label)
def test_the_stacked_index_equals_each_members_own_index_stacked(geometry):
    entries = _manifest(geometry)
    plan = _stacked_plan(entries, 1 << 20)
    wire, stacks = _derive_wire_plan(plan)
    assert stacks, geometry.label
    by_name = {entry.name: entry for entry in entries}
    for stack in stacks:
        virtual = stack.entry
        members = [by_name[name] for name in stack.members]
        globals_ = [_global_shaped(member) for member in members]
        stacked = torch.stack(globals_)
        for side, mesh, placements in (
            ("src", virtual.src_mesh, virtual.src_placements),
            ("dst", virtual.dst_mesh, virtual.dst_placements),
        ):
            for rank in mesh.ranks():
                wire_slice = stacked[
                    _expected_index(virtual.global_shape, mesh, placements, rank)
                ]
                own = torch.stack(
                    [
                        g[
                            _expected_index(
                                m.global_shape,
                                mesh,
                                getattr(m, f"{side}_placements"),
                                rank,
                            )
                        ]
                        for g, m in zip(globals_, members, strict=True)
                    ]
                )
                assert torch.equal(wire_slice, own), (side, rank, stack.name)
    assert len(wire.bulk) < len(entries)


class _FakeStream:
    cuda_stream = 7

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def synchronize(self):
        pass


class _FakeComm:
    def __init__(self, rank):
        self.rank = rank

    def abort(self):
        pass


class _FakeWire:
    """Global-tensor semantics for ``nccl.m2n.reshard``, one call per wire entry.

    Calls pair up by their position in the wire plan, which is the property the
    real collective relies on: every rank walks the same entry sequence.
    """

    def __init__(self, entries):
        self.entries = entries
        self.deposits = {}
        self.calls = defaultdict(int)

    def reshard(self, src, dst, handle, **_kwargs):
        rank = handle.rank
        side = "src" if dst is None else "dst"
        position = self.calls[(side, rank)]
        self.calls[(side, rank)] += 1
        entry = self.entries[position]
        if dst is None:
            expected = _expected_index(
                entry.global_shape, entry.src_mesh, entry.src_placements, rank
            )
            assert tuple(src.shape) == tuple(
                axis.stop - axis.start for axis in expected
            )
            self.deposits[(position, rank)] = src.clone()
            return
        assembled = torch.zeros(entry.global_shape, dtype=torch.bfloat16)
        for source_rank in entry.src_mesh.ranks():
            index = _expected_index(
                entry.global_shape,
                entry.src_mesh,
                entry.src_placements,
                source_rank,
            )
            assembled[index] = self.deposits[(position, source_rank)]
        index = _expected_index(
            entry.global_shape, entry.dst_mesh, entry.dst_placements, rank
        )
        assert tuple(dst.shape) == tuple(axis.stop - axis.start for axis in index)
        dst.copy_(assembled[index])


@pytest.fixture
def fake_wire(monkeypatch):
    holder = {}
    module = ModuleType("nccl.m2n")
    module.reshard = lambda src, dst, handle, **kw: holder["wire"].reshard(
        src, dst, handle, **kw
    )
    monkeypatch.setitem(sys.modules, "nccl", ModuleType("nccl"))
    monkeypatch.setitem(sys.modules, "nccl.m2n", module)
    monkeypatch.setattr(backend, "loaded_nccl_version", lambda: backend.MIN_NCCL)
    return holder


def _half(cls, rank, plan, specs, world):
    cache = CommunicatorCache()
    for lane_id in (0, 1):
        cache._lanes[LaneKey("g", 1, lane_id)] = LaneCommunicator(
            _FakeComm(rank), rank=rank, world_size=world, stream=_FakeStream()
        )
    kwargs = {
        "plan": plan,
        "specs": specs,
        "group_id": "g",
        "epoch": 1,
        "cache": cache,
    }
    if cls is NcclM2nSender:
        kwargs["source_partition"] = 0
    return cls(**kwargs)


def _round(fake_wire, wire_plan, trainer_specs, receiver_specs, geometry):
    """Drive every trainer rank, then every receiver rank, through one round."""
    src_mesh, dst_mesh = geometry.meshes()
    wire = _FakeWire(_canonical(wire_plan.bulk))
    fake_wire["wire"] = wire
    world = src_mesh.size + dst_mesh.size
    for cls, mesh, all_specs in (
        (NcclM2nSender, src_mesh, trainer_specs),
        (NcclM2nReceiver, dst_mesh, receiver_specs),
    ):
        for rank in mesh.ranks():
            half = _half(cls, rank, wire_plan, all_specs[rank], world)
            half.start_weight_update("v1")
            for group_id in half.layer_group_ids:
                if cls is NcclM2nSender:
                    half.publish_weights(group_id)
                else:
                    half.update_weights(group_id)
            half.finish_weight_update(broadcast_lane_id=1)
    return wire


def _local(entry, mesh, placements, rank, full):
    return full[_expected_index(entry.global_shape, mesh, placements, rank)].clone()


@pytest.mark.parametrize("geometry", GEOMETRIES, ids=lambda g: g.label)
def test_stacked_and_unstacked_rounds_deliver_identical_bytes(fake_wire, geometry):
    entries = _manifest(geometry)
    src_mesh, dst_mesh = geometry.meshes()
    fulls = {entry.name: _global_shaped(entry) for entry in entries}
    by_name = {entry.name: entry for entry in entries}

    def trainer_tensors(rank):
        return {
            entry.name: _local(
                entry, src_mesh, entry.src_placements, rank, fulls[entry.name]
            )
            for entry in entries
        }

    # --- unstacked: today's per-tensor wire plan ---
    plain_plan = ReshardPlan(bulk=entries)
    plain_receive = {
        rank: {
            entry.name: torch.zeros_like(
                _local(entry, dst_mesh, entry.dst_placements, rank, fulls[entry.name])
            )
            for entry in entries
        }
        for rank in dst_mesh.ranks()
    }
    plain_wire = _round(
        fake_wire,
        plain_plan,
        {
            rank: {
                name: LocalParamSpec(base=t)
                for name, t in trainer_tensors(rank).items()
            }
            for rank in src_mesh.ranks()
        },
        {
            rank: {n: LocalParamSpec(base=t) for n, t in plain_receive[rank].items()}
            for rank in dst_mesh.ranks()
        },
        geometry,
    )

    # --- stacked ---
    stacked_plan = _stacked_plan(entries, 1 << 20)
    wire_plan, stacks = _derive_wire_plan(stacked_plan)
    stacked_trainer = {}
    trainer_members = {}
    for rank in src_mesh.ranks():
        tensors = trainer_tensors(rank)
        storage = _stack_trainer_buffers(stacks, tensors)
        trainer_members[rank] = tensors
        member_names = {name for stack in stacks for name in stack.members}
        specs = {
            name: LocalParamSpec(base=t)
            for name, t in tensors.items()
            if name not in member_names
        }
        specs.update({name: LocalParamSpec(base=t) for name, t in storage.items()})
        stacked_trainer[rank] = specs
    stack_by_name = {stack.name: stack for stack in stacks}
    stacked_receive = {}
    for rank in dst_mesh.ranks():
        buffers = {}
        for wire_entry in wire_plan.bulk:
            local = _expected_index(
                wire_entry.global_shape,
                wire_entry.dst_mesh,
                wire_entry.dst_placements,
                rank,
            )
            shape = tuple(axis.stop - axis.start for axis in local)
            buffers[wire_entry.name] = torch.zeros(shape, dtype=torch.bfloat16)
        stacked_receive[rank] = buffers
    stacked_wire = _round(
        fake_wire,
        wire_plan,
        stacked_trainer,
        {
            rank: {n: LocalParamSpec(base=t) for n, t in buffers.items()}
            for rank, buffers in stacked_receive.items()
        },
        geometry,
    )

    # The trainer's members are views of the stacks it sends.
    for rank, tensors in trainer_members.items():
        for stack in stacks:
            storage = stacked_trainer[rank][stack.name].base
            for index, name in enumerate(stack.members):
                assert tensors[name].untyped_storage().data_ptr() == (
                    storage.untyped_storage().data_ptr()
                )
                assert tensors[name].data_ptr() == (
                    storage.data_ptr() + index * tensors[name].numel() * 2
                )

    # Same bytes at every receiver, for every member, and equal to the truth.
    for rank in dst_mesh.ranks():
        for entry in entries:
            expected = _local(
                entry, dst_mesh, entry.dst_placements, rank, fulls[entry.name]
            )
            assert torch.equal(plain_receive[rank][entry.name], expected)
        for wire_entry in wire_plan.bulk:
            buffer = stacked_receive[rank][wire_entry.name]
            if wire_entry.name in stack_by_name:
                members = stack_by_name[wire_entry.name].members
                for name, piece in zip(members, buffer.unbind(0), strict=True):
                    expected = _local(
                        by_name[name],
                        dst_mesh,
                        by_name[name].dst_placements,
                        rank,
                        fulls[name],
                    )
                    assert torch.equal(piece, expected), (name, rank)
            else:
                assert torch.equal(
                    buffer,
                    _local(
                        by_name[wire_entry.name],
                        dst_mesh,
                        by_name[wire_entry.name].dst_placements,
                        rank,
                        fulls[wire_entry.name],
                    ),
                )

    # One call per wire entry per rank, against one per tensor today.
    for rank in src_mesh.ranks():
        assert plain_wire.calls[("src", rank)] == len(entries)
        assert stacked_wire.calls[("src", rank)] == len(wire_plan.bulk)
    for rank in dst_mesh.ranks():
        assert stacked_wire.calls[("dst", rank)] == len(wire_plan.bulk)
    assert len(wire_plan.bulk) == len(stacks) + 1 + 2  # singleton plus the rank-3 pair


def test_the_geometry_matrix_covers_tp1_tp2_tp4_and_dp2_tp2():
    shapes = {(g.dp, g.tp, g.engine_tp) for g in GEOMETRIES}
    assert {(1, 1, 1), (1, 2, 2), (1, 4, 4), (2, 2, 2)} <= shapes
    assert len({g.label for g in GEOMETRIES}) == len(GEOMETRIES)


class TestPackStagingBucketSizes:
    """envs.pack_staging_largest_bucket_bytes bounds one native PACK request."""

    def test_unset_is_the_two_gib_default_bucket(self, monkeypatch):
        monkeypatch.delenv("NCCL_RESHARD_PACK_BUFFSIZES", raising=False)
        assert envs.pack_staging_largest_bucket_bytes() == 2048 * 1024 * 1024

    def test_the_largest_listed_bucket_wins(self, monkeypatch):
        monkeypatch.setenv("NCCL_RESHARD_PACK_BUFFSIZES", "512M:2,2048k,1M:3")
        assert envs.pack_staging_largest_bucket_bytes() == 512 * 1024 * 1024

    def test_a_bare_size_is_bytes(self, monkeypatch):
        monkeypatch.setenv("NCCL_RESHARD_PACK_BUFFSIZES", "4096")
        assert envs.pack_staging_largest_bucket_bytes() == 4096

    def test_an_invalid_value_keeps_the_default_like_the_native_parser(
        self, monkeypatch
    ):
        monkeypatch.setenv("NCCL_RESHARD_PACK_BUFFSIZES", "2G:8")
        assert envs.pack_staging_largest_bucket_bytes() == 2048 * 1024 * 1024

    def test_the_parser_returns_slots_and_sizes_in_listed_order(self):
        assert envs._parse_pack_buffsizes("4M:2,2048k") == (3, (4 << 20, 2048 << 10))
        assert envs._parse_pack_buffsizes("1K:4") is None

    def test_non_ascii_digits_and_spaces_are_rejected_like_the_native_parser(self):
        # Python's \d and int() accept Arabic-Indic digits and \s matches a
        # non-breaking space; the native grammar is ASCII-only, so the mirror
        # refuses both instead of reporting a bucket the native pool rejected.
        assert envs._parse_pack_buffsizes("\u0664M:2") is None
        assert envs._parse_pack_buffsizes("4M\xa0:2") is None
