# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from itertools import count
from types import SimpleNamespace
from typing import ClassVar

import pytest
import torch

import modelexpress_rl.collective.integrations.sglang as sglang_integration
from modelexpress_rl.collective import MeshSpec, ParamPlan, Placement, ReshardPlan
from modelexpress_rl.collective.integrations.miles import CollectiveTopology
from modelexpress_rl.collective.integrations.sglang import (
    SglangGeneratorSession,
    SglangLoader,
)

_NAMES = (
    "model.layers.0.mlp.gate_proj.weight",
    "model.layers.0.self_attn.q_proj.weight",
)


class FakeTensor:
    def __init__(self, shape, *, address):
        self.shape = tuple(shape)
        self.dtype = "torch.bfloat16"
        self.device = "cuda:0"
        self.address = address

    def data_ptr(self):
        return self.address

    def is_contiguous(self):
        return True

    def __getitem__(self, index):
        # stack[i]: a contiguous view of the leading dimension.
        inner = self.shape[1:]
        size = 1
        for extent in inner:
            size *= extent
        return FakeTensor(inner, address=self.address + index * size * 2)


@pytest.fixture
def fake_buffers(monkeypatch):
    """Allocate CUDA-labelled stand-ins so the loader runs without a GPU."""
    addresses = count(0x1000, 0x1000)
    allocated = []

    def empty(shape, *, dtype, device):
        assert dtype is torch.bfloat16
        assert device == "cuda:0"
        tensor = FakeTensor(shape, address=next(addresses))
        allocated.append(tensor)
        return tensor

    monkeypatch.setattr(torch, "empty", empty)
    return allocated


def _entry(name, *, dst_placement=None):
    return ParamPlan(
        name=name,
        global_shape=(4, 4),
        dtype="bfloat16",
        partition_id=0,
        src_mesh=MeshSpec(shape=(1,), rank_offset=0),
        src_placements=(Placement.replicate(),),
        dst_mesh=MeshSpec(shape=(2,), rank_offset=1),
        dst_placements=(dst_placement or Placement.replicate(),),
        group_key="layer-0",
    )


def _plan(**kwargs):
    return ReshardPlan(
        bulk=[_entry(name, **kwargs) for name in _NAMES],
        source_partition_count=1,
    )


def _topology():
    return CollectiveTopology(
        model_name="qwen",
        trainer_slots=("run:trainer-0",),
        generator_slots=("run:generator-0", "run:generator-1"),
        source_partition_count=1,
        m2n_abi_version="nccl-m2n-2.30.7",
    )


class FakeModel:
    def __init__(self, *, error=None):
        self.loaded = []
        self.error = error

    def modules(self):
        # Stands in for the live nn.Module; it carries no submodules and
        # therefore no derived weight caches.
        return iter(())

    def load_weights(self, weights):
        weights = list(weights)
        self.loaded.append(weights)
        if self.error is not None:
            raise self.error


def _loader(model=None, *, per_entry=True):
    return SglangLoader(
        plan=_plan(),
        model=FakeModel() if model is None else model,
        device="cuda:0",
        layer_groups=tuple((name,) for name in _NAMES) if per_entry else (),
    )


class TestSglangLoader:
    def test_noncanonical_plan_is_rejected_before_allocating(self, fake_buffers):
        plan = _plan()
        plan.bulk.reverse()
        with pytest.raises(ValueError, match="canonical wire order"):
            SglangLoader(plan=plan, model=FakeModel(), device="cuda:0")
        assert fake_buffers == []

    def test_one_receive_buffer_per_entry_backs_the_local_params(self, fake_buffers):
        loader = _loader()

        specs = loader.local_params()

        assert list(specs) == list(_NAMES)
        assert [specs[name].base for name in _NAMES] == fake_buffers
        assert all(buffer.shape == (4, 4) for buffer in fake_buffers)
        assert loader.device == "cuda:0"

    def test_install_hands_each_group_to_the_model_loader_in_plan_order(
        self, fake_buffers
    ):
        model = FakeModel()
        loader = _loader(model)

        loader.start_new_round("v1")
        loader.install(0)
        loader.install(1)
        loader.finish()

        assert model.loaded == [
            [(_NAMES[0], fake_buffers[0])],
            [(_NAMES[1], fake_buffers[1])],
        ]

    def test_default_grouping_installs_the_whole_plan_at_once(self, fake_buffers):
        model = FakeModel()
        loader = _loader(model, per_entry=False)

        loader.start_new_round("v1")
        loader.install(0)

        assert model.loaded == [list(zip(_NAMES, fake_buffers, strict=True))]

    def test_install_requires_an_open_round_and_a_declared_group(self, fake_buffers):
        loader = _loader()
        with pytest.raises(RuntimeError, match="start_new_round must run"):
            loader.install(0)
        loader.start_new_round("v1")
        with pytest.raises(ValueError, match="outside the 2 declared"):
            loader.install(2)
        with pytest.raises(RuntimeError, match="already in flight"):
            loader.start_new_round("v2")

    def test_a_failed_install_poisons_every_later_round(self, fake_buffers):
        loader = _loader(FakeModel(error=RuntimeError("bad weights")))
        loader.start_new_round("v1")

        with pytest.raises(RuntimeError, match="bad weights"):
            loader.install(0)

        assert loader.poisoned
        loader.fail_round(possibly_mutated=False)
        with pytest.raises(RuntimeError, match="poisoned"):
            loader.start_new_round("v2")
        with pytest.raises(RuntimeError, match="poisoned"):
            loader.local_params()

    def test_storage_drift_is_refused_before_a_new_round(self, fake_buffers):
        loader = _loader()
        fake_buffers[1].address += 8

        with pytest.raises(RuntimeError, match="storage address changed"):
            loader.start_new_round("v1")

    def test_sharded_plans_need_the_engine_rank_and_a_provable_view(self, fake_buffers):
        plan = _plan(dst_placement=Placement.shard(0))
        with pytest.raises(ValueError, match="needs this generator's index"):
            SglangLoader(plan=plan, model=FakeModel(), device="cuda:0")
        with pytest.raises(ValueError, match="engine runs TP 1"):
            SglangLoader(
                plan=plan, model=FakeModel(), device="cuda:0", generator_index=0
            )
        with pytest.raises(ValueError, match="needs the SGLang model's HF config"):
            SglangLoader(
                plan=plan,
                model=FakeModel(),
                device="cuda:0",
                generator_index=1,
                tp_rank=1,
                tp_size=2,
            )
        assert fake_buffers == []

    def test_topology_must_match_the_plan_meshes(self, fake_buffers):
        loader = _loader()
        loader.validate_topology(_topology())
        wider = CollectiveTopology(
            model_name="qwen",
            trainer_slots=("t0",),
            generator_slots=("g0", "g1", "g2"),
            source_partition_count=1,
            m2n_abi_version="abi",
        )
        with pytest.raises(ValueError, match="one generator slot per"):
            loader.validate_topology(wider)


_STACK_NAMES = tuple(f"model.layers.0.p{index}.weight" for index in range(4))
_STACK_KEYS = ("m2n-stack1-0", "m2n-stack1-0", "m2n-stack1-1", "m2n-stack1-1")


def _stacked_plan(keys=_STACK_KEYS):
    return ReshardPlan(
        bulk=[
            replace(_entry(name), group_key=key)
            for name, key in zip(_STACK_NAMES, keys, strict=True)
        ],
        source_partition_count=1,
    )


def _stack_topology(abi_suffix="+stack1"):
    base = _topology()
    return replace(base, m2n_abi_version="miles-sglang-bf16-replicated-v1" + abi_suffix)


class TestStackedLoader:
    def test_a_stack_is_one_receive_buffer_and_its_members_are_views(
        self, fake_buffers
    ):
        loader = SglangLoader(plan=_stacked_plan(), model=FakeModel(), device="cuda:0")

        specs = loader.local_params()
        wire = loader.capture()

        assert loader.parameter_names() == [entry.name for entry in wire.bulk]
        assert all(name.startswith("m2n-stack1/") for name in specs)
        assert len(specs) == 2
        # Two stack tensors were allocated, not one buffer per tensor.
        stacks = [spec.base for spec in specs.values()]
        assert [tuple(stack.shape) for stack in stacks] == [(2, 4, 4), (2, 4, 4)]
        assert [stack in fake_buffers for stack in stacks] == [True, True]
        assert len(fake_buffers) == 2
        # Members read the stack in place.
        assert loader._buffers[_STACK_NAMES[1]].address == stacks[0].address + 4 * 4 * 2
        assert loader._buffers[_STACK_NAMES[3]].address == stacks[1].address + 4 * 4 * 2

    def test_each_stack_is_one_load_weights_call_over_its_member_views(
        self, fake_buffers
    ):
        model = FakeModel()
        loader = SglangLoader(plan=_stacked_plan(), model=model, device="cuda:0")
        assert loader.layer_groups == (tuple(loader.parameter_names()),)
        wire_names = loader.parameter_names()
        loader = SglangLoader(
            plan=_stacked_plan(),
            model=model,
            device="cuda:0",
            layer_groups=tuple((name,) for name in wire_names),
        )

        loader.start_new_round("v1")
        assert loader._staged_members(0)
        loader.install(0)
        loader.install(1)
        loader.finish()

        assert [[name for name, _ in call] for call in model.loaded] == [
            list(_STACK_NAMES[:2]),
            list(_STACK_NAMES[2:]),
        ]
        for call, members in zip(
            model.loaded, (_STACK_NAMES[:2], _STACK_NAMES[2:]), strict=True
        ):
            assert [tensor.address for _, tensor in call] == [
                loader._buffers[name].address for name in members
            ]

    def test_the_loader_digests_the_same_wire_plan_as_the_trainer(self, fake_buffers):
        from modelexpress_rl.collective.integrations._common import _derive_wire_plan

        loader = SglangLoader(plan=_stacked_plan(), model=FakeModel(), device="cuda:0")
        trainer_wire, _ = _derive_wire_plan(_stacked_plan())
        assert loader.capture().bulk == trainer_wire.bulk

    def test_a_stacked_plan_needs_the_stack_abi_and_an_unstacked_one_refuses_it(
        self, fake_buffers
    ):
        stacked = SglangLoader(plan=_stacked_plan(), model=FakeModel(), device="cuda:0")
        stacked.validate_topology(_stack_topology())
        with pytest.raises(ValueError, match="requires the"):
            stacked.validate_topology(_stack_topology(""))
        plain = _loader()
        with pytest.raises(ValueError, match="disagree on stacking"):
            plain.validate_topology(_stack_topology())

    def test_a_reserved_name_or_a_bad_stack_plan_fails_before_allocating(
        self, fake_buffers
    ):
        reserved = ReshardPlan(
            bulk=[_entry("m2n-stack-real"), _entry("zz")], source_partition_count=1
        )
        with pytest.raises(ValueError, match="reserved"):
            SglangLoader(plan=reserved, model=FakeModel(), device="cuda:0")
        gap = _stacked_plan(
            ("m2n-stack1-0", "m2n-stack1-0", "m2n-stack1-2", "m2n-stack1-2")
        )
        with pytest.raises(ValueError, match="exactly 0..K-1"):
            SglangLoader(plan=gap, model=FakeModel(), device="cuda:0")
        assert fake_buffers == []

    def test_a_stack_over_the_pack_bucket_is_refused_at_construction(
        self, fake_buffers, monkeypatch
    ):
        big = ReshardPlan(
            bulk=[
                replace(
                    _entry(name),
                    global_shape=(64, 64),
                    group_key="m2n-stack1-0",
                )
                for name in _STACK_NAMES[:2]
            ],
            source_partition_count=1,
        )
        monkeypatch.setenv("NCCL_RESHARD_PACK_BUFFSIZES", "2048")
        with pytest.raises(ValueError, match="largest PACK staging bucket of 2048"):
            SglangLoader(plan=big, model=FakeModel(), device="cuda:0")
        assert fake_buffers == []


class FakeRendezvous:
    def __init__(self, *, report_error=None):
        self.reported = []
        self.closed = 0
        self.report_error = report_error

    def report(self, **kwargs):
        self.reported.append(kwargs)
        if self.report_error is not None:
            raise self.report_error

    def close(self):
        self.closed += 1


class FakeGeneratorClient:
    def __init__(self, *, update_error=None, compute_error=None):
        self.membership = SimpleNamespace(group_id="group-1", epoch=3)
        self.events = []
        self.update_error = update_error
        self.compute_error = compute_error

    def initialize(self, loader):
        self.events.append(("initialize", loader))

    def setup_layer_groups(self, groups):
        self.events.append(("groups", groups))

    def compute_plan(self):
        self.events.append(("compute",))
        if self.compute_error is not None:
            raise self.compute_error
        return self.membership

    def start_weight_update(self, version):
        self.events.append(("start", version))

    def update_weights(self, version, layer_group_id):
        self.events.append(("update", version, layer_group_id))
        if self.update_error is not None:
            raise self.update_error

    def finish_weight_update(self, version, operation_id=None):
        self.events.append(("finish", version, operation_id))

    def cleanup(self):
        self.events.append(("cleanup",))


def _session(client, rendezvous, loader):
    return SglangGeneratorSession(
        client=client,
        rendezvous=rendezvous,
        loader=loader,
        worker_id="sglang-run:generator-0-abc",
    )


class TestSglangGeneratorSession:
    def test_a_round_receives_every_group_then_finishes_under_the_operation(
        self, fake_buffers
    ):
        client = FakeGeneratorClient()
        rendezvous = FakeRendezvous()
        loader = _loader()
        session = _session(client, rendezvous, loader)

        assert session.prepare() is client.membership
        session.run_round(version="1", operation_id="miles-g-weight-version-1")

        assert client.events[0] == ("initialize", loader)
        assert client.events[1] == ("groups", [[_NAMES[0]], [_NAMES[1]]])
        assert [event[0] for event in client.events[2:]] == [
            "compute",
            "start",
            "update",
            "update",
            "finish",
        ]
        assert client.events[-1] == ("finish", "1", "miles-g-weight-version-1")
        # A successful round is reported by the client's own finish, not here.
        assert rendezvous.reported == []
        assert rendezvous.closed == 0

    def test_a_round_requires_prepare(self, fake_buffers):
        session = _session(FakeGeneratorClient(), FakeRendezvous(), _loader())
        with pytest.raises(RuntimeError, match="prepare must complete"):
            session.run_round(version="1", operation_id="op-1")

    def test_a_failed_round_poisons_reports_and_closes(self, fake_buffers):
        client = FakeGeneratorClient(update_error=RuntimeError("lane died"))
        rendezvous = FakeRendezvous()
        loader = _loader()
        session = _session(client, rendezvous, loader)
        session.prepare()

        with pytest.raises(RuntimeError, match="lane died"):
            session.run_round(version="1", operation_id="op-1")

        assert loader.poisoned
        assert client.events[-1] == ("cleanup",)
        assert rendezvous.reported == [
            {
                "operation_id": "op-1",
                "group_id": "group-1",
                "epoch": 3,
                "worker_id": "sglang-run:generator-0-abc",
                "succeeded": False,
                "message": "RuntimeError('lane died')",
            }
        ]
        assert rendezvous.closed == 1
        with pytest.raises(RuntimeError, match="session is closed"):
            session.run_round(version="2", operation_id="op-2")

    def test_a_failed_report_does_not_mask_the_round_error(self, fake_buffers):
        client = FakeGeneratorClient(update_error=RuntimeError("lane died"))
        rendezvous = FakeRendezvous(report_error=RuntimeError("server gone"))
        session = _session(client, rendezvous, _loader())
        session.prepare()

        with pytest.raises(RuntimeError, match="lane died"):
            session.run_round(version="1", operation_id="op-1")
        assert rendezvous.closed == 1

    def test_a_failed_prepare_closes_the_session(self, fake_buffers):
        client = FakeGeneratorClient(compute_error=TimeoutError("not ready"))
        rendezvous = FakeRendezvous()
        session = _session(client, rendezvous, _loader())

        with pytest.raises(TimeoutError, match="not ready"):
            session.prepare()
        assert client.events[-1] == ("cleanup",)
        assert rendezvous.closed == 1
        with pytest.raises(RuntimeError, match="session is closed"):
            session.prepare()

    def test_close_is_one_shot(self, fake_buffers):
        client = FakeGeneratorClient()
        rendezvous = FakeRendezvous()
        session = _session(client, rendezvous, _loader())
        session.close()
        session.close()
        assert client.events == [("cleanup",)]
        assert rendezvous.closed == 1

    def test_the_factory_builds_the_generator_client_from_the_topology(
        self, fake_buffers, monkeypatch
    ):
        captured = {}

        class Client(FakeGeneratorClient):
            def __init__(self, **kwargs):
                super().__init__()
                captured.update(kwargs)

        monkeypatch.setattr(sglang_integration, "RefitClientGenerator", Client)
        session = SglangGeneratorSession.create(
            rendezvous=FakeRendezvous(),
            topology=_topology(),
            loader=_loader(),
            slot_id="run:generator-1",
            worker_id="sglang-run:generator-1-abc",
            index_in_role=1,
            device="cuda:0",
            streams=[None],
        )

        assert isinstance(session, SglangGeneratorSession)
        assert captured["model_name"] == "qwen"
        assert captured["trainer_slots"] == ["run:trainer-0"]
        assert captured["generator_slots"] == ["run:generator-0", "run:generator-1"]
        assert captured["slot_id"] == "run:generator-1"
        assert captured["index_in_role"] == 1
        assert captured["m2n_abi_version"] == "nccl-m2n-2.30.7"
        assert captured["device"] == "cuda:0"


class FakeChannel:
    def __init__(self, endpoint):
        self.endpoint = endpoint
        self.closed = 0

    def close(self):
        self.closed += 1


class FakeSession:
    instances: ClassVar[list] = []

    def __init__(self, *, round_error=None, **kwargs):
        self.kwargs = kwargs
        self.rounds = []
        self.closed = 0
        self.round_error = round_error
        FakeSession.instances.append(self)

    def prepare(self):
        return SimpleNamespace(group_id="g", epoch=1)

    def run_round(self, *, version, operation_id):
        self.rounds.append((version, operation_id))
        if self.round_error is not None:
            raise self.round_error

    def close(self):
        self.closed += 1
