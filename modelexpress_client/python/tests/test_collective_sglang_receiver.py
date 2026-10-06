# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
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

    def test_sharded_plans_are_refused(self, fake_buffers):
        with pytest.raises(ValueError, match="replicated placements only"):
            SglangLoader(
                plan=_plan(dst_placement=Placement.shard(0)),
                model=FakeModel(),
                device="cuda:0",
            )

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
        with pytest.raises(ValueError, match="dst_mesh ranks"):
            loader.validate_topology(wider)


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


# --- counted truncation in the shared helpers --------------------------------


def test_layer_group_coverage_errors_count_the_unshown_names(fake_buffers):
    plan = ReshardPlan(
        bulk=[_entry(f"model.{index}") for index in range(7)],
        source_partition_count=1,
    )
    with pytest.raises(ValueError, match="missing=6 total") as raised:
        SglangLoader(
            plan=plan,
            model=FakeModel(),
            device="cuda:0",
            layer_groups=(("model.0",),),
        )

    # Not a silent [:5] cut: the detail counts the set and names the elision.
    assert "(+1 more)" in str(raised.value)
    assert "model.5" in str(raised.value)
    assert "model.6" not in str(raised.value)
    assert "unknown=0 total: -" in str(raised.value)


# --- tied-alias detection (Qwen3-0.6B class: lm_head IS embed_tokens) --------


class _TiedEmbeddingModel(torch.nn.Module):
    """A tied-embedding stand-in: ``lm_head`` IS ``model.embed_tokens``.

    Mirrors SGLang's Qwen3 wiring (``self.lm_head = self.model.embed_tokens``),
    so ``named_parameters()`` dedupes ``lm_head.weight`` away and the model's
    ``load_weights`` resolution cannot see a write aimed at it.
    """

    def __init__(self):
        super().__init__()
        inner = torch.nn.Module()
        inner.embed_tokens = torch.nn.Module()
        # A plain registered parameter: the same ``_parameters`` shape an
        # nn.Embedding carries, without a device allocation.
        inner.embed_tokens.weight = torch.nn.Parameter(
            torch.zeros((4, 4), dtype=torch.bfloat16)
        )
        self.model = inner
        self.lm_head = inner.embed_tokens

    def load_weights(self, weights):
        # Same contract the real models give: a deduplicated name lookup.
        params = dict(self.named_parameters())
        self.loaded = [(name, name in params) for name, _weight in weights]


def _alias_plan(*names):
    return ReshardPlan(
        bulk=sorted((_entry(name) for name in names), key=lambda entry: entry.canonical()),
        source_partition_count=1,
    )


def _alias_context(model, plan):
    import modelexpress_rl.collective.integrations.sglang_receiver as receiver_module
    from modelexpress_rl.collective.integrations.manifest import manifest_to_wire

    return receiver_module, SimpleNamespace(
        master_address="mx-server",
        master_port=8001,
        rank_offset=0,
        world_size=2,
        init_payload=manifest_to_wire(plan, _topology()),
        model=model,
        device="cuda:0",
        tp_rank=1,
        tp_size=2,
    )


@pytest.fixture
def fake_receiver_backend(monkeypatch):
    """Fake the server/engine boundary so create_receiver runs off-cluster."""
    import modelexpress_rl.collective.integrations.sglang_receiver as receiver_module

    FakeSession.instances = []

    monkeypatch.setattr(
        receiver_module.SglangGeneratorSession, "create", staticmethod(FakeSession)
    )
    monkeypatch.setattr(
        receiver_module.grpc, "insecure_channel", FakeChannel
    )
    monkeypatch.setattr(receiver_module.auth, "with_auth", lambda channel: channel)
    monkeypatch.setattr(
        receiver_module, "CollectiveRendezvous", lambda channel: FakeRendezvous()
    )


@pytest.mark.usefixtures("fake_buffers", "fake_receiver_backend")
class TestTiedAliasReceiver:
    def test_the_alias_map_names_each_alias_load_visible_registration(self):
        import modelexpress_rl.collective.integrations.sglang_receiver as receiver_module

        assert receiver_module._registered_parameter_aliases(
            _TiedEmbeddingModel()
        ) == {"lm_head.weight": "model.embed_tokens.weight"}

    def test_a_tied_lm_head_model_is_accepted(self, caplog):
        # Qwen3-0.6B's shape: the trainer stream carries the shared table
        # under its load-visible name; lm_head.weight is the registered alias.
        model = _TiedEmbeddingModel()
        plan = _alias_plan("model.embed_tokens.weight", "model.norm.weight")
        receiver_module, context = _alias_context(model, plan)

        with caplog.at_level(logging.DEBUG, logger=receiver_module.logger.name):
            receiver = receiver_module.create_receiver(context)

        assert isinstance(receiver, receiver_module.ModelExpressM2NReceiver)
        assert "lm_head.weight<-model.embed_tokens.weight" in caplog.text

    def test_a_plan_carrying_both_tied_names_is_accepted(self):
        model = _TiedEmbeddingModel()
        plan = _alias_plan("model.embed_tokens.weight", "lm_head.weight")
        receiver_module, context = _alias_context(model, plan)

        receiver = receiver_module.create_receiver(context)

        assert isinstance(receiver, receiver_module.ModelExpressM2NReceiver)

    def test_a_plan_naming_only_the_load_hidden_alias_is_refused(self):
        # The engine's deduplicated lookup would silently drop the received
        # alias bytes; with the shared registration absent from the plan the
        # table would never be written. Refuse by name instead.
        model = _TiedEmbeddingModel()
        plan = _alias_plan("lm_head.weight", "model.norm.weight")
        receiver_module, context = _alias_context(model, plan)

        with pytest.raises(ValueError, match="silently dropped") as raised:
            receiver_module.create_receiver(context)

        assert "lm_head.weight (alias of model.embed_tokens.weight)" in str(
            raised.value
        )
