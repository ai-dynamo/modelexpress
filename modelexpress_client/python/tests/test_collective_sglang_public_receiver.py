# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ModelExpress receiver factory for SGLang's external receiver contract.

The factory runs against fake SGLang contexts always, and against SG-1's own
``WeightUpdateReceiverContext`` and ``WeightUpdater`` when
``MX_TEST_SGLANG_PYTHON`` names an SGLang ``python/`` tree that has the
allowlisted external receiver extension (and its dependencies are installed):
``MX_TEST_SGLANG_PYTHON=<sglang>/python pytest tests/test_collective_sglang_public_receiver.py``.
"""

from __future__ import annotations

import importlib
import os
import sys
from contextlib import contextmanager
from itertools import count
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

import modelexpress_rl.collective.integrations.sglang_receiver as receiver_module
from modelexpress_rl.collective.integrations.sglang_receiver import (
    RECEIVER_PATH,
    ModelExpressM2NReceiver,
    create_receiver,
    mx_server_endpoint,
)
from modelexpress_rl.collective.integrations.manifest import (
    MANIFEST_SCHEMA,
    manifest_from_wire,
    manifest_to_wire,
)
from tests.test_collective_sglang_receiver import (
    _NAMES,
    _STACK_KEYS,
    _stack_topology,
    _stacked_plan,
    FakeChannel,
    FakeTensor,
    FakeModel,
    FakeSession,
    FakeRendezvous,
    _plan,
    _topology,
    fake_buffers,  # noqa: F401
)


def _manifest(plan=None):
    return manifest_to_wire(_plan() if plan is None else plan, _topology())


def _context(**overrides):
    values = {
        "master_address": "mx-server",
        "master_port": 8001,
        "rank_offset": 0,
        "world_size": 2,
        "group_name": "mx-m2n-g",
        "init_payload": _manifest(),
        "model": FakeModel(),
        "device": "cuda:0",
        "tp_rank": 1,
        "tp_size": 2,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.fixture
def fake_mx(monkeypatch):
    FakeSession.instances = []
    channels = []
    rendezvous = []
    prepare_errors = []

    def create(**kwargs):
        session = FakeSession(**kwargs)
        if prepare_errors:
            error = prepare_errors.pop(0)

            def failing_prepare(error=error):
                raise error

            session.prepare = failing_prepare
        return session

    def insecure_channel(endpoint):
        channel = FakeChannel(endpoint)
        channels.append(channel)
        return channel

    def new_rendezvous(channel):
        fake = FakeRendezvous()
        rendezvous.append(fake)
        return fake

    monkeypatch.setattr(
        receiver_module.SglangGeneratorSession, "create", staticmethod(create)
    )
    monkeypatch.setattr(receiver_module.grpc, "insecure_channel", insecure_channel)
    monkeypatch.setattr(receiver_module.auth, "with_auth", lambda channel: channel)
    monkeypatch.setattr(receiver_module, "CollectiveRendezvous", new_rendezvous)
    return SimpleNamespace(
        channels=channels, rendezvous=rendezvous, prepare_errors=prepare_errors
    )


@pytest.fixture
def torch_device_buffers(monkeypatch):
    """Like ``fake_buffers`` but for the ``torch.device`` SGLang's context carries."""
    addresses = count(0x1000, 0x1000)

    def empty(shape, *, dtype, device):
        assert dtype is torch.bfloat16
        assert isinstance(device, torch.device) and str(device) == "cuda:0"
        return FakeTensor(shape, address=next(addresses))

    monkeypatch.setattr(torch, "empty", empty)


def _round(version="3", operation_id="op-3"):
    return {"operation_id": operation_id, "version": version}


@pytest.mark.usefixtures("fake_buffers")
class TestFactory:
    def test_the_receiver_path_names_the_factory(self):
        module, _, name = RECEIVER_PATH.rpartition(".")
        assert getattr(importlib.import_module(module), name) is create_receiver

    def test_it_builds_one_session_from_the_manifest_context(
        self,
        fake_mx,
    ):
        receiver = create_receiver(_context(rank_offset=0))

        assert isinstance(receiver, ModelExpressM2NReceiver)
        assert receiver.slot_id == "run:generator-1"
        (session,) = FakeSession.instances
        assert session.kwargs["slot_id"] == "run:generator-1"
        assert session.kwargs["index_in_role"] == 1
        assert session.kwargs["worker_id"].startswith("sglang-run:generator-1-")
        assert session.kwargs["device"] == "cuda:0"
        assert session.kwargs["loader"].layer_groups == tuple((n,) for n in _NAMES)
        assert session.kwargs["topology"] == _topology()
        assert fake_mx.channels[0].endpoint == "mx-server:8001"

    def test_rank_offset_selects_the_generator_slot(self, fake_mx):
        context = _context(rank_offset=1, tp_rank=0, tp_size=2)
        assert create_receiver(context).slot_id == "run:generator-1"

    def test_a_stacking_plan_uses_one_group_per_wire_entry(self, fake_mx):
        manifest = manifest_to_wire(_stacked_plan(_STACK_KEYS), _stack_topology())
        create_receiver(_context(init_payload=manifest))
        kwargs = FakeSession.instances[0].kwargs
        loader = kwargs["loader"]
        assert loader.layer_groups == tuple((n,) for n in loader.parameter_names())
        assert len(loader.layer_groups) == 2

    def test_a_round_runs_under_its_version_and_operation(
        self,
        fake_mx,
    ):
        receiver = create_receiver(_context())
        receiver.receive(_round("3", "op-3"))
        receiver.receive(_round("4", "op-4"))
        assert FakeSession.instances[0].rounds == [("3", "op-3"), ("4", "op-4")]

    @pytest.mark.parametrize(
        "bad",
        [
            None,
            {},
            {"operation_id": "op"},
            {"version": "1"},
            {"operation_id": "op", "version": "1", "extra": 1},
            {"operation_id": "", "version": "1"},
            {"operation_id": "op", "version": " "},
            {"operation_id": 7, "version": "1"},
            {"operation_id": "op", "version": 1},
        ],
    )
    def test_a_malformed_round_is_refused_before_the_session(
        self,
        fake_mx,
        bad,
    ):
        receiver = create_receiver(_context())
        with pytest.raises(ValueError, match="receiver_payload"):
            receiver.receive(bad)
        assert FakeSession.instances[0].rounds == []
        # A bad request is not a failed round: the receiver stays usable.
        receiver.receive(_round())
        assert FakeSession.instances[0].rounds == [("3", "op-3")]

    def test_a_failed_round_poisons_the_receiver_until_destroy(
        self,
        fake_mx,
    ):
        receiver = create_receiver(_context())
        session = FakeSession.instances[0]
        session.round_error = RuntimeError("lane failed")
        with pytest.raises(RuntimeError, match="lane failed"):
            receiver.receive(_round())
        assert fake_mx.channels[0].closed == 1
        assert fake_mx.rendezvous[0].closed == 1

        session.round_error = None
        with pytest.raises(RuntimeError, match="poisons this receiver"):
            receiver.receive(_round("4", "op-4"))
        assert session.rounds == [("3", "op-3")]

        receiver.destroy()
        assert session.closed == 1

    def test_destroy_closes_everything_once(self, fake_mx):
        receiver = create_receiver(_context())
        receiver.destroy()
        receiver.destroy()
        assert FakeSession.instances[0].closed == 1
        assert fake_mx.channels[0].closed == 1
        assert fake_mx.rendezvous[0].closed == 1
        with pytest.raises(RuntimeError, match="destroyed"):
            receiver.receive(_round())

    def test_a_failed_destroy_raises_and_is_not_retried(
        self,
        fake_mx,
    ):
        receiver = create_receiver(_context())
        session = FakeSession.instances[0]

        def broken_close():
            raise RuntimeError("close failed")

        session.close = broken_close
        with pytest.raises(RuntimeError, match="close failed"):
            receiver.destroy()
        # The channel and rendezvous are still released after the failure.
        assert fake_mx.channels[0].closed == 1
        receiver.destroy()
        assert fake_mx.channels[0].closed == 1

    def test_a_failed_prepare_releases_the_connection(
        self,
        fake_mx,
    ):
        fake_mx.prepare_errors.append(RuntimeError("join refused"))
        with pytest.raises(RuntimeError, match="join refused"):
            create_receiver(_context())
        assert fake_mx.channels[0].closed == 1
        assert FakeSession.instances[0].closed == 1


@pytest.mark.usefixtures("fake_buffers")
class TestFailClosed:
    @pytest.fixture(autouse=True)
    def _fake(self, fake_mx):
        self.fake_mx = fake_mx

    def _refused(self, match, **overrides):
        with pytest.raises((ValueError, RuntimeError), match=match):
            create_receiver(_context(**overrides))
        # Nothing was connected or left behind.
        assert FakeSession.instances == []
        assert self.fake_mx.channels == []

    def test_a_missing_manifest_is_refused(self):
        self._refused("object", init_payload=None)

    def test_a_foreign_manifest_schema_is_refused(self):
        manifest = _manifest()
        manifest["schema"] = "miles.nccl_m2n/v1"
        self._refused("schema", init_payload=manifest)

    def test_a_default_receiver_manifest_is_refused(self):
        self._refused("schema", init_payload={"schema_version": 1, "entries": []})

    def test_manifest_extra_or_missing_fields_are_refused(self):
        extra = {**_manifest(), "endpoint": "other:1"}
        self._refused("unexpected fields", init_payload=extra)
        missing = _manifest()
        del missing["topology"]
        self._refused("topology", init_payload=missing)

    def test_a_non_object_manifest_is_refused(self):
        self._refused("object", init_payload=["plan"])

    def test_world_size_must_match_the_generator_slots(self):
        self._refused("world_size", world_size=4)

    def test_a_rank_beyond_the_generator_slots_is_refused(self):
        self._refused("does not contain generator index", rank_offset=2)

    @pytest.mark.parametrize("offset", [-1, True, "0", 0.0])
    def test_a_bad_rank_offset_is_refused(self, offset):
        self._refused("rank_offset", rank_offset=offset)

    @pytest.mark.parametrize(
        "overrides",
        [
            {"tp_rank": None, "tp_size": 2},
            {"tp_rank": "0", "tp_size": 2},
            {"tp_rank": True, "tp_size": 2},
            {"tp_rank": 2, "tp_size": 2},
            {"tp_rank": -1, "tp_size": 2},
        ],
    )
    def test_a_bad_engine_tp_coordinate_is_refused(self, overrides):
        self._refused("tp_", **overrides)

    @pytest.mark.parametrize(
        ("host", "port"),
        [("", 1), ("grpc://mx", 1), (None, 1), ("mx", 0), ("mx", 70000), ("mx", True)],
    )
    def test_a_bad_server_address_is_refused(self, host, port):
        self._refused("master_", master_address=host, master_port=port)


class TestManifestAndEndpoint:
    def test_the_manifest_round_trips(self):
        plan, topology = manifest_from_wire(_manifest())
        assert plan == _plan()
        assert topology == _topology()
        assert _manifest()["schema"] == MANIFEST_SCHEMA

    @pytest.mark.parametrize(
        ("host", "port", "expected"),
        [
            ("mx-server", 8001, "mx-server:8001"),
            ("10.0.0.5", 50051, "10.0.0.5:50051"),
            ("::1", 8001, "[::1]:8001"),
            ("[::1]", 8001, "[::1]:8001"),
        ],
    )
    def test_the_server_endpoint_comes_from_the_request(self, host, port, expected):
        assert mx_server_endpoint(host, port) == expected


# --- SGLang's own classes -------------------------------------------------------


def _sglang_weight_updater():
    path = os.environ.get("MX_TEST_SGLANG_PYTHON")
    if not path:
        return None
    if path not in sys.path:
        sys.path.insert(0, path)
    try:
        from sglang.srt.model_executor.model_runner_components import (
            weight_updater,
        )
        from sglang.srt.weight_sync import external_receiver
    except Exception:
        return None
    if not hasattr(external_receiver, "WeightUpdateReceiverContext"):
        return None
    return weight_updater


_SGLANG = _sglang_weight_updater()
_needs_sglang = pytest.mark.skipif(
    _SGLANG is None,
    reason=(
        "set MX_TEST_SGLANG_PYTHON to an SGLang python/ tree with the "
        "allowlisted external weight-update receiver hook (SG-1)"
    ),
)


@_needs_sglang
class TestAgainstSglang:
    def test_the_real_context_is_what_the_factory_consumes(
        self, fake_mx, torch_device_buffers
    ):
        from sglang.srt.weight_sync.external_receiver import (
            WeightUpdateReceiverContext,
        )

        context = WeightUpdateReceiverContext(
            model=FakeModel(),
            device=torch.device("cuda:0"),
            tp_rank=1,
            tp_size=2,
            group_name="mx-m2n-g",
            master_address="mx-server",
            master_port=8001,
            world_size=2,
            rank_offset=0,
            init_payload=_manifest(),
        )
        receiver = create_receiver(context)
        # The external-receiver protocol: receive(payload), destroy().
        receiver.receive(_round())
        receiver.destroy()
        assert FakeSession.instances[0].rounds == [("3", "op-3")]

    @contextmanager
    def _updater(self, allowlist):
        wu = _SGLANG
        updater = object.__new__(wu.WeightUpdater)
        for name, value in {
            "tp_rank": 1,
            "device": "cuda:0",
            "get_model": lambda: FakeModel(),
            "weight_update_receivers": list(allowlist),
            "_model_update_group": {},
            "_external_receivers": {},
        }.items():
            object.__setattr__(updater, name, value)
        with (
            patch("torch.distributed.is_initialized", return_value=True),
            patch.object(wu, "get_parallel", return_value=SimpleNamespace(tp_size=2)),
            # The in-place write guards read the runtime-context model
            # namespace; a real server publishes it — run with the weight
            # cache off, as SG-1's own external-receiver tests do.
            patch.object(
                wu, "get_model", return_value=SimpleNamespace(weight_cache_mode="off")
            ),
            patch.object(wu, "init_custom_process_group") as init_pg,
            patch("torch.distributed.destroy_process_group") as destroy_pg,
        ):
            yield updater, init_pg, destroy_pg

    def _init(self, updater, **kwargs):
        return updater.init_weights_update_group(
            "mx-server", 8001, 0, 2, "mx-m2n-g", "nccl", **kwargs
        )

    def test_the_allowlisted_factory_runs_rounds_and_is_destroyed_by_sglang(
        self, fake_mx, torch_device_buffers
    ):
        with self._updater((RECEIVER_PATH,)) as (updater, init_pg, destroy_pg):
            success, message = self._init(
                updater, receiver=RECEIVER_PATH, receiver_init_payload=_manifest()
            )
            assert success, message
            receiver = updater._external_receivers["mx-m2n-g"]
            assert isinstance(receiver, ModelExpressM2NReceiver)
            updater.receive_weights_from_distributed(
                names=[],
                dtypes=[],
                shapes=[],
                group_name="mx-m2n-g",
                receiver_payload=_round("5", "op-5"),
            )
            assert FakeSession.instances[0].rounds == [("5", "op-5")]
            # No torch process group exists for a ModelExpress group.
            init_pg.assert_not_called()
            assert updater.destroy_weights_update_group("mx-m2n-g")[0]
            destroy_pg.assert_not_called()
        assert FakeSession.instances[0].closed == 1
        assert updater._external_receivers == {}

    def test_an_unlisted_factory_is_refused_by_sglang(self, fake_mx):
        with self._updater(()) as (updater, _, _):
            success, message = self._init(
                updater, receiver=RECEIVER_PATH, receiver_init_payload=_manifest()
            )
        assert not success and "--weight-update-receivers" in message
        assert FakeSession.instances == []

    def test_a_refused_manifest_surfaces_as_a_failed_init(self, fake_mx):
        manifest = _manifest()
        manifest["schema"] = "other"
        with self._updater((RECEIVER_PATH,)) as (updater, _, _):
            success, message = self._init(
                updater, receiver=RECEIVER_PATH, receiver_init_payload=manifest
            )
        assert not success and "schema" in message
        assert updater._external_receivers == {}
