# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.distributed.weight_transfer.nccl_engine import NCCLWeightTransferEngine

from modelexpress_rl.inference.engines.vllm import nccl_publication_engine as engine_mod
from modelexpress_rl.inference.engines.vllm.nccl_publication_engine import (
    ModelExpressNCCLWeightTransferEngine,
)


@pytest.fixture
def events(monkeypatch):
    events = []
    monkeypatch.setattr(
        NCCLWeightTransferEngine,
        "start_weight_update",
        lambda self: events.append("nccl_start"),
    )
    monkeypatch.setattr(
        NCCLWeightTransferEngine,
        "finish_weight_update",
        lambda self: events.append("nccl_finish"),
    )
    return events


class _Publication:
    def __init__(self, events, *, fail_unpublish=False, fail_publish=False):
        self.events = events
        self.fail_unpublish = fail_unpublish
        self.fail_publish = fail_publish

    def unpublish_runtime_tensors(self, drain_timeout):
        self.events.append(("unpublish", drain_timeout))
        if self.fail_unpublish:
            raise TimeoutError("active readers")

    def publish_runtime_tensors(self, version_id):
        self.events.append(("publish", version_id))
        if self.fail_publish:
            raise RuntimeError("server unavailable")


def _engine(monkeypatch, *, loader=None, load_format="auto", with_draft=False):
    monkeypatch.setenv("MX_RUNTIME_PUBLISH_REVISION", "fleet")
    monkeypatch.setattr(engine_mod, "get_model_loader", lambda device_id: loader)
    speculative_config = (
        SimpleNamespace(
            draft_model_config=SimpleNamespace(model="test/draft"),
            prefill_context_kv_only=False,
        )
        if with_draft
        else None
    )
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(),
        model_config=SimpleNamespace(model="test/model"),
        load_config=SimpleNamespace(load_format=load_format),
        speculative_config=speculative_config,
    )
    return ModelExpressNCCLWeightTransferEngine(
        SimpleNamespace(),
        vllm_config,
        torch.device("cuda", 0),
        torch.nn.Linear(2, 2),
    )


def test_update_drains_before_nccl_and_republishes_after(monkeypatch, events):
    engine = _engine(monkeypatch, loader=_Publication(events))
    drain_timeout = engine_mod.runtime_tensor_drain_timeout_seconds()

    engine.start_weight_update()
    engine.finish_weight_update()

    assert events == [
        ("unpublish", drain_timeout),
        "nccl_start",
        "nccl_finish",
        ("publish", "fleet"),
    ]


def test_drain_outlasts_reader_lease(monkeypatch):
    monkeypatch.setenv("MX_TRANSFER_TIMEOUT", "60")

    assert engine_mod.runtime_tensor_drain_timeout_seconds() == 2 * 120 + 30 + 30


def test_failed_drain_blocks_nccl_mutation(monkeypatch, events):
    engine = _engine(monkeypatch, loader=_Publication(events, fail_unpublish=True))

    with pytest.raises(TimeoutError, match="active readers"):
        engine.start_weight_update()

    assert "nccl_start" not in events


def test_failed_publish_does_not_fail_installed_update(monkeypatch, events, caplog):
    engine = _engine(monkeypatch, loader=_Publication(events, fail_publish=True))

    engine.finish_weight_update()

    assert events == ["nccl_finish", ("publish", "fleet")]
    assert "continuing without this worker as a P2P source" in caplog.text


def test_publication_waits_for_target_and_draft_updates(monkeypatch, events):
    engine = _engine(monkeypatch, loader=_Publication(events), with_draft=True)

    engine.start_weight_update()
    engine.finish_weight_update()

    engine.model = torch.nn.Linear(2, 2)
    engine.start_weight_update()
    engine.finish_weight_update()

    assert events == [
        ("unpublish", engine_mod.runtime_tensor_drain_timeout_seconds()),
        "nccl_start",
        "nccl_finish",
        "nccl_start",
        "nccl_finish",
        ("publish", "fleet"),
    ]


def test_publication_waits_for_draft_and_target_in_either_order(monkeypatch, events):
    engine = _engine(monkeypatch, loader=_Publication(events), with_draft=True)
    engine.model = torch.nn.Linear(2, 2)

    engine.start_weight_update()
    engine.finish_weight_update()
    engine.model = engine._default_model
    engine.start_weight_update()
    engine.finish_weight_update()

    assert events == [
        ("unpublish", engine_mod.runtime_tensor_drain_timeout_seconds()),
        "nccl_start",
        "nccl_finish",
        "nccl_start",
        "nccl_finish",
        ("publish", "fleet"),
    ]


def test_worker_without_publication_is_plain_nccl(monkeypatch, events):
    engine = _engine(monkeypatch)

    engine.start_weight_update()
    engine.finish_weight_update()

    assert events == ["nccl_start", "nccl_finish"]


def test_requires_publish_revision(monkeypatch):
    monkeypatch.setenv("MX_RUNTIME_PUBLISH_REVISION", " ")
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(),
        model_config=SimpleNamespace(model="test/model"),
        load_config=SimpleNamespace(load_format="auto"),
    )

    with pytest.raises(ValueError, match="MX_RUNTIME_PUBLISH_REVISION"):
        ModelExpressNCCLWeightTransferEngine(
            SimpleNamespace(), vllm_config, torch.device("cpu"), torch.nn.Linear(2, 2)
        )


def test_dummy_load_registers_and_publishes_tensors(monkeypatch, events):
    from modelexpress.engines.vllm import adapter

    ctx = SimpleNamespace(
        nixl_manager=object(),
        tensors={"weight": object()},
        identity=SimpleNamespace(revision=""),
    )
    register = MagicMock()
    monkeypatch.setattr(adapter, "build_vllm_load_context", lambda *args: ctx)
    monkeypatch.setattr(engine_mod, "register_tensors", register)
    monkeypatch.setattr(
        engine_mod,
        "drain_tensor_readers",
        lambda received, *, timeout: events.append(("drain", received, timeout)),
    )
    monkeypatch.setattr(
        engine_mod, "unpublish_metadata", lambda received: events.append(("unpublish", received))
    )
    monkeypatch.setattr(
        engine_mod, "publish_metadata", lambda received: events.append(("publish", received))
    )

    engine = _engine(monkeypatch, load_format="dummy")
    engine.start_weight_update()
    engine.finish_weight_update()

    register.assert_called_once_with(engine.model, ctx)
    assert events == [
        ("drain", ctx, engine_mod.runtime_tensor_drain_timeout_seconds()),
        ("unpublish", ctx),
        "nccl_start",
        "nccl_finish",
        ("publish", ctx),
    ]
    assert ctx.identity.revision == "fleet"


@pytest.mark.parametrize("failure", ["raises", "no_nixl"])
def test_dummy_registration_failure_leaves_plain_nccl(monkeypatch, events, failure):
    from modelexpress.engines.vllm import adapter

    ctx = SimpleNamespace(nixl_manager=None, tensors={})
    if failure == "raises":
        build = MagicMock(side_effect=RuntimeError("no NIXL"))
    else:
        build = MagicMock(return_value=ctx)
    monkeypatch.setattr(adapter, "build_vllm_load_context", build)
    monkeypatch.setattr(engine_mod, "register_tensors", MagicMock())

    engine = _engine(monkeypatch, load_format="dummy")
    engine.start_weight_update()
    engine.finish_weight_update()

    assert events == ["nccl_start", "nccl_finish"]
