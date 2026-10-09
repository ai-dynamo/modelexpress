# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
from modelexpress_rl.inference import runtime
from modelexpress_rl.inference.adapter import (
    TrainerSourceShard,
)
from modelexpress_rl.inference.plan import (
    PreparedStreamingTensors,
    TrainerSourceSnapshot,
    StreamingSettings,
)
from modelexpress_rl.train import WeightPayloadFormat


def setup_method(monkeypatch, *, staging_device="cuda", staging_buffers=1) -> tuple:
    events = []
    arena = torch.ones(2, 2)

    class Transfer:
        fail_prepare = False

        def __init__(self, **kwargs) -> None:
            self.arena = arena
            self.settings = kwargs["streaming"]

        def prepare_streaming(self, **kwargs) -> SimpleNamespace:
            events.append(("prepare", kwargs))
            if self.fail_prepare:
                raise RuntimeError("preparation failed")
            return SimpleNamespace(
                metrics={},
                batches=[SimpleNamespace(layouts=({"weight": None},))],
            )

        def iter_bounded(self, prepared, metrics):
            events.append(("read", self.arena))
            yield {"weight": self.arena}
            events.append(("drained", None))

        def reset_workspace(self):
            events.append(("reset", None))

        def close(self):
            events.append(("close", None))
            self.arena = None

    monkeypatch.setattr(runtime, "_NixlStagedTransfer", Transfer)
    method = runtime._create_load_time_tensor_method(
        capability=SimpleNamespace(device_id=0, device="cpu", capture_layout=None),
        worker_id="receiver",
        streaming=StreamingSettings(512, staging_device, staging_buffers),
    )
    source = TrainerSourceSnapshot(
        mesh_id="mesh",
        mesh_generation=1,
        shards=(
            TrainerSourceShard(
                source_slot_id="slot",
                worker_id="trainer",
                manifest_digest="digest",
                manifest_endpoint="trainer:19000",
                manifest=b"manifest",
                structural_digest="structure",
            ),
        ),
    )
    return method, source, events, arena


def prepare(method, source) -> PreparedStreamingTensors:
    return method.prepare_streaming(
        version=SimpleNamespace(version_id="v:1"),
        source=source,
    )


@pytest.mark.parametrize(
    "staging_device,staging_buffers", [("cuda", 1), ("cpu", 1), ("cuda", 2), ("cpu", 2)]
)
def test_streaming_prepare_is_lazy_and_keeps_receive_options(
    monkeypatch, staging_device, staging_buffers
) -> None:
    method, source, events, arena = setup_method(
        monkeypatch, staging_device=staging_device, staging_buffers=staging_buffers
    )
    previous_owner = None
    for value in (5, 7, 11):
        arena.fill_(value)
        events.clear()
        prepared = prepare(method, source)
        assert type(prepared) is PreparedStreamingTensors
        assert prepared is method._active_streamed
        assert prepared.ownership is not previous_owner
        assert not prepared.ownership.release_blocked
        assert [name for name, _ in events] == ["prepare"]
        assert prepared.parameter_names == frozenset({"weight"})
        with method.installation_context(prepared):
            batches = list(prepared.batches())
            assert torch.equal(batches[0]["weight"], torch.full((2, 2), value))
        method.release(prepared)
        assert method._active_streamed is None
        assert [name for name, _ in events] == ["prepare", "read", "drained"]
        previous_owner = prepared.ownership


@pytest.mark.parametrize(
    "failure", ["drain_failed", "close_failed", "source_failed", "in_progress"]
)
def test_uncertain_stream_retains_iterator_and_receive_workspace(monkeypatch, failure):
    method, source, events, arena = setup_method(monkeypatch)
    prepared = prepare(method, source)
    iterator = prepared.batches()
    prepared.ownership.iterator = iterator
    assert next(iterator)["weight"] is arena
    if failure != "in_progress":
        setattr(prepared.ownership, failure, True)
    for cleanup in (lambda: method.release(prepared), method.close):
        with pytest.raises(RuntimeError, match="reset the process"):
            cleanup()
        assert method._active_streamed is prepared
        assert prepared.ownership.iterator is iterator
        assert method._transfer.arena is arena
    with pytest.raises(RuntimeError, match="active update"):
        prepare(method, source)
    with (
        pytest.raises(RuntimeError, match="reset the process"),
        method.installation_context(prepared),
    ):
        pytest.fail("uncertain streaming source entered installation")
    assert [name for name, _ in events] == ["prepare", "read"]
    iterator.close()


def test_foreign_or_released_stream_cannot_enter_or_release_active_source(monkeypatch):
    method, source, events, _ = setup_method(monkeypatch)
    prepared = prepare(method, source)
    foreign = PreparedStreamingTensors(prepared.batches, prepared.parameter_names, {})
    with (
        pytest.raises(RuntimeError, match="active source"),
        method.installation_context(foreign),
    ):
        pytest.fail("foreign artifact entered installation")
    with pytest.raises(RuntimeError, match="no longer active"):
        method.release(foreign)
    assert method._active_streamed is prepared
    assert [name for name, _ in events] == ["prepare"]
    method.release(prepared)
    with (
        pytest.raises(RuntimeError, match="active source"),
        method.installation_context(prepared),
    ):
        pytest.fail("released artifact entered installation")
    method.close()
    assert method._transfer.arena is None


def test_failed_unread_preparation_resets_workspace_and_allows_retry(
    monkeypatch,
) -> None:
    method, source, events, _ = setup_method(monkeypatch)
    method._transfer.fail_prepare = True
    with pytest.raises(RuntimeError, match="preparation failed"):
        prepare(method, source)
    assert method._active_streamed is None
    assert [name for name, _ in events] == ["prepare", "reset"]
    method._transfer.fail_prepare = False
    prepared = prepare(method, source)
    assert method._active_streamed is prepared
    method.release(prepared)
    assert [name for name, _ in events] == ["prepare", "reset", "prepare"]
