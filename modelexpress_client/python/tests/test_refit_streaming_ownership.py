# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from uuid import uuid4

from tests.helpers.refit_replay import consume_stream

import pytest
import torch
from modelexpress_rl.inference import runtime
from modelexpress_rl.inference.adapter import (
    TrainerSourceShard,
)
from modelexpress_rl.inference.plan import (
    PreparedStreamingTensors,
    TrainerSourceSnapshot,
    ResolvedTrainerSource,
    StreamingSettings,
)
from modelexpress_rl.train import WeightPayloadFormat
from modelexpress_rl.refit.reshard.types import IncompleteRefit
from modelexpress_rl.version import MeshIdentity


def setup_method(monkeypatch, *, staging_device="cuda", staging_buffers=1) -> tuple:
    events = []
    arena = torch.ones(2, 2)

    class Transfer:
        bounded_staging = True
        fail_prepare = False

        def __init__(self, **kwargs):
            self.arena = arena
            self.settings = kwargs["streaming"]

        def prepare(self, **kwargs) -> SimpleNamespace:
            events.append(("prepare", kwargs))
            if self.fail_prepare:
                raise RuntimeError("preparation failed")
            return SimpleNamespace(
                metrics={},
                replay_id=uuid4(),
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
        capability=SimpleNamespace(
            device_id=0,
            device="cpu",
            capture_layout=lambda manifest: None,
            generator_fingerprint=None,
        ),
        worker_id="receiver",
        streaming=StreamingSettings(512, staging_device, staging_buffers),
    )
    source = TrainerSourceSnapshot(
        mesh_identity=MeshIdentity("mesh", 1),
        shards=(
            TrainerSourceShard(
                source_slot_id="slot",
                worker_id="trainer",
                stable_metadata_digest="digest",
                metadata_endpoint="trainer:19000",
                metadata=b"metadata",
            ),
        ),
    )
    return method, source, events, arena


def prepare(method, source) -> PreparedStreamingTensors:
    return method.prepare(
        version=SimpleNamespace(version_id="v:1"),
        source=ResolvedTrainerSource(source),
    )


@pytest.mark.parametrize(
    "staging_device,staging_buffers", [("cuda", 1), ("cpu", 1), ("cuda", 2), ("cpu", 2)]
)
def test_streaming_prepare_is_lazy_and_keeps_receive_options(
    monkeypatch, staging_device, staging_buffers
):
    method, source, events, arena = setup_method(
        monkeypatch, staging_device=staging_device, staging_buffers=staging_buffers
    )
    previous = None
    for value in (5, 7, 11):
        arena.fill_(value)
        events.clear()
        prepared = prepare(method, source)
        assert type(prepared) is PreparedStreamingTensors
        assert prepared is method._active
        assert prepared is not previous
        assert method.can_release(prepared)
        assert [name for name, _ in events] == ["prepare"]
        assert prepared.replay_id is not None
        with method.installation_context(prepared):

            def install(batch):
                assert torch.equal(batch["weight"], torch.full((2, 2), value))

            consume_stream(prepared, install)
        method.release(prepared)
        assert method._active is None
        assert [name for name, _ in events] == ["prepare", "read", "drained"]
        previous = prepared


@pytest.mark.parametrize("failure", ["drain", "close", "source", "in_progress"])
def test_uncertain_stream_retains_iterator_and_receive_workspace(monkeypatch, failure):
    method, source, events, arena = setup_method(monkeypatch)
    prepared = prepare(method, source)

    class Source:
        reads = 0

        def __iter__(self):
            return self

        def __next__(self):
            self.reads += 1
            events.append(("read", arena))
            if self.reads > 1:
                if failure == "source":
                    raise RuntimeError("source failed")
                raise StopIteration
            return {"weight": arena}

        def close(self):
            if failure == "close":
                raise RuntimeError("close failed")

    iterator = Source()
    prepared._batch_factory = lambda: iterator

    def drain():
        if failure == "drain":
            raise RuntimeError("drain failed")

    consumption = prepared.consume(drain)
    if failure == "in_progress":
        consumption.__enter__()
        prepared.apply(lambda batch: None)
    else:
        with pytest.raises(RuntimeError, match=f"{failure} failed"):
            with consumption:
                prepared.apply(lambda batch: None)
    assert not method.can_release(prepared)
    for cleanup in (lambda: method.release(prepared), method.close):
        with pytest.raises(RuntimeError, match="reset the process"):
            cleanup()
        assert method._active is prepared
        assert prepared._iterator is iterator
        assert method._transfer.arena is arena
    with pytest.raises(RuntimeError, match="active update"):
        prepare(method, source)
    with pytest.raises(RuntimeError, match="reset the process"):
        with method.installation_context(prepared):
            pytest.fail("uncertain stream entered installation")
    if failure == "in_progress":
        consumption.__exit__(None, None, None)


def test_foreign_or_released_stream_cannot_enter_or_release_active_source(monkeypatch):
    method, source, events, _ = setup_method(monkeypatch)
    prepared = prepare(method, source)
    foreign = PreparedStreamingTensors(lambda: iter(()), prepared.replay_id, {})
    with (
        pytest.raises(RuntimeError, match="active source"),
        method.installation_context(foreign),
    ):
        pytest.fail("foreign artifact entered installation")
    with pytest.raises(RuntimeError, match="no longer active"):
        method.release(foreign)
    assert method._active is prepared
    assert [name for name, _ in events] == ["prepare"]
    method.release(prepared)
    with (
        pytest.raises(RuntimeError, match="active source"),
        method.installation_context(prepared),
    ):
        pytest.fail("released artifact entered installation")
    method.close()
    assert method._transfer.arena is None


def test_failed_unread_preparation_preserves_workspace_and_allows_retry(
    monkeypatch,
) -> None:
    method, source, events, _ = setup_method(monkeypatch)
    method._transfer.fail_prepare = True
    with pytest.raises(RuntimeError, match="preparation failed"):
        prepare(method, source)
    assert method._active is None
    assert [name for name, _ in events] == ["prepare"]
    method._transfer.fail_prepare = False
    prepared = prepare(method, source)
    assert method._active is prepared
    method.release(prepared)
    assert [name for name, _ in events] == ["prepare", "prepare"]


@pytest.mark.parametrize("failure", [None, "consumer", "drain"])
def test_stream_owns_borrowed_batch_until_gpu_reads_finish(failure):
    import weakref

    events = []
    source_ref = None

    class Source:
        sent = False

        def __iter__(self):
            return self

        def __next__(self):
            nonlocal source_ref
            if self.sent:
                raise StopIteration
            self.sent = True
            value = torch.ones(2)
            source_ref = weakref.ref(value)
            weakref.finalize(value, events.append, "source_released")
            return value

        def close(self):
            events.append("close")

    prepared = PreparedStreamingTensors(Source, uuid4(), {})

    def install(value):
        events.append("install")
        assert source_ref() is value
        if failure == "consumer":
            raise RuntimeError("consumer failed")

    def drain():
        events.append("drain")
        assert source_ref() is not None or "source_released" in events
        if failure == "drain":
            raise RuntimeError("drain failed")

    if failure is None:
        with prepared.consume(drain):
            prepared.apply(install)
    else:
        with pytest.raises(RuntimeError, match=f"{failure} failed"):
            with prepared.consume(drain):
                prepared.apply(install)
    if failure == "drain":
        assert source_ref() is prepared._batch
        assert prepared.release_blocked
        assert events == ["install", "drain"]
    else:
        assert events.index("drain") < events.index("source_released")
        assert events.index("drain") < events.index("close")
        assert not prepared.release_blocked


def test_stream_rejects_consumption_outside_or_inside_an_active_context():
    opened = []

    def source():
        opened.append(True)
        return iter(())

    prepared = PreparedStreamingTensors(source, uuid4(), {})
    with pytest.raises(RuntimeError, match="installation context"):
        prepared.apply(lambda batch: None)
    assert opened == []
    with prepared.consume(lambda: None):
        with pytest.raises(IncompleteRefit, match="retains transfer resources"):
            with prepared.consume(lambda: None):
                pytest.fail("nested consumption must fail before opening the source")
    assert opened == []
