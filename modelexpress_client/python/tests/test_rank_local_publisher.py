# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for :mod:`modelexpress.rank_local_publisher` and the verl integration.

Uses a fake :class:`MxTrainingPublisher` (no NIXL / no GRPC) and synthetic
tensors so the no-allgather contract can be exercised on a CPU-only host.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest
import torch

from modelexpress.integrations.verl_checkpoint_engine import (
    VerlMxCheckpointEngine,
    VerlMxRolloutLoader,
    VerlPublishConfig,
)
from modelexpress.rank_local_publisher import (
    PlacementDescriptor,
    RankLocalPublisher,
)
from modelexpress.rl_slice_descriptors import SliceOwnership


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


@dataclass
class _FakePublisher:
    """In-memory stand-in for :class:`MxTrainingPublisher`.

    Records every :meth:`publish_weights` call so tests can assert that
    the wrapper passed the right tensors through (and only this rank's
    local shards — no gather).
    """
    seen: list[tuple[dict[str, Any], int, int]] = field(default_factory=list)
    next_source_id: str = "src-0"
    ready_called: bool = False

    def publish_weights(
        self,
        named_tensors: dict[str, Any],
        step: int,
        worker_rank: int,
    ) -> str:
        # Copy keys (not tensors) so tests can mutate the input freely.
        self.seen.append((dict(named_tensors), step, worker_rank))
        return self.next_source_id

    def mark_ready(self, worker_rank: int) -> bool:
        self.ready_called = True
        return True


@dataclass
class _FakeReceiver:
    """In-memory stand-in for :class:`MxRefitReceiver`.

    Records every :meth:`receive_segment` call so tests can verify that
    the loader issued the right NIXL reads (one per :class:`SegmentPlan`).
    """
    segments_received: list[dict[str, Any]] = field(default_factory=list)

    def receive_segment(
        self,
        *,
        source_id: str,
        source_rank: int,
        source_addr: int,
        source_offset: int,
        target_addr: int,
        byte_count: int,
    ) -> None:
        self.segments_received.append({
            "source_id": source_id,
            "source_rank": source_rank,
            "source_addr": source_addr,
            "source_offset": source_offset,
            "target_addr": target_addr,
            "byte_count": byte_count,
        })


# ---------------------------------------------------------------------------
# RankLocalPublisher: explicit-shard path
# ---------------------------------------------------------------------------


def test_add_explicit_shard_records_ownership():
    pub = _FakePublisher()
    rlp = RankLocalPublisher(pub, model_name="m", worker_rank=2)

    local = torch.zeros(1024, 4096, dtype=torch.bfloat16)
    rlp.add_explicit_shard(
        "model.layers.0.q_proj.weight",
        local,
        PlacementDescriptor(
            placement_kind="SHARD",
            global_shape=(4096, 4096),
            shard_axis=0,
            local_shard_range=(2048, 3072),  # rank 2 of FSDP-4
        ),
    )

    source_id = rlp.publish(step=7)
    assert source_id == "src-0"
    assert len(pub.seen) == 1
    named, step, rank = pub.seen[0]
    assert step == 7 and rank == 2
    # Crucial: only the local shard is passed through — no gather happened.
    assert named["model.layers.0.q_proj.weight"].shape == (1024, 4096)

    ownerships = rlp.drain_slice_ownerships()
    assert len(ownerships) == 1
    own = ownerships[0]
    assert own.worker_rank == 2
    assert own.local_shard_range == (2048, 3072)
    assert own.global_shape == (4096, 4096)
    assert own.byte_size == 1024 * 4096 * 2


def test_replicate_explicit_shard():
    pub = _FakePublisher()
    rlp = RankLocalPublisher(pub, model_name="m", worker_rank=0)
    norm = torch.zeros(4096, dtype=torch.bfloat16)
    rlp.add_explicit_shard(
        "model.norm.weight",
        norm,
        PlacementDescriptor(placement_kind="REPLICATE", global_shape=(4096,)),
    )
    rlp.publish(step=0)
    own = rlp.drain_slice_ownerships()[0]
    assert own.placement_kind == "REPLICATE"
    assert own.shard_axis is None


def test_publish_with_nothing_recorded_raises():
    pub = _FakePublisher()
    rlp = RankLocalPublisher(pub, model_name="m", worker_rank=0)
    with pytest.raises(RuntimeError, match="nothing recorded"):
        rlp.publish(step=0)


def test_drain_clears_pending():
    pub = _FakePublisher()
    rlp = RankLocalPublisher(pub, model_name="m", worker_rank=0)
    rlp.add_explicit_shard(
        "t", torch.zeros(8),
        PlacementDescriptor(placement_kind="REPLICATE", global_shape=(8,)),
    )
    rlp.publish(step=0)
    assert rlp.drain_slice_ownerships()  # non-empty
    assert rlp.drain_slice_ownerships() == []  # second drain is empty


def test_compile_target_metadata_propagates():
    pub = _FakePublisher()
    rlp = RankLocalPublisher(pub, model_name="m", worker_rank=0)
    rlp.add_explicit_shard(
        # torch.empty (vs torch.zeros) — the FP8 fill_cpu kernel doesn't exist
        # in stock CPU torch builds, and this test only inspects the shape +
        # dtype metadata that lands in SliceOwnership, not the tensor contents.
        "w", torch.empty(1024, 1024, dtype=torch.float8_e4m3fn),
        PlacementDescriptor(
            placement_kind="SHARD", global_shape=(4096, 1024),
            shard_axis=0, local_shard_range=(0, 1024),
        ),
        compile_target="cutlass_fp8",
        compile_metadata={"block_size": 128, "scale_layout": "per_channel"},
    )
    rlp.publish(step=0)
    own = rlp.drain_slice_ownerships()[0]
    assert own.compile_target == "cutlass_fp8"
    assert own.compile_metadata == {"block_size": 128, "scale_layout": "per_channel"}


# ---------------------------------------------------------------------------
# VerlMxCheckpointEngine: trainer-side end-to-end
# ---------------------------------------------------------------------------


def test_checkpoint_engine_publishes_plain_tensor_as_replicate():
    pub = _FakePublisher()
    engine = VerlMxCheckpointEngine(pub, worker_rank=0)
    state_dict = {"model.norm.weight": torch.zeros(4096, dtype=torch.bfloat16)}
    sid = engine.publish_weights(state_dict, VerlPublishConfig(model_name="m", step=1))
    assert sid == "src-0"
    assert engine.last_source_id == "src-0"
    assert len(engine.last_ownerships) == 1
    assert engine.last_ownerships[0].placement_kind == "REPLICATE"


def test_checkpoint_engine_with_explicit_placement_override():
    pub = _FakePublisher()
    engine = VerlMxCheckpointEngine(pub, worker_rank=1)
    local = torch.zeros(1024, 4096, dtype=torch.bfloat16)
    state_dict = {"model.layers.0.q_proj.weight": local}
    overrides = {
        "model.layers.0.q_proj.weight": PlacementDescriptor(
            placement_kind="SHARD", global_shape=(4096, 4096),
            shard_axis=0, local_shard_range=(1024, 2048),
        ),
    }
    engine.publish_weights(
        state_dict,
        VerlPublishConfig(model_name="m", step=3),
        placement_overrides=overrides,
    )
    own = engine.last_ownerships[0]
    assert own.worker_rank == 1
    assert own.local_shard_range == (1024, 2048)


def test_checkpoint_engine_notifies_rollout_actor():
    """When a rollout_actor handle is provided, publish should call notify_new_source."""
    pub = _FakePublisher()
    notifications: list[tuple[str, int, list]] = []

    class _FakeRolloutHandle:
        class _RemoteMethod:
            def __init__(self, outer):
                self.outer = outer

            def remote(self, source_id, step, ownerships):
                self.outer.notifications.append((source_id, step, list(ownerships)))

        def __init__(self):
            self.notifications = notifications
            self.notify_new_source = self._RemoteMethod(self)

    actor = _FakeRolloutHandle()
    engine = VerlMxCheckpointEngine(pub, worker_rank=0, rollout_actor=actor)
    engine.publish_weights(
        {"x": torch.zeros(8)},
        VerlPublishConfig(model_name="m", step=42),
    )
    assert len(notifications) == 1
    sid, step, owns = notifications[0]
    assert sid == "src-0" and step == 42 and len(owns) == 1


def test_checkpoint_engine_ack_timeout():
    """With wait_for_inference_ack=True and no ack, publish should time out."""
    pub = _FakePublisher()
    engine = VerlMxCheckpointEngine(pub, worker_rank=0)
    with pytest.raises(TimeoutError, match="no ack"):
        engine.publish_weights(
            {"x": torch.zeros(8)},
            VerlPublishConfig(
                model_name="m", step=1,
                wait_for_inference_ack=True,
                ack_timeout_s=0.05,
            ),
        )


def test_checkpoint_engine_ack_arrives_in_time(monkeypatch):
    """Ack from rollout side should unblock publish."""
    import threading
    pub = _FakePublisher()
    engine = VerlMxCheckpointEngine(pub, worker_rank=0)

    def _ack_after_delay():
        import time
        time.sleep(0.1)
        engine.record_ack(step=1)

    threading.Thread(target=_ack_after_delay, daemon=True).start()
    engine.publish_weights(
        {"x": torch.zeros(8)},
        VerlPublishConfig(
            model_name="m", step=1,
            wait_for_inference_ack=True,
            ack_timeout_s=2.0,
        ),
    )


# ---------------------------------------------------------------------------
# VerlMxRolloutLoader: inference-side end-to-end
# ---------------------------------------------------------------------------


def test_rollout_loader_executes_plan():
    receiver = _FakeReceiver()
    loader = VerlMxRolloutLoader(
        receiver, receiver_rank=0, receiver_tp_size=2,
    )

    # Trainer FSDP-2 publishes two shards covering the full tensor.
    ownerships = [
        SliceOwnership(
            model_name="m",
            tensor_name="model.layers.0.q_proj.weight",
            global_shape=(4096, 4096),
            dtype="torch.bfloat16",
            placement_kind="SHARD",
            shard_axis=0,
            local_shard_range=(0, 2048),
            worker_rank=0,
            nixl_addr=0x10000,
            byte_size=2048 * 4096 * 2,
        ),
        SliceOwnership(
            model_name="m",
            tensor_name="model.layers.0.q_proj.weight",
            global_shape=(4096, 4096),
            dtype="torch.bfloat16",
            placement_kind="SHARD",
            shard_axis=0,
            local_shard_range=(2048, 4096),
            worker_rank=1,
            nixl_addr=0x20000,
            byte_size=2048 * 4096 * 2,
        ),
    ]
    loader.notify_new_source("src-0", step=1, ownerships=ownerships)

    # Inference TP-2; rank 0 holds the lower half — same as trainer rank 0.
    local_sd = {
        "model.layers.0.q_proj.weight": torch.zeros(2048, 4096, dtype=torch.bfloat16),
    }
    plan = loader.load_step(local_sd, timeout_s=1.0)
    assert plan.complete
    # Should issue exactly one segment (same-rank source covers full request).
    assert len(receiver.segments_received) == 1
    seg = receiver.segments_received[0]
    assert seg["source_id"] == "src-0"
    assert seg["source_rank"] == 0
    assert seg["byte_count"] == 2048 * 4096 * 2


def test_rollout_loader_falls_back_to_v1_receive_weights():
    """Receiver without receive_segment should drive the v1 receive_weights path."""
    calls: list[str] = []

    class _V1OnlyReceiver:
        def receive_weights(self, source_id):
            calls.append(source_id)
            return iter([])

    loader = VerlMxRolloutLoader(
        _V1OnlyReceiver(), receiver_rank=0, receiver_tp_size=1,
    )
    ownerships = [
        SliceOwnership(
            model_name="m",
            tensor_name="model.norm.weight",
            global_shape=(4096,),
            dtype="torch.bfloat16",
            placement_kind="REPLICATE",
            worker_rank=0, nixl_addr=0x1000, byte_size=8192,
        ),
    ]
    loader.notify_new_source("src-1", step=1, ownerships=ownerships)
    loader.load_step(
        {"model.norm.weight": torch.zeros(4096, dtype=torch.bfloat16)},
        timeout_s=1.0,
    )
    assert calls == ["src-1"]


def test_rollout_loader_timeout_when_no_source():
    receiver = _FakeReceiver()
    loader = VerlMxRolloutLoader(
        receiver, receiver_rank=0, receiver_tp_size=1,
    )
    with pytest.raises(TimeoutError, match="no source"):
        loader.load_step({"x": torch.zeros(8)}, timeout_s=0.01)


def test_rollout_loader_skips_stale_source():
    """A new source with step <= pending step should be dropped."""
    receiver = _FakeReceiver()
    loader = VerlMxRolloutLoader(
        receiver, receiver_rank=0, receiver_tp_size=1,
    )
    own_step10 = SliceOwnership(
        model_name="m", tensor_name="x",
        global_shape=(8,), dtype="torch.bfloat16",
        placement_kind="REPLICATE",
        worker_rank=0, byte_size=16,
    )
    loader.notify_new_source("src-10", step=10, ownerships=[own_step10])
    # Stale notification — should be silently dropped, not overwrite pending.
    loader.notify_new_source("src-5", step=5, ownerships=[own_step10])
    plan = loader.load_step(
        {"x": torch.zeros(8, dtype=torch.bfloat16)}, timeout_s=1.0,
    )
    assert plan.complete
    assert receiver.segments_received[0]["source_id"] == "src-10"


def test_rollout_loader_sends_ack_when_requested():
    receiver = _FakeReceiver()
    acks: list[int] = []

    class _FakeTrainerHandle:
        class _RemoteMethod:
            def __init__(self, outer):
                self.outer = outer
            def remote(self, step):
                self.outer.acks.append(step)
        def __init__(self):
            self.acks = acks
            self.record_ack = self._RemoteMethod(self)

    actor = _FakeTrainerHandle()
    loader = VerlMxRolloutLoader(
        receiver, receiver_rank=0, receiver_tp_size=1,
        trainer_actor=actor,
    )
    own = SliceOwnership(
        model_name="m", tensor_name="x",
        global_shape=(8,), dtype="torch.bfloat16",
        placement_kind="REPLICATE",
        worker_rank=0, byte_size=16,
    )
    loader.notify_new_source("src-0", step=99, ownerships=[own])
    loader.load_step(
        {"x": torch.zeros(8, dtype=torch.bfloat16)},
        timeout_s=1.0, send_ack=True,
    )
    assert acks == [99]
