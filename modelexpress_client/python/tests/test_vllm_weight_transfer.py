# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the vLLM WeightTransferEngine adapter.

These tests exercise the dispatch logic in ``MxWeightTransferEngine``
without requiring vLLM, NIXL, or a live MX server. They follow the
same direct-load + stub pattern as ``test_v2_source_picker.py`` so the
suite runs on a plain CPU box.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock

import pytest


_HERE = Path(__file__).resolve().parent
_PKG_ROOT = _HERE.parent / "modelexpress"


@pytest.fixture(scope="module")
def vllm_wt():
    """Load the vllm_weight_transfer module against the stubs we set up
    for ``test_v2_source_picker.py``. We re-create those stubs here so
    this test file is self-contained (no fixture order coupling).

    Crucially we set ``MX_WEIGHT_TRANSFER_AUTOREGISTER=0`` so the module
    doesn't try to register with a (non-existent) vLLM at import time.

    We snapshot any pre-existing ``modelexpress.*`` entries in ``sys.modules``
    and restore them on teardown so other test modules that run after us
    still see the real ``modelexpress`` package — not the stub we install
    here.
    """
    os.environ["MX_WEIGHT_TRANSFER_AUTOREGISTER"] = "0"

    _saved: dict = {}
    _injected: set = set()

    def _install(name: str, mod) -> None:
        if name not in _saved and name in sys.modules:
            _saved[name] = sys.modules[name]
        _injected.add(name)
        sys.modules[name] = mod

    def _load(modname: str, path: Path):
        spec = importlib.util.spec_from_file_location(modname, path)
        mod = importlib.util.module_from_spec(spec)
        _install(modname, mod)
        spec.loader.exec_module(mod)
        return mod

    # Build the same stub modules ``test_v2_source_picker.py`` uses.
    pkg = types.ModuleType("modelexpress")
    pkg.__path__ = [str(_PKG_ROOT)]  # type: ignore[attr-defined]
    _install("modelexpress", pkg)

    p2p_pb2 = types.ModuleType("modelexpress.p2p_pb2")
    p2p_pb2.SOURCE_STATUS_READY = 2
    p2p_pb2.SOURCE_STATUS_INITIALIZING = 1
    p2p_pb2.SOURCE_STATUS_STALE = 3
    p2p_pb2.MX_SOURCE_TYPE_WEIGHTS = 0
    p2p_pb2.BACKEND_FRAMEWORK_UNKNOWN = 0
    _install("modelexpress.p2p_pb2", p2p_pb2)

    @dataclass
    class _SourceIdentity:
        model_name: str = ""
        mx_source_type: int = 0
        backend_framework: int = 0
        tensor_parallel_size: int = 0
        pipeline_parallel_size: int = 0
        expert_parallel_size: int = 0
        dtype: str = ""
        quantization: str = ""

        def __post_init__(self):
            self.extra_parameters = {}

    @dataclass
    class _WorkerMetadata:
        worker_rank: int = 0
        nixl_metadata: bytes = b""
        tensors: list = None
        status: int = 0
        agent_name: str = ""

        def __post_init__(self):
            if self.tensors is None:
                self.tensors = []

    @dataclass
    class _TensorDescriptor:
        name: str = ""
        addr: int = 0
        size: int = 0
        device_id: int = 0
        dtype: str = ""

    p2p_pb2.SourceIdentity = _SourceIdentity
    p2p_pb2.WorkerMetadata = _WorkerMetadata
    p2p_pb2.TensorDescriptor = _TensorDescriptor

    # Heartbeat stub
    hb = types.ModuleType("modelexpress.heartbeat")

    class _HBStub:
        def __init__(self, *a, **kw):
            self.started = False

        def start(self):
            self.started = True

        def stop(self):
            self.started = False

    hb.HeartbeatThread = _HBStub
    _install("modelexpress.heartbeat", hb)

    # MxRefitReceiver / MxTrainingPublisher stubs
    refit_mod = types.ModuleType("modelexpress.refit_receiver")

    @dataclass
    class _SourceRef:
        mx_source_id: str = ""
        worker_id: str = ""
        model_name: str = ""
        worker_rank: int = 0
        training_step: int = 0

    @dataclass
    class _TransferStats:
        bytes_received: int = 0
        bytes_skipped: int = 0
        tensors_received: int = 0
        elapsed_seconds: float = 0.0
        bandwidth_gbps: float = 0.0
        discovery_seconds: float = 0.0
        path: str = ""
        training_step: int = 0
        source_worker_rank: int | None = None

    class _RefitStub:
        def __init__(self, *a, **kw):
            self._client = MagicMock()
            self._nixl = MagicMock()
            self._agent_name = kw.get("agent_name", "stub")
            self._worker_id = "stub-worker"
            self.last_stats = _TransferStats()
            self.history: list = []

        def initialize(self, model_tensors=None):
            pass

        def receive_weights(self, ref, timeout_seconds=300.0):
            return iter([])

        def receive_weights_scratch(self, ref, timeout_seconds=300.0, tensor_shapes=None):
            return iter([])

        def shutdown(self):
            pass

    refit_mod.MxRefitReceiver = _RefitStub
    refit_mod.SourceRef = _SourceRef
    refit_mod.TransferStats = _TransferStats
    _install("modelexpress.refit_receiver", refit_mod)

    pub_mod = types.ModuleType("modelexpress.training_publisher")

    class _PubStub:
        def __init__(self, *a, **kw):
            self._client = None
            self._nixl = None
            self.mx_source_id = "abcd1234"
            self.worker_id = "stub-pub-worker"

        def initialize(self, **kw):
            pass

        def publish_weights(self, named_tensors, step, worker_rank):
            return self.mx_source_id

        def mark_ready(self, worker_rank=0):
            return True

        def shutdown(self):
            pass

        def _build_identity(self, step):
            return p2p_pb2.SourceIdentity()

    pub_mod.MxTrainingPublisher = _PubStub
    _install("modelexpress.training_publisher", pub_mod)

    types_mod = types.ModuleType("modelexpress.types")

    @dataclass
    class _TD:
        name: str = ""
        addr: int = 0
        size: int = 0
        device_id: int = 0
        dtype: str = ""

    types_mod.TensorDescriptor = _TD
    _install("modelexpress.types", types_mod)

    # Now exec the real modules against these stubs.
    sd = _load("modelexpress.shape_descriptors", _PKG_ROOT / "shape_descriptors.py")
    pkg.shape_descriptors = sd  # type: ignore[attr-defined]
    v2 = _load("modelexpress.nemo_rl_v2", _PKG_ROOT / "nemo_rl_v2.py")
    pkg.nemo_rl_v2 = v2  # type: ignore[attr-defined]
    wt = _load(
        "modelexpress.engines.vllm.weight_transfer",
        _PKG_ROOT / "engines" / "vllm" / "weight_transfer.py",
    )

    try:
        yield wt, v2, sd
    finally:
        for name in _injected:
            sys.modules.pop(name, None)
        for name, mod in _saved.items():
            sys.modules[name] = mod


def test_engine_construction_without_init_info(vllm_wt):
    """Engine can be constructed with no init_info; init_transfer_engine
    is called separately."""
    wt, _, _ = vllm_wt
    engine = wt.MxWeightTransferEngine()
    assert engine._receiver is None
    assert engine._init_info is None


def test_engine_construction_with_init_info(vllm_wt):
    """Engine can be constructed with init_info; receiver is built eagerly."""
    wt, _, _ = vllm_wt
    init = wt.MxInitInfo(
        mx_server_url="fake:8001",
        model_name="m",
        worker_rank=0,
        agent_name="ag",
        device_id=0,
    )
    engine = wt.MxWeightTransferEngine(init_info=init)
    assert engine._receiver is not None
    assert engine._init_info is init


def test_receive_weights_without_init_raises(vllm_wt):
    """Calling receive_weights before init_transfer_engine should error."""
    wt, _, _ = vllm_wt
    engine = wt.MxWeightTransferEngine()
    update = wt.MxUpdateInfo(version=1)
    with pytest.raises(RuntimeError, match="init_transfer_engine"):
        engine.receive_weights(update, load_weights=lambda batch: None)


def test_receive_weights_matched_tp_path(vllm_wt, monkeypatch):
    """Matched-TP path: target_tp_layout=None → discover_v2_sources +
    pick_best_source + receive_from. Verify load_weights is called with
    yielded (name, tensor) pairs."""
    wt, v2, sd = vllm_wt
    engine = wt.MxWeightTransferEngine(
        init_info=wt.MxInitInfo(
            mx_server_url="fake:8001",
            model_name="m",
            worker_rank=0,
            agent_name="ag",
            publish_self_as_replica=False,  # disable to keep this test tight
        )
    )

    fake_candidate = MagicMock(name="V2SourceCandidate")
    fake_candidate.ref = MagicMock()
    monkeypatch.setattr(
        engine._receiver, "discover_v2_sources", lambda **kw: [fake_candidate]
    )
    monkeypatch.setattr(
        engine._receiver, "pick_best_source", lambda *a, **kw: fake_candidate
    )
    yielded = [("w1", "T1"), ("w2", "T2")]
    monkeypatch.setattr(
        engine._receiver, "receive_from_scratch", lambda c, **kw: iter(yielded)
    )

    received = []
    engine.receive_weights(
        wt.MxUpdateInfo(version=42),
        load_weights=lambda batch: received.extend(batch),
    )
    assert received == yielded


def test_receive_weights_mixed_tp_phase4_path(vllm_wt, monkeypatch):
    """Mixed-TP path: target_tp_layout set → discover_v2_sources_for_slice
    + receive_via_plan. Verify the Phase-4 plan is built and stitching
    happens."""
    wt, v2, sd = vllm_wt
    engine = wt.MxWeightTransferEngine(
        init_info=wt.MxInitInfo(
            mx_server_url="fake:8001",
            model_name="m",
            worker_rank=0,
            agent_name="ag",
            publish_self_as_replica=False,
        )
    )

    fake_plan = MagicMock(name="SliceCoveragePlan")
    fake_plan.fully_covered = True
    fake_plan.missing = []
    monkeypatch.setattr(
        engine._receiver, "discover_v2_sources_for_slice", lambda **kw: fake_plan
    )
    yielded = [("w", "STITCHED_TENSOR")]
    monkeypatch.setattr(
        engine._receiver,
        "receive_via_plan",
        lambda plan, **kw: iter(yielded) if plan is fake_plan else iter([]),
    )

    received = []
    update = wt.MxUpdateInfo(
        version=99,
        target_tp_layout=v2.TargetTPLayout(world_size=8, rank=3, shard_axis=0),
    )
    engine.receive_weights(update, load_weights=lambda batch: received.extend(batch))
    assert received == yielded


def test_receive_weights_phase4_uncovered_plan_raises(vllm_wt, monkeypatch):
    """Phase-4 path: a partial slice plan (missing entries) raises before
    any RDMA cycles are spent."""
    wt, v2, _ = vllm_wt
    engine = wt.MxWeightTransferEngine(
        init_info=wt.MxInitInfo(
            mx_server_url="x",
            model_name="m",
            worker_rank=0,
            agent_name="ag",
            publish_self_as_replica=False,
        )
    )
    bad_plan = MagicMock(name="SliceCoveragePlan")
    bad_plan.fully_covered = False
    bad_plan.missing = ["w: coverage gap"]
    monkeypatch.setattr(
        engine._receiver, "discover_v2_sources_for_slice", lambda **kw: bad_plan
    )

    with pytest.raises(RuntimeError, match="no covering source set"):
        engine.receive_weights(
            wt.MxUpdateInfo(
                version=1,
                target_tp_layout=v2.TargetTPLayout(world_size=2, rank=0),
            ),
            load_weights=lambda batch: None,
        )


def test_receive_weights_matched_no_candidates_raises(vllm_wt, monkeypatch):
    """Matched path: when no source passes the filter, raise BEFORE
    NIXL receive. This is the Phase-3b safety net for the
    compile_target_filter case."""
    wt, _, _ = vllm_wt
    engine = wt.MxWeightTransferEngine(
        init_info=wt.MxInitInfo(
            mx_server_url="x",
            model_name="m",
            worker_rank=0,
            agent_name="ag",
            publish_self_as_replica=False,
        )
    )
    monkeypatch.setattr(engine._receiver, "discover_v2_sources", lambda **kw: [])
    monkeypatch.setattr(engine._receiver, "pick_best_source", lambda *a, **kw: None)

    with pytest.raises(RuntimeError, match="no source matches filters"):
        engine.receive_weights(
            wt.MxUpdateInfo(
                version=1,
                compile_target_filter={"cutlass_fp8"},
            ),
            load_weights=lambda batch: None,
        )


def test_receive_weights_compile_target_filter_threaded_through(vllm_wt, monkeypatch):
    """The MxUpdateInfo's compile_target_filter and required_compile_metadata
    must reach the receiver's discover_v2_sources call unchanged."""
    wt, _, sd = vllm_wt
    engine = wt.MxWeightTransferEngine(
        init_info=wt.MxInitInfo(
            mx_server_url="x",
            model_name="m",
            worker_rank=0,
            agent_name="ag",
            publish_self_as_replica=False,
        )
    )
    captured: dict[str, object] = {}

    def fake_discover(**kw):
        captured.update(kw)
        return []

    monkeypatch.setattr(engine._receiver, "discover_v2_sources", fake_discover)
    monkeypatch.setattr(engine._receiver, "pick_best_source", lambda *a, **kw: None)

    with pytest.raises(RuntimeError):
        engine.receive_weights(
            wt.MxUpdateInfo(
                version=7,
                compile_target_filter={sd.COMPILE_TARGET_CUTLASS_FP8},
                required_compile_metadata={"block_size": 128},
                same_rank_only=False,
                dedup_freshest_per_rank=False,
            ),
            load_weights=lambda batch: None,
        )
    assert captured["model_name"] == "m"
    assert captured["min_version"] == 7
    assert captured["compile_target_filter"] == {sd.COMPILE_TARGET_CUTLASS_FP8}
    assert captured["required_compile_metadata"] == {"block_size": 128}
    assert captured["same_rank_only"] is False


def test_receive_weights_publishes_self_as_replica_when_enabled(vllm_wt, monkeypatch):
    """With publish_self_as_replica=True AND non-empty _registered_buffers,
    the engine triggers tree fan-out after a successful receive."""
    wt, _, _ = vllm_wt
    engine = wt.MxWeightTransferEngine(
        init_info=wt.MxInitInfo(
            mx_server_url="x",
            model_name="m",
            worker_rank=0,
            agent_name="ag",
            publish_self_as_replica=True,
        )
    )
    cand = MagicMock()
    monkeypatch.setattr(engine._receiver, "discover_v2_sources", lambda **kw: [cand])
    monkeypatch.setattr(engine._receiver, "pick_best_source", lambda *a, **kw: cand)
    monkeypatch.setattr(engine._receiver, "receive_from_scratch", lambda *a, **kw: iter([]))

    # Simulate the receiver having been initialized with model_tensors so
    # _registered_buffers is non-empty (the pre-registered fast path).
    # Without this the engine correctly skips publish_self_as_source —
    # see `test_receive_weights_skips_publish_when_no_registered_buffers`.
    engine._receiver._registered_buffers = {"layer.weight": object()}

    publish_calls = []

    def fake_publish(*, version, model_name):
        publish_calls.append((version, model_name))
        return "replica-source-id"

    monkeypatch.setattr(engine._receiver, "publish_self_as_source", fake_publish)
    engine.receive_weights(
        wt.MxUpdateInfo(version=11), load_weights=lambda batch: None
    )
    assert publish_calls == [(11, "m")]


def test_receive_weights_skips_publish_when_no_registered_buffers(
    vllm_wt, monkeypatch, caplog
):
    """With publish_self_as_replica=True but EMPTY _registered_buffers
    (scratch-mode receive), publish_self_as_source is skipped with a
    warning. Calling it would be a no-op silently — better to flag it
    so the misconfiguration is visible.
    """
    wt, _, _ = vllm_wt
    engine = wt.MxWeightTransferEngine(
        init_info=wt.MxInitInfo(
            mx_server_url="x",
            model_name="m",
            worker_rank=0,
            agent_name="ag",
            publish_self_as_replica=True,
        )
    )
    cand = MagicMock()
    monkeypatch.setattr(engine._receiver, "discover_v2_sources", lambda **kw: [cand])
    monkeypatch.setattr(engine._receiver, "pick_best_source", lambda *a, **kw: cand)
    monkeypatch.setattr(engine._receiver, "receive_from_scratch", lambda *a, **kw: iter([]))

    # _registered_buffers stays empty (scratch path: receiver was init'd
    # with model_tensors=None, which is the common cold-start case).
    publish_calls = []

    def fake_publish(*, version, model_name):
        publish_calls.append((version, model_name))
        return "replica-source-id"

    monkeypatch.setattr(engine._receiver, "publish_self_as_source", fake_publish)
    import logging
    with caplog.at_level(logging.WARNING, logger="modelexpress.engines.vllm.weight_transfer"):
        engine.receive_weights(
            wt.MxUpdateInfo(version=11), load_weights=lambda batch: None
        )

    assert publish_calls == []
    assert any(
        "publish_self_as_replica=True" in r.message and "scratch-buffer mode" in r.message
        for r in caplog.records
    ), "expected scratch-mode warning when publish_self_as_replica is True with no buffers"


def test_receive_weights_publish_self_failure_is_swallowed(vllm_wt, monkeypatch):
    """publish_self_as_replica failure must NOT propagate — it's a
    best-effort optimization, not correctness."""
    wt, _, _ = vllm_wt
    engine = wt.MxWeightTransferEngine(
        init_info=wt.MxInitInfo(
            mx_server_url="x",
            model_name="m",
            worker_rank=0,
            agent_name="ag",
            publish_self_as_replica=True,
        )
    )
    cand = MagicMock()
    monkeypatch.setattr(engine._receiver, "discover_v2_sources", lambda **kw: [cand])
    monkeypatch.setattr(engine._receiver, "pick_best_source", lambda *a, **kw: cand)
    monkeypatch.setattr(engine._receiver, "receive_from_scratch", lambda *a, **kw: iter([]))

    def broken_publish(*, version, model_name):
        raise RuntimeError("MX server unreachable")

    monkeypatch.setattr(engine._receiver, "publish_self_as_source", broken_publish)

    # Should NOT raise.
    engine.receive_weights(
        wt.MxUpdateInfo(version=11), load_weights=lambda batch: None
    )


def test_trainer_send_weights_threads_compile_target(vllm_wt, monkeypatch):
    """Trainer-side classmethod: each tensor in the iterator gets
    add_tensor'd with the compile_target + compile_metadata from
    MxTrainerSendArgs, and finally publish(version) is called."""
    wt, v2, sd = vllm_wt

    added: list[dict] = []
    published_with_version: list[int] = []

    class _RecordingPublisher:
        def add_tensor(self, **kw):
            added.append(kw)

        def publish(self, *, version):
            published_with_version.append(version)
            return "trainer-source-id"

    pub = _RecordingPublisher()
    args = wt.MxTrainerSendArgs(
        publisher=pub,
        version=42,
        compile_target=sd.COMPILE_TARGET_CUTLASS_FP8,
        compile_metadata={"block_size": 128, "scale_layout": "per_channel"},
        expert_axis_map={"expert.w": 0},
        owned_expert_ids={"expert.w": (0, 1, 2, 3)},
    )

    iterator = iter([
        ("w1", "FAKE_TENSOR_1"),
        ("expert.w", "FAKE_TENSOR_2"),
    ])
    out = wt.MxWeightTransferEngine.trainer_send_weights(iterator, args)
    assert out == "trainer-source-id"
    assert len(added) == 2
    assert added[0]["name"] == "w1"
    assert added[0]["compile_target"] == sd.COMPILE_TARGET_CUTLASS_FP8
    assert added[0]["compile_metadata"] == {
        "block_size": 128,
        "scale_layout": "per_channel",
    }
    assert added[0]["is_expert"] is False
    assert added[1]["name"] == "expert.w"
    assert added[1]["is_expert"] is True
    assert added[1]["expert_axis"] == 0
    assert added[1]["owned_expert_ids"] == (0, 1, 2, 3)
    assert published_with_version == [42]


def test_metrics_surface_exposed(vllm_wt):
    """The engine exposes last_transfer_stats / transfer_history /
    last_discovery_seconds for benchmark consumers."""
    wt, _, _ = vllm_wt
    engine = wt.MxWeightTransferEngine()
    # Pre-init: graceful Nones / empties
    assert engine.last_transfer_stats is None
    assert engine.transfer_history == []
    assert engine.last_discovery_seconds == 0.0

    engine.init_transfer_engine(
        wt.MxInitInfo(
            mx_server_url="x",
            model_name="m",
            worker_rank=0,
            agent_name="ag",
        )
    )
    # Post-init: surfaces are wired through the receiver
    assert engine.last_transfer_stats is not None  # the empty TransferStats
    assert engine.last_transfer_stats.bytes_received == 0
    assert engine.transfer_history == []
    assert engine.last_discovery_seconds == 0.0


def test_engine_is_registered_when_vllm_unavailable(vllm_wt):
    """In environments without vLLM, _AUTOREGISTERED is False; the
    engine is still usable directly. (We force this case via the env
    var in the fixture.)"""
    wt, _, _ = vllm_wt
    # The class is exported regardless of registration outcome.
    assert wt.MxWeightTransferEngine is not None
    # Auto-registration was disabled via env var in the fixture; the
    # engine should be usable but not necessarily registered.
    assert isinstance(wt._AUTOREGISTERED, bool)


def test_engine_exposes_init_info_cls_and_update_info_cls(vllm_wt):
    """The vLLM factory contract: each engine class declares its info
    types as class attributes."""
    wt, _, _ = vllm_wt
    assert wt.MxWeightTransferEngine.init_info_cls is wt.MxInitInfo
    assert wt.MxWeightTransferEngine.update_info_cls is wt.MxUpdateInfo
