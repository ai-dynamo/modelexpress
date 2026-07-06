# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for v2 same-rank source filtering / freshest-per-rank dedup / tree
fan-out picker logic.

Mocks out the underlying NIXL / gRPC layer so we can drive the V2 receiver's
`discover_v2_sources` + `pick_best_source` purely from Python.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest


_HERE = Path(__file__).resolve().parent
_PKG_ROOT = _HERE.parent / "modelexpress"


def _load(modname: str, path: Path):
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def v2():
    """Load shape_descriptors and nemo_rl_v2 in isolation.

    nemo_rl_v2 normally imports MxRefitReceiver / MxTrainingPublisher /
    HeartbeatThread at module top. We mock those out before exec'ing the
    v2 module so its imports succeed without NIXL / gRPC.
    """
    # Pre-create stub modules for the dependencies that nemo_rl_v2 imports.
    pkg = types.ModuleType("modelexpress")
    pkg.__path__ = [str(_PKG_ROOT)]  # type: ignore[attr-defined]
    sys.modules["modelexpress"] = pkg

    # p2p_pb2 stub: just the constants & message classes used.
    p2p_pb2 = types.ModuleType("modelexpress.p2p_pb2")
    p2p_pb2.SOURCE_STATUS_READY = 2
    p2p_pb2.SOURCE_STATUS_INITIALIZING = 1
    p2p_pb2.SOURCE_STATUS_STALE = 3
    p2p_pb2.MX_SOURCE_TYPE_WEIGHTS = 0
    p2p_pb2.BACKEND_FRAMEWORK_UNKNOWN = 0
    sys.modules["modelexpress.p2p_pb2"] = p2p_pb2

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

    p2p_pb2.SourceIdentity = _SourceIdentity  # type: ignore[attr-defined]
    p2p_pb2.WorkerMetadata = _WorkerMetadata  # type: ignore[attr-defined]
    p2p_pb2.TensorDescriptor = _TensorDescriptor  # type: ignore[attr-defined]

    # Heartbeat stub: no-op start/stop.
    hb = types.ModuleType("modelexpress.heartbeat")

    class _HBStub:
        def __init__(self, *a, **kw):
            self.started = False

        def start(self):
            self.started = True

        def stop(self):
            self.started = False

    hb.HeartbeatThread = _HBStub
    sys.modules["modelexpress.heartbeat"] = hb

    # MxRefitReceiver / MxTrainingPublisher stubs.
    refit_mod = types.ModuleType("modelexpress.refit_receiver")

    @dataclass
    class _SourceRef:
        mx_source_id: str = ""
        worker_id: str = ""
        model_name: str = ""
        worker_rank: int = 0
        training_step: int = 0

    class _RefitStub:
        def __init__(self, *a, **kw):
            self._client = MagicMock()
            self._nixl = MagicMock()
            self._agent_name = kw.get("agent_name", "stub")
            self._worker_id = "stub-worker"

        def initialize(self, model_tensors=None):
            pass

        def receive_weights(self, ref, timeout_seconds=300.0):
            return iter([])

    refit_mod.MxRefitReceiver = _RefitStub
    refit_mod.SourceRef = _SourceRef
    sys.modules["modelexpress.refit_receiver"] = refit_mod

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
            ident = p2p_pb2.SourceIdentity()
            return ident

    pub_mod.MxTrainingPublisher = _PubStub
    sys.modules["modelexpress.training_publisher"] = pub_mod

    # types.TensorDescriptor stub
    types_mod = types.ModuleType("modelexpress.types")

    @dataclass
    class _TD:
        name: str = ""
        addr: int = 0
        size: int = 0
        device_id: int = 0
        dtype: str = ""

    types_mod.TensorDescriptor = _TD
    sys.modules["modelexpress.types"] = types_mod

    # Now exec shape_descriptors + nemo_rl_v2 against this module space.
    sd = _load("modelexpress.shape_descriptors", _PKG_ROOT / "shape_descriptors.py")
    pkg.shape_descriptors = sd  # type: ignore[attr-defined]
    v2 = _load("modelexpress.nemo_rl_v2", _PKG_ROOT / "nemo_rl_v2.py")
    return v2


def _fake_instance(model_name, mx_source_id, worker_id):
    return types.SimpleNamespace(
        model_name=model_name, mx_source_id=mx_source_id, worker_id=worker_id
    )


def _fake_meta(role, worker_rank, training_step, updated_at, registry_blob=""):
    """Build a fake get_metadata response with v2 metadata in extra_parameters."""

    @dataclass
    class _Meta:
        found: bool = True

        def __post_init__(self):
            from modelexpress.p2p_pb2 import SourceIdentity, WorkerMetadata

            self.identity = SourceIdentity(model_name="m")
            self.identity.extra_parameters.update(
                {
                    "mx_v2": "1",
                    "role": role,
                    "worker_rank": str(worker_rank),
                    "training_step": str(training_step),
                    "shape_registry": registry_blob,
                }
            )
            self.worker = WorkerMetadata()
            # we tack updated_at as an attribute since the proto stub doesn't
            # natively expose it
            self.worker.updated_at = updated_at

    return _Meta()


def test_same_rank_filter_dedup_freshest(v2):
    """Multiple sources at the same rank → keep only freshest by updated_at."""
    receiver = v2.MxV2RefitReceiver(
        agent_name="test-recv",
        device_id=0,
        mx_server_url="fake:8001",
        worker_rank=2,
    )
    receiver.initialize()
    # Inject 4 fake sources at MX:
    #  rank 0 trainer (irrelevant — different rank)
    #  rank 2 trainer, version 5, updated_at 100
    #  rank 2 trainer, version 5, updated_at 200 (FRESHER, should win)
    #  rank 2 inference_replica, version 5, updated_at 50 (excluded, replicas not preferred over trainer)
    response = MagicMock()
    response.instances = [
        _fake_instance("m", "s0", "w_r0"),
        _fake_instance("m", "s2", "w_r2_old"),
        _fake_instance("m", "s2", "w_r2_new"),
        _fake_instance("m", "s2", "w_r2_replica"),
    ]
    metas = {
        "w_r0": _fake_meta("trainer", 0, 5, 1000),
        "w_r2_old": _fake_meta("trainer", 2, 5, 100),
        "w_r2_new": _fake_meta("trainer", 2, 5, 200),
        "w_r2_replica": _fake_meta("inference_replica", 2, 5, 50),
    }
    receiver._receiver._client.list_sources.return_value = response
    receiver._receiver._client.get_metadata = lambda mx_source_id, worker_id: metas[
        worker_id
    ]

    candidates = receiver.discover_v2_sources(
        model_name="m", min_version=0, same_rank_only=True
    )
    # rank 0 is filtered out (same_rank_only=True; receiver is rank 2)
    assert all(c.worker_rank == 2 for c in candidates), candidates
    # All 3 remaining (2 trainers + 1 replica) are returned, but trainer comes first
    assert len(candidates) == 3
    assert candidates[0].role == "trainer"
    # Among trainers, freshest first
    assert candidates[0].ref.worker_id == "w_r2_new"
    # Replica comes last
    assert candidates[-1].role == "inference_replica"


def test_min_version_filter(v2):
    """Sources whose version is below min_version are excluded."""
    receiver = v2.MxV2RefitReceiver(
        agent_name="t", device_id=0, mx_server_url="x", worker_rank=0
    )
    receiver.initialize()
    response = MagicMock()
    response.instances = [
        _fake_instance("m", "s", "w_old"),
        _fake_instance("m", "s", "w_cur"),
        _fake_instance("m", "s", "w_new"),
    ]
    metas = {
        "w_old": _fake_meta("trainer", 0, 1, 100),
        "w_cur": _fake_meta("trainer", 0, 5, 200),
        "w_new": _fake_meta("trainer", 0, 7, 300),
    }
    receiver._receiver._client.list_sources.return_value = response
    receiver._receiver._client.get_metadata = lambda mx_source_id, worker_id: metas[
        worker_id
    ]

    cands = receiver.discover_v2_sources(model_name="m", min_version=5)
    versions = sorted(c.ref.training_step for c in cands)
    assert versions == [5, 7]


def test_non_v2_sources_ignored(v2):
    """Sources lacking ``mx_v2`` marker are ignored entirely."""
    receiver = v2.MxV2RefitReceiver(
        agent_name="t", device_id=0, mx_server_url="x", worker_rank=0
    )
    receiver.initialize()
    response = MagicMock()
    response.instances = [
        _fake_instance("m", "s", "v2_worker"),
        _fake_instance("m", "s", "v1_worker"),
    ]

    @dataclass
    class _MetaV1:
        found: bool = True

        def __post_init__(self):
            from modelexpress.p2p_pb2 import SourceIdentity, WorkerMetadata

            self.identity = SourceIdentity(model_name="m")
            # No mx_v2 marker → v1 source
            self.identity.extra_parameters.update(
                {"role": "trainer", "training_step": "1"}
            )
            self.worker = WorkerMetadata()
            self.worker.updated_at = 999

    metas = {
        "v2_worker": _fake_meta("trainer", 0, 1, 100),
        "v1_worker": _MetaV1(),
    }
    receiver._receiver._client.list_sources.return_value = response
    receiver._receiver._client.get_metadata = lambda mx_source_id, worker_id: metas[
        worker_id
    ]

    cands = receiver.discover_v2_sources(model_name="m")
    assert len(cands) == 1
    assert cands[0].ref.worker_id == "v2_worker"


def test_pick_best_with_expert_filter(v2):
    """When MoE expert filter is set, candidate must own all needed experts."""
    receiver = v2.MxV2RefitReceiver(
        agent_name="t", device_id=0, mx_server_url="x", worker_rank=2
    )
    # Build candidates manually to test pick_best_source in isolation.
    candidates = [
        v2.V2SourceCandidate(
            ref=type(
                "Ref",
                (),
                {
                    "mx_source_id": "s",
                    "worker_id": "replica_partial",
                    "model_name": "m",
                    "worker_rank": 2,
                    "training_step": 5,
                },
            )(),
            role="inference_replica",
            worker_rank=2,
            registry=None,
            owned_experts_per_layer={5: {48, 49, 50}},  # only 3 of needed 6
            updated_at=200,
        ),
        v2.V2SourceCandidate(
            ref=type(
                "Ref",
                (),
                {
                    "mx_source_id": "s",
                    "worker_id": "replica_full",
                    "model_name": "m",
                    "worker_rank": 2,
                    "training_step": 5,
                },
            )(),
            role="inference_replica",
            worker_rank=2,
            registry=None,
            owned_experts_per_layer={5: {48, 49, 50, 51, 52, 53, 54, 55}},
            updated_at=100,  # older but covers all needed
        ),
    ]
    needed = {5: {48, 49, 50, 51, 52, 53}}
    chosen = receiver.pick_best_source(
        candidates, needed_experts_per_layer=needed
    )
    assert chosen is not None
    assert chosen.ref.worker_id == "replica_full"


def test_pick_best_falls_back_to_trainer(v2):
    """Trainer always covers all experts (its registry is authoritative)."""
    receiver = v2.MxV2RefitReceiver(
        agent_name="t", device_id=0, mx_server_url="x", worker_rank=2
    )
    candidates = [
        v2.V2SourceCandidate(
            ref=type(
                "Ref",
                (),
                {
                    "mx_source_id": "s",
                    "worker_id": "trainer-2",
                    "model_name": "m",
                    "worker_rank": 2,
                    "training_step": 5,
                },
            )(),
            role="trainer",
            worker_rank=2,
            registry={"version": 5, "tensors": []},
            owned_experts_per_layer={},  # not populated for trainers
            updated_at=300,
        ),
    ]
    chosen = receiver.pick_best_source(
        candidates, needed_experts_per_layer={5: {0, 1, 2, 3}}
    )
    assert chosen is not None
    assert chosen.ref.worker_id == "trainer-2"


def test_world_layout_round_trip(v2):
    layout = v2.TrainerWorldLayout(fsdp_world_size=4, ep_world_size=8)
    encoded = layout.encode()
    assert encoded == "fsdp:4,tp:1,pp:1,ep:8"
    rt = v2.TrainerWorldLayout.decode(encoded)
    assert rt == layout


def test_agent_name_fallback_when_identity_missing(v2):
    """Older servers don't return SourceIdentity in GetMetadataResponse;
    the v2 receiver must fall back to parsing the v2 marker from
    WorkerMetadata.agent_name."""
    receiver = v2.MxV2RefitReceiver(
        agent_name="t", device_id=0, mx_server_url="x", worker_rank=2
    )
    receiver.initialize()

    response = MagicMock()
    response.instances = [
        _fake_instance("m", "s", "v2_via_agent_name"),
    ]

    class _MetaNoIdentity:
        """Mimics an old-server GetMetadataResponse: no `identity` attribute."""

        found = True

        def __init__(self):
            from modelexpress.p2p_pb2 import WorkerMetadata

            self.worker = WorkerMetadata()
            self.worker.updated_at = 12345
            # The publisher writes v2 markers into agent_name as a fallback
            self.worker.agent_name = (
                "mx_v2|trainer|rank=2|version=42|orig=nemo-rl-trainer-r2"
            )

    receiver._receiver._client.list_sources.return_value = response
    receiver._receiver._client.get_metadata = lambda mx_source_id, worker_id: (
        _MetaNoIdentity()
    )

    candidates = receiver.discover_v2_sources(model_name="m", min_version=0)
    assert len(candidates) == 1
    cand = candidates[0]
    assert cand.role == "trainer"
    assert cand.worker_rank == 2
    assert cand.ref.training_step == 42
    assert cand.updated_at == 12345
