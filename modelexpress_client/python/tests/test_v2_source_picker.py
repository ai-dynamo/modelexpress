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

    We snapshot any pre-existing ``modelexpress.*`` entries in ``sys.modules``
    and restore them on teardown so other test modules that run after us
    (test_vllm_adapter, test_vllm_loader, etc.) still see the real
    ``modelexpress`` package — not the stub we install here.
    """
    _saved: dict[str, Any] = {}
    _injected: set[str] = set()

    def _install(name: str, mod) -> None:
        if name not in _saved and name in sys.modules:
            _saved[name] = sys.modules[name]
        _injected.add(name)
        sys.modules[name] = mod

    # Pre-create stub modules for the dependencies that nemo_rl_v2 imports.
    pkg = types.ModuleType("modelexpress")
    pkg.__path__ = [str(_PKG_ROOT)]  # type: ignore[attr-defined]
    _install("modelexpress", pkg)

    # p2p_pb2 stub: just the constants & message classes used.
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
    _install("modelexpress.heartbeat", hb)

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
            # Tests inject scratch payloads via this dict: worker_id → {name: tensor}.
            self._scratch_payloads: dict[str, dict[str, Any]] = {}

        def initialize(self, model_tensors=None):
            pass

        def receive_weights(self, ref, timeout_seconds=300.0):
            return iter([])

        def receive_weights_scratch(
            self, ref, timeout_seconds=300.0, tensor_shapes=None
        ):
            payload = self._scratch_payloads.get(ref.worker_id, {})
            for name, tensor in payload.items():
                yield name, tensor

    refit_mod.MxRefitReceiver = _RefitStub
    refit_mod.SourceRef = _SourceRef
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
            ident = p2p_pb2.SourceIdentity()
            return ident

    pub_mod.MxTrainingPublisher = _PubStub
    _install("modelexpress.training_publisher", pub_mod)

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
    _install("modelexpress.types", types_mod)

    # Now exec shape_descriptors + nemo_rl_v2 against this module space.
    # _load also writes into sys.modules, so route through _install for cleanup.
    sd_spec = importlib.util.spec_from_file_location(
        "modelexpress.shape_descriptors", _PKG_ROOT / "shape_descriptors.py"
    )
    sd = importlib.util.module_from_spec(sd_spec)
    _install("modelexpress.shape_descriptors", sd)
    sd_spec.loader.exec_module(sd)
    pkg.shape_descriptors = sd  # type: ignore[attr-defined]

    v2_spec = importlib.util.spec_from_file_location(
        "modelexpress.nemo_rl_v2", _PKG_ROOT / "nemo_rl_v2.py"
    )
    v2 = importlib.util.module_from_spec(v2_spec)
    _install("modelexpress.nemo_rl_v2", v2)
    v2_spec.loader.exec_module(v2)

    try:
        yield v2
    finally:
        # Restore the real modules so test modules that run after this one
        # (test_vllm_adapter, test_vllm_loader, ...) get the genuine
        # ``modelexpress`` package back, not our stubs.
        for name in _injected:
            sys.modules.pop(name, None)
        for name, mod in _saved.items():
            sys.modules[name] = mod


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


def test_pick_best_requires_ownership_for_trainer(v2):
    """Trainer candidates must still own the experts they're picked for.

    Previously the picker returned a ROLE_TRAINER candidate unconditionally,
    on the assumption the trainer's registry was authoritative. That breaks
    on EP > 1 where each trainer rank only holds a slice of the experts —
    picking a trainer rank that doesn't own the requested experts loads the
    wrong shard. The receiver must instead return None and let the caller
    multi-source via discover_v2_sources_for_slice.
    """
    receiver = v2.MxV2RefitReceiver(
        agent_name="t", device_id=0, mx_server_url="x", worker_rank=2
    )
    # Trainer publishes empty ownership — it owns nothing claimable here.
    trainer_no_ownership = v2.V2SourceCandidate(
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
        owned_experts_per_layer={},
        updated_at=300,
    )
    chosen = receiver.pick_best_source(
        [trainer_no_ownership],
        needed_experts_per_layer={5: {0, 1, 2, 3}},
    )
    assert chosen is None, (
        "trainer with no declared ownership must NOT be picked for "
        "expert-coverage queries — caller has to multi-source"
    )

    # Trainer that DOES own all needed experts is picked.
    trainer_full_ownership = v2.V2SourceCandidate(
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
        owned_experts_per_layer={5: {0, 1, 2, 3}},
        updated_at=300,
    )
    chosen = receiver.pick_best_source(
        [trainer_full_ownership],
        needed_experts_per_layer={5: {0, 1, 2, 3}},
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


# ----------------------------------------------------------------------------
# Phase 3b — compile_target_filter on discover_v2_sources
# ----------------------------------------------------------------------------


def _registry_blob(v2, tensors):
    """Helper: encode a registry from a list of TensorDescriptorV2 dicts."""
    sd = sys.modules["modelexpress.shape_descriptors"]
    descriptors = [
        sd.TensorDescriptorV2(
            name=t["name"],
            global_shape=t.get("global_shape", (8, 16)),
            dtype=t.get("dtype", "bfloat16"),
            placement_kind=t.get("placement_kind", sd.PLACEMENT_REPLICATE),
            shard_axis=t.get("shard_axis", 0),
            local_shard_range=t.get("local_shard_range"),
            compile_target=t.get("compile_target", sd.COMPILE_TARGET_HF_RAW),
            compile_metadata=t.get("compile_metadata", {}),
        )
        for t in tensors
    ]
    return sd.encode_registry(descriptors, version=1, trainer_world_layout="fsdp:1")


def _set_two_compile_sources(v2, receiver, hf_blob, fp8_blob):
    response = MagicMock()
    response.instances = [
        _fake_instance("m", "s", "trainer_hf"),
        _fake_instance("m", "s", "trainer_fp8"),
    ]
    metas = {
        "trainer_hf": _fake_meta("trainer", 0, 7, 200, registry_blob=hf_blob),
        "trainer_fp8": _fake_meta("trainer", 0, 7, 300, registry_blob=fp8_blob),
    }
    receiver._receiver._client.list_sources.return_value = response
    receiver._receiver._client.get_metadata = lambda mx_source_id, worker_id: metas[worker_id]


def test_compile_target_filter_accepts_only_matching(v2):
    sd = sys.modules["modelexpress.shape_descriptors"]
    receiver = v2.MxV2RefitReceiver(
        agent_name="t", device_id=0, mx_server_url="x", worker_rank=0
    )
    receiver.initialize()

    hf_blob = _registry_blob(v2, [{"name": "w"}])
    fp8_blob = _registry_blob(
        v2,
        [{"name": "w", "compile_target": sd.COMPILE_TARGET_DEEPGEMM_FP8}],
    )
    _set_two_compile_sources(v2, receiver, hf_blob, fp8_blob)

    only_hf = receiver.discover_v2_sources(
        model_name="m",
        min_version=0,
        compile_target_filter={sd.COMPILE_TARGET_HF_RAW},
    )
    assert [c.ref.worker_id for c in only_hf] == ["trainer_hf"]

    only_fp8 = receiver.discover_v2_sources(
        model_name="m",
        min_version=0,
        compile_target_filter={sd.COMPILE_TARGET_DEEPGEMM_FP8},
    )
    assert [c.ref.worker_id for c in only_fp8] == ["trainer_fp8"]

    both = receiver.discover_v2_sources(
        model_name="m",
        min_version=0,
        compile_target_filter={
            sd.COMPILE_TARGET_HF_RAW,
            sd.COMPILE_TARGET_DEEPGEMM_FP8,
        },
    )
    assert {c.ref.worker_id for c in both} == {"trainer_hf", "trainer_fp8"}


def test_compile_target_filter_unset_admits_all(v2):
    sd = sys.modules["modelexpress.shape_descriptors"]
    receiver = v2.MxV2RefitReceiver(
        agent_name="t", device_id=0, mx_server_url="x", worker_rank=0
    )
    receiver.initialize()

    hf_blob = _registry_blob(v2, [{"name": "w"}])
    fp8_blob = _registry_blob(
        v2,
        [{"name": "w", "compile_target": sd.COMPILE_TARGET_DEEPGEMM_FP8}],
    )
    _set_two_compile_sources(v2, receiver, hf_blob, fp8_blob)

    out = receiver.discover_v2_sources(model_name="m", min_version=0)
    assert {c.ref.worker_id for c in out} == {"trainer_hf", "trainer_fp8"}


def test_compile_target_filter_rejects_when_no_registry(v2):
    """If the candidate has no registry but caller wants compile filtering,
    we MUST reject (we can't certify the bytes blindly)."""
    sd = sys.modules["modelexpress.shape_descriptors"]
    receiver = v2.MxV2RefitReceiver(
        agent_name="t", device_id=0, mx_server_url="x", worker_rank=0
    )
    receiver.initialize()

    response = MagicMock()
    response.instances = [_fake_instance("m", "s", "no_registry")]
    metas = {
        "no_registry": _fake_meta("trainer", 0, 1, 100, registry_blob=""),
    }
    receiver._receiver._client.list_sources.return_value = response
    receiver._receiver._client.get_metadata = lambda mx_source_id, worker_id: metas[worker_id]

    # With no filter, candidate is admitted (back-compat).
    assert len(receiver.discover_v2_sources(model_name="m")) == 1
    # With a filter, candidate is rejected (no registry → unknowable target).
    filtered = receiver.discover_v2_sources(
        model_name="m",
        compile_target_filter={sd.COMPILE_TARGET_HF_RAW},
    )
    assert filtered == []


def test_compile_target_filter_required_metadata(v2):
    sd = sys.modules["modelexpress.shape_descriptors"]
    receiver = v2.MxV2RefitReceiver(
        agent_name="t", device_id=0, mx_server_url="x", worker_rank=0
    )
    receiver.initialize()

    blob128 = _registry_blob(
        v2,
        [{
            "name": "w",
            "compile_target": sd.COMPILE_TARGET_DEEPGEMM_FP8,
            "compile_metadata": {"block_size": 128},
        }],
    )
    blob256 = _registry_blob(
        v2,
        [{
            "name": "w",
            "compile_target": sd.COMPILE_TARGET_DEEPGEMM_FP8,
            "compile_metadata": {"block_size": 256},
        }],
    )
    response = MagicMock()
    response.instances = [
        _fake_instance("m", "s", "blk128"),
        _fake_instance("m", "s", "blk256"),
    ]
    metas = {
        "blk128": _fake_meta("trainer", 0, 1, 100, registry_blob=blob128),
        "blk256": _fake_meta("trainer", 0, 1, 200, registry_blob=blob256),
    }
    receiver._receiver._client.list_sources.return_value = response
    receiver._receiver._client.get_metadata = lambda mx_source_id, worker_id: metas[worker_id]

    keep_128 = receiver.discover_v2_sources(
        model_name="m",
        compile_target_filter={sd.COMPILE_TARGET_DEEPGEMM_FP8},
        required_compile_metadata={"block_size": 128},
    )
    assert [c.ref.worker_id for c in keep_128] == ["blk128"]


# ----------------------------------------------------------------------------
# Phase 4 — multi-source slice discovery (discover_v2_sources_for_slice)
# ----------------------------------------------------------------------------


def _build_two_trainers_tp4_mixed_tp8(v2):
    """Two trainers at TP=4: rank 0 holds rows [0,2048), rank 1 holds [2048,4096)
    on a tensor of global axis-0 extent 4096. Receiver at TP=8 rank N wants
    rows [N*512, (N+1)*512)."""
    sd = sys.modules["modelexpress.shape_descriptors"]
    blob_r0 = _registry_blob(
        v2,
        [{
            "name": "w",
            "global_shape": (4096, 1024),
            "placement_kind": sd.PLACEMENT_SHARD,
            "shard_axis": 0,
            "local_shard_range": (0, 2048),
        }],
    )
    blob_r1 = _registry_blob(
        v2,
        [{
            "name": "w",
            "global_shape": (4096, 1024),
            "placement_kind": sd.PLACEMENT_SHARD,
            "shard_axis": 0,
            "local_shard_range": (2048, 4096),
        }],
    )
    return blob_r0, blob_r1


def _set_two_trainers(v2, receiver, blob_r0, blob_r1):
    response = MagicMock()
    response.instances = [
        _fake_instance("m", "s", "trainer_r0"),
        _fake_instance("m", "s", "trainer_r1"),
    ]
    metas = {
        "trainer_r0": _fake_meta("trainer", 0, 7, 200, registry_blob=blob_r0),
        "trainer_r1": _fake_meta("trainer", 1, 7, 200, registry_blob=blob_r1),
    }
    receiver._receiver._client.list_sources.return_value = response
    receiver._receiver._client.get_metadata = lambda mx_source_id, worker_id: metas[worker_id]


def test_phase4_slice_within_one_trainer_shard(v2):
    """Receiver TP=8 rank=1 wants rows [512, 1024) — fully inside trainer rank 0."""
    receiver = v2.MxV2RefitReceiver(
        agent_name="t", device_id=0, mx_server_url="x", worker_rank=1
    )
    receiver.initialize()
    blob_r0, blob_r1 = _build_two_trainers_tp4_mixed_tp8(v2)
    _set_two_trainers(v2, receiver, blob_r0, blob_r1)

    plan = receiver.discover_v2_sources_for_slice(
        model_name="m",
        target_layout=v2.TargetTPLayout(world_size=8, rank=1, shard_axis=0),
        same_rank_only=False,
    )
    assert plan.fully_covered
    contributions = plan.per_tensor_sources["w"]
    assert len(contributions) == 1
    src = contributions[0]
    assert src.candidate.ref.worker_id == "trainer_r0"
    assert src.src_range == (512, 1024)
    assert src.dst_range == (0, 512)


def test_phase4_slice_spans_two_trainer_shards(v2):
    """Receiver TP=2 rank=0 wants [0, 2048) — exactly trainer rank 0 alone.
    Receiver TP=2 rank=1 wants [2048, 4096) — exactly trainer rank 1 alone.
    Now flip: receiver TP=4 rank=1 wants [1024, 2048) — split case stays inside
    trainer rank 0.  Real cross-shard case: receiver TP=2 wants exact halves
    (use a receiver that explicitly straddles the boundary)."""
    receiver = v2.MxV2RefitReceiver(
        agent_name="t", device_id=0, mx_server_url="x", worker_rank=0
    )
    receiver.initialize()
    blob_r0, blob_r1 = _build_two_trainers_tp4_mixed_tp8(v2)
    _set_two_trainers(v2, receiver, blob_r0, blob_r1)

    # target_range=(1500, 2500) straddles the trainer rank 0/1 boundary at 2048.
    plan = receiver.discover_v2_sources_for_slice(
        model_name="m",
        target_layout=v2.TargetTPLayout(
            world_size=1, rank=0, shard_axis=0, target_range=(1500, 2500)
        ),
        same_rank_only=False,
    )
    assert plan.fully_covered
    contributions = plan.per_tensor_sources["w"]
    assert len(contributions) == 2
    # First contribution from trainer rank 0: [1500, 2048) in src → [0, 548) in dst
    first = contributions[0]
    assert first.candidate.ref.worker_id == "trainer_r0"
    assert first.src_range == (1500, 2048)
    assert first.dst_range == (0, 548)
    # Second contribution from trainer rank 1: [0, 452) in src → [548, 1000) in dst
    second = contributions[1]
    assert second.candidate.ref.worker_id == "trainer_r1"
    assert second.src_range == (0, 452)
    assert second.dst_range == (548, 1000)


def test_phase4_replicate_picks_one_candidate(v2):
    """REPLICATE tensor: any candidate works; planner picks the freshest one."""
    sd = sys.modules["modelexpress.shape_descriptors"]
    receiver = v2.MxV2RefitReceiver(
        agent_name="t", device_id=0, mx_server_url="x", worker_rank=0
    )
    receiver.initialize()

    blob = _registry_blob(
        v2,
        [{"name": "lm_head.weight", "global_shape": (1024, 4096), "placement_kind": sd.PLACEMENT_REPLICATE}],
    )
    response = MagicMock()
    response.instances = [
        _fake_instance("m", "s", "trainer_a"),
        _fake_instance("m", "s", "trainer_b"),
    ]
    metas = {
        "trainer_a": _fake_meta("trainer", 0, 7, 100, registry_blob=blob),
        "trainer_b": _fake_meta("trainer", 1, 7, 300, registry_blob=blob),
    }
    receiver._receiver._client.list_sources.return_value = response
    receiver._receiver._client.get_metadata = lambda mx_source_id, worker_id: metas[worker_id]

    plan = receiver.discover_v2_sources_for_slice(
        model_name="m",
        target_layout=v2.TargetTPLayout(world_size=2, rank=0, shard_axis=0),
        same_rank_only=False,
    )
    assert plan.fully_covered
    contribs = plan.per_tensor_sources["lm_head.weight"]
    assert len(contribs) == 1
    # Freshest trainer (updated_at=300) preferred; either trainer is correct, but
    # the picker sorts by freshness so trainer_b wins.
    assert contribs[0].candidate.ref.worker_id == "trainer_b"


def test_phase4_coverage_gap_is_missing(v2):
    """If trainers don't fully cover the receiver's slice, plan.missing fires."""
    sd = sys.modules["modelexpress.shape_descriptors"]
    receiver = v2.MxV2RefitReceiver(
        agent_name="t", device_id=0, mx_server_url="x", worker_rank=0
    )
    receiver.initialize()
    # Single trainer with [0, 1024); receiver wants [0, 4096).
    blob = _registry_blob(
        v2,
        [{
            "name": "w",
            "global_shape": (4096,),
            "placement_kind": sd.PLACEMENT_SHARD,
            "shard_axis": 0,
            "local_shard_range": (0, 1024),
        }],
    )
    response = MagicMock()
    response.instances = [_fake_instance("m", "s", "only_one")]
    metas = {"only_one": _fake_meta("trainer", 0, 7, 200, registry_blob=blob)}
    receiver._receiver._client.list_sources.return_value = response
    receiver._receiver._client.get_metadata = lambda mx_source_id, worker_id: metas[worker_id]

    plan = receiver.discover_v2_sources_for_slice(
        model_name="m",
        target_layout=v2.TargetTPLayout(
            world_size=1, rank=0, shard_axis=0, target_range=(0, 4096)
        ),
        same_rank_only=False,
    )
    assert not plan.fully_covered
    assert plan.missing
    assert "coverage gap" in plan.missing[0]


def test_phase4_shard_axis_mismatch_is_missing(v2):
    """Publisher shards on axis 0 but receiver wants axis 1: caller's problem."""
    sd = sys.modules["modelexpress.shape_descriptors"]
    receiver = v2.MxV2RefitReceiver(
        agent_name="t", device_id=0, mx_server_url="x", worker_rank=0
    )
    receiver.initialize()
    blob = _registry_blob(
        v2,
        [{
            "name": "w",
            "global_shape": (4096, 1024),
            "placement_kind": sd.PLACEMENT_SHARD,
            "shard_axis": 0,
            "local_shard_range": (0, 4096),
        }],
    )
    response = MagicMock()
    response.instances = [_fake_instance("m", "s", "trainer")]
    metas = {"trainer": _fake_meta("trainer", 0, 7, 200, registry_blob=blob)}
    receiver._receiver._client.list_sources.return_value = response
    receiver._receiver._client.get_metadata = lambda mx_source_id, worker_id: metas[worker_id]

    plan = receiver.discover_v2_sources_for_slice(
        model_name="m",
        target_layout=v2.TargetTPLayout(
            world_size=2, rank=0, shard_axis=1, target_range=(0, 512)
        ),
        same_rank_only=False,
    )
    assert not plan.fully_covered
    assert any("shard_axis mismatch" in m for m in plan.missing)


def test_phase4_receive_via_plan_stitches_two_sources(v2):
    """End-to-end: planner + receive_via_plan correctly stitches two shards."""
    import torch

    sd = sys.modules["modelexpress.shape_descriptors"]
    receiver = v2.MxV2RefitReceiver(
        agent_name="t", device_id=0, mx_server_url="x", worker_rank=0
    )
    receiver.initialize()
    blob_r0, blob_r1 = _build_two_trainers_tp4_mixed_tp8(v2)
    _set_two_trainers(v2, receiver, blob_r0, blob_r1)

    # Build fake scratch tensors that match what the publishers would expose:
    # trainer r0 owns [0, 2048), trainer r1 owns [2048, 4096). Each is a
    # (local_extent, 1024) bf16 tensor (we'd actually be returning floats here
    # so the in-process slice/cat math is observable).
    r0_buf = torch.arange(0, 2048).repeat_interleave(1024).view(2048, 1024).float()
    r1_buf = torch.arange(2048, 4096).repeat_interleave(1024).view(2048, 1024).float()
    receiver._receiver._scratch_payloads = {
        "trainer_r0": {"w": r0_buf},
        "trainer_r1": {"w": r1_buf},
    }

    plan = receiver.discover_v2_sources_for_slice(
        model_name="m",
        target_layout=v2.TargetTPLayout(
            world_size=1, rank=0, shard_axis=0, target_range=(1500, 2500)
        ),
        same_rank_only=False,
    )
    assert plan.fully_covered

    out = dict(receiver.receive_via_plan(plan))
    assert "w" in out
    stitched = out["w"]
    # Expect shape (1000, 1024); first 548 rows come from r0_buf[1500:2048],
    # next 452 rows come from r1_buf[0:452] (which are global rows [2048,2500)).
    assert stitched.shape == (1000, 1024)
    expected = torch.cat([r0_buf[1500:2048], r1_buf[0:452]], dim=0)
    assert torch.equal(stitched, expected)


def test_phase4_receive_via_plan_single_source_passthrough(v2):
    """When one trainer covers the slice, no torch.cat happens."""
    import torch

    receiver = v2.MxV2RefitReceiver(
        agent_name="t", device_id=0, mx_server_url="x", worker_rank=1
    )
    receiver.initialize()
    blob_r0, blob_r1 = _build_two_trainers_tp4_mixed_tp8(v2)
    _set_two_trainers(v2, receiver, blob_r0, blob_r1)

    r0_buf = torch.arange(0, 2048).repeat_interleave(1024).view(2048, 1024).float()
    receiver._receiver._scratch_payloads = {"trainer_r0": {"w": r0_buf}}

    # Receiver TP=8 rank=1: wants rows [512, 1024) — fully inside trainer r0.
    plan = receiver.discover_v2_sources_for_slice(
        model_name="m",
        target_layout=v2.TargetTPLayout(world_size=8, rank=1, shard_axis=0),
        same_rank_only=False,
    )
    assert plan.fully_covered
    out = dict(receiver.receive_via_plan(plan))
    assert out["w"].shape == (512, 1024)
    assert torch.equal(out["w"], r0_buf[512:1024])


# ----------------------------------------------------------------------------
# Phase 3 graduation glue — MxV2TrainingPublisher.add_tensor takes the new
# compile_target / compile_metadata kwargs and they flow into the registry.
# ----------------------------------------------------------------------------


def test_phase3_add_tensor_threads_compile_target(v2):
    """add_tensor's new compile_target + compile_metadata kwargs must surface
    on the resulting TensorDescriptorV2 in the publisher's internal registry."""
    import torch

    sd = sys.modules["modelexpress.shape_descriptors"]

    # Stand up a publisher pointed at the stub MX client; we won't actually
    # publish, just inspect the registry after add_tensor calls.
    pub = v2.MxV2TrainingPublisher(
        agent_name="t",
        device_id=0,
        mx_server_url="x",
        worker_rank=0,
        world_layout=v2.TrainerWorldLayout(fsdp_world_size=1),
        heartbeat=False,
    )
    pub.initialize(model_name="m", dtype="bfloat16")

    class _FakeCudaTensor:
        # Minimal stand-in: ``describe_tensor`` reads .dtype, .shape,
        # optional .placements; ``add_tensor`` checks .is_cuda.
        def __init__(self, shape, dtype=torch.bfloat16):
            self.shape = torch.Size(shape)
            self.dtype = dtype
            self.is_cuda = True

    pub.add_tensor(
        name="lm_head.weight",
        tensor=_FakeCudaTensor([2048, 4096]),
    )
    pub.add_tensor(
        name="model.layers.0.mlp.gate_proj.weight",
        tensor=_FakeCudaTensor([512, 2048]),
        compile_target=sd.COMPILE_TARGET_CUTLASS_FP8,
        compile_metadata={
            "dtype": "e4m3",
            "scale_layout": "per_channel",
            "scale_axis": -1,
            "activation_scheme": "dynamic",
        },
    )
    pub.add_tensor(
        name="model.layers.0.mlp.experts.weight",
        tensor=_FakeCudaTensor([24, 4096, 12288]),
        is_expert=True,
        owned_expert_ids=(0, 1, 2, 3),
        compile_target=sd.COMPILE_TARGET_DEEPGEMM_FP8,
        compile_metadata={
            "dtype": "e4m3",
            "scale_layout": "blockwise",
            "block_size": [128, 128],
        },
    )

    by_name = {d.name: d for d in pub._registry}
    assert by_name["lm_head.weight"].compile_target == sd.COMPILE_TARGET_HF_RAW
    assert by_name["lm_head.weight"].compile_metadata == {}

    gp = by_name["model.layers.0.mlp.gate_proj.weight"]
    assert gp.compile_target == sd.COMPILE_TARGET_CUTLASS_FP8
    assert gp.compile_metadata == {
        "dtype": "e4m3",
        "scale_layout": "per_channel",
        "scale_axis": -1,
        "activation_scheme": "dynamic",
    }

    ex = by_name["model.layers.0.mlp.experts.weight"]
    assert ex.compile_target == sd.COMPILE_TARGET_DEEPGEMM_FP8
    assert ex.compile_metadata == {
        "dtype": "e4m3",
        "scale_layout": "blockwise",
        "block_size": [128, 128],
    }
    assert ex.is_expert
    assert set(ex.owned_expert_ids) == {0, 1, 2, 3}


def test_phase3_add_tensor_compile_target_survives_encode_decode(v2):
    """Round-trip: tagged tensors → encode_registry → decode_registry preserves
    compile_target + compile_metadata. Asserts the wire format is intact end-to-end."""
    import torch

    sd = sys.modules["modelexpress.shape_descriptors"]
    pub = v2.MxV2TrainingPublisher(
        agent_name="t",
        device_id=0,
        mx_server_url="x",
        worker_rank=0,
        world_layout=v2.TrainerWorldLayout(fsdp_world_size=1),
        heartbeat=False,
    )
    pub.initialize(model_name="m", dtype="bfloat16")

    class _FakeCudaTensor:
        def __init__(self, shape, dtype=torch.bfloat16):
            self.shape = torch.Size(shape)
            self.dtype = dtype
            self.is_cuda = True

    pub.add_tensor(
        name="w",
        tensor=_FakeCudaTensor([64, 128]),
        compile_target=sd.COMPILE_TARGET_CUTLASS_FP8,
        compile_metadata={"dtype": "e4m3", "scale_layout": "per_channel"},
    )

    blob = sd.encode_registry(
        pub._registry, version=42, trainer_world_layout="fsdp:1"
    )
    parsed = sd.decode_registry(blob)
    out = parsed["tensors"][0]
    assert out.compile_target == sd.COMPILE_TARGET_CUTLASS_FP8
    assert out.compile_metadata == {"dtype": "e4m3", "scale_layout": "per_channel"}


def test_phase4_receive_via_plan_rejects_uncovered(v2):
    """receive_via_plan refuses to run a partial plan."""
    sd = sys.modules["modelexpress.shape_descriptors"]
    receiver = v2.MxV2RefitReceiver(
        agent_name="t", device_id=0, mx_server_url="x", worker_rank=0
    )
    receiver.initialize()
    blob = _registry_blob(
        v2,
        [{
            "name": "w",
            "global_shape": (4096,),
            "placement_kind": sd.PLACEMENT_SHARD,
            "shard_axis": 0,
            "local_shard_range": (0, 1024),
        }],
    )
    response = MagicMock()
    response.instances = [_fake_instance("m", "s", "only_one")]
    metas = {"only_one": _fake_meta("trainer", 0, 7, 200, registry_blob=blob)}
    receiver._receiver._client.list_sources.return_value = response
    receiver._receiver._client.get_metadata = lambda mx_source_id, worker_id: metas[worker_id]

    plan = receiver.discover_v2_sources_for_slice(
        model_name="m",
        target_layout=v2.TargetTPLayout(
            world_size=1, rank=0, shard_axis=0, target_range=(0, 4096)
        ),
        same_rank_only=False,
    )
    with pytest.raises(RuntimeError, match="not fully covered"):
        list(receiver.receive_via_plan(plan))
