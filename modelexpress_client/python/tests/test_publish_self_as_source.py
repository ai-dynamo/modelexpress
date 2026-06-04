# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``MxV2RefitReceiver.publish_self_as_source``.

Pinned by two regressions found while validating tree fan-out on GB200:

* Bug A — ``MxRefitReceiver.initialize`` was not assigning ``_worker_id``,
  so ``publish_self_as_source`` always forwarded an empty ``worker_id``
  to the MX server, which rejects with
  ``PublishMetadata failed: worker_id is required``. The exception was
  swallowed and the method silently returned ``None``.

* Bug B — ``publish_self_as_source`` only emitted v2 metadata via
  ``identity.extra_parameters``. The Rust server line that strips
  ``identity.extra_parameters`` on ``GetMetadataResponse`` therefore
  saw replica entries with no v2 markers and ``discover_v2_sources``
  filtered them out. The trainer's ``publish()`` defends against this
  by emitting all three transports in parallel; replicas should too.

These tests exercise both fixes in isolation, without NIXL / gRPC,
mirroring the stubbing pattern in ``test_v2_source_picker.py``.
"""

from __future__ import annotations

import importlib.util
import json
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
    """Load nemo_rl_v2 with NIXL / gRPC / heartbeat stubbed out."""
    pkg = types.ModuleType("modelexpress")
    pkg.__path__ = [str(_PKG_ROOT)]  # type: ignore[attr-defined]
    sys.modules["modelexpress"] = pkg

    # p2p_pb2 stub.
    p2p_pb2 = types.ModuleType("modelexpress.p2p_pb2")
    p2p_pb2.SOURCE_STATUS_READY = 2
    p2p_pb2.SOURCE_STATUS_INITIALIZING = 1
    p2p_pb2.SOURCE_STATUS_STALE = 3
    p2p_pb2.MX_SOURCE_TYPE_WEIGHTS = 0
    p2p_pb2.BACKEND_FRAMEWORK_UNKNOWN = 0
    sys.modules["modelexpress.p2p_pb2"] = p2p_pb2

    class _SourceIdentity:
        """Mirror just enough of the real protobuf SourceIdentity for tests.

        Real protobuf accepts every field as a kwarg, including the
        ``extra_parameters`` map. The dataclass-with-post_init pattern used
        in test_v2_source_picker.py forces callers to set
        ``ident.extra_parameters[k] = v`` after construction; the
        publish_self_as_source code path here passes extras directly to
        the constructor (matching real protobuf behaviour).
        """

        def __init__(self, **kwargs):
            self.model_name = kwargs.get("model_name", "")
            self.mx_source_type = kwargs.get("mx_source_type", 0)
            self.backend_framework = kwargs.get("backend_framework", 0)
            self.tensor_parallel_size = kwargs.get("tensor_parallel_size", 0)
            self.pipeline_parallel_size = kwargs.get("pipeline_parallel_size", 0)
            self.expert_parallel_size = kwargs.get("expert_parallel_size", 0)
            self.dtype = kwargs.get("dtype", "")
            self.quantization = kwargs.get("quantization", "")
            self.extra_parameters = dict(kwargs.get("extra_parameters") or {})

    class _TensorDescriptor:
        def __init__(self, **kwargs):
            self.name = kwargs.get("name", "")
            self.addr = kwargs.get("addr", 0)
            self.size = kwargs.get("size", 0)
            self.device_id = kwargs.get("device_id", 0)
            self.dtype = kwargs.get("dtype", "")

    class _WorkerMetadata:
        def __init__(self, **kwargs):
            self.worker_rank = kwargs.get("worker_rank", 0)
            self.nixl_metadata = kwargs.get("nixl_metadata", b"")
            self.tensors = list(kwargs.get("tensors") or [])
            self.status = kwargs.get("status", 0)
            self.agent_name = kwargs.get("agent_name", "")

    p2p_pb2.SourceIdentity = _SourceIdentity  # type: ignore[attr-defined]
    p2p_pb2.WorkerMetadata = _WorkerMetadata  # type: ignore[attr-defined]
    p2p_pb2.TensorDescriptor = _TensorDescriptor  # type: ignore[attr-defined]

    # Heartbeat stub.
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

    # MxRefitReceiver stub. _client / _nixl populated by tests.
    refit_mod = types.ModuleType("modelexpress.refit_receiver")

    @dataclass
    class _SourceRef:
        mx_source_id: str = ""
        worker_id: str = ""
        model_name: str = ""
        worker_rank: int = 0
        training_step: int = 0

    class _RefitStub:
        def __init__(self, agent_name="stub", **kw):
            self._client = None
            self._nixl = None
            self._agent_name = agent_name
            # Mirror the (post-fix) MxRefitReceiver.__init__ behaviour: a
            # stable per-instance worker_id is set even before initialize().
            self._worker_id = f"{agent_name}-stubworker"

        def initialize(self, model_tensors=None):
            self._client = MagicMock()
            self._nixl = MagicMock()

        def receive_weights(self, ref, timeout_seconds=300.0):
            return iter([])

    refit_mod.MxRefitReceiver = _RefitStub
    refit_mod.SourceRef = _SourceRef
    sys.modules["modelexpress.refit_receiver"] = refit_mod

    # MxTrainingPublisher stub (only constructor matters for our tests).
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
    sys.modules["modelexpress.training_publisher"] = pub_mod

    # types.TensorDescriptor stub.
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

    sd = _load("modelexpress.shape_descriptors", _PKG_ROOT / "shape_descriptors.py")
    pkg.shape_descriptors = sd  # type: ignore[attr-defined]
    v2 = _load("modelexpress.nemo_rl_v2", _PKG_ROOT / "nemo_rl_v2.py")
    return v2


def _make_receiver_with_buffers(v2, *, agent_name="rcv-test", worker_rank=0):
    """Build a fully-initialized MxV2RefitReceiver with mocked deps and one
    registered tensor descriptor, ready for ``publish_self_as_source``."""
    rcv = v2.MxV2RefitReceiver(
        agent_name=agent_name,
        device_id=0,
        mx_server_url="mx-stub:8001",
        worker_rank=worker_rank,
    )
    rcv.initialize(model_tensors={"foo": object()})

    # Underlying v1 receiver has _client / _nixl; we replace _client with a
    # MagicMock that records publish_metadata calls and seed _nixl with the
    # attributes publish_self_as_source reads (tensor_descriptors,
    # nixl_metadata).
    rcv._receiver._client = MagicMock()
    rcv._receiver._client.publish_metadata.return_value = "replica-sid-xyz"

    fake_descriptor = types.SimpleNamespace(
        name="foo", addr=0xdeadbeef, size=1024,
        device_id=0, dtype="bfloat16",
    )
    rcv._receiver._nixl = MagicMock()
    rcv._receiver._nixl.tensor_descriptors = [fake_descriptor]
    rcv._receiver._nixl.nixl_metadata = b"fake-nixl-md"

    # MxV2RefitReceiver.initialize sets _registered_buffers; the gate in
    # publish_self_as_source needs it truthy.
    rcv._registered_buffers = {"foo": object()}
    return rcv


# ---------------------------------------------------------------------------
# Bug A — _worker_id assignment
# ---------------------------------------------------------------------------


def test_underlying_receiver_has_worker_id_after_init(v2):
    """The stubbed MxRefitReceiver mirrors the post-fix invariant: a stable
    ``_worker_id`` is set on construction (no separate initialize call
    required). The real ``MxRefitReceiver`` does the same."""
    rcv = v2.MxV2RefitReceiver(
        agent_name="bug-a", device_id=0, mx_server_url="mx-stub:8001",
        worker_rank=0,
    )
    inner = rcv._receiver
    assert hasattr(inner, "_worker_id"), \
        "Bug A regressed: MxRefitReceiver.__init__ no longer assigns _worker_id"
    assert inner._worker_id, \
        "Bug A regressed: _worker_id is empty (server rejects empty worker_id)"
    assert inner._worker_id.startswith("bug-a-"), \
        "_worker_id should be prefixed with agent_name for traceability"


def test_worker_id_passed_through_to_publish_metadata(v2):
    """``publish_self_as_source`` forwards the receiver's ``_worker_id``
    verbatim to ``client.publish_metadata``. This is the call site that
    used to forward an empty string when ``hasattr(...) == False``."""
    rcv = _make_receiver_with_buffers(v2, agent_name="bug-a-pass")
    sid = rcv.publish_self_as_source(version=7, model_name="m")
    assert sid == "replica-sid-xyz"
    kwargs = rcv._receiver._client.publish_metadata.call_args.kwargs
    assert kwargs["worker_id"], \
        "publish_self_as_source forwarded an empty worker_id"
    assert kwargs["worker_id"] == rcv._receiver._worker_id


# ---------------------------------------------------------------------------
# Bug B — three v2-metadata transports
# ---------------------------------------------------------------------------


def test_emits_identity_extra_parameters_transport(v2):
    """Transport (1): identity.extra_parameters carries the v2 markers."""
    rcv = _make_receiver_with_buffers(v2, worker_rank=3)
    rcv.publish_self_as_source(version=11, model_name="my-model")
    kwargs = rcv._receiver._client.publish_metadata.call_args.kwargs
    ep = kwargs["identity"].extra_parameters
    assert ep["mx_v2"] == "1"
    assert ep["role"] == v2.ROLE_INFERENCE_REPLICA
    assert ep["worker_rank"] == "3"
    assert ep["training_step"] == "11"
    assert kwargs["identity"].model_name == "my-model"


def test_emits_v2_sidecar_tensor_descriptor(v2):
    """Transport (2): worker.tensors contains a __mx_v2_meta__ descriptor
    whose dtype field carries the v2 metadata as a JSON blob. This
    survives the older Rust server's identity-extras strip."""
    rcv = _make_receiver_with_buffers(v2, worker_rank=2)
    rcv.publish_self_as_source(version=42, model_name="m")
    kwargs = rcv._receiver._client.publish_metadata.call_args.kwargs
    sidecars = [
        t for t in kwargs["worker"].tensors
        if t.name == v2._V2_SIDECAR_NAME
    ]
    assert len(sidecars) == 1, "expected exactly one __mx_v2_meta__ sidecar"
    sidecar = sidecars[0]
    # Sidecar is a marker descriptor; addr/size/device_id are zero by
    # convention (the data lives in the dtype JSON).
    assert sidecar.size == 0
    assert sidecar.addr == 0
    payload = json.loads(sidecar.dtype)
    assert payload["mx_v2"] == "1"
    assert payload["role"] == v2.ROLE_INFERENCE_REPLICA
    assert payload["worker_rank"] == 2
    assert payload["training_step"] == 42
    assert payload["framework"] == "nemo_rl"


def test_emits_v2_agent_name_marker_transport(v2):
    """Transport (3): worker.agent_name starts with mx_v2|<role>|... so a
    discover_v2_sources fallback path can still parse it even if both
    the identity and the sidecar are stripped."""
    rcv = _make_receiver_with_buffers(v2, agent_name="rcv-7", worker_rank=7)
    rcv.publish_self_as_source(version=5, model_name="m")
    kwargs = rcv._receiver._client.publish_metadata.call_args.kwargs
    agent = kwargs["worker"].agent_name
    assert agent.startswith(f"mx_v2|{v2.ROLE_INFERENCE_REPLICA}|"), \
        f"agent_name not prefixed with mx_v2 marker: {agent!r}"
    assert "rank=7" in agent
    assert "version=5" in agent
    assert "orig=rcv-7" in agent


def test_payload_tensors_include_registered_buffers(v2):
    """Sanity: alongside the sidecar, the regular registered tensors are
    still present so receivers can actually pull them via NIXL."""
    rcv = _make_receiver_with_buffers(v2)
    rcv.publish_self_as_source(version=1, model_name="m")
    tensors = rcv._receiver._client.publish_metadata.call_args.kwargs["worker"].tensors
    names = [t.name for t in tensors]
    assert "foo" in names, "registered buffer 'foo' missing from worker.tensors"
    assert v2._V2_SIDECAR_NAME in names
    # exactly one of each
    assert names.count("foo") == 1
    assert names.count(v2._V2_SIDECAR_NAME) == 1


# ---------------------------------------------------------------------------
# Error handling — silent failures should not regress
# ---------------------------------------------------------------------------


def test_publish_failure_propagates(v2):
    """Server-side failure (non-empty worker_id but the server rejects for
    any other reason) re-raises rather than silently returning None.
    Earlier versions swallowed the exception, which was the reason
    Bug B took so long to find — every caller saw a clean return value
    while the catalog stayed empty."""
    rcv = _make_receiver_with_buffers(v2)
    rcv._receiver._client.publish_metadata.side_effect = RuntimeError(
        "PublishMetadata failed: simulated server-side rejection"
    )
    with pytest.raises(RuntimeError, match="simulated server-side rejection"):
        rcv.publish_self_as_source(version=1, model_name="m")


def test_skips_when_no_registered_buffers(v2):
    """If the receiver has no buffers (e.g. ``initialize(model_tensors=None)``),
    publish_self_as_source returns None without contacting the server.
    This is the "skipped" case, distinct from "errored"."""
    rcv = _make_receiver_with_buffers(v2)
    rcv._registered_buffers = None
    sid = rcv.publish_self_as_source(version=1, model_name="m")
    assert sid is None
    rcv._receiver._client.publish_metadata.assert_not_called()
