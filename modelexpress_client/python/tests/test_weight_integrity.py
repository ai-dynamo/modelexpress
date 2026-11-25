# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import struct
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from modelexpress import p2p_pb2
from modelexpress.adapter import StrategyFailed
from modelexpress.load_strategy import base, rdma_strategy
from modelexpress.load_strategy.context import LoadResult
from modelexpress.load_strategy.rdma_strategy import RdmaStrategy
from modelexpress.metadata import weight_integrity
from modelexpress.metadata.worker_server import (
    WorkerGrpcServer,
    fetch_tensor_manifest_response,
)
from modelexpress.types import TensorDescriptor


def _summary(scope="main", **kwargs):
    fields = {
        "scope": scope,
        "format_version": 1,
        "digest": b"x" * 32,
        "generation": "generation",
    }
    return p2p_pb2.WeightDigest(**(fields | kwargs))


def _context(**kwargs):
    fields = {
        "p2p_role": "main",
        "p2p_enabled": True,
        "require_exact_tensor_catalog": False,
        "is_reload": False,
        "desired_version_uid": None,
        "worker_id": "target",
        "worker_rank": 0,
        "global_rank": 0,
        "device_id": 0,
        "mx_client": Mock(REQUIRES_P2P_METADATA=False),
        "mx_server_url": "mx:8001",
        "identity": p2p_pb2.SourceIdentity(
            mx_source_type=p2p_pb2.MX_SOURCE_TYPE_WEIGHTS,
            backend_framework=p2p_pb2.BACKEND_FRAMEWORK_SGLANG,
        ),
        "accelerator_backend": Mock(),
        "source_ready_fn": None,
        "weight_load_from_legacy": False,
        "verified_weight_digest": None,
        "nixl_manager": Mock(),
        "tensors": {},
        "draft_published": False,
        "draft_tensor_namespace": "mx_draft::",
    }
    ctx = SimpleNamespace(**(fields | kwargs))
    ctx.accelerator_backend.name = "cuda"
    return ctx


def test_worker_manifest_publishes_draft_tensors_and_digest_together():
    server = WorkerGrpcServer(
        [p2p_pb2.TensorDescriptor(name="main.w")],
        "source",
        worker_id="worker",
        weight_digests=(_summary(),),
    )
    port = server.start()
    try:

        def fetch():
            return fetch_tensor_manifest_response(
                f"localhost:{port}", "source", worker_id="worker"
            )

        assert [entry.scope for entry in fetch().weight_digests] == ["main"]
        draft = [p2p_pb2.TensorDescriptor(name="mx_draft::w")]
        server.extend_tensors(draft, (_summary("mx_draft::"),))
        with pytest.raises(ValueError, match="already published"):
            server.extend_tensors(draft, (_summary("mx_draft::"),))
        manifest = fetch()
        assert [entry.name for entry in manifest.tensors] == ["main.w", "mx_draft::w"]
        assert [entry.scope for entry in manifest.weight_digests] == [
            "main",
            "mx_draft::",
        ]
    finally:
        server.stop(grace=0)


def test_manifest_accepts_absent_scope_and_rejects_invalid_present_digest():
    ctx = _context()
    manifest = p2p_pb2.GetTensorManifestResponse(
        mx_source_id="source",
        worker_id="worker",
        agent_name="agent",
        metadata_endpoint="host:26155",
        weight_digests=[_summary("mx_draft::")],
    )
    assert weight_integrity.manifest_digest(manifest, ctx, "source", "worker") is None
    manifest.weight_digests.append(_summary(format_version=2))
    with pytest.raises(ValueError, match="unsupported"):
        weight_integrity.manifest_digest(manifest, ctx, "source", "worker")
    manifest.weight_digests[-1].format_version = 1
    with pytest.raises(ValueError, match="identity"):
        weight_integrity.manifest_digest(manifest, ctx, "source", "replaced_worker")
    assert (
        weight_integrity.manifest_digest(manifest, ctx, "source", "worker").scope
        == "main"
    )


def test_transfer_requires_exact_nonempty_registered_ranges():
    tensor = torch.arange(4, dtype=torch.int32)[1:]
    descriptor = TensorDescriptor("w", tensor.data_ptr(), 12, 0, "torch.int32")
    ctx = _context(tensors={"w": tensor})
    ctx.nixl_manager.tensor_descriptors = [descriptor]
    registered = weight_integrity.registered_tensors(ctx)
    assert weight_integrity.validate_transfer([descriptor], registered) == [descriptor]
    with pytest.raises(ValueError, match="range mismatch"):
        weight_integrity.validate_transfer([], registered)
    with pytest.raises(ValueError, match="duplicate"):
        weight_integrity.validate_transfer([descriptor, descriptor], registered)
    ctx.tensors["w"] = tensor.clone()
    with pytest.raises(ValueError, match="registration"):
        weight_integrity.registered_tensors(ctx)


def test_source_digest_failure_skips_only_publication(monkeypatch):
    monkeypatch.setenv("MX_WEIGHT_INTEGRITY", "1")
    monkeypatch.setattr(
        base, "source_digest", Mock(side_effect=RuntimeError("GPU hash failed"))
    )
    publish = Mock()
    monkeypatch.setattr(base, "publish_metadata_and_ready", publish)
    ctx = _context()
    model = torch.nn.Module()
    result = LoadResult(value=model, model=model)
    base.publish_source_if_supported(result, ctx)
    publish.assert_not_called()
    assert result.value is model
    ctx.weight_load_from_legacy = True
    base.publish_source_if_supported(result, ctx)
    publish.assert_called_once()
    assert "weight_digest" not in publish.call_args.kwargs


def test_mismatch_rebuilds_target_before_trying_a_legacy_source(monkeypatch):
    strategy = RdmaStrategy()
    ctx = _context(adapter=Mock())
    ctx.adapter.all_gather_state.side_effect = lambda status: (status,)
    first, second = [
        p2p_pb2.SourceInstanceRef(worker_id=name) for name in ("first", "second")
    ]
    workers = p2p_pb2.WorkerMetadata(accelerator="cuda")
    fetch = Mock(side_effect=[(workers, _summary()), (workers, None)])
    monkeypatch.setattr(strategy, "_fetch_worker_snapshot", fetch)
    result = LoadResult(value=object())
    rebuilt = LoadResult(value=object())
    ctx.adapter.reinit_for_retry.return_value = rebuilt
    transfer = Mock(
        side_effect=[
            StrategyFailed("RDMA weight digest mismatch", mutated=True),
            result,
        ]
    )
    monkeypatch.setattr(strategy, "_load_as_target", transfer)

    assert strategy._load_checked(result, ctx, [first, second]).value is rebuilt.value
    ctx.adapter.reinit_for_retry.assert_called_once()
    assert ctx.nixl_manager is None
    assert transfer.call_args_list[0].kwargs["weight_digest"] == _summary()
    assert transfer.call_args_list[1].kwargs["weight_digest"] is None
    assert ctx.weight_load_from_legacy


@pytest.mark.parametrize("corrupt", [False, True])
def test_rdma_verifies_bytes_after_read_before_postprocessing(monkeypatch, corrupt):
    source = torch.arange(24, dtype=torch.uint8)
    target = torch.empty_like(source)
    local = TensorDescriptor("w", target.data_ptr(), 24, 0, "torch.uint8")
    remote = p2p_pb2.TensorDescriptor(name="w", addr=4096, size=24, dtype="torch.uint8")
    worker = p2p_pb2.WorkerMetadata(
        accelerator="cuda",
        metadata_endpoint="host:26155",
        agent_name="agent",
        worker_grpc_endpoint="host:6555",
        tensor_source=p2p_pb2.TensorSourceMetadata(tensors=[remote]),
    )
    ctx = _context(tensors={"w": target}, adapter=Mock())
    ctx.nixl_manager.tensor_descriptors = [local]
    result = LoadResult(value=object())
    ctx.adapter.prepare_rdma_target.return_value = result
    ctx.adapter.before_rdma_receive.return_value = result
    events = []

    def receive(**kwargs):
        events.append("read")
        dest = kwargs["destination_tensors"]["w"]
        dest.copy_(source)
        if corrupt:
            dest[0] ^= 1
        return 24, 1, 0.01

    def digest(tensors, rank, scope):
        events.append("verify")
        return _cpu_reference({"w": bytes(tensors["w"])}, rank, scope)

    ctx.nixl_manager.receive_from_source.side_effect = receive
    ctx.accelerator_backend.synchronize.side_effect = lambda: events.append(
        "synchronize"
    )
    ctx.adapter.after_rdma_receive.side_effect = lambda loaded: (
        events.append("postprocess") or loaded
    )
    monkeypatch.setattr(rdma_strategy, "register_tensors", Mock())
    monkeypatch.setattr(weight_integrity, "content_digest", digest)

    strategy = RdmaStrategy()
    expected = _summary(digest=_cpu_reference({"w": bytes(source)}, 0, "main"))
    if corrupt:
        with pytest.raises(StrategyFailed, match="RDMA weight digest mismatch") as error:
            strategy._load_as_target(
                result, ctx, worker, "source", "worker", weight_digest=expected
            )
        assert error.value.mutated
        assert events == ["read", "synchronize", "verify"]
    else:
        assert strategy._load_as_target(
            result, ctx, worker, "source", "worker", weight_digest=expected
        ) is result
        assert events == ["read", "synchronize", "verify", "postprocess"]
    transfer = ctx.nixl_manager.receive_from_source.call_args.kwargs
    assert transfer["require_exact_match"]
    assert transfer["destination_tensors"]["w"] is target
    assert transfer["source_tensors"][0].addr == remote.addr

    events.clear()
    worker.tensor_source.tensors[0].size = 12
    with pytest.raises(StrategyFailed, match="incompatible source tensor") as error:
        strategy._load_as_target(
            result, ctx, worker, "source", "worker", weight_digest=expected
        )
    assert error.value.mutated
    assert events == []


def test_peer_failure_rejects_a_locally_successful_receive():
    ctx = _context(adapter=Mock())
    ctx.adapter.all_gather_state.side_effect = lambda state: (
        state, (state[0], ("digest mismatch", False, True)),
    )
    with (
        pytest.raises(StrategyFailed, match="digest mismatch") as error,
        weight_integrity.agree_load_phase(ctx, True, "RDMA receive"),
    ):
        pass
    assert error.value.mutated


def _cpu_reference(payloads, rank, scope):
    def encode(value):
        encoded = value.encode()
        return struct.pack(">I", len(encoded)) + encoded

    leaves = []
    for name, data in sorted(payloads.items(), key=lambda pair: pair[0].encode()):
        prefix = (
            struct.pack(">III", 0, 1, rank)
            + encode(scope)
            + encode(name)
            + struct.pack(">Q", len(data))
        )
        prefix = prefix.ljust((len(prefix) + 63) // 64 * 64, b"\0")
        for offset in range(0, len(data), 16 * 1024):
            chunk = data[offset : offset + 16 * 1024]
            header = struct.pack(">QI", offset, len(chunk)).ljust(64, b"\0")
            leaves.append(hashlib.sha256(prefix + header + chunk).digest())
    while len(leaves) > 1:
        leaves = [
            hashlib.sha256(b"\0\0\0\1" + left + right).digest()
            for left, right in zip(leaves[::2], leaves[1::2])
        ] + (leaves[-1:] if len(leaves) % 2 else [])
    return leaves[0]


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required to execute GPU digests"
)
@pytest.mark.parametrize(
    "length", [1, 55, 56, 63, 64, 65, 16 * 1024 - 1, 16 * 1024, 16 * 1024 + 1]
)
def test_gpu_digest_matches_sha256_reference(length):
    pytest.importorskip("triton")
    payload = bytes((index * 13) % 256 for index in range(length))
    payloads = {"weight.层": payload, "an.aligned.weight": b"abc"}
    tensor = torch.frombuffer(bytearray(b"!" + payload), dtype=torch.uint8).to("cuda")[
        1:
    ]
    aligned = torch.tensor([97, 98, 99], dtype=torch.uint8, device="cuda")
    tensors = {"weight.层": tensor, "an.aligned.weight": aligned}
    expected = _cpu_reference(payloads, 2, "mx_draft::")
    assert weight_integrity.content_digest(tensors, 2, "mx_draft::") == expected
    tensor[0] ^= 1
    assert weight_integrity.content_digest(tensors, 2, "mx_draft::") != expected
