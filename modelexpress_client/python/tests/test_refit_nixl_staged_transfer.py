# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext

import modelexpress_rl.inference.nixl_staged_transfer as transfer_module
import pytest
import torch
from modelexpress.refit.reshard.rendezvous import (
    PublishedShard,
    PublishedTensor,
    wrap_rendezvous_blob,
)
from modelexpress.refit.reshard.slice_plan import PullSegment, Shard
from modelexpress.refit.reshard.transfer_plan import SourceInfo, TransferPlan
from modelexpress.refit.reshard.types import (
    CaptureResult,
    IncompleteRefit,
    RecordedCopy,
)
from modelexpress.refit.reshard.verify import tensor_digest
from modelexpress_rl.inference.nixl_staged_transfer import (
    _load_agent_metadata,
    _NixlStagedTransfer,
    _plan_staged_transfer,
    _PreparedNixlTransfer,
    _required_agent_metadata,
    _resolve_sources,
    _ResolvedSources,
)


def _manifest(*, agent_name: str, endpoint: str, offset: int, address: int) -> bytes:
    return wrap_rendezvous_blob(
        b"nixl-metadata",
        agent_name,
        endpoint,
        [
            PublishedTensor(
                name="weight",
                dtype="torch.float32",
                elsize=4,
                full_shape=(4,),
                shards=[
                    PublishedShard(
                        agent_name=agent_name,
                        device_id=0,
                        addr=address,
                        shard_offset=(offset,),
                        shape=(2,),
                    )
                ],
            )
        ],
    )


def test_exact_manifests_resolve_without_legacy_source_discovery():
    resolved = _resolve_sources(
        [
            _manifest(
                agent_name="trainer-0",
                endpoint="trainer-0:19000",
                offset=0,
                address=100,
            ),
            _manifest(
                agent_name="trainer-1",
                endpoint="trainer-1:19001",
                offset=2,
                address=200,
            ),
        ]
    )

    assert resolved.sources["weight"].global_shape == (4,)
    assert [shard.addr for shard in resolved.sources["weight"].shards] == [100, 200]
    assert resolved.session_to_agent == {
        "trainer-0": "trainer-0",
        "trainer-1": "trainer-1",
    }
    assert resolved.agent_metadata == {
        "trainer-0": b"nixl-metadata",
        "trainer-1": b"nixl-metadata",
    }


def test_required_agent_metadata_rejects_incomplete_source_metadata():
    plan = TransferPlan(segments=[PullSegment("session-a", 1, "weight", 0, 4)])
    resolved = _ResolvedSources(
        sources={},
        session_to_agent={"session-a": "agent-a"},
        session_to_device={},
        agent_metadata={"agent-a": b"metadata"},
    )
    assert _required_agent_metadata(plan, resolved) == {"agent-a": b"metadata"}

    with pytest.raises(RuntimeError, match="unknown source sessions"):
        _required_agent_metadata(
            plan,
            _ResolvedSources({}, {}, {}, {}),
        )
    with pytest.raises(RuntimeError, match="without NIXL metadata"):
        _required_agent_metadata(
            plan,
            _ResolvedSources({}, {"session-a": "agent-a"}, {}, {}),
        )


def test_load_agent_metadata_validates_embedded_agent_identity():
    calls = []

    class _Manager:
        def add_remote_agent(self, metadata):
            calls.append(metadata)
            return b"agent-a"

    _load_agent_metadata(_Manager(), {"agent-a": b"metadata"})
    assert calls == [b"metadata"]

    with pytest.raises(RuntimeError, match="does not match its manifest"):
        _load_agent_metadata(_Manager(), {"agent-b": b"metadata"})


def test_transformed_source_is_fully_reconstructed_for_verification(monkeypatch):
    # Full reconstruction + verification is the digest mode; default is minimal reads.
    monkeypatch.setenv("MX_RESHARD_PUBLISH_DIGEST", "1")
    source = SourceInfo(
        global_shape=(4, 4),
        dtype=torch.float32,
        elsize=4,
        shards=[
            Shard((0, 0), (4, 2), "left", 0, 4),
            Shard((0, 2), (4, 2), "right", 32, 4),
        ],
    )
    copy = RecordedCopy(
        src_name="weight",
        op_chain=(("narrow", (1, 0, 2), ()),),
        param_name="fused_weight",
        dest_offset=0,
        dest_shape=(4, 2),
        dest_stride=(2, 1),
        dest_dtype=torch.float32,
    )

    plan = _plan_staged_transfer(CaptureResult(copies=[copy]), {"weight": source})

    assert plan.segments == []
    assert len(plan.full_pulls) == 1
    assert plan.full_pulls[0].copies == [copy]
    assert sum(segment.nbytes for segment in plan.full_pulls[0].segments) == 64
    assert {segment.session for segment in plan.full_pulls[0].segments} == {
        "left",
        "right",
    }


def _prepared(tensor: torch.Tensor, digest: str | None) -> _PreparedNixlTransfer:
    copy = RecordedCopy(
        src_name="weight",
        op_chain=(),
        param_name="weight",
        dest_offset=0,
        dest_shape=tuple(tensor.shape),
        dest_stride=tuple(tensor.stride()),
        dest_dtype=tensor.dtype,
    )
    source = SourceInfo(
        global_shape=tuple(tensor.shape),
        dtype=tensor.dtype,
        elsize=tensor.element_size(),
        shards=[
            Shard(
                shard_offset=(0,),
                shape=tuple(tensor.shape),
                session="trainer",
                addr=0,
                elsize=tensor.element_size(),
                digest=digest,
            )
        ],
    )
    return _PreparedNixlTransfer(
        plan=TransferPlan(),
        capture=CaptureResult(copies=[copy]),
        sources={"weight": source},
        descriptors=(),
        transport=object(),
        plan_revision=1,
    )


def test_staged_verification_rejects_missing_or_mismatched_digest():
    tensor = torch.arange(64, dtype=torch.int32)
    transfer = object.__new__(_NixlStagedTransfer)
    transfer._recv_buffers = {"weight": tensor}
    transfer._convert_buffers = {}
    transfer._full_buffers = {}

    transfer._verify(_prepared(tensor, tensor_digest(tensor)))
    with pytest.raises(RuntimeError, match="digest mismatch"):
        transfer._verify(_prepared(tensor, tensor_digest(tensor + 1)))
    with pytest.raises(RuntimeError, match="did not publish"):
        transfer._verify(_prepared(tensor, None))


def test_full_tensor_plan_fails_before_transfer_when_capture_has_holes():
    capture = CaptureResult(copies=[])
    with pytest.raises(IncompleteRefit, match="must cover every engine parameter"):
        _NixlStagedTransfer._validate_complete(
            capture,
            {"weight": ((4,), torch.float32)},
            TransferPlan(),
        )


def test_transfer_manager_is_closed_after_failed_init_and_only_once(monkeypatch):
    calls = []

    class _Manager:
        def __init__(self, **_kwargs):
            pass

        def initialize(self):
            calls.append("initialize")
            raise RuntimeError("init failed")

        def shutdown(self):
            calls.append("shutdown")

    monkeypatch.setattr(transfer_module, "NixlTransferManager", _Manager)
    with pytest.raises(RuntimeError, match="init failed"):
        _NixlStagedTransfer(
            agent_name="generator",
            device_id=0,
            device=torch.device("cpu"),
        )
    assert calls == ["initialize", "shutdown"]

    transfer = object.__new__(_NixlStagedTransfer)
    transfer._manager = _Manager()
    transfer._closed = False
    transfer.close()
    transfer.close()
    assert calls == ["initialize", "shutdown", "shutdown"]


def test_registered_workspace_is_reused_only_for_the_same_layout(monkeypatch):
    monkeypatch.setattr(transfer_module, "classic_cuda_alloc", nullcontext)
    transfer = object.__new__(_NixlStagedTransfer)
    transfer._device = torch.device("cpu")
    buffers = {}
    layout = {"weight": ((4,), torch.float32)}

    transfer._ensure_buffers(buffers, layout, label="receive-buffer")
    pointer = buffers["weight"].data_ptr()
    transfer._ensure_buffers(buffers, layout, label="receive-buffer")
    assert buffers["weight"].data_ptr() == pointer

    with pytest.raises(RuntimeError, match="layout changed"):
        transfer._ensure_buffers(
            buffers,
            {"weight": ((8,), torch.float32)},
            label="receive-buffer",
        )
