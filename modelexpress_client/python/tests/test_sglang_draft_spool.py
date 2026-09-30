# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A cold SGLang source can replay only its draft weights without another read."""

import pytest
import torch
from modelexpress import p2p_pb2
from modelexpress.engines.sglang.draft_spool import DraftSpool, SpoolLimitExceeded
from modelexpress.engines.sglang.draft_weights import (
    Qwen35MtpWeights,
    draft_tensor_namespace,
    draft_weight_adapter_for,
)


def test_draft_spool_copies_streamer_buffer_before_it_is_reused(tmp_path):
    source_buffer = torch.tensor([1.0, 2.0])
    spool = DraftSpool(tmp_path, max_bytes=1024, memory_bytes=1024)

    spool.capture("mtp.fc.weight", source_buffer)
    source_buffer.fill_(99)
    spool.finish()

    assert [(name, tensor.tolist()) for name, tensor in spool.replay()] == [
        ("mtp.fc.weight", [1.0, 2.0])
    ]
    spool.close()


def test_draft_spool_spills_to_disk_and_preserves_stream_order(tmp_path):
    spool = DraftSpool(tmp_path, max_bytes=1024, memory_bytes=4)
    spool.capture("mtp.fc.weight", torch.tensor([1.0, 2.0]))
    spool.capture("mtp.layers.0.w", torch.tensor([3.0, 4.0]))
    spool.finish()

    assert [(name, tensor.tolist()) for name, tensor in spool.replay()] == [
        ("mtp.fc.weight", [1.0, 2.0]),
        ("mtp.layers.0.w", [3.0, 4.0]),
    ]
    assert list(spool.path.iterdir())
    spool.close()
    assert not spool.path.exists()


def test_draft_spool_is_unavailable_until_complete(tmp_path):
    spool = DraftSpool(tmp_path, max_bytes=1024, memory_bytes=1024)
    spool.capture("mtp.fc.weight", torch.ones(2))

    with pytest.raises(RuntimeError, match="not complete"):
        list(spool.replay())

    spool.close()


def test_draft_spool_rejects_bytes_over_limit(tmp_path):
    spool = DraftSpool(tmp_path, max_bytes=4, memory_bytes=4)

    with pytest.raises(SpoolLimitExceeded):
        spool.capture("mtp.fc.weight", torch.ones(2))

    spool.close()


def test_draft_spool_cannot_finish_without_draft_tensors(tmp_path):
    spool = DraftSpool(tmp_path, max_bytes=1024, memory_bytes=1024)

    with pytest.raises(RuntimeError, match="no draft tensors"):
        spool.finish()

    spool.close()


def test_qwen35_selector_uses_its_sglang_model_rule_not_a_generic_prefix():
    selector = Qwen35MtpWeights()

    assert selector.supports_main("Qwen3_5MoeForCausalLM")
    assert selector.supports_draft("Qwen3_5ForCausalLMMTP")
    assert selector.includes("mtp.fc.weight")
    assert selector.includes("model.mtp.layers.0.w")
    assert not selector.includes("model.layers.0.w")
    assert not selector.supports_main("DeepseekV3ForCausalLM")


def test_qwen35_shared_weights_are_identified_by_tensor_not_name():
    selector = Qwen35MtpWeights()
    embed = torch.nn.Parameter(torch.ones(2))
    head = torch.nn.Parameter(torch.ones(2))
    own = torch.nn.Parameter(torch.ones(2))

    class Draft(torch.nn.Module):
        def get_embed_and_head(self):
            return embed, head

    tensors = {"renamed.embedding": embed, "renamed.head": head, "layer": own}
    assert selector.transferable_tensors(Draft(), tensors) == {"layer": own}


def test_qwen35_shared_storage_view_is_not_registered_for_peer_transfer():
    selector = Qwen35MtpWeights()
    embed = torch.nn.Parameter(torch.arange(4, dtype=torch.float32).reshape(2, 2))
    head = torch.nn.Parameter(torch.ones(2))

    class Draft(torch.nn.Module):
        def get_embed_and_head(self):
            return embed, head

    raw_storage_view = embed.detach().view(torch.uint8)
    own = torch.ones(2)
    assert selector.transferable_tensors(
        Draft(), {"renamed.embedding.__storage": raw_storage_view, "layer": own}
    ) == {"layer": own}


def test_qwen35_unrelated_empty_tensor_is_not_treated_as_shared():
    selector = Qwen35MtpWeights()
    embed = torch.nn.Parameter(torch.empty(0))
    head = torch.nn.Parameter(torch.empty(0))

    class Draft(torch.nn.Module):
        def get_embed_and_head(self):
            return embed, head

    assert list(selector.transferable_tensors(Draft(), {"other": torch.empty(0)})) == [
        "other"
    ]


def test_unknown_model_has_no_draft_weight_adapter():
    assert isinstance(
        draft_weight_adapter_for("Qwen3_5MoeForCausalLM", role="main"),
        Qwen35MtpWeights,
    )
    assert draft_weight_adapter_for("DeepseekV3ForCausalLM", role="main") is None


def test_draft_namespace_changes_with_revision_or_sglang_version():
    identity = p2p_pb2.SourceIdentity(model_name="org/qwen", revision="rev-a")
    selector = Qwen35MtpWeights()
    one = draft_tensor_namespace(
        identity, selector, "Qwen3_5ForCausalLMMTP", "0.5.16", "s3://bucket/qwen"
    )
    assert one.startswith("mx_draft::") and one.endswith("::")
    identity.revision = "rev-b"
    assert (
        draft_tensor_namespace(
            identity, selector, "Qwen3_5ForCausalLMMTP", "0.5.16", "s3://bucket/qwen"
        )
        != one
    )
    identity.revision = "rev-a"
    assert (
        draft_tensor_namespace(
            identity, selector, "Qwen3_5ForCausalLMMTP", "0.5.17", "s3://bucket/qwen"
        )
        != one
    )


def test_draft_namespace_requires_explicit_revision_and_engine_version():
    identity = p2p_pb2.SourceIdentity(model_name="org/qwen")
    selector = Qwen35MtpWeights()
    assert (
        draft_tensor_namespace(
            identity, selector, "Qwen3_5ForCausalLMMTP", "0.5.16", "s3://bucket/qwen"
        )
        is None
    )
    identity.revision = "rev-a"
    assert (
        draft_tensor_namespace(
            identity, selector, "Qwen3_5ForCausalLMMTP", "", "s3://bucket/qwen"
        )
        is None
    )
