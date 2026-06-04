# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import struct

from modelexpress.resharding import QuantizationScope
from modelexpress.resharding_manifest import (
    classify_qwen_moe_tensor,
    extract_qwen_moe_manifest,
    extract_qwen_moe_manifest_from_safetensors_files,
    extract_qwen_moe_manifest_from_safetensors_metadata,
    manifest_coverage_summary,
    read_safetensors_header,
    safetensors_header_to_tensor_metadata,
    write_manifest_coverage_artifact,
    write_manifest_artifact,
)


def test_qwen_moe_expert_tensor_is_marked_expert_axis_shard():
    entry = classify_qwen_moe_tensor(
        "model.layers.0.mlp.experts.w1.weight",
        {"shape": (8, 4096, 14336), "dtype": "torch.bfloat16"},
        model_name="qwen3-moe",
        model_version="step-1",
    )

    assert entry.tensor_family == "moe-expert-axis-shard"
    assert entry.quantization_scope == QuantizationScope.ABSENT
    assert entry.layout_tags["moe_expert_axis"] == 0
    assert entry.layout_tags["num_experts"] == 8
    assert entry.requires_special_handling is True


def test_qwen_per_expert_tensor_is_not_labeled_as_shape_expert_axis():
    entry = classify_qwen_moe_tensor(
        "model.layers.0.mlp.experts.3.gate_proj.weight",
        {"shape": (14336, 4096), "dtype": "torch.bfloat16"},
        model_name="qwen3-moe",
        model_version="step-1",
    )

    assert entry.tensor_family == "moe-expert-axis-shard"
    assert entry.layout_tags["expert_storage"] == "per-expert-tensor"
    assert entry.layout_tags["moe_expert_index"] == 3
    assert "moe_expert_axis" not in entry.layout_tags
    assert "num_experts" not in entry.layout_tags


def test_qwen_quantization_metadata_requires_global_scope():
    entry = classify_qwen_moe_tensor(
        "model.layers.0.mlp.experts.w1.weight_scale_inv",
        {"shape": (8, 1), "dtype": "float32"},
        model_name="qwen3-moe",
        model_version="step-1",
    )

    assert entry.quantization_scope == QuantizationScope.GLOBAL_REQUIRED
    assert entry.tensor_family == "quantization-global-required-fallback"
    assert entry.layout_tags["tensor_role"] == "expert-quant-metadata"
    assert "global quantization metadata" in entry.reason


def test_qwen_per_expert_quantization_metadata_preserves_expert_index():
    entry = classify_qwen_moe_tensor(
        "model.layers.0.mlp.experts.127.down_proj.weight_scale_inv",
        {"shape": (1,), "dtype": "float32"},
        model_name="qwen3-moe-fp8",
        model_version="step-1",
    )

    assert entry.quantization_scope == QuantizationScope.GLOBAL_REQUIRED
    assert entry.tensor_family == "quantization-global-required-fallback"
    assert entry.layout_tags["expert_storage"] == "per-expert-tensor"
    assert entry.layout_tags["moe_expert_index"] == 127
    assert entry.layout_tags["tensor_role"] == "expert-quant-metadata"


def test_qwen_generated_tensor_is_classified_for_target_generation():
    entry = classify_qwen_moe_tensor(
        "model.rotary_emb.inv_freq",
        {"shape": (128,), "dtype": "float32"},
        model_name="qwen3-moe",
        model_version="step-1",
    )

    assert entry.quantization_scope == QuantizationScope.GENERATED_ON_TARGET
    assert entry.tensor_family == "generated-on-target"
    assert entry.requires_special_handling is True


def test_extract_qwen_moe_manifest_and_write_artifact(tmp_path):
    manifest = extract_qwen_moe_manifest(
        {
            "model.layers.0.mlp.experts.w2.weight": {
                "shape": (8, 14336, 4096),
                "dtype": "bfloat16",
            },
            "model.layers.0.mlp.shared_expert.gate_proj.weight": {
                "shape": (4096, 14336),
                "dtype": "bfloat16",
            },
            "model.rotary_emb.inv_freq": {
                "shape": (128,),
                "dtype": "float32",
            },
        },
        model_name="qwen3-moe",
        model_version="step-1",
    )

    assert [entry.tensor_name for entry in manifest] == sorted(
        entry.tensor_name for entry in manifest
    )
    assert {
        entry.tensor_name: entry.tensor_family for entry in manifest
    } == {
        "model.layers.0.mlp.experts.w2.weight": "moe-expert-axis-shard",
        "model.layers.0.mlp.shared_expert.gate_proj.weight": "layout-sensitive-slice",
        "model.rotary_emb.inv_freq": "generated-on-target",
    }

    artifact = tmp_path / "qwen-manifest.json"
    write_manifest_artifact(manifest, artifact)
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    assert payload[0]["model_name"] == "qwen3-moe"
    assert payload[0]["model_version"] == "step-1"


def test_safetensors_header_metadata_is_classified_without_payload():
    header = {
        "__metadata__": {"format": "pt"},
        "model.layers.0.mlp.experts.w1.weight": {
            "dtype": "BF16",
            "shape": [8, 4096, 14336],
            "data_offsets": [0, 128],
        },
        "model.layers.0.mlp.experts.w1.weight_scale_inv": {
            "dtype": "F32",
            "shape": [8, 1],
            "data_offsets": [128, 160],
        },
        "model.rotary_emb.inv_freq": {
            "dtype": "F32",
            "shape": [128],
            "data_offsets": [160, 192],
        },
    }

    metadata = safetensors_header_to_tensor_metadata(
        header,
        source_file="model-00001-of-00002.safetensors",
    )
    manifest = extract_qwen_moe_manifest_from_safetensors_metadata(
        metadata,
        model_name="qwen3-moe",
        model_version="step-1",
    )

    by_name = {entry.tensor_name: entry for entry in manifest}
    assert set(by_name) == {
        "model.layers.0.mlp.experts.w1.weight",
        "model.layers.0.mlp.experts.w1.weight_scale_inv",
        "model.rotary_emb.inv_freq",
    }
    assert by_name["model.layers.0.mlp.experts.w1.weight"].dtype == "bfloat16"
    assert (
        by_name["model.layers.0.mlp.experts.w1.weight"].layout_tags["source_file"]
        == "model-00001-of-00002.safetensors"
    )
    assert (
        by_name["model.layers.0.mlp.experts.w1.weight_scale_inv"].quantization_scope
        == QuantizationScope.GLOBAL_REQUIRED
    )
    assert (
        by_name["model.rotary_emb.inv_freq"].quantization_scope
        == QuantizationScope.GENERATED_ON_TARGET
    )


def test_local_safetensors_header_read_and_coverage_artifact(tmp_path):
    header = {
        "__metadata__": {"format": "pt"},
        "model.layers.1.mlp.experts.w2.weight": {
            "dtype": "BF16",
            "shape": [8, 14336, 4096],
            "data_offsets": [0, 64],
        },
        "model.layers.1.mlp.shared_expert.down_proj.weight": {
            "dtype": "BF16",
            "shape": [14336, 4096],
            "data_offsets": [64, 128],
        },
    }
    header_bytes = json.dumps(header).encode("utf-8")
    safetensors_path = tmp_path / "model-00002-of-00002.safetensors"
    safetensors_path.write_bytes(
        struct.pack("<Q", len(header_bytes)) + header_bytes + (b"\0" * 128)
    )

    assert read_safetensors_header(safetensors_path) == header

    manifest = extract_qwen_moe_manifest_from_safetensors_files(
        [safetensors_path],
        model_name="qwen3-moe",
        model_version="step-1",
    )
    summary = manifest_coverage_summary(manifest)

    assert summary["tensor_count"] == 2
    assert summary["tensor_family_counts"] == {
        "layout-sensitive-slice": 1,
        "moe-expert-axis-shard": 1,
    }
    assert summary["source_file_counts"] == {str(safetensors_path): 2}

    artifact = tmp_path / "qwen-coverage.json"
    write_manifest_coverage_artifact(
        manifest,
        artifact,
        source={"kind": "local-safetensors"},
    )
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    assert payload["source"] == {"kind": "local-safetensors"}
    assert payload["summary"]["tensor_count"] == 2
    assert len(payload["entries"]) == 2
