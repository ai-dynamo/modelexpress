# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused CPU checks for the native MILES broadcast transport microbench."""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).parent
SPEC = importlib.util.spec_from_file_location(
    "native_broadcast_microbench_under_test",
    HERE / "native_broadcast_microbench.py",
)
assert SPEC is not None and SPEC.loader is not None
microbench = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = microbench
SPEC.loader.exec_module(microbench)


def test_qwen_3b_manifest_matches_gathered_production_payload():
    manifest = microbench.qwen_25_3b_manifest()

    assert len(manifest) == 434
    assert sum(item.bytes for item in manifest) == 6_171_877_376
    assert manifest[0] == microbench.TensorSpec(
        "model.norm.weight",
        (2_048,),
        None,
        "decoder.final_layernorm.weight",
    )
    assert manifest[1] == microbench.TensorSpec(
        "model.embed_tokens.weight",
        (151_936, 2_048),
        None,
        "embedding.word_embeddings.weight",
    )
    assert [(item.name, item.shape) for item in manifest if item.layer == 0] == [
        ("model.layers.0.post_attention_layernorm.weight", (2_048,)),
        ("model.layers.0.mlp.gate_proj.weight", (11_008, 2_048)),
        ("model.layers.0.mlp.up_proj.weight", (11_008, 2_048)),
        ("model.layers.0.mlp.down_proj.weight", (2_048, 11_008)),
        ("model.layers.0.self_attn.o_proj.weight", (2_048, 2_048)),
        ("model.layers.0.self_attn.q_proj.bias", (2_048,)),
        ("model.layers.0.self_attn.k_proj.bias", (256,)),
        ("model.layers.0.self_attn.v_proj.bias", (256,)),
        ("model.layers.0.input_layernorm.weight", (2_048,)),
        ("model.layers.0.self_attn.q_proj.weight", (2_048, 2_048)),
        ("model.layers.0.self_attn.k_proj.weight", (256, 2_048)),
        ("model.layers.0.self_attn.v_proj.weight", (256, 2_048)),
    ]
    microbench.validate_manifest(manifest)


def test_topology_is_one_gathered_source_and_four_receivers():
    topology = microbench.Topology()

    assert topology.source_rank == 0
    assert topology.receiver_ranks == (1, 2, 3, 4)
    assert topology.group_ranks == (0, 1, 2, 3, 4)
    assert topology.world_size == 5
    topology.validate(rank=4, world_size=5)

    with pytest.raises(ValueError, match="world size 5"):
        topology.validate(rank=0, world_size=8)
    with pytest.raises(ValueError, match="outside"):
        topology.validate(rank=5, world_size=5)


def test_default_packing_matches_production_source_units_and_order():
    manifest = microbench.qwen_25_3b_manifest()
    buckets = microbench.pack_manifest(manifest, microbench.DEFAULT_BUCKET_BYTES)

    assert len(buckets) == 14
    assert [len(bucket) for bucket in buckets] == [
        1,
        1,
        37,
        36,
        36,
        36,
        36,
        36,
        36,
        36,
        36,
        36,
        36,
        35,
    ]
    assert [item.name for bucket in buckets for item in bucket] == [
        item.name for item in manifest
    ]
    source_unit_buckets: dict[str, set[int]] = {}
    for bucket_index, bucket in enumerate(buckets):
        for item in bucket:
            source_unit_buckets.setdefault(item.source_unit, set()).add(bucket_index)
    assert all(
        len(bucket_indexes) == 1 for bucket_indexes in source_unit_buckets.values()
    )


def test_validating_client_checks_exact_metadata():
    manifest = microbench.qwen_25_3b_manifest()[:2]
    expected_names = [item.name for item in manifest]
    expected_dtypes = ["bf16", "bf16"]
    expected_shapes = [item.shape for item in manifest]
    client = microbench.ValidatingMetadataClient(
        client_id="receiver-1",
        expected_names=expected_names,
        expected_dtypes=expected_dtypes,
        expected_shapes=expected_shapes,
    )

    asyncio.run(
        client.update_weights_from_distributed(
            names=expected_names,
            dtypes=expected_dtypes,
            shapes=expected_shapes,
            selector="all",
            group_name="miles-pp_0",
        )
    )

    assert client.calls == 1
    rejecting_client = microbench.ValidatingMetadataClient(
        client_id="receiver-1",
        expected_names=expected_names,
        expected_dtypes=expected_dtypes,
        expected_shapes=expected_shapes,
    )
    with pytest.raises(AssertionError, match="names"):
        asyncio.run(
            rejecting_client.update_weights_from_distributed(
                names=list(reversed(expected_names)),
                dtypes=expected_dtypes,
                shapes=expected_shapes,
                selector="all",
                group_name="miles-pp_0",
            )
        )


def test_source_invocation_calls_actual_miles_signature_without_reimplementation():
    calls = []
    futures = [object(), object(), object(), object()]

    def broadcast_fn(group_name, group, clients, tensors, selector):
        calls.append((group_name, group, clients, tensors, selector))
        return futures

    clients = [object()] * 4
    tensors = [("weight", object())]

    returned = microbench.invoke_miles_broadcast(
        broadcast_fn=broadcast_fn,
        group="nccl-group",
        clients=clients,
        tensors=tensors,
        group_name="miles-pp_0",
        selector="all",
    )

    assert returned is futures
    assert calls == [
        ("miles-pp_0", "nccl-group", clients, tensors, "all"),
    ]


@pytest.mark.parametrize(
    "path",
    (
        Path("/tmp/result.json"),
        Path("/dev/shm/result.json"),
        Path("/run/result.json"),
        Path("relative/result.json"),
    ),
)
def test_output_path_rejects_known_ephemeral_or_relative_locations(path):
    with pytest.raises(ValueError):
        microbench.validate_output_path(path)


def test_output_path_accepts_absolute_persistent_location():
    assert microbench.validate_output_path(
        Path("/workspace/results/native.json")
    ) == Path("/workspace/results/native.json")
