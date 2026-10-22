# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import logging

import pytest

from modelexpress_rl.refit.reshard.rendezvous import (
    PublishedShard,
    PublishedTensor,
    build_sources,
    encode_shard_table,
    decode_shard_table,
    unwrap_rendezvous_blob,
    structural_manifest_digest,
    wrap_rendezvous_blob,
)


def _one_tensor(agent_name="trainer-agent"):
    """The smallest publishable shard table: one tensor, one shard."""
    return [
        PublishedTensor(
            name="weight",
            dtype="torch.bfloat16",
            elsize=2,
            full_shape=(4, 4),
            shards=[
                PublishedShard(
                    agent_name=agent_name,
                    device_id=0,
                    addr=4096,
                    shard_offset=(0, 0),
                    shape=(4, 4),
                )
            ],
        )
    ]


def test_structural_digest_ignores_content_digest():
    first = _one_tensor()
    first[0].shards[0].digest = "version-a"
    second = _one_tensor()
    second[0].shards[0].digest = "version-b"

    first_blob = wrap_rendezvous_blob(
        b"nixl",
        "trainer-agent",
        "trainer:1234",
        first,
    )
    second_blob = wrap_rendezvous_blob(
        b"nixl",
        "trainer-agent",
        "trainer:1234",
        second,
    )

    assert structural_manifest_digest(first_blob) == structural_manifest_digest(
        second_blob
    )
    moved = _one_tensor()
    moved[0].shards[0].addr = 8192
    moved_blob = wrap_rendezvous_blob(
        b"nixl",
        "trainer-agent",
        "trainer:1234",
        moved,
    )
    restarted_blob = wrap_rendezvous_blob(
        b"new-registration",
        "trainer-agent",
        "trainer:1234",
        second,
    )
    assert structural_manifest_digest(first_blob) != structural_manifest_digest(
        moved_blob
    )
    assert structural_manifest_digest(first_blob) != structural_manifest_digest(
        restarted_blob
    )


def test_host_memory_kind_roundtrips_and_requires_host_aware_reader():
    device = _one_tensor()
    host = _one_tensor()
    host[0].shards[0].memory_type = "DRAM"
    device_blob = wrap_rendezvous_blob(b"nixl", "trainer-agent", "host:1234", device)
    host_blob = wrap_rendezvous_blob(b"nixl", "trainer-agent", "host:1234", host)
    assert json.loads(device_blob)["schema"] == "mx.reshard.shard_table.v1"
    assert "memory_type" not in json.loads(device_blob)["tensors"][0]["shards"][0]
    assert json.loads(host_blob)["schema"] == "mx.reshard.shard_table.v2"
    assert structural_manifest_digest(device_blob) != structural_manifest_digest(
        host_blob
    )
    decoded = unwrap_rendezvous_blob(host_blob).tensors
    assert (
        decode_shard_table(encode_shard_table(host))[0].shards[0].memory_type == "DRAM"
    )
    assert decoded[0].shards[0].memory_type == "DRAM"
    with pytest.raises(ValueError, match="memory-type-aware receiver"):
        build_sources(decoded)
    memory = {}
    sources, agents, devices = build_sources(decoded, session_to_memory=memory)
    assert memory == {"trainer-agent": "DRAM"}
    assert agents == {"trainer-agent": "trainer-agent"}
    assert devices == {"trainer-agent": 0}
    assert sources["weight"].shards[0].addr == 4096


@pytest.mark.parametrize("change", ["missing_counts", "null_counts", "checksum"])
def test_stable_metadata_validation_preserves_legacy_blob_support(change):
    payload = json.loads(wrap_rendezvous_blob(b"nixl", "a", "a:1", _one_tensor()))
    if change == "checksum":
        payload["tensors"][0]["shards"][0]["digest"] = "version-checksum"
    else:
        for name in ("tensor_count", "total_bytes"):
            if change == "missing_counts":
                del payload[name]
            else:
                payload[name] = None
    blob = json.dumps(payload).encode()
    assert unwrap_rendezvous_blob(blob).tensor_count == 1
    with pytest.raises((KeyError, ValueError)):
        unwrap_rendezvous_blob(blob, stable=True)


def test_rejects_unknown_source_memory_type():
    blob = json.loads(wrap_rendezvous_blob(b"nixl", "a", "a:1", _one_tensor()))
    blob["tensors"][0]["shards"][0]["memory_type"] = "typo"
    with pytest.raises(ValueError, match="unsupported source memory type"):
        unwrap_rendezvous_blob(json.dumps(blob).encode())


@pytest.mark.parametrize("field,value", [("memory_type", "DRAM"), ("device_id", 1)])
def test_one_source_session_cannot_advertise_conflicting_memory(field, value):
    tensors = _one_tensor() + _one_tensor()
    tensors[1].name = "other"
    setattr(tensors[1].shards[0], field, value)
    with pytest.raises(ValueError, match="conflicting device or memory"):
        build_sources(tensors, session_to_memory={})


@pytest.mark.parametrize(
    "blob", [b"\xff\xfe not json", b"{ definitely not json", b'["a", "list"]']
)
def test_an_unreadable_manifest_says_it_disabled_plan_reuse(caplog, blob):
    """The raw-byte fallback moves with the per-shard content digests, so every
    version looks structurally different and plan reuse stops. Correct, but the
    symptom is a warm refit priced like a cold one, which needs a reason."""
    with caplog.at_level(logging.WARNING):
        assert structural_manifest_digest(blob) == hashlib.sha256(blob).hexdigest()

    assert [m for m in caplog.messages if "disables transfer-plan reuse" in m], (
        f"an unreadable manifest must say so, got {caplog.messages}"
    )


def test_decoding_parsed_entries_matches_decoding_the_blob():
    """The shard table used to be re-serialized only to be parsed again. Removing
    that round-trip must not change a single decoded field."""
    import json

    from modelexpress_rl.refit.reshard.rendezvous import (
        _SCHEMA,
        decode_shard_entries,
        decode_shard_table,
        encode_shard_table,
    )

    blob = encode_shard_table(_one_tensor())
    entries = json.loads(blob.decode("utf-8"))["tensors"]

    assert decode_shard_entries(entries) == decode_shard_table(blob)
    assert decode_shard_table(blob) == _one_tensor()
    assert json.loads(blob.decode("utf-8"))["schema"] == _SCHEMA


def test_built_sources_own_immutable_geometry():
    published = _one_tensor()
    published[0].full_shape = [4, 4]
    published[0].shards[0].shape = [4, 4]
    published[0].shards[0].shard_offset = [0, 0]
    sources, agents, devices = build_sources(published)
    source = sources["weight"]
    shard = source.shards[0]
    published[0].full_shape[0] = 8
    published[0].shards[0].shape[0] = 8
    published[0].shards[0].shard_offset[0] = 4
    published[0].shards[0].addr = 8192
    published[0].shards.clear()
    assert source.global_shape == shard.shape == (4, 4)
    assert shard.shard_offset == (0, 0)
    assert shard.addr == 4096
    for mapping in (sources, agents, devices):
        with pytest.raises(TypeError):
            mapping["other"] = None
    with pytest.raises(AttributeError):
        source.shards += (shard,)
    with pytest.raises(AttributeError):
        shard.addr = 0
