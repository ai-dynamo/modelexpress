# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for the shard digest and its wire format.

The digest exists to catch a refit that installs the wrong bytes while reporting
healthy timings, so the tests that matter are the ones proving it *changes* when it
should: a corrupted byte, and - the case plain checksums miss - a pure permutation of
correct bytes.

The rest cover the wire format, where the property worth pinning is that the field is
optional in both directions. A fleet is upgraded one image at a time, so a publisher
that does not digest must produce the blob an older client would have, and a reader
must not invent agreement from a missing digest.
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from modelexpress import envs  # noqa: E402
from modelexpress.refit.reshard.rendezvous import (  # noqa: E402
    PublishedShard,
    PublishedTensor,
    build_sources,
    decode_shard_table,
    encode_shard_table,
    unwrap_rendezvous_blob,
    wrap_rendezvous_blob,
)
from modelexpress.refit.reshard.verify import (  # noqa: E402
    published_digest,
    shard_region,
    tensor_digest,
)


# ------------------------------------------------------------------- the digest
def test_digest_is_stable_for_identical_bytes():
    a = torch.arange(4096, dtype=torch.int32)
    assert tensor_digest(a) == tensor_digest(a.clone())


def test_digest_changes_when_a_single_value_changes():
    a = torch.arange(4096, dtype=torch.int32)
    b = a.clone()
    b[1234] += 1
    assert tensor_digest(a) != tensor_digest(b)


def test_digest_detects_a_permutation():
    """The reason this is not a plain sum.

    A plan that copies the right bytes to the wrong offsets preserves every
    order-independent statistic, so a sum-based digest would call it equal.
    """
    a = torch.arange(8192, dtype=torch.int32)
    b = a.clone()
    # swap two whole digest rows, so the multiset of values is untouched
    row = 1024
    b[0:row], b[row : 2 * row] = a[row : 2 * row].clone(), a[0:row].clone()
    assert a.sum() == b.sum(), "precondition: sums must agree, else the test is trivial"
    assert tensor_digest(a) != tensor_digest(b)


def test_digest_covers_a_non_word_multiple_tail():
    """Sizes that are not a multiple of the row or word width must still be digested.

    An implementation that silently drops the ragged tail would report equal for
    tensors differing only in their last bytes.
    """
    a = torch.arange(1024 + 7, dtype=torch.int16)
    b = a.clone()
    b[-1] += 1
    assert tensor_digest(a) != tensor_digest(b)


def test_digest_handles_a_strided_input():
    dense = torch.arange(64, dtype=torch.int32).reshape(8, 8)
    strided = dense[:, ::2]
    assert tensor_digest(strided) == tensor_digest(strided.contiguous())


# -------------------------------------------------------------------- the region
def test_shard_region_extracts_the_publishers_box():
    full = torch.arange(24, dtype=torch.int32).reshape(4, 6)
    region = shard_region(full.reshape(-1), (4, 6), (1, 2), (2, 3))
    assert torch.equal(region, full[1:3, 2:5])


def test_a_region_digests_the_same_as_the_shard_its_publisher_held():
    """The two halves of the eventual comparison have to agree on what they hash.

    A publisher digests its own contiguous shard; a receiver recovers the box out of
    a staging buffer it assembled. If those disagreed the gate would report every
    shard as corrupt.
    """
    full = torch.arange(24, dtype=torch.int32).reshape(4, 6)
    publisher_shard = full[1:3, 2:5].contiguous()

    region = shard_region(full.reshape(-1), (4, 6), (1, 2), (2, 3))

    assert tensor_digest(region) == tensor_digest(publisher_shard)


# --------------------------------------------------------------- publication gate
def test_publication_is_off_by_default(monkeypatch):
    monkeypatch.delenv("MX_RESHARD_PUBLISH_DIGEST", raising=False)
    assert published_digest(torch.arange(64, dtype=torch.int32)) is None


def test_publication_is_read_live_not_frozen_at_import(monkeypatch):
    """So a publisher can be switched without reimporting, and so these tests do not
    have to reload the module the way a captured module constant would force."""
    tensor = torch.arange(64, dtype=torch.int32)
    monkeypatch.setenv("MX_RESHARD_PUBLISH_DIGEST", "1")
    assert published_digest(tensor) == tensor_digest(tensor)

    monkeypatch.setenv("MX_RESHARD_PUBLISH_DIGEST", "0")
    assert published_digest(tensor) is None


def test_the_flag_comes_from_the_env_registry(monkeypatch):
    monkeypatch.setenv("MX_RESHARD_PUBLISH_DIGEST", "1")

    assert envs.MX_RESHARD_PUBLISH_DIGEST is True


# ------------------------------------------------------------------ the wire format
def _table(digest=None):
    return [
        PublishedTensor(
            name="layers.0.weight",
            dtype="torch.bfloat16",
            elsize=2,
            full_shape=(8, 4),
            shards=[
                PublishedShard(
                    agent_name="trainer-r0",
                    device_id=0,
                    addr=4096,
                    shard_offset=(0, 0),
                    shape=(4, 4),
                    digest=digest,
                )
            ],
        )
    ]


def _encoded_shard(table) -> dict:
    """The one shard's JSON, out of the encoded table."""
    payload = json.loads(encode_shard_table(table).decode("utf-8"))
    return payload["tensors"][0]["shards"][0]


def test_the_digest_round_trips():
    (decoded,) = decode_shard_table(encode_shard_table(_table(digest="abc123")))
    assert decoded.shards[0].digest == "abc123"


def test_a_publisher_without_a_digest_decodes_as_none_not_as_agreement():
    (decoded,) = decode_shard_table(encode_shard_table(_table()))
    assert decoded.shards[0].digest is None


def test_the_key_is_omitted_rather_than_nulled_when_absent():
    """So an undigested publisher emits the blob an older client would have."""
    assert "digest" not in _encoded_shard(_table())


def test_the_digest_is_the_only_added_key():
    """A wire-format change should be auditable by diffing the key sets."""
    without = set(_encoded_shard(_table()))
    with_digest = set(_encoded_shard(_table(digest="d")))

    assert with_digest - without == {"digest"}


def test_an_older_publishers_blob_is_still_readable():
    """Forward compatibility: a blob written before this field existed."""
    payload = json.loads(encode_shard_table(_table(digest="d")).decode("utf-8"))
    del payload["tensors"][0]["shards"][0]["digest"]

    (decoded,) = decode_shard_table(json.dumps(payload).encode("utf-8"))

    assert decoded.shards[0].digest is None


# --------------------------------------------------------------------- the read path
def test_the_digest_survives_into_the_planning_inputs():
    """The planner ignores it, but it has to arrive. ``build_sources`` is where a
    published table becomes planning input, and a digest dropped here would leave a
    future check with no expectation to compare against - the hole this field exists
    to close."""
    sources, _agents, _devices = build_sources(_table(digest="abc123"))

    assert sources["layers.0.weight"].shards[0].digest == "abc123"


def test_an_undigested_table_yields_no_expectation():
    sources, _agents, _devices = build_sources(_table())

    assert sources["layers.0.weight"].shards[0].digest is None


def test_a_full_publish_to_discover_round_trip_keeps_the_digest():
    """End to end across the two hops it has to survive: the shard-table encoding
    inside the rendezvous blob, and the conversion into planning input."""
    blob = wrap_rendezvous_blob(b"meta", "trainer-r0", "h:1", _table(digest="abc123"))

    payload = unwrap_rendezvous_blob(blob)
    sources, _agents, _devices = build_sources(payload.tensors)

    assert sources["layers.0.weight"].shards[0].digest == "abc123"
