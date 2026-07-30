# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for the publisher step stamp on the rendezvous blob.

The stamp answers a question a receiver otherwise has to guess at: does the shard
table I just discovered describe the step I am refitting, or the one before it? It
matters because the table carries the per-shard digests a verify gate compares
against, so a table one step behind turns correctly delivered bytes into a reported
corruption.

What is worth testing here is not that a field round-trips. It is that the stamp is
optional in *both* directions, because a fleet is upgraded one image at a time and
neither side may assume the other has it, and that a missing stamp reads as "unknown"
rather than as step 0.
"""

from __future__ import annotations

import json

from modelexpress.refit.reshard.rendezvous import (
    unwrap_rendezvous_blob,
    wrap_rendezvous_blob,
)


def _blob(step=None, tensors=None):
    return wrap_rendezvous_blob(b"meta", "agent-r0", "h:1", tensors or [], step)


def test_the_stamp_round_trips():
    assert unwrap_rendezvous_blob(_blob(step=7)).publisher_step == 7


def test_an_unstamped_blob_reads_as_unknown_not_step_zero():
    """The distinction a staleness check has to rest on. Read as 0, an unstamped
    publisher looks permanently behind, and a consumer that excuses lagging
    publishers would excuse every one of its shards."""
    assert unwrap_rendezvous_blob(_blob()).publisher_step is None


def test_step_zero_is_carried_not_dropped():
    """0 is a legitimate step and must not collapse into "no stamp"."""
    assert unwrap_rendezvous_blob(_blob(step=0)).publisher_step == 0


def test_the_stamp_is_omitted_rather_than_nulled_when_absent():
    """So an unstamped publisher emits the blob an older client would have."""
    payload = json.loads(_blob().decode("utf-8"))

    assert "publisher_step" not in payload


def test_the_stamp_is_the_only_added_key():
    """A wire-format change should be auditable by diffing the key sets."""
    without = set(json.loads(_blob().decode("utf-8")))
    with_step = set(json.loads(_blob(step=4).decode("utf-8")))

    assert with_step - without == {"publisher_step"}


def test_the_first_four_fields_keep_their_positions():
    """The stamp is appended, so existing positional reads still land on the same
    values. Unpacking now has to name five, which is the intended cost of one
    payload type over a second unwrap function."""
    payload = unwrap_rendezvous_blob(_blob(step=3))

    assert payload[:4] == (b"meta", "agent-r0", "h:1", [])


def test_a_new_reader_handles_an_unstamped_publishers_blob():
    """Forward compatibility: no `publisher_step` key at all."""
    payload = json.loads(_blob().decode("utf-8"))
    payload.pop("publisher_step", None)

    unwrapped = unwrap_rendezvous_blob(json.dumps(payload).encode("utf-8"))

    assert unwrapped.publisher_step is None


def test_an_old_reader_ignores_a_stamped_publishers_blob():
    """Backward compatibility: the extra key must not be rejected."""
    assert unwrap_rendezvous_blob(_blob(step=11))[1] == "agent-r0"


def test_the_stamp_does_not_disturb_the_shard_table():
    """The stamp describes the table; it must not alter how the table decodes."""
    from modelexpress.refit.reshard.rendezvous import PublishedShard, PublishedTensor

    tensors = [
        PublishedTensor(
            name="layers.0.weight",
            dtype="torch.bfloat16",
            elsize=2,
            full_shape=(8, 4),
            shards=[
                PublishedShard(
                    agent_name="agent-r0",
                    device_id=0,
                    addr=4096,
                    shard_offset=(0, 0),
                    shape=(4, 4),
                )
            ],
        )
    ]

    stamped = unwrap_rendezvous_blob(_blob(step=5, tensors=tensors))
    plain = unwrap_rendezvous_blob(_blob(tensors=tensors))

    assert stamped[:4] == plain[:4]
    assert stamped.tensors[0].shards[0].addr == 4096
