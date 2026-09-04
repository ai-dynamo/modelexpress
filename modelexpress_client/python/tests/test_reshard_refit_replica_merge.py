# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Replica dedup in merge_shard_tables - pure data, no torch.

Under DP/EDP the same geometric shard is published by several trainer ranks, at
distinct addresses on distinct NICs, holding byte-identical bytes. The merge used
to retain every one of those offers, which is a read per replica into the same
destination: correct bytes, but wire and descriptor count scaling with the DP
degree, plus a P2P handshake per redundant owner. It now retains exactly one
representative per ``(shard_offset, shape)``.

The tests that matter are the ones that would catch a wrong merge:

  * duplicates collapse, and the resulting plan reads each region once;
  * genuine cross-rank fan-in (distinct geometry) is still fully retained, for
    every source - collapsing that would silently drop half the model;
  * the default is the previous first-writer-wins choice.

Run: pytest tests/test_reshard_refit_replica_merge.py
"""

import pytest

from modelexpress.refit.reshard.rendezvous import (
    PublishedShard,
    PublishedTensor,
    build_sources,
    merge_shard_tables,
)
from modelexpress.refit.reshard.slice_plan import plan_pull
from modelexpress.refit.reshard.types import RecordedCopy

EL = 2  # bytes per bfloat16 element


def _dp_replica_tables(replicas: int, shards_per_rank: int = 2):
    """``replicas`` publishers each advertising the same geometries.

    This is the DP8 / EDP2 shape: byte-identical copies of a shard on distinct
    ranks, at distinct addresses.
    """
    return [
        [
            PublishedTensor(
                name="weight",
                dtype="torch.bfloat16",
                elsize=EL,
                full_shape=(8, 4),
                shards=[
                    PublishedShard(
                        agent_name=f"dp{dp}",
                        device_id=dp,
                        addr=1000 * (dp + 1) + 10 * i,
                        shard_offset=(4 * i, 0),
                        shape=(4, 4),
                    )
                    for i in range(shards_per_rank)
                ],
            )
        ]
        for dp in range(replicas)
    ]


def _fan_in_tables(ranks: int = 4):
    """``ranks`` publishers each owning a distinct row block: real fan-in."""
    return [
        [
            PublishedTensor(
                name="weight",
                dtype="torch.bfloat16",
                elsize=EL,
                full_shape=(16, 4),
                shards=[
                    PublishedShard(
                        agent_name=f"r{rank}",
                        device_id=rank,
                        addr=1000 * (rank + 1),
                        shard_offset=(4 * rank, 0),
                        shape=(4, 4),
                    )
                ],
            )
        ]
        for rank in range(ranks)
    ]


def test_duplicate_replicas_collapse_to_one_owner():
    """The core of the change: eight offers of a geometry become one."""
    merged = merge_shard_tables(_dp_replica_tables(replicas=8))

    assert len(merged) == 1
    # Two geometries, not sixteen offers.
    assert len(merged[0].shards) == 2
    assert [tuple(s.shard_offset) for s in merged[0].shards] == [(0, 0), (4, 0)]


def test_dedup_removes_the_redundant_reads_not_just_the_offers():
    """The point is wire bytes, so assert it where the reads are planned.

    Without dedup a DP8 publish plans eight reads of every region into the same
    destination bytes. The segment count and byte total must instead match what a
    single publisher would produce.
    """
    def segments_for(replicas):
        merged = merge_shard_tables(_dp_replica_tables(replicas=replicas))
        sources, _agents, _devices = build_sources(merged)
        source = sources["weight"]
        # build_sources resolves the published dtype string to the real dtype, and
        # plan_pull refuses a cross-dtype read, so take the dest dtype from there.
        copy = RecordedCopy(
            src_name="weight",
            op_chain=(),
            param_name="weight",
            dest_offset=0,
            dest_shape=(8, 4),
            dest_stride=(4, 1),
            dest_dtype=source.dtype,
        )
        return plan_pull(
            copy,
            global_shape=source.global_shape,
            src_dtype=source.dtype,
            elsize=source.elsize,
            shards=source.shards,
        )

    single = segments_for(1)
    assert len(single) == 2  # one contiguous run per row block
    assert sum(s.nbytes for s in single) == 32 * EL  # the whole 8x4 tensor, once

    for replicas in (2, 8):
        spread = segments_for(replicas)
        assert len(spread) == len(single), replicas
        assert sum(s.nbytes for s in spread) == sum(s.nbytes for s in single), replicas
        # One owner serves the tensor, so one peer needs a handshake.
        assert len({s.session for s in spread}) == 1, replicas


def test_default_offset_takes_the_first_publisher():
    """The default must be the previous first-writer-wins choice."""
    merged = merge_shard_tables(_dp_replica_tables(replicas=4))
    assert [s.agent_name for s in merged[0].shards] == ["dp0", "dp0"]


def test_merge_still_fans_in_distinct_geometry_across_ranks():
    """Non-replica shards are real fan-in: collapsing them drops model bytes."""
    tables = _fan_in_tables(ranks=4)

    merged = merge_shard_tables(tables)
    assert len(merged[0].shards) == 4
    assert sorted(s.agent_name for s in merged[0].shards) == [
        "r0",
        "r1",
        "r2",
        "r3",
    ]


def test_replicated_fan_in_keeps_every_region_once():
    """DP replication on top of fan-in: four regions, one owner each."""
    tables = []
    for _dp in range(3):
        tables.extend(_fan_in_tables(ranks=4))

    merged = merge_shard_tables(tables)

    assert len(merged[0].shards) == 4
    assert sorted(tuple(s.shard_offset) for s in merged[0].shards) == [
        (0, 0),
        (4, 0),
        (8, 0),
        (12, 0),
    ]


def test_inconsistent_shape_or_dtype_still_raises():
    """Preserved from before the rewrite: disagreeing publishers are a hard error."""
    tables = _dp_replica_tables(replicas=1)
    conflicting = _dp_replica_tables(replicas=1)
    conflicting[0][0].full_shape = (16, 4)

    with pytest.raises(ValueError, match="inconsistent shape/dtype"):
        merge_shard_tables(tables + conflicting)


def test_distinct_tensor_names_are_independent():
    """Dedup is per name: two names sharing a geometry must both survive."""
    tables = []
    for dp in range(2):
        tables.append(
            [
                PublishedTensor(
                    name=name,
                    dtype="torch.bfloat16",
                    elsize=EL,
                    full_shape=(4, 4),
                    shards=[
                        PublishedShard(
                            agent_name=f"dp{dp}",
                            device_id=dp,
                            addr=1000 * (dp + 1),
                            shard_offset=(0, 0),
                            shape=(4, 4),
                        )
                    ],
                )
                for name in ("a", "b")
            ]
        )

    merged = merge_shard_tables(tables)

    assert sorted(t.name for t in merged) == ["a", "b"]
    assert all(len(t.shards) == 1 for t in merged)
