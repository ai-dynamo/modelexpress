# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from modelexpress.rl_fanout import RlTreeFanoutPolicy
from modelexpress.rl_metadata import RlSourceRole


def test_tree_fanout_roots_first_wave_at_trainer_rank():
    policy = RlTreeFanoutPolicy(
        receiver_rank=1,
        replica_world_size=8,
        fanout=2,
        root_source_rank=3,
    )

    assert policy.parent_replica_rank is None
    assert policy.roles == (RlSourceRole.TRAINER,)
    assert policy.source_ranks_by_role == {RlSourceRole.TRAINER: (3,)}


def test_tree_fanout_assigns_later_replicas_to_prior_replica_parent():
    parents = [
        RlTreeFanoutPolicy(
            receiver_rank=rank,
            replica_world_size=8,
            fanout=2,
        ).parent_replica_rank
        for rank in range(8)
    ]

    assert parents == [None, None, 0, 0, 1, 1, 2, 2]
    policy = RlTreeFanoutPolicy(receiver_rank=6, replica_world_size=8, fanout=2)
    assert policy.roles == (RlSourceRole.INFERENCE_REPLICA,)
    assert policy.source_ranks_by_role == {RlSourceRole.INFERENCE_REPLICA: (2,)}


def test_tree_fanout_rejects_invalid_topology_values():
    with pytest.raises(ValueError, match="replica_world_size"):
        RlTreeFanoutPolicy(receiver_rank=0, replica_world_size=0)
    with pytest.raises(ValueError, match="receiver_rank"):
        RlTreeFanoutPolicy(receiver_rank=2, replica_world_size=2)
    with pytest.raises(ValueError, match="fanout"):
        RlTreeFanoutPolicy(receiver_rank=0, replica_world_size=1, fanout=0)
    with pytest.raises(ValueError, match="root_source_rank"):
        RlTreeFanoutPolicy(receiver_rank=0, replica_world_size=1, root_source_rank=-1)
