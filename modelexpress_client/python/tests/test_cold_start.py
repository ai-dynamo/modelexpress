# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from modelexpress import cold_start, p2p_pb2
from modelexpress.client import MxClient


def instance(group, rank, accelerator="cuda"):
    return p2p_pb2.SourceInstanceRef(
        worker_id=f"{group}-{rank}", worker_rank=rank, accelerator=accelerator
    )


def context():
    client = MagicMock(spec=MxClient)
    adapter = MagicMock()
    adapter.all_gather_state.side_effect = lambda value: (value,)
    adapter.broadcast_state.side_effect = lambda value: value
    return SimpleNamespace(
        mx_client=client,
        adapter=adapter,
        p2p_enabled=True,
        is_reload=False,
        p2p_role="main",
        global_rank=0,
        worker_rank=0,
        identity=SimpleNamespace(mx_source_type="", quantization="", dtype=""),
        accelerator_backend=SimpleNamespace(name="cuda"),
    )


def config(ranks=2):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=ranks,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            enable_expert_parallel=False,
        ),
        speculative_config=None,
    )


def test_only_complete_coherent_groups_are_ready():
    a = "mx-cold-0123456789abcdef0123456789abcdef"
    b = "mx-cold-abcdef0123456789abcdef0123456789"
    assert cold_start.ready_groups([instance(a, 0), instance(b, 1)], 2) == []
    assert cold_start.ready_groups(
        [instance(a, 0), instance(a, 1), instance(b, 1)], 2
    ) == [a]
    assert cold_start.ready_groups([instance(a, 0), instance(a, 2)], 2) == []


def test_ready_group_rejects_incompatible_accelerator():
    group = "mx-cold-0123456789abcdef0123456789abcdef"
    with patch.object(cold_start, "accelerators_compatible", return_value=False):
        assert not cold_start.ready_groups([instance(group, 0)], 1, context())


def test_elects_one_donor_and_uses_lease():
    ctx = context()
    ctx.mx_client.list_sources.return_value.instances = []
    ctx.mx_client.donor_lease.return_value = True
    with patch.object(cold_start, "DonorLease") as lease:
        cold_start.coordinate(ctx, config())
    assert ctx.cold_start_role == "donor"
    assert ctx.worker_id == f"{ctx.cold_start_group}-0"
    assert ctx.donor_lease_token
    lease.assert_called_once()
    assert (
        ctx.mx_client.donor_lease.call_args.args[2]
        == p2p_pb2.DONOR_LEASE_ACTION_ACQUIRE
    )


def test_election_rechecks_source_after_acquiring():
    ctx = context()
    group = "mx-cold-0123456789abcdef0123456789abcdef"
    ctx.mx_client.list_sources.side_effect = [
        SimpleNamespace(instances=[]),
        SimpleNamespace(instances=[instance(group, 0), instance(group, 1)]),
    ]
    ctx.mx_client.donor_lease.return_value = True
    cold_start.coordinate(ctx, config())
    assert ctx.cold_start_role == "follower"
    assert ctx.cold_start_group == group
    assert (
        ctx.mx_client.donor_lease.call_args.args[2]
        == p2p_pb2.DONOR_LEASE_ACTION_RELEASE
    )


def test_busy_lease_waits_then_uses_newly_published_source():
    ctx = context()
    group = "mx-cold-0123456789abcdef0123456789abcdef"
    ctx.mx_client.list_sources.side_effect = [
        SimpleNamespace(instances=[]),
        SimpleNamespace(instances=[instance(group, 0), instance(group, 1)]),
    ]
    ctx.mx_client.donor_lease.return_value = False
    with patch.object(cold_start.time, "sleep"):
        cold_start.coordinate(ctx, config())
    assert ctx.cold_start_role == "follower"


def test_discovery_error_fails_closed():
    ctx = context()
    ctx.mx_client.list_sources.side_effect = RuntimeError("server unavailable")
    with pytest.raises(RuntimeError, match="server unavailable"):
        cold_start.coordinate(ctx, config())
    ctx.mx_client.donor_lease.assert_not_called()


def test_unsupported_configuration_rejected():
    ctx = context()
    ctx.p2p_enabled = False
    with pytest.raises(RuntimeError, match="P2P must be enabled"):
        cold_start.coordinate(ctx, config())
