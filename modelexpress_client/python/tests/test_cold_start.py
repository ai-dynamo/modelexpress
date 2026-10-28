# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from modelexpress import cold_start, p2p_pb2
from modelexpress.client import MxClient


def instance(rank):
    return p2p_pb2.SourceInstanceRef(
        worker_id=f"worker-{rank}", worker_rank=rank, accelerator="cuda"
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
        donor_lease=None,
        identity=p2p_pb2.SourceIdentity(),
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


def sources(*ranks):
    return SimpleNamespace(instances=[instance(rank) for rank in ranks])


def test_ready_requires_all_compatible_ranks():
    ctx = context()
    ctx.mx_client.list_sources.return_value = sources(0, 1)
    with patch.object(cold_start, "accelerators_compatible", return_value=True):
        assert cold_start._ready(ctx, 2)
        assert not cold_start._ready(ctx, 3)
    with patch.object(cold_start, "accelerators_compatible", return_value=False):
        assert not cold_start._ready(ctx, 2)
    assert ctx.mx_client.list_sources.call_args.kwargs["strict"]


def test_existing_sources_skip_election():
    ctx = context()
    with patch.object(cold_start, "_ready", return_value=True):
        cold_start.coordinate(ctx, config())
    assert ctx.adapter.broadcast_state.call_args.args[0][0] == "follower"
    ctx.mx_client.donor_lease.assert_not_called()


def test_first_contender_becomes_donor():
    ctx = context()
    ctx.mx_client.donor_lease.return_value = True
    with (
        patch.object(cold_start, "_ready", return_value=False),
        patch.object(cold_start, "DonorLease") as lease,
    ):
        cold_start.coordinate(ctx, config())
    assert ctx.adapter.broadcast_state.call_args.args[0][0] == "donor"
    assert ctx.donor_lease is lease.return_value
    assert (
        ctx.mx_client.donor_lease.call_args.args[2]
        == p2p_pb2.DONOR_LEASE_ACTION_ACQUIRE
    )


def test_busy_lease_waits_for_ready():
    ctx = context()
    ctx.mx_client.donor_lease.return_value = False
    with (
        patch.object(cold_start, "_ready", side_effect=[False, True]),
        patch.object(cold_start.time, "sleep"),
    ):
        cold_start.coordinate(ctx, config())
    assert ctx.adapter.broadcast_state.call_args.args[0][0] == "follower"


def test_busy_then_acquires_after_expiry():
    ctx = context()
    ctx.mx_client.donor_lease.side_effect = [False, True]
    with (
        patch.object(cold_start, "_ready", return_value=False),
        patch.object(cold_start.time, "sleep"),
        patch.object(cold_start, "DonorLease"),
    ):
        cold_start.coordinate(ctx, config())
    assert ctx.adapter.broadcast_state.call_args.args[0][0] == "donor"


def test_recheck_after_acquire_releases_lease():
    ctx = context()
    ctx.mx_client.donor_lease.return_value = True
    with patch.object(cold_start, "_ready", side_effect=[False, True]):
        cold_start.coordinate(ctx, config())
    assert ctx.adapter.broadcast_state.call_args.args[0][0] == "follower"
    assert (
        ctx.mx_client.donor_lease.call_args.args[2]
        == p2p_pb2.DONOR_LEASE_ACTION_RELEASE
    )


@pytest.mark.parametrize("operation", ["_ready", "donor_lease"])
def test_election_errors_fail_closed(operation):
    ctx = context()
    with patch.object(
        cold_start,
        "_ready",
        side_effect=RuntimeError("unavailable") if operation == "_ready" else None,
        return_value=False,
    ):
        if operation == "donor_lease":
            ctx.mx_client.donor_lease.side_effect = RuntimeError("unavailable")
        with pytest.raises(RuntimeError, match="unavailable"):
            cold_start.coordinate(ctx, config())
    assert ctx.adapter.broadcast_state.call_args.args[0][0] == "error"


def test_unsupported_configuration_rejected():
    ctx = context()
    ctx.p2p_enabled = False
    with pytest.raises(RuntimeError, match="P2P"):
        cold_start.coordinate(ctx, config())


def test_renewal_loss_is_recorded():
    ctx = context()
    ctx.mx_client.donor_lease.return_value = False
    with (
        patch.object(cold_start, "RENEW_INTERVAL", 0.001),
        patch.object(cold_start, "_ready", return_value=False),
    ):
        lease = cold_start.DonorLease(ctx, "token", 2)
        lease.thread.join(timeout=1)
        assert lease.lost.is_set()
        lease.stop()


def test_ready_releases_lease():
    ctx = context()
    ctx.mx_client.donor_lease.return_value = True
    with (
        patch.object(cold_start, "RENEW_INTERVAL", 0.001),
        patch.object(cold_start, "_ready", return_value=True),
    ):
        lease = cold_start.DonorLease(ctx, "token", 2)
        lease.thread.join(timeout=1)
        lease.stop()
    assert (
        ctx.mx_client.donor_lease.call_args.args[2]
        == p2p_pb2.DONOR_LEASE_ACTION_RELEASE
    )


def test_nonzero_rank_uses_broadcast_decision():
    ctx = context()
    ctx.global_rank = 1
    ctx.adapter.broadcast_state.side_effect = None
    ctx.adapter.broadcast_state.return_value = ("follower", "")
    cold_start.coordinate(ctx, config())
    ctx.mx_client.list_sources.assert_not_called()
    ctx.mx_client.donor_lease.assert_not_called()
    ctx.adapter.broadcast_state.assert_called_once_with(None)


def test_renews_before_checking_ready():
    ctx = context()
    ctx.mx_client.donor_lease.return_value = True
    with (
        patch.object(cold_start, "RENEW_INTERVAL", 0.001),
        patch.object(cold_start, "_ready", return_value=True) as ready,
    ):
        lease = cold_start.DonorLease(ctx, "token", 2)
        lease.thread.join(timeout=1)
        lease.stop()
    assert ready.called
    assert [call.args[2] for call in ctx.mx_client.donor_lease.call_args_list] == [
        p2p_pb2.DONOR_LEASE_ACTION_RENEW,
        p2p_pb2.DONOR_LEASE_ACTION_RELEASE,
    ]
