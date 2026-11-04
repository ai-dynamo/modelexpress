# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import grpc
import pytest

from modelexpress import cold_start, p2p_pb2
from modelexpress.client import MxClient
from modelexpress.engines.vllm.adapter import cold_start_expected_workers


def instance(rank):
    return p2p_pb2.SourceInstanceRef(
        worker_id=f"worker-{rank}", worker_rank=rank, accelerator="cuda"
    )


def context():
    client = MagicMock(spec=MxClient)
    adapter = MagicMock()
    adapter.all_gather_state.side_effect = lambda value: (
        (value[0], 0, 0), (value[0], 1, 1)
    )
    adapter.broadcast_state.side_effect = lambda value: value
    return SimpleNamespace(
        mx_client=client,
        adapter=adapter,
        p2p_enabled=True,
        is_reload=False,
        p2p_role="main",
        global_rank=0,
        worker_rank=0,
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
        cold_start.coordinate(ctx, 2)
    assert ctx.adapter.broadcast_state.call_args.args[0][0] == "follower"
    assert ctx.skip_rdma is False
    ctx.mx_client.donor_lease.assert_not_called()


def test_first_contender_becomes_donor():
    ctx = context()
    ctx.mx_client.donor_lease.return_value = True
    with (
        patch.object(cold_start, "_ready", return_value=False),
        patch.object(cold_start, "DonorLease") as lease,
    ):
        cold_start.coordinate(ctx, 2)
    assert ctx.adapter.broadcast_state.call_args.args[0][0] == "donor"
    assert ctx.skip_rdma is True
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
        cold_start.coordinate(ctx, 2)
    assert ctx.adapter.broadcast_state.call_args.args[0][0] == "follower"


def test_busy_then_acquires_after_expiry():
    ctx = context()
    ctx.mx_client.donor_lease.side_effect = [False, True]
    with (
        patch.object(cold_start, "_ready", return_value=False),
        patch.object(cold_start.time, "sleep"),
        patch.object(cold_start, "DonorLease"),
    ):
        cold_start.coordinate(ctx, 2)
    assert ctx.adapter.broadcast_state.call_args.args[0][0] == "donor"


def test_recheck_after_acquire_releases_lease():
    ctx = context()
    ctx.mx_client.donor_lease.return_value = True
    with patch.object(cold_start, "_ready", side_effect=[False, True]):
        cold_start.coordinate(ctx, 2)
    assert ctx.adapter.broadcast_state.call_args.args[0][0] == "follower"
    assert (
        ctx.mx_client.donor_lease.call_args.args[2]
        == p2p_pb2.DONOR_LEASE_ACTION_RELEASE
    )


@pytest.mark.parametrize("operation", ["_ready", "donor_lease"])
def test_election_errors_fall_back(operation, caplog):
    ctx = context()
    with patch.object(
        cold_start,
        "_ready",
        side_effect=RuntimeError("unavailable") if operation == "_ready" else None,
        return_value=False,
    ):
        if operation == "donor_lease":
            ctx.mx_client.donor_lease.side_effect = RuntimeError("unavailable")
        cold_start.coordinate(ctx, 2)
    assert ctx.adapter.broadcast_state.call_args.args[0] == ("fallback", "")
    assert ctx.skip_rdma is True
    assert ctx.p2p_enabled is True
    assert "MX cold start election failed (unavailable)" in caplog.text


def test_source_discovery_error_falls_back(caplog):
    ctx = context()
    ctx.mx_client.list_sources.side_effect = RuntimeError("coordinator unavailable")

    cold_start.coordinate(ctx, 2)

    assert ctx.mx_client.list_sources.call_args.kwargs["strict"]
    assert ctx.adapter.broadcast_state.call_args.args[0] == ("fallback", "")
    assert "MX cold start election failed (coordinator unavailable)" in caplog.text


def test_unsupported_configuration_rejected():
    ctx = context()
    ctx.p2p_enabled = False
    with pytest.raises(RuntimeError, match="P2P"):
        cold_start.coordinate(ctx, 2)


def test_renewal_loss_stops_without_release(caplog):
    ctx = context()
    ctx.mx_client.donor_lease.return_value = False
    with (
        patch.object(cold_start, "RENEW_INTERVAL", 0.001),
        patch.object(cold_start, "_ready", return_value=False),
    ):
        lease = cold_start.DonorLease(ctx, "token", 2, cold_start.time.monotonic())
        lease.thread.join(timeout=1)
        assert not lease.thread.is_alive()
        lease.stop()
    assert [call.args[2] for call in ctx.mx_client.donor_lease.call_args_list] == [
        p2p_pb2.DONOR_LEASE_ACTION_RENEW,
    ]
    assert "MX donor lease was lost" in caplog.text


def test_ready_releases_lease():
    ctx = context()
    ctx.mx_client.donor_lease.return_value = True
    with (
        patch.object(cold_start, "RENEW_INTERVAL", 0.001),
        patch.object(cold_start, "_ready", return_value=True),
    ):
        lease = cold_start.DonorLease(ctx, "token", 2, cold_start.time.monotonic())
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
    cold_start.coordinate(ctx, 2)
    ctx.mx_client.list_sources.assert_not_called()
    ctx.mx_client.donor_lease.assert_not_called()
    ctx.adapter.broadcast_state.assert_called_once_with(None)
    assert ctx.skip_rdma is False


def test_nonzero_donor_rank_skips_rdma():
    ctx = context()
    ctx.global_rank = 1
    ctx.adapter.broadcast_state.side_effect = None
    ctx.adapter.broadcast_state.return_value = ("donor", "")
    cold_start.coordinate(ctx, 2)
    ctx.mx_client.list_sources.assert_not_called()
    ctx.mx_client.donor_lease.assert_not_called()
    ctx.adapter.broadcast_state.assert_called_once_with(None)
    assert ctx.skip_rdma is True


def test_renews_before_checking_ready():
    ctx = context()
    ctx.mx_client.donor_lease.return_value = True
    with (
        patch.object(cold_start, "RENEW_INTERVAL", 0.001),
        patch.object(cold_start, "_ready", return_value=True) as ready,
    ):
        lease = cold_start.DonorLease(ctx, "token", 2, cold_start.time.monotonic())
        lease.thread.join(timeout=1)
        lease.stop()
    assert ready.called
    assert [call.args[2] for call in ctx.mx_client.donor_lease.call_args_list] == [
        p2p_pb2.DONOR_LEASE_ACTION_RENEW,
        p2p_pb2.DONOR_LEASE_ACTION_RELEASE,
    ]


def run_lease(ctx, renew, ready, wait=None):
    lease = cold_start.DonorLease.__new__(cold_start.DonorLease)
    lease.ctx, lease.token, lease.ranks = ctx, "token", 2
    lease.deadline = 60
    lease.renewal_end = 600
    lease.stopped = MagicMock()
    lease.stopped.wait.side_effect = wait or [False, False, True]
    with (
        patch.object(cold_start.time, "monotonic", side_effect=range(20)),
        patch.object(cold_start, "_ready", side_effect=ready),
    ):
        ctx.mx_client.donor_lease.side_effect = renew
        lease._run()
    return lease


def test_transient_renewal_failure_retries_and_releases():
    ctx = context()
    run_lease(ctx, [RuntimeError("unavailable"), True, True], [True])
    assert [call.args[2] for call in ctx.mx_client.donor_lease.call_args_list] == [
        p2p_pb2.DONOR_LEASE_ACTION_RENEW,
        p2p_pb2.DONOR_LEASE_ACTION_RENEW,
        p2p_pb2.DONOR_LEASE_ACTION_RELEASE,
    ]


def test_readiness_error_does_not_abandon_lease():
    ctx = context()
    run_lease(ctx, [True, True, True], [RuntimeError("unavailable"), True])
    assert [call.args[2] for call in ctx.mx_client.donor_lease.call_args_list] == [
        p2p_pb2.DONOR_LEASE_ACTION_RENEW,
        p2p_pb2.DONOR_LEASE_ACTION_RENEW,
        p2p_pb2.DONOR_LEASE_ACTION_RELEASE,
    ]


def test_persistent_renewal_failure_stops_at_deadline(caplog):
    ctx = context()
    lease = cold_start.DonorLease.__new__(cold_start.DonorLease)
    lease.ctx, lease.token, lease.ranks = ctx, "token", 2
    lease.deadline = 5
    lease.renewal_end = 600
    lease.stopped = MagicMock()
    lease.stopped.wait.return_value = False
    ctx.mx_client.donor_lease.side_effect = RuntimeError("unavailable")
    with patch.object(cold_start.time, "monotonic", side_effect=[0, 1, 3, 5]):
        lease._run()
    assert ctx.mx_client.donor_lease.call_count == 1
    assert "MX donor lease renewal deadline passed" in caplog.text


def test_stop_during_retry_releases_lease():
    ctx = context()
    run_lease(ctx, [RuntimeError("unavailable"), True], [], [False, True])
    assert [call.args[2] for call in ctx.mx_client.donor_lease.call_args_list] == [
        p2p_pb2.DONOR_LEASE_ACTION_RENEW,
        p2p_pb2.DONOR_LEASE_ACTION_RELEASE,
    ]


def test_slow_readiness_check_does_not_delay_next_renewal():
    ctx = context()
    lease = cold_start.DonorLease.__new__(cold_start.DonorLease)
    lease.ctx, lease.token, lease.ranks = ctx, "token", 2
    lease.deadline = 60
    lease.renewal_end = 600
    lease.stopped = MagicMock()
    lease.stopped.wait.side_effect = [False, True]
    ctx.mx_client.donor_lease.return_value = True
    with (
        patch.object(cold_start.time, "monotonic", side_effect=[0, 1, 1, 31, 31]),
        patch.object(cold_start, "_ready", return_value=False),
    ):
        lease._run()
    assert lease.stopped.wait.call_args_list[1].args == (0,)




def test_donor_renewal_cap_uses_configured_timeout(monkeypatch):
    ctx = context()
    monkeypatch.setattr(cold_start.envs, "MX_P2P_COLD_START_WAIT_TIMEOUT_SECS", 45)
    with (
        patch.object(cold_start.threading, "Thread") as thread,
        patch.object(cold_start.atexit, "register"),
    ):
        lease = cold_start.DonorLease(ctx, "token", 2, 100)
    assert lease.renewal_end == 145
    thread.return_value.start.assert_called_once()


def test_renewal_cap_releases_without_stopping_donor(monkeypatch, caplog):
    ctx = context()
    clock = FakeClock()
    monkeypatch.setattr(cold_start.time, "monotonic", clock.monotonic)
    lease = cold_start.DonorLease.__new__(cold_start.DonorLease)
    lease.ctx, lease.token, lease.ranks = ctx, "token", 2
    lease.deadline = 60
    lease.renewal_end = 45
    lease.stopped = MagicMock()
    lease.stopped.wait.side_effect = lambda seconds: clock.sleep(seconds) or False
    ctx.mx_client.donor_lease.return_value = True
    with patch.object(cold_start, "_ready", return_value=False):
        lease._run()
    assert clock.now == 45
    assert [call.args[2] for call in ctx.mx_client.donor_lease.call_args_list] == [
        p2p_pb2.DONOR_LEASE_ACTION_RENEW,
        p2p_pb2.DONOR_LEASE_ACTION_RENEW,
        p2p_pb2.DONOR_LEASE_ACTION_RELEASE,
    ]
    assert ctx.mx_client.donor_lease.call_args.args[1] == "token"
    assert "renewal limit reached" in caplog.text
    lease.stopped.set.assert_not_called()


def test_renewal_rpc_is_bounded_by_remaining_cap(monkeypatch):
    ctx = context()
    clock = FakeClock()
    monkeypatch.setattr(cold_start.time, "monotonic", clock.monotonic)
    lease = cold_start.DonorLease.__new__(cold_start.DonorLease)
    lease.ctx, lease.token, lease.ranks = ctx, "token", 2
    lease.deadline = 60
    lease.renewal_end = 25
    lease.stopped = MagicMock()
    lease.stopped.wait.side_effect = lambda seconds: clock.sleep(seconds) or False

    def rpc(*args, timeout=10):
        if args[2] == p2p_pb2.DONOR_LEASE_ACTION_RENEW:
            assert timeout == 5
            clock.sleep(timeout)
            raise RuntimeError("RPC timed out")
        return True

    ctx.mx_client.donor_lease.side_effect = rpc
    with patch.object(cold_start, "_ready") as ready:
        lease._run()
    ready.assert_not_called()
    assert [call.args[2] for call in ctx.mx_client.donor_lease.call_args_list] == [
        p2p_pb2.DONOR_LEASE_ACTION_RENEW,
        p2p_pb2.DONOR_LEASE_ACTION_RELEASE,
    ]
    assert clock.now == 25


def test_cap_release_failure_does_not_raise(monkeypatch, caplog):
    ctx = context()
    lease = cold_start.DonorLease.__new__(cold_start.DonorLease)
    lease.ctx, lease.token, lease.ranks = ctx, "token", 2
    lease.deadline = 60
    lease.renewal_end = 5
    lease.stopped = MagicMock()
    lease.stopped.wait.return_value = False
    ctx.mx_client.donor_lease.side_effect = RuntimeError("unavailable")
    monkeypatch.setattr(cold_start.time, "monotonic", lambda: 5)
    lease._run()
    assert ctx.mx_client.donor_lease.call_args.args[2] == p2p_pb2.DONOR_LEASE_ACTION_RELEASE
    assert "it will expire" in caplog.text


def test_acquisition_deadline_starts_before_acquire_rpc():
    ctx = context()
    ctx.mx_client.donor_lease.return_value = True
    with (
        patch.object(cold_start, "_ready", return_value=False),
        patch.object(cold_start.time, "monotonic", return_value=123),
        patch.object(cold_start, "DonorLease") as lease,
    ):
        cold_start.coordinate(ctx, 2)
    lease.assert_called_once_with(ctx, lease.call_args.args[1], 2, 123)


@pytest.mark.parametrize("overrides", [
    {"pipeline_parallel_size": 2},
    {"data_parallel_size": 2},
    {"data_parallel_size": 2, "enable_expert_parallel": True},
])
def test_parallel_worker_groups(overrides):
    ctx = context()
    cfg = config()
    for name, value in overrides.items():
        setattr(cfg.parallel_config, name, value)
    count = 4
    ctx.adapter.all_gather_state.side_effect = lambda value: tuple(
        (value[0], rank, rank) for rank in range(count)
    )
    with patch.object(cold_start, "_ready", return_value=True) as ready:
        cold_start.coordinate(ctx, cold_start_expected_workers(cfg))
    assert ready.call_args.args == (ctx, count)
    assert 0 < ready.call_args.kwargs["timeout"] <= 30


def test_mismatched_worker_group_is_rejected():
    ctx = context()
    ctx.adapter.all_gather_state.return_value = [([], 0, 0)]
    ctx.adapter.all_gather_state.side_effect = None
    with pytest.raises(RuntimeError, match="worker group"):
        cold_start.coordinate(ctx, 2)
    ctx.mx_client.list_sources.assert_not_called()


def test_elastic_ep_rejected():
    cfg = config()
    cfg.parallel_config.enable_elastic_ep = True
    with pytest.raises(RuntimeError, match="elastic"):
        cold_start_expected_workers(cfg)


@pytest.mark.parametrize("name", [
    "prefill_context_parallel_size",
    "decode_context_parallel_size",
])
def test_context_parallel_rejected(name):
    cfg = config()
    setattr(cfg.parallel_config, name, 2)
    with pytest.raises(RuntimeError, match="context parallelism"):
        cold_start_expected_workers(cfg)


@pytest.mark.parametrize("name", [
    "prefill_context_parallel_size",
    "decode_context_parallel_size",
])
def test_context_parallel_size_one_allowed(name):
    cfg = config()
    setattr(cfg.parallel_config, name, 1)
    assert cold_start_expected_workers(cfg) == 2


def test_speculative_config_does_not_reject_main():
    ctx = context()
    cfg = config()
    cfg.speculative_config = object()
    with patch.object(cold_start, "_ready", return_value=True):
        cold_start.coordinate(ctx, cold_start_expected_workers(cfg))


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def monotonic(self):
        return self.now

    def sleep(self, duration):
        self.sleeps.append(duration)
        self.now += duration


def test_busy_donor_timeout_falls_back(monkeypatch, caplog):
    ctx = context()
    ctx.mx_client.donor_lease.return_value = False
    clock = FakeClock()
    monkeypatch.setenv("MX_P2P_COLD_START_WAIT_TIMEOUT_SECS", "3")
    with (
        patch.object(cold_start.time, "monotonic", clock.monotonic),
        patch.object(cold_start.time, "sleep", clock.sleep),
        patch.object(cold_start, "_ready", return_value=False),
    ):
        cold_start.coordinate(ctx, 2)
    assert clock.sleeps == [2, 1]
    assert ctx.adapter.broadcast_state.call_args.args[0] == ("fallback", "")
    assert "timed out" in caplog.text
    assert ctx.skip_rdma is True
    assert ctx.p2p_enabled is True


@pytest.mark.parametrize("operation", ["ready", "acquire", "recheck"])
def test_rpc_deadline_falls_back_and_bounds_timeout(monkeypatch, operation):
    ctx = context()
    clock = FakeClock()
    monkeypatch.setenv("MX_P2P_COLD_START_WAIT_TIMEOUT_SECS", "5")
    calls = []

    def ready(ctx, ranks, timeout):
        calls.append(timeout)
        if operation == "ready" or (operation == "recheck" and len(calls) == 2):
            clock.now += timeout
            raise RuntimeError("RPC deadline exceeded")
        return False

    def lease(identity, token, action, ttl, timeout):
        assert 0 < timeout <= 5
        if operation == "acquire" and action == p2p_pb2.DONOR_LEASE_ACTION_ACQUIRE:
            clock.now += timeout
            raise RuntimeError("RPC deadline exceeded")
        return True

    ctx.mx_client.donor_lease.side_effect = lease
    with (
        patch.object(cold_start.time, "monotonic", clock.monotonic),
        patch.object(cold_start, "_ready", side_effect=ready),
    ):
        cold_start.coordinate(ctx, 2)
    assert ctx.adapter.broadcast_state.call_args.args[0] == ("fallback", "")
    assert all(0 < t <= 5 for t in calls)


@pytest.mark.parametrize("release_fails", [False, True])
def test_failed_acquire_releases_token_after_deadline(monkeypatch, release_fails):
    ctx = context()
    clock = FakeClock()
    monkeypatch.setenv("MX_P2P_COLD_START_WAIT_TIMEOUT_SECS", "5")

    def lease(identity, token, action, ttl, timeout):
        if action == p2p_pb2.DONOR_LEASE_ACTION_ACQUIRE:
            clock.now += timeout
            raise RuntimeError("acquire response lost")
        if release_fails:
            raise RuntimeError("release failed")
        return True

    ctx.mx_client.donor_lease.side_effect = lease
    with (
        patch.object(cold_start.time, "monotonic", clock.monotonic),
        patch.object(cold_start, "_ready", return_value=False),
    ):
        cold_start.coordinate(ctx, 2)

    acquire, release = ctx.mx_client.donor_lease.call_args_list
    assert acquire.args[2] == p2p_pb2.DONOR_LEASE_ACTION_ACQUIRE
    assert release.args[2] == p2p_pb2.DONOR_LEASE_ACTION_RELEASE
    assert acquire.args[1] == release.args[1]
    assert release.kwargs["timeout"] == 1
    assert ctx.adapter.broadcast_state.call_args.args[0] == ("fallback", "")


def test_failed_initial_discovery_does_not_release():
    ctx = context()
    with patch.object(cold_start, "_ready", side_effect=RuntimeError("unavailable")):
        cold_start.coordinate(ctx, 2)
    ctx.mx_client.donor_lease.assert_not_called()
    assert ctx.adapter.broadcast_state.call_args.args[0] == ("fallback", "")


def test_timeout_decision_broadcast_to_other_ranks():
    ctx = context()
    ctx.global_rank = ctx.worker_rank = 1
    ctx.adapter.broadcast_state.side_effect = None
    ctx.adapter.broadcast_state.return_value = ("fallback", "")
    cold_start.coordinate(ctx, 2)
    ctx.mx_client.donor_lease.assert_not_called()
    ctx.adapter.broadcast_state.assert_called_once_with(None)
    assert ctx.skip_rdma is True
    assert ctx.p2p_enabled is True


@pytest.mark.parametrize("release_fails", [False, True])
def test_acquired_lease_released_when_recheck_fails(release_fails, caplog):
    ctx = context()
    ctx.mx_client.donor_lease.side_effect = [
        True, RuntimeError("release failed") if release_fails else True
    ]
    with patch.object(cold_start, "_ready", side_effect=[False, RuntimeError("failed")]):
        cold_start.coordinate(ctx, 2)
    assert ctx.mx_client.donor_lease.call_args.args[2] == p2p_pb2.DONOR_LEASE_ACTION_RELEASE
    assert ctx.adapter.broadcast_state.call_args.args[0] == ("fallback", "")
    assert "MX cold start election failed (failed)" in caplog.text
    if release_fails:
        assert "MX donor lease release failed or timed out" in caplog.text


class DeadlineExceeded(grpc.RpcError):
    def code(self):
        return grpc.StatusCode.DEADLINE_EXCEEDED


class RpcFailure(grpc.RpcError):
    def __init__(self, status):
        self.status = status

    def code(self):
        return self.status


def test_transient_source_discovery_failure_retries():
    ctx = context()
    clock = FakeClock()
    ctx.mx_client.list_sources.side_effect = [
        RpcFailure(grpc.StatusCode.UNAVAILABLE),
        sources(0, 1),
    ]
    with (
        patch.object(cold_start.time, "monotonic", clock.monotonic),
        patch.object(cold_start.time, "sleep", clock.sleep),
        patch.object(cold_start, "accelerators_compatible", return_value=True),
    ):
        cold_start.coordinate(ctx, 2)
    assert ctx.adapter.broadcast_state.call_args.args[0] == ("follower", "")
    assert clock.sleeps == [1]
    assert ctx.mx_client.list_sources.call_count == 2
    assert ctx.mx_client.list_sources.call_args.kwargs["timeout"] == 2


@pytest.mark.parametrize("owned", [True, False])
def test_failed_acquire_reconciles_ownership(owned):
    ctx = context()
    clock = FakeClock()
    ctx.mx_client.donor_lease.side_effect = [
        RpcFailure(grpc.StatusCode.UNAVAILABLE),
        owned,
    ]
    with (
        patch.object(cold_start.time, "monotonic", clock.monotonic),
        patch.object(cold_start.time, "sleep", clock.sleep),
        patch.object(cold_start, "_ready", side_effect=[False, not owned, False]),
        patch.object(cold_start, "DonorLease"),
    ):
        cold_start.coordinate(ctx, 2)
    calls = ctx.mx_client.donor_lease.call_args_list
    assert [call.args[2] for call in calls] == [
        p2p_pb2.DONOR_LEASE_ACTION_ACQUIRE,
        p2p_pb2.DONOR_LEASE_ACTION_RENEW,
    ]
    assert calls[0].args[1] == calls[1].args[1]
    assert ctx.adapter.broadcast_state.call_args.args[0][0] == (
        "donor" if owned else "follower"
    )


def test_persistent_rpc_failure_falls_back_after_retry_window():
    ctx = context()
    clock = FakeClock()
    ctx.mx_client.list_sources.side_effect = RpcFailure(grpc.StatusCode.UNAVAILABLE)
    with (
        patch.object(cold_start.time, "monotonic", clock.monotonic),
        patch.object(cold_start.time, "sleep", clock.sleep),
    ):
        cold_start.coordinate(ctx, 2)
    assert clock.now == cold_start.RPC_RETRY_WINDOW
    assert ctx.mx_client.list_sources.call_count == 3
    assert ctx.adapter.broadcast_state.call_args.args[0] == ("fallback", "")
    assert ctx.skip_rdma


def test_persistent_acquire_failure_falls_back():
    ctx = context()
    clock = FakeClock()
    ctx.mx_client.donor_lease.side_effect = RpcFailure(grpc.StatusCode.UNAVAILABLE)
    with (
        patch.object(cold_start.time, "monotonic", clock.monotonic),
        patch.object(cold_start.time, "sleep", clock.sleep),
        patch.object(cold_start, "_ready", return_value=False),
    ):
        cold_start.coordinate(ctx, 2)
    calls = ctx.mx_client.donor_lease.call_args_list
    assert [call.args[2] for call in calls] == [
        p2p_pb2.DONOR_LEASE_ACTION_ACQUIRE,
        p2p_pb2.DONOR_LEASE_ACTION_RENEW,
        p2p_pb2.DONOR_LEASE_ACTION_RENEW,
        p2p_pb2.DONOR_LEASE_ACTION_RELEASE,
    ]
    assert len({call.args[1] for call in calls}) == 1
    assert clock.now == cold_start.RPC_RETRY_WINDOW
    assert ctx.skip_rdma


def test_permanent_rpc_error_is_not_retried():
    ctx = context()
    ctx.mx_client.list_sources.side_effect = RpcFailure(grpc.StatusCode.PERMISSION_DENIED)
    with patch.object(cold_start.time, "sleep") as sleep:
        cold_start.coordinate(ctx, 2)
    ctx.mx_client.list_sources.assert_called_once()
    sleep.assert_not_called()
    assert ctx.skip_rdma


def test_retry_stays_within_cold_start_deadline(monkeypatch):
    ctx = context()
    clock = FakeClock()
    monkeypatch.setenv("MX_P2P_COLD_START_WAIT_TIMEOUT_SECS", "2")
    ctx.mx_client.list_sources.side_effect = RpcFailure(grpc.StatusCode.UNAVAILABLE)
    with (
        patch.object(cold_start.time, "monotonic", clock.monotonic),
        patch.object(cold_start.time, "sleep", clock.sleep),
    ):
        cold_start.coordinate(ctx, 2)
    assert clock.now == 2
    assert all(
        call.kwargs["timeout"] <= 2
        for call in ctx.mx_client.list_sources.call_args_list
    )
    assert ctx.skip_rdma


@pytest.mark.parametrize("operation", ["ready", "acquire", "recheck"])
def test_rpc_deadline_just_before_local_deadline_falls_back(monkeypatch, operation):
    ctx = context()
    clock = FakeClock()
    monkeypatch.setenv("MX_P2P_COLD_START_WAIT_TIMEOUT_SECS", "5")
    error = DeadlineExceeded()
    calls = 0

    def ready(ctx, ranks, timeout):
        nonlocal calls
        calls += 1
        if operation == "ready" or (operation == "recheck" and calls == 2):
            clock.now += timeout - 0.001
            raise error
        return False

    def lease(identity, token, action, ttl, timeout):
        if operation == "acquire" and action == p2p_pb2.DONOR_LEASE_ACTION_ACQUIRE:
            clock.now += timeout - 0.001
            raise error
        return True

    ctx.mx_client.donor_lease.side_effect = lease
    with (
        patch.object(cold_start.time, "monotonic", clock.monotonic),
        patch.object(cold_start.time, "sleep", clock.sleep),
        patch.object(cold_start, "_ready", side_effect=ready),
    ):
        cold_start.coordinate(ctx, 2)
    assert ctx.adapter.broadcast_state.call_args.args[0] == ("fallback", "")


def test_rpc_timeout_before_wait_deadline_falls_back(monkeypatch, caplog):
    ctx = context()
    clock = FakeClock()
    monkeypatch.setenv("MX_P2P_COLD_START_WAIT_TIMEOUT_SECS", "60")
    error = DeadlineExceeded()
    with (
        patch.object(cold_start.time, "monotonic", clock.monotonic),
        patch.object(cold_start.time, "sleep", clock.sleep),
        patch.object(cold_start, "_ready", side_effect=error),
    ):
        cold_start.coordinate(ctx, 2)
    assert ctx.adapter.broadcast_state.call_args.args[0] == ("fallback", "")
    assert "MX cold start election failed" in caplog.text
