# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Engine ranks walk the same strategies and leave each one the same way.

Each rank runs the strategy chain on its own, but later strategies
(InstantTensor, distributed ModelStreamer) and the engine's post-load barrier
are collective. The chain first keeps only the strategies every rank can run,
in the same order, and stops every rank if one could not check its own.
For each strategy it then gathers every rank's outcome after prepare(), before
any rank loads (a rank that could not recover from the previous strategy
reports that here instead):

    all prepared         -> every rank loads
    any rank failed      -> every rank aborts and falls back together
    any rank fatal       -> every rank raises StrategyRecoveryError

after load(), before finalize():

    all loaded           -> every rank finalizes
    any rank failed      -> every rank falls back to the next strategy together
    any rank fatal       -> every rank raises StrategyRecoveryError

and again after finalize(), before any rank publishes:

    all finalized        -> every rank publishes and returns
    every rank failed    -> every rank falls back to the next strategy together
    otherwise            -> every rank raises StrategyRecoveryError

Run: pytest tests/test_rank_agreement.py
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from modelexpress.adapter import StrategyFailed, StrategyRecoveryError
from modelexpress.load_strategy import execute_load_strategies
from modelexpress.load_strategy.base import LoadStrategy


class _PeerRanks:
    """Adapter stand-in: this rank plus one peer with fixed answers."""

    def __init__(
        self,
        peer_outcomes=None,
        *,
        peer_eligible=None,
        peer_availability_error=None,
        peer_phase=None,
    ):
        self.peer_outcomes = peer_outcomes or {}
        self.peer_eligible = peer_eligible
        self.peer_availability_error = peer_availability_error
        self.peer_phase = peer_phase
        self.eligibility = []
        self.outcomes = []
        self.reinit_for_retry = MagicMock(side_effect=lambda result: result)

    def all_gather_state(self, state):
        phase, value = state
        if phase == "eligible":
            self.eligibility.append(value)
            names, _error = value
            peer_names = names if self.peer_eligible is None else self.peer_eligible
            peer_value = (peer_names, self.peer_availability_error)
        else:
            self.outcomes.append((phase, value))
            # A peer prepares and finalizes unless the test says otherwise.
            default = "succeeded" if phase.startswith(("prepare:", "finalize:")) else None
            peer_value = self.peer_outcomes.get(phase, default)
            assert peer_value is not None, f"no peer outcome for {phase}"
        return (state, (self.peer_phase or phase, peer_value))


class _Strategy(LoadStrategy):
    def __init__(self, name, *, error=None, prepare_error=None, finalize_error=None):
        self.name = name
        self.error = error
        self.prepare_error = prepare_error
        self.finalize_error = finalize_error
        self.prepared = 0
        self.loaded = 0
        self.finalized = 0
        self.aborted = 0
        self.rolled_back = 0

    def is_available(self, ctx):
        return True

    def prepare(self, result, ctx):
        self.prepared += 1
        if self.prepare_error is not None:
            raise self.prepare_error

    def load(self, result, ctx):
        self.loaded += 1
        if self.error is not None:
            raise self.error
        return result

    def finalize(self, result, ctx):
        self.finalized += 1
        if self.finalize_error is not None:
            raise self.finalize_error

    def abort(self, result, ctx):
        self.aborted += 1

    def rollback(self, ctx):
        self.rolled_back += 1


def _ctx(adapter, *, is_reload=False):
    return SimpleNamespace(
        global_rank=0,
        is_reload=is_reload,
        p2p_enabled=False,
        adapter=adapter,
        identity=SimpleNamespace(model_name="m"),
    )


@pytest.fixture
def published(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "modelexpress.load_strategy.publish_source_if_supported",
        lambda result, ctx: calls.append(result.value),
    )
    return calls


def test_all_ranks_loaded_publishes_and_returns(published):
    adapter = _PeerRanks({"load:rdma": "succeeded"})
    model = MagicMock()

    assert execute_load_strategies(model, _ctx(adapter), [_Strategy("rdma")]) is model

    assert published == [model]
    adapter.reinit_for_retry.assert_not_called()


def test_a_rank_that_loaded_falls_back_when_another_rank_failed(published):
    """The #901/#966 split: this rank's RDMA read finished, a peer's did not."""
    adapter = _PeerRanks({"load:rdma": "failed", "load:disk": "succeeded"})
    rdma, disk = _Strategy("rdma"), _Strategy("disk")
    model = MagicMock()

    assert execute_load_strategies(model, _ctx(adapter), [rdma, disk]) is model

    assert adapter.outcomes == [
        ("prepare:rdma", "succeeded"),
        ("load:rdma", "succeeded"),
        ("prepare:disk", "succeeded"),
        ("load:disk", "succeeded"),
        ("finalize:disk", "succeeded"),
    ]
    # The finished rank aborts the attempt before finalize() (no registration),
    # undoes its RDMA state, rebuilds the model it wrote, and publishes only
    # once every rank has loaded.
    assert (rdma.finalized, rdma.aborted, rdma.rolled_back) == (0, 1, 1)
    assert disk.finalized == 1
    adapter.reinit_for_retry.assert_called_once()
    assert published == [model]


def test_a_failed_rank_falls_back_with_a_rank_that_loaded(published):
    adapter = _PeerRanks({"load:rdma": "succeeded", "load:disk": "succeeded"})
    rdma = _Strategy("rdma", error=StrategyFailed("source died", mutated=True))

    execute_load_strategies(MagicMock(), _ctx(adapter), [rdma, _Strategy("disk")])

    assert adapter.outcomes == [
        ("prepare:rdma", "succeeded"),
        ("load:rdma", "failed"),
        ("prepare:disk", "succeeded"),
        ("load:disk", "succeeded"),
        ("finalize:disk", "succeeded"),
    ]
    adapter.reinit_for_retry.assert_called_once()
    assert len(published) == 1


def test_an_unmutated_miss_is_not_reinitialized(published):
    adapter = _PeerRanks({"load:rdma": "succeeded", "load:disk": "succeeded"})
    rdma = _Strategy("rdma", error=StrategyFailed("no source", mutated=False))

    execute_load_strategies(MagicMock(), _ctx(adapter), [rdma, _Strategy("disk")])

    adapter.reinit_for_retry.assert_not_called()


def test_a_reload_falls_back_together_without_reinitializing(published):
    adapter = _PeerRanks({"load:rdma": "failed", "load:disk": "succeeded"})
    rdma = _Strategy("rdma")

    execute_load_strategies(
        MagicMock(), _ctx(adapter, is_reload=True), [rdma, _Strategy("disk")]
    )

    assert rdma.rolled_back == 1
    adapter.reinit_for_retry.assert_not_called()


def test_every_rank_fails_closed_when_another_rank_is_fatal(published):
    adapter = _PeerRanks({"load:rdma": "fatal"})

    with pytest.raises(StrategyRecoveryError, match="Another rank could not recover"):
        execute_load_strategies(
            MagicMock(), _ctx(adapter), [_Strategy("rdma"), _Strategy("disk")]
        )

    assert adapter.outcomes == [("prepare:rdma", "succeeded"), ("load:rdma", "succeeded")]
    assert published == []


def test_a_fatal_rank_still_joins_the_agreement_before_raising(published):
    """Otherwise the other ranks would wait in the gather forever."""
    adapter = _PeerRanks({"load:rdma": "succeeded"})
    error = StrategyRecoveryError("model cannot be restored")
    rdma = _Strategy("rdma", error=error)

    with pytest.raises(StrategyRecoveryError) as raised:
        execute_load_strategies(MagicMock(), _ctx(adapter), [rdma, _Strategy("disk")])

    assert raised.value is error
    assert adapter.outcomes == [("prepare:rdma", "succeeded"), ("load:rdma", "fatal")]
    assert rdma.rolled_back == 1


def test_no_strategy_succeeds_when_every_rank_fails_everywhere(published):
    adapter = _PeerRanks({"load:rdma": "failed", "load:disk": "failed"})
    disk = _Strategy("disk", error=StrategyFailed("disk read failed"))

    with pytest.raises(RuntimeError, match="No loading strategy succeeded"):
        execute_load_strategies(MagicMock(), _ctx(adapter), [_Strategy("rdma"), disk])

    assert published == []


def test_a_strategy_another_rank_cannot_run_is_skipped_on_every_rank(published):
    """GDS (or NIXL) can be available on one node and not another."""
    adapter = _PeerRanks({"load:disk": "succeeded"}, peer_eligible=["rdma", "disk"])
    gds, disk = _Strategy("gds"), _Strategy("disk")

    execute_load_strategies(MagicMock(), _ctx(adapter), [gds, disk])

    assert gds.loaded == 0
    assert adapter.outcomes == [
        ("prepare:disk", "succeeded"),
        ("load:disk", "succeeded"),
        ("finalize:disk", "succeeded"),
    ]
    assert len(published) == 1


def test_no_strategy_eligible_on_every_rank_aborts_once(published):
    adapter = _PeerRanks(peer_eligible=["disk"])
    gds = _Strategy("gds")

    with pytest.raises(RuntimeError, match="No loading strategy is eligible on every rank"):
        execute_load_strategies(MagicMock(), _ctx(adapter), [gds])

    assert gds.loaded == 0
    assert adapter.outcomes == []


def test_ranks_out_of_step_fail_closed(published):
    adapter = _PeerRanks({"load:rdma": "succeeded"}, peer_phase="load:disk")

    with pytest.raises(StrategyRecoveryError, match="Ranks are out of step"):
        execute_load_strategies(MagicMock(), _ctx(adapter), [_Strategy("rdma")])

    assert published == []


def test_a_failed_rank_fails_closed_when_another_rank_is_fatal(published):
    adapter = _PeerRanks({"load:rdma": "fatal"})
    rdma = _Strategy("rdma", error=StrategyFailed("source died", mutated=True))

    with pytest.raises(StrategyRecoveryError, match="Another rank could not recover"):
        execute_load_strategies(MagicMock(), _ctx(adapter), [rdma, _Strategy("disk")])

    assert adapter.outcomes == [("prepare:rdma", "succeeded"), ("load:rdma", "failed")]
    adapter.reinit_for_retry.assert_not_called()
    assert published == []


def test_no_rank_publishes_until_every_rank_finalized(published):
    """This rank post-processed its weights; a peer's post-processing failed."""
    adapter = _PeerRanks({"load:disk": "succeeded", "finalize:disk": "failed"})
    disk, default = _Strategy("disk"), _Strategy("default")

    with pytest.raises(StrategyRecoveryError, match="finalized on only 1 of 2 ranks"):
        execute_load_strategies(MagicMock(), _ctx(adapter), [disk, default])

    # This rank already registered its tensors, so it stops rather than fall back.
    assert adapter.outcomes == [
        ("prepare:disk", "succeeded"),
        ("load:disk", "succeeded"),
        ("finalize:disk", "succeeded"),
    ]
    assert default.loaded == 0
    adapter.reinit_for_retry.assert_not_called()
    assert published == []


def test_a_rank_whose_finalize_failed_reports_it_and_stops(published):
    adapter = _PeerRanks({"load:disk": "succeeded", "finalize:disk": "succeeded"})
    disk = _Strategy("disk", finalize_error=StrategyFailed("OOM", mutated=True))
    default = _Strategy("default")

    with pytest.raises(StrategyRecoveryError, match="finalized on only 1 of 2 ranks"):
        execute_load_strategies(MagicMock(), _ctx(adapter), [disk, default])

    assert adapter.outcomes == [
        ("prepare:disk", "succeeded"),
        ("load:disk", "succeeded"),
        ("finalize:disk", "failed"),
    ]
    assert default.loaded == 0
    assert published == []


def test_ranks_fall_back_together_when_every_finalize_failed(published):
    adapter = _PeerRanks(
        {
            "load:disk": "succeeded",
            "finalize:disk": "failed",
            "load:default": "succeeded",
        }
    )
    disk = _Strategy("disk", finalize_error=StrategyFailed("OOM", mutated=True))
    model = MagicMock()

    assert execute_load_strategies(model, _ctx(adapter), [disk, _Strategy("default")]) is model

    assert adapter.outcomes[-2:] == [
        ("load:default", "succeeded"),
        ("finalize:default", "succeeded"),
    ]
    adapter.reinit_for_retry.assert_called_once()
    assert published == [model]


def test_no_rank_loads_when_a_peer_failed_to_prepare(published):
    """load() can be collective, so a rank that cannot load says so first."""
    adapter = _PeerRanks({"prepare:disk": "failed", "load:default": "succeeded"})
    disk, default = _Strategy("disk"), _Strategy("default")

    execute_load_strategies(MagicMock(), _ctx(adapter), [disk, default])

    assert (disk.prepared, disk.loaded, disk.aborted) == (1, 0, 1)
    assert default.loaded == 1
    # abort() undid prepare(), so the model needs no rebuild.
    adapter.reinit_for_retry.assert_not_called()
    assert len(published) == 1


def test_a_rank_that_failed_to_prepare_reports_it_and_falls_back(published):
    adapter = _PeerRanks({"load:default": "succeeded"})
    disk = _Strategy("disk", prepare_error=StrategyFailed("reload", mutated=True))
    default = _Strategy("default")

    execute_load_strategies(MagicMock(), _ctx(adapter), [disk, default])

    assert adapter.outcomes[0] == ("prepare:disk", "failed")
    assert (disk.loaded, default.loaded) == (0, 1)
    adapter.reinit_for_retry.assert_called_once()


def test_a_rank_that_could_not_recover_stops_every_rank_before_the_next_load(published):
    """The rebuild after a failed strategy can fail on one rank, e.g. out of memory."""
    adapter = _PeerRanks({"load:rdma": "failed"})
    adapter.reinit_for_retry.side_effect = RuntimeError("CUDA out of memory")
    rdma = _Strategy("rdma", error=StrategyFailed("source died", mutated=True))
    disk = _Strategy("disk")

    with pytest.raises(StrategyRecoveryError, match="could not recover") as raised:
        execute_load_strategies(MagicMock(), _ctx(adapter), [rdma, disk])

    assert adapter.outcomes == [
        ("prepare:rdma", "succeeded"),
        ("load:rdma", "failed"),
        ("prepare:disk", "fatal"),
    ]
    assert isinstance(raised.value.__cause__, RuntimeError)
    assert (disk.prepared, disk.loaded) == (0, 0)
    assert published == []


def test_every_rank_stops_when_a_peer_could_not_recover(published):
    adapter = _PeerRanks({"load:rdma": "failed", "prepare:disk": "fatal"})
    disk = _Strategy("disk")

    with pytest.raises(StrategyRecoveryError, match="could not recover"):
        execute_load_strategies(MagicMock(), _ctx(adapter), [_Strategy("rdma"), disk])

    assert (disk.prepared, disk.loaded, disk.aborted) == (1, 0, 1)
    assert published == []


def test_a_failed_recovery_after_the_last_strategy_is_the_cause(published):
    adapter = _PeerRanks({"load:rdma": "failed"})
    adapter.reinit_for_retry.side_effect = RuntimeError("CUDA out of memory")
    rdma = _Strategy("rdma", error=StrategyFailed("source died", mutated=True))

    with pytest.raises(RuntimeError, match="No loading strategy succeeded") as raised:
        execute_load_strategies(MagicMock(), _ctx(adapter), [rdma])

    assert isinstance(raised.value.__cause__, RuntimeError)


def test_a_rank_whose_prepare_was_fatal_does_not_abort(published):
    """abort() undoes a prepare() that succeeded; this one did not."""
    adapter = _PeerRanks()
    disk = _Strategy("disk", prepare_error=StrategyRecoveryError("reload state lost"))

    with pytest.raises(StrategyRecoveryError, match="reload state lost"):
        execute_load_strategies(MagicMock(), _ctx(adapter), [disk, _Strategy("default")])

    assert adapter.outcomes == [("prepare:disk", "fatal")]
    assert (disk.prepared, disk.loaded, disk.aborted) == (1, 0, 0)


class _Unavailable(_Strategy):
    def is_available(self, ctx):
        raise RuntimeError("cannot reach the MX server")


def test_an_availability_error_reaches_every_rank_before_loading(published):
    """A rank whose check raised still joins the eligibility gather first."""
    adapter = _PeerRanks()
    broken, disk = _Unavailable("broken"), _Strategy("disk")

    with pytest.raises(RuntimeError, match="cannot reach the MX server"):
        execute_load_strategies(MagicMock(), _ctx(adapter), [broken, disk])

    assert adapter.eligibility == [(["disk"], "broken: cannot reach the MX server")]
    assert (disk.prepared, disk.loaded) == (0, 0)
    assert adapter.outcomes == []


def test_a_peer_availability_error_stops_every_rank_before_loading(published):
    adapter = _PeerRanks(peer_availability_error="gds: driver missing")
    disk = _Strategy("disk")

    with pytest.raises(RuntimeError, match="Rank 1 could not check .*gds: driver missing"):
        execute_load_strategies(MagicMock(), _ctx(adapter), [disk])

    assert (disk.prepared, disk.loaded) == (0, 0)
    assert adapter.outcomes == []


def test_ranks_that_order_the_loaders_differently_stop_before_loading(published):
    """MX_LOAD_PLUGIN_BEFORE set differently per node reorders the chain."""
    adapter = _PeerRanks(peer_eligible=["disk", "plugin"])
    plugin, disk = _Strategy("plugin"), _Strategy("disk")

    with pytest.raises(RuntimeError, match="order the loading strategies differently"):
        execute_load_strategies(MagicMock(), _ctx(adapter), [plugin, disk])

    assert (plugin.prepared, disk.prepared) == (0, 0)
    assert adapter.outcomes == []


def test_rank_local_loaders_do_not_change_the_shared_order(published):
    adapter = _PeerRanks({"load:rdma": "succeeded"}, peer_eligible=["rdma", "gds", "disk"])
    rdma, disk = _Strategy("rdma"), _Strategy("disk")

    execute_load_strategies(MagicMock(), _ctx(adapter), [rdma, disk])

    assert rdma.loaded == 1
    assert len(published) == 1


@pytest.mark.parametrize("rejects_manifest", [False, True])
def test_missing_tensor_on_one_rank_forces_fallback_before_publication(
    monkeypatch, rejects_manifest
):
    import torch

    from modelexpress.nixl_transfer import ManifestMismatchError, NixlTransferManager

    monkeypatch.setattr(torch.cuda, "set_device", lambda *_: None)
    weight = torch.full((2,), -1.0)
    manager = NixlTransferManager(agent_name="test", device_id=0)
    manager._agent = MagicMock()
    manager._tensors = {"lm_head": weight}
    adapter = _PeerRanks({
        "load:rdma": "succeeded" if rejects_manifest else "failed",
        "load:disk": "succeeded",
    })
    rdma, disk = _Strategy("rdma"), _Strategy("disk")
    publications = []
    monkeypatch.setattr(
        "modelexpress.load_strategy.publish_source_if_supported",
        lambda *_: publications.append(weight.clone()),
    )

    def receive(result, ctx):
        rdma.loaded += 1
        if rejects_manifest:
            try:
                manager.receive_from_source(
                    source_metadata=b"", source_tensors=[], remote_agent_name="source"
                )
            except ManifestMismatchError as error:
                # The RDMA strategy reports exhausted source candidates as a miss.
                raise StrategyFailed("No RDMA source succeeded", mutated=False) from error
        else:
            weight.fill_(7.0)
        return result

    def fallback(result, ctx):
        assert publications == []
        assert rdma.finalized == 0
        disk.loaded += 1
        weight.fill_(42.0)
        return result

    monkeypatch.setattr(rdma, "load", receive)
    monkeypatch.setattr(disk, "load", fallback)
    execute_load_strategies(MagicMock(), _ctx(adapter), [rdma, disk])

    assert adapter.outcomes == [
        ("prepare:rdma", "succeeded"),
        ("load:rdma", "failed" if rejects_manifest else "succeeded"),
        ("prepare:disk", "succeeded"),
        ("load:disk", "succeeded"),
        ("finalize:disk", "succeeded"),
    ]
    assert (rdma.loaded, rdma.finalized, rdma.aborted, rdma.rolled_back) == (1, 0, 1, 1)
    assert (disk.loaded, disk.finalized) == (1, 1)
    assert adapter.reinit_for_retry.call_count == (0 if rejects_manifest else 1)
    manager._agent.prep_xfer_dlist.assert_not_called()
    assert len(publications) == 1
    assert torch.equal(publications[0], torch.full((2,), 42.0))
