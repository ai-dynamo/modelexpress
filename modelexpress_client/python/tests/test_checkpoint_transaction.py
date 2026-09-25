# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from threading import Condition
from types import SimpleNamespace

import pytest
from modelexpress_rl.inference.checkpoint_selection import CheckpointChanges
from modelexpress_rl.inference.checkpoint_transaction import (
    CheckpointTransactionError,
    refit_checkpoint_collectively,
)
from modelexpress_rl.inference.receiver import PreparedCheckpoint


class Collective:
    def __init__(self, size=2):
        self.size = size
        self.condition = Condition()
        self.rounds = {}
        self.sequences = [0] * size

    def gather(self, rank, value):
        with self.condition:
            sequence = self.sequences[rank]
            self.sequences[rank] += 1
            current = self.rounds.setdefault(sequence, {})
            current[rank] = value
            self.condition.notify_all()
            if not self.condition.wait_for(
                lambda: len(current) == self.size, timeout=3
            ):
                raise TimeoutError("test collective timed out")
            return tuple(current[i] for i in range(self.size))


def run_ranks(tmp_path, *, unsupported=False, failure=None, metadata=True, drift=False):
    collective = Collective()
    events = []
    state = [{"weight": "base", "version": "base", "fenced": False} for _ in range(2)]

    def run(rank):
        def step(phase):
            events.append((phase, rank))
            if failure == (phase, rank):
                raise RuntimeError(f"injected {phase}")

        @contextmanager
        def locked():
            step("enter")
            try:
                yield
            finally:
                step("exit")

        def prepare(checkpoint):
            step("stage")
            assert checkpoint.changes.names == {"weight"}
            if unsupported and rank == 1:
                return None
            return SimpleNamespace(install=lambda: install("partial"))

        def install(mode):
            state[rank]["weight"] = "target"
            step(mode)

        def activate():
            assert events.count(("sync", 0)) == events.count(("sync", 1)) == 1
            assert ("exit", 0) in events and ("exit", 1) in events
            step("activate")

        def commit():
            assert ("activate", 0) in events and ("activate", 1) in events
            state[rank]["version"] = "target"
            step("commit")

        def fence():
            state[rank]["fenced"] = True
            events.append(("fence", rank))

        checkpoint = PreparedCheckpoint(
            "target",
            tmp_path,
            {},
            CheckpointChanges("base", "target", frozenset({"weight"}))
            if metadata and rank == 0
            else None,
        )
        try:
            mode = refit_checkpoint_collectively(
                checkpoint,
                serving_version="other" if drift and rank == 1 else "base",
                world_size=2,
                all_gather=lambda value: collective.gather(rank, value),
                installation_context=locked,
                prepare_partial=prepare,
                install_full=lambda: install("full"),
                synchronize=lambda: step("sync"),
                activate=activate,
                commit_version=commit,
                fence=fence,
            )
            events.append(("returned", rank))
            return mode
        except (CheckpointTransactionError, RuntimeError) as error:
            return error

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run, rank) for rank in range(2)]
        results = [future.result(timeout=10) for future in futures]
    return results, events, state


def test_partial_commits_only_after_all_ranks_install_and_release_locks(tmp_path):
    results, events, state = run_ranks(tmp_path)
    assert results == ["partial", "partial"]
    assert all(
        item == {"weight": "target", "version": "target", "fenced": False}
        for item in state
    )
    assert not any(phase == "full" for phase, _ in events)


@pytest.mark.parametrize("metadata,unsupported", [(True, True), (False, False)])
def test_collective_full_fallback_precedes_any_partial_mutation(
    tmp_path, metadata, unsupported
):
    results, events, state = run_ranks(
        tmp_path, metadata=metadata, unsupported=unsupported
    )
    assert results == ["full", "full"]
    assert not any(phase == "partial" for phase, _ in events)
    assert all(item["version"] == "target" for item in state)


@pytest.mark.parametrize(
    "phase", ["enter", "stage", "partial", "sync", "exit", "activate", "commit"]
)
def test_rank_failure_fences_every_rank_without_resuming(tmp_path, phase):
    results, events, state = run_ranks(tmp_path, failure=(phase, 1))
    assert all(isinstance(result, CheckpointTransactionError) for result in results)
    assert all(item["fenced"] for item in state)
    assert not any(event in ("returned", "full") for event, _ in events)
    if phase in ("enter", "stage"):
        assert all(item["weight"] == "base" for item in state)
    if phase != "commit":
        assert all(item["version"] == "base" for item in state)


def test_version_disagreement_never_enters_installation(tmp_path):
    results, events, state = run_ranks(tmp_path, drift=True)
    assert all(isinstance(result, CheckpointTransactionError) for result in results)
    assert all(item["weight"] == "base" and item["fenced"] for item in state)
    assert not any(event == "enter" for event, _ in events)


def test_failed_full_fallback_also_fences_both_ranks(tmp_path):
    results, events, state = run_ranks(tmp_path, unsupported=True, failure=("full", 1))
    assert all(isinstance(result, CheckpointTransactionError) for result in results)
    assert all(item["fenced"] and item["version"] == "base" for item in state)
    assert not any(event in ("activate", "commit", "returned") for event, _ in events)


def test_transport_timeout_fences_without_commit(tmp_path):
    events = []

    def timed_out(_value):
        raise TimeoutError("rank lost")

    with pytest.raises(TimeoutError, match="rank lost"):
        refit_checkpoint_collectively(
            PreparedCheckpoint("target", tmp_path, {}),
            serving_version="base",
            world_size=2,
            all_gather=timed_out,
            installation_context=lambda: None,
            prepare_partial=lambda _: None,
            install_full=lambda: events.append("install"),
            synchronize=lambda: None,
            activate=lambda: events.append("activate"),
            commit_version=lambda: events.append("commit"),
            fence=lambda: events.append("fence"),
        )
    assert events == ["fence"]
