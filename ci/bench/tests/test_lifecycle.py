# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise benchmark process and evidence handling without a cluster."""

import subprocess

import pytest


@pytest.fixture
def lifecycle(monkeypatch):
    from harness import lifecycle

    return lifecycle


@pytest.mark.parametrize("collection_fails", [False, True])
def test_run_failure_collects_evidence_and_stops_local_processes(
    tmp_path, lifecycle, collection_fails
):
    from types import SimpleNamespace

    events = []
    bench = object.__new__(lifecycle.Benchmark)
    bench.root = tmp_path
    bench.k = SimpleNamespace(call=lambda *args: None)
    bench.processes = [
        SimpleNamespace(
            poll=lambda: None,
            terminate=lambda: events.append("terminate"),
            wait=lambda **kwargs: events.append("wait"),
        )
    ]

    def fail_apply(name):
        raise RuntimeError("deployment failed")

    def collect():
        events.append("collect")
        if collection_fails:
            raise RuntimeError("collection failed")

    bench.apply = fail_apply
    bench.collect = collect
    with pytest.raises(RuntimeError):
        bench.run()
    assert events == ["collect", "terminate", "wait"]
    assert (tmp_path / "started").exists()


def test_failed_collection_preserves_existing_evidence(
    tmp_path, lifecycle, monkeypatch
):
    from types import SimpleNamespace

    bench = object.__new__(lifecycle.Benchmark)
    bench.root = tmp_path
    bench.k = SimpleNamespace(command=["kubectl"])
    destination = tmp_path / "publication.json"
    destination.write_text('{"complete": true}')

    def fail(command, **kwargs):
        kwargs["stdout"].write(b"partial")
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(lifecycle.subprocess, "run", fail)
    bench.capture("publication.json", "exec", "test")
    assert destination.read_text() == '{"complete": true}'


@pytest.mark.parametrize("codes", [[None, 1], [1, None]])
def test_publication_failure_does_not_wait_for_other_process(
    lifecycle, monkeypatch, codes
):
    from types import SimpleNamespace

    processes = [SimpleNamespace(poll=lambda code=code: code) for code in codes]
    monkeypatch.setattr(
        lifecycle.time, "sleep", lambda _: pytest.fail("Unexpected wait")
    )
    with pytest.raises(RuntimeError, match="publication failed"):
        lifecycle.wait_for_publications(processes, float("inf"))


def test_publications_share_one_deadline(lifecycle, monkeypatch):
    from types import SimpleNamespace

    clock = [0]
    monkeypatch.setattr(lifecycle.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        lifecycle.time,
        "sleep",
        lambda seconds: clock.__setitem__(0, clock[0] + seconds),
    )
    processes = [
        SimpleNamespace(poll=lambda: 0 if clock[0] >= 5 else None),
        SimpleNamespace(poll=lambda: None),
    ]
    with pytest.raises(TimeoutError, match="publication timed out"):
        lifecycle.wait_for_publications(processes, 10)
    assert clock[0] == 10


def test_completed_publications_do_not_wait(lifecycle, monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(
        lifecycle.time, "sleep", lambda _: pytest.fail("Unexpected wait")
    )
    lifecycle.wait_for_publications([SimpleNamespace(poll=lambda: 0)], 0)
