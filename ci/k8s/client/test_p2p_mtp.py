# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

import test_p2p_k8s


def _phase(rank, role, *, p2p=True):
    return [
        f"[Worker {rank}] MxModelLoader starting "
        f"(model=test, p2p_enabled={p2p}, p2p_role={role})",
        f"[Worker {rank}] [TIMING] RDMA transfer complete: 21 tensors",
        f"[Worker {rank}] MxModelLoader.load_model() COMPLETE in 1.0s",
    ]


def _check(monkeypatch, lines, tp_size=1):
    monkeypatch.setattr(
        test_p2p_k8s, "_all_pod_logs", lambda *args: "\n".join(lines)
    )
    test_p2p_k8s.test_mtp_load_phases("test", True, tp_size)


def _format_log(message, *, style="vllm", second=48, pid=626):
    prefix = f"(EngineCore_DP0 pid={pid})"
    if style == "vllm":
        return f"{prefix} INFO 10-12 23:35:{second} [loader.py:159] {message}"
    return f"{prefix} [2026-10-12 23:35:{second}] INFO loader.py:159: {message}"


def _dual_handler_logs(lines, *, reverse=False):
    styles = ("mx", "vllm") if reverse else ("vllm", "mx")
    return [_format_log(line, style=style) for line in lines for style in styles]


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("tp_size", [1, 2])
def test_mtp_accepts_dual_handler_logs(monkeypatch, reverse, tp_size):
    ranks = [_phase(rank, "main") + _phase(rank, "draft") for rank in range(tp_size)]
    lines = [line for events in zip(*ranks) for line in events]
    _check(monkeypatch, _dual_handler_logs(lines, reverse=reverse), tp_size)


@pytest.mark.parametrize("duplicate", ["same_handler", "later_time", "other_process"])
def test_mtp_rejects_distinct_repeated_start(monkeypatch, duplicate):
    main = [_format_log(line) for line in _phase(0, "main")]
    main.insert(1, _format_log(
        _phase(0, "main")[0],
        style="vllm" if duplicate == "same_handler" else "mx",
        second=49 if duplicate == "later_time" else 48,
        pid=627 if duplicate == "other_process" else 626,
    ))
    with pytest.raises(AssertionError, match="before main completed"):
        _check(monkeypatch, main + _phase(0, "draft"))


def test_mtp_rejects_repeated_start_with_both_handlers(monkeypatch):
    main = _phase(0, "main")
    main.insert(1, main[0])
    with pytest.raises(AssertionError, match="before main completed"):
        _check(monkeypatch, _dual_handler_logs(main + _phase(0, "draft")))


def test_mtp_accepts_interleaved_rank_logs(monkeypatch):
    rank0 = _phase(0, "main") + _phase(0, "draft")
    rank1 = _phase(1, "main") + _phase(1, "draft")
    lines = [line for pair in zip(rank0, rank1) for line in pair]
    _check(monkeypatch, lines, tp_size=2)


@pytest.mark.parametrize("dual_handlers", [False, True])
@pytest.mark.parametrize("role", ["main", "draft"])
def test_mtp_rejects_missing_role_transfer(monkeypatch, role, dual_handlers):
    main = _phase(0, "main")
    draft = _phase(0, "draft")
    (main if role == "main" else draft).pop(1)
    with pytest.raises(AssertionError, match=f"no {role} RDMA transfer"):
        lines = main + draft
        _check(monkeypatch, _dual_handler_logs(lines) if dual_handlers else lines)


def test_mtp_rejects_local_draft(monkeypatch):
    with pytest.raises(AssertionError, match="expected P2P-enabled draft"):
        _check(monkeypatch, _phase(0, "main") + _phase(0, "draft", p2p=False))


@pytest.mark.parametrize("dual_handlers", [False, True])
def test_mtp_rejects_storage_fallback_after_transfer(monkeypatch, dual_handlers):
    draft = _phase(0, "draft")
    draft.insert(2, "[Worker 0] Streaming weights from s3://checkpoint")
    with pytest.raises(AssertionError, match="fell back to ModelStreamer"):
        lines = _phase(0, "main") + draft
        _check(monkeypatch, _dual_handler_logs(lines) if dual_handlers else lines)


def test_mtp_does_not_borrow_transfer_from_another_rank(monkeypatch):
    rank0 = _phase(0, "main") + _phase(0, "draft")
    rank1 = _phase(1, "main") + _phase(1, "draft")
    rank1.pop(4)
    with pytest.raises(AssertionError, match="Rank 1: no draft RDMA transfer"):
        _check(monkeypatch, rank0 + rank1, tp_size=2)


def test_mtp_rejects_incomplete_main(monkeypatch):
    with pytest.raises(AssertionError, match="before main completed"):
        _check(monkeypatch, _phase(0, "main")[:-1] + _phase(0, "draft"))


def test_mtp_skips_when_not_requested():
    with pytest.raises(pytest.skip.Exception):
        test_p2p_k8s.test_mtp_load_phases("test", False, 1)
