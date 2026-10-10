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


def test_mtp_accepts_interleaved_rank_logs(monkeypatch):
    rank0 = _phase(0, "main") + _phase(0, "draft")
    rank1 = _phase(1, "main") + _phase(1, "draft")
    lines = [line for pair in zip(rank0, rank1) for line in pair]
    _check(monkeypatch, lines, tp_size=2)


@pytest.mark.parametrize("role", ["main", "draft"])
def test_mtp_rejects_missing_role_transfer(monkeypatch, role):
    main = _phase(0, "main")
    draft = _phase(0, "draft")
    (main if role == "main" else draft).pop(1)
    with pytest.raises(AssertionError, match=f"no {role} RDMA transfer"):
        _check(monkeypatch, main + draft)


def test_mtp_rejects_local_draft(monkeypatch):
    with pytest.raises(AssertionError, match="expected P2P-enabled draft"):
        _check(monkeypatch, _phase(0, "main") + _phase(0, "draft", p2p=False))


def test_mtp_rejects_storage_fallback_after_transfer(monkeypatch):
    draft = _phase(0, "draft")
    draft.insert(2, "[Worker 0] Streaming weights from s3://checkpoint")
    with pytest.raises(AssertionError, match="fell back to ModelStreamer"):
        _check(monkeypatch, _phase(0, "main") + draft)


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
