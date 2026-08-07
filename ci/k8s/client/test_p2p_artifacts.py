# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

import test_p2p_k8s

_VLLM_INSTALL = (
    "[Worker {rank}] [TIMING] vLLM artifact install complete: "
    "name=torch_compile_cache artifact_id=a1 mx_source_id={source_id} "
    "size=30.45 MiB elapsed=1.204s"
)
_EFFECTIVE = (
    "[Worker 0] vLLM selected torch.compile cache directory "
    "/root/.cache/vllm/torch_compile_cache/a531dd9a8f, which ModelExpress installed"
)
_INEFFECTIVE = (
    "[Worker 0] ModelExpress installed torch.compile cache directory/ies "
    "['a531dd9a8f'] but vLLM selected "
    "/root/.cache/vllm/torch_compile_cache/0249c1b5c6, so the transferred cache "
    "was not reused and the engine recompiled."
)


class _NoSleep:
    """Drop-in for the `time` module so polling tests run instantly.

    Each `monotonic` call advances a quarter of the timeout, so the poll loop
    runs a few iterations and then expires. Real sleeping is skipped entirely —
    without this the never-arrives case would block for the full timeout.
    """

    def __init__(self) -> None:
        self._now = 0.0

    def monotonic(self) -> float:
        self._now += test_p2p_k8s.EFFECTIVENESS_TIMEOUT_SECS / 4
        return self._now

    def sleep(self, _seconds: float) -> None:
        return None


def _patch_clock(monkeypatch) -> None:
    monkeypatch.setattr(test_p2p_k8s, "EFFECTIVENESS_POLL_SECS", 0)
    monkeypatch.setattr(test_p2p_k8s, "time", _NoSleep())


def _run_against_logs(monkeypatch, logs: str) -> None:
    _patch_clock(monkeypatch)
    monkeypatch.setattr(
        test_p2p_k8s,
        "_ready_artifact_source_types",
        lambda namespace: {"triton_cache", "torch_compile_cache"},
    )
    monkeypatch.setattr(
        test_p2p_k8s,
        "_pod_logs_by_pod",
        lambda namespace, job_name, container: {"mx-target-0": logs},
    )
    test_p2p_k8s.test_artifact_transfer(
        namespace="mx-ci-vllm",
        require_artifact_transfer=True,
        expected_artifact_sources=1,
        expected_artifact_source_types=set(),
    )


def test_artifact_transfer_accepts_sglang_install_log(monkeypatch) -> None:
    monkeypatch.setattr(
        test_p2p_k8s,
        "_ready_artifact_source_types",
        lambda namespace: {"triton_cache"},
    )
    monkeypatch.setattr(
        test_p2p_k8s,
        "_pod_logs_by_pod",
        lambda namespace, job_name, container: {
            "mx-target-0": "SGLang artifact install complete: name=triton_cache"
        },
    )

    test_p2p_k8s.test_artifact_transfer(
        namespace="mx-ci-sglang",
        require_artifact_transfer=True,
        expected_artifact_sources=1,
        expected_artifact_source_types=set(),
    )


def test_artifact_transfer_accepts_an_effective_compile_cache(monkeypatch) -> None:
    _run_against_logs(
        monkeypatch,
        "\n".join(
            [
                _VLLM_INSTALL.format(rank=0, source_id="623926a656633a00"),
                _VLLM_INSTALL.format(rank=1, source_id="623926a656633a00"),
                _EFFECTIVE,
            ]
        ),
    )


def test_artifact_transfer_rejects_a_cache_the_engine_did_not_reuse(monkeypatch) -> None:
    """The install succeeds and the bytes land; vLLM recompiles anyway."""
    logs = "\n".join(
        [_VLLM_INSTALL.format(rank=0, source_id="623926a656633a00"), _INEFFECTIVE]
    )

    with pytest.raises(AssertionError, match="MX_ARTIFACT_COMPILE_CONFIG_DIGEST"):
        _run_against_logs(monkeypatch, logs)


def test_artifact_transfer_rejects_a_missing_effectiveness_check(monkeypatch) -> None:
    logs = _VLLM_INSTALL.format(rank=0, source_id="623926a656633a00")

    with pytest.raises(AssertionError, match="effectiveness check"):
        _run_against_logs(monkeypatch, logs)


def test_artifact_transfer_waits_for_a_late_effectiveness_check(monkeypatch) -> None:
    """The publisher thread is gated on /health plus a cache-settle interval.

    pytest starts as soon as the target is healthy, so the first log snapshot can
    legitimately predate the effectiveness line. The assertion must poll, not
    fail on the first read.
    """
    install_only = _VLLM_INSTALL.format(rank=0, source_id="623926a656633a00")
    snapshots = iter([install_only, install_only, install_only + "\n" + _EFFECTIVE])

    _patch_clock(monkeypatch)
    monkeypatch.setattr(
        test_p2p_k8s,
        "_ready_artifact_source_types",
        lambda namespace: {"triton_cache", "torch_compile_cache"},
    )
    monkeypatch.setattr(
        test_p2p_k8s,
        "_pod_logs_by_pod",
        lambda namespace, job_name, container: {"mx-target-0": next(snapshots)},
    )

    test_p2p_k8s.test_artifact_transfer(
        namespace="mx-ci-vllm",
        require_artifact_transfer=True,
        expected_artifact_sources=1,
        expected_artifact_source_types=set(),
    )


def test_artifact_transfer_gives_up_when_the_check_never_arrives(monkeypatch) -> None:
    """A publisher that never runs must still fail, not hang."""
    install_only = _VLLM_INSTALL.format(rank=0, source_id="623926a656633a00")

    _patch_clock(monkeypatch)
    monkeypatch.setattr(
        test_p2p_k8s,
        "_ready_artifact_source_types",
        lambda namespace: {"torch_compile_cache"},
    )
    monkeypatch.setattr(
        test_p2p_k8s,
        "_pod_logs_by_pod",
        lambda namespace, job_name, container: {"mx-target-0": install_only},
    )

    with pytest.raises(AssertionError, match="effectiveness check"):
        test_p2p_k8s.test_artifact_transfer(
            namespace="mx-ci-vllm",
            require_artifact_transfer=True,
            expected_artifact_sources=1,
            expected_artifact_source_types=set(),
        )


def test_artifact_transfer_rejects_divergent_source_ids(monkeypatch) -> None:
    """Identically configured targets must agree on the artifact identity."""
    logs = "\n".join(
        [
            _VLLM_INSTALL.format(rank=0, source_id="623926a656633a00"),
            _VLLM_INSTALL.format(rank=1, source_id="0f1e2d3c4b5a6978"),
            _EFFECTIVE,
        ]
    )

    with pytest.raises(AssertionError, match="distinct mx_source_ids"):
        _run_against_logs(monkeypatch, logs)


def test_artifact_transfer_rejects_a_pod_that_skipped_the_check(monkeypatch) -> None:
    """One matching line must not vouch for every pod.

    Three targets install a torch.compile cache; only one logs the effectiveness
    check. On the concatenated string this passed, because a substring test is
    satisfied by a single occurrence anywhere.
    """
    install = _VLLM_INSTALL.format(rank=0, source_id="623926a656633a00")
    per_pod = {
        "mx-target-0": install + "\n" + _EFFECTIVE,
        "mx-target-1": install,
        "mx-target-2": install,
    }

    _patch_clock(monkeypatch)
    monkeypatch.setattr(
        test_p2p_k8s,
        "_ready_artifact_source_types",
        lambda namespace: {"torch_compile_cache"},
    )
    monkeypatch.setattr(
        test_p2p_k8s,
        "_pod_logs_by_pod",
        lambda namespace, job_name, container: per_pod,
    )

    with pytest.raises(AssertionError, match=r"mx-target-1.*mx-target-2"):
        test_p2p_k8s.test_artifact_transfer(
            namespace="mx-ci-vllm",
            require_artifact_transfer=True,
            expected_artifact_sources=1,
            expected_artifact_source_types=set(),
        )


def test_artifact_transfer_accepts_one_check_per_pod(monkeypatch) -> None:
    """A TP>1 pod logs exactly one check for many installing ranks.

    `mark_publish_scheduled` is pod-scoped, so requiring a line per rank would
    fail every multi-rank deployment. Per pod is the invariant that holds.
    """
    per_pod = {
        f"mx-target-{i}": "\n".join(
            [_VLLM_INSTALL.format(rank=r, source_id="623926a656633a00") for r in range(4)]
            + [_EFFECTIVE]
        )
        for i in range(2)
    }

    _patch_clock(monkeypatch)
    monkeypatch.setattr(
        test_p2p_k8s,
        "_ready_artifact_source_types",
        lambda namespace: {"torch_compile_cache"},
    )
    monkeypatch.setattr(
        test_p2p_k8s,
        "_pod_logs_by_pod",
        lambda namespace, job_name, container: per_pod,
    )

    test_p2p_k8s.test_artifact_transfer(
        namespace="mx-ci-vllm",
        require_artifact_transfer=True,
        expected_artifact_sources=1,
        expected_artifact_source_types=set(),
    )
