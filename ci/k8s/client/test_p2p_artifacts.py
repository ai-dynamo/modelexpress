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


def _run_against_logs(monkeypatch, logs: str) -> None:
    monkeypatch.setattr(
        test_p2p_k8s,
        "_ready_artifact_source_types",
        lambda namespace: {"triton_cache", "torch_compile_cache"},
    )
    monkeypatch.setattr(
        test_p2p_k8s,
        "_all_pod_logs",
        lambda namespace, job_name, container: logs,
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
        "_all_pod_logs",
        lambda namespace, job_name, container: (
            "SGLang artifact install complete: name=triton_cache"
        ),
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
