# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import test_p2p_k8s


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
