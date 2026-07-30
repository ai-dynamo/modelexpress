# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path


_REPO_ROOT = Path(__file__).parents[3]
_TRTLLM_EXAMPLE = (
    _REPO_ROOT / "examples" / "p2p_transfer_k8s" / "client" / "trtllm"
)


def test_example_is_a_normal_scalable_deployment() -> None:
    manifest = (_TRTLLM_EXAMPLE / "trtllm-single-node-p2p.yaml").read_text()

    assert manifest.count("kind: Deployment") == 1
    assert "name: mx-trtllm\n" in manifest
    assert "replicas: 1" in manifest
    assert "initContainers:" not in manifest
    assert "prepare-model-metadata" not in manifest
    assert "mx-trtllm-source" not in manifest
    assert "mx-trtllm-target" not in manifest
    assert "claimName: model-cache" in manifest
    assert "mountPath: /models" in manifest
    assert "MODEL_EXPRESS_SOURCE" not in manifest
    assert "name: nvcr-imagepullsecret" in manifest
    assert "command:\n            - trtllm-serve" in manifest
    assert "checkpoint_format: MX" in manifest
    assert "server_url: modelexpress-server:8001" in manifest


def test_legacy_trtllm_examples_and_patch_bundle_are_removed() -> None:
    legacy_paths = [
        _TRTLLM_EXAMPLE / "Dockerfile.ph3-gcp-gb200",
        _TRTLLM_EXAMPLE / "kimi-disagg-mx-tp8-dgd.yaml",
        _TRTLLM_EXAMPLE / "kimi-source-decode-dgd.yaml",
        _TRTLLM_EXAMPLE / "mx-infra-decode.yaml",
        _TRTLLM_EXAMPLE / "llama-p2p.yaml",
        _REPO_ROOT / "ci" / "k8s" / "client" / "trt-llm",
        _REPO_ROOT / "trtllm_patches" / "v1.3.0rc5",
    ]

    assert not any(path.exists() for path in legacy_paths)


def test_documentation_points_to_the_native_trtllm_example() -> None:
    documentation = [
        _REPO_ROOT / "README.md",
        _REPO_ROOT / "CONTRIBUTING.md",
        _REPO_ROOT / "ci" / "TEST_PLAN.md",
        _REPO_ROOT / "modelexpress_client" / "python" / "README.md",
        _REPO_ROOT / "examples" / "p2p_transfer_k8s" / "README.md",
        _REPO_ROOT / "examples" / "p2p_transfer_k8s" / "client" / "README.md",
        _TRTLLM_EXAMPLE / "README.md",
    ]

    for path in documentation:
        text = path.read_text()
        assert "llama-p2p.yaml" not in text, path
        assert "PRESHARDED" not in text, path

    assert "trtllm-single-node-p2p.yaml" in documentation[-2].read_text()
    assert "checkpoint_format=\"MX\"" in documentation[-1].read_text()
