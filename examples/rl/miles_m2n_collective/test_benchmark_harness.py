# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for the matched MILES benchmark harness."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml

EXAMPLE_DIR = Path(__file__).parent
RUNTIME_IMAGE = "example.invalid/runtime@sha256:" + "1" * 64
MX_SERVER_IMAGE = "example.invalid/server@sha256:" + "2" * 64
ACTOR_PREFIX = "(ActorModel pid=4132, ip=10.0.0.7, actor_name=actor_cell0_rank0) "


def _load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, EXAMPLE_DIR / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


run_miles = _load_module("run_miles_under_test", "run_miles.py")
parse_timings = _load_module("parse_miles_timings_under_test", "parse_miles_timings.py")


def _timer(name: str, elapsed: float, *, rank: int = 0) -> str:
    prefix = ACTOR_PREFIX.replace("actor_cell0_rank0", f"actor_cell0_rank{rank}")
    return f"{prefix}INFO Timer {name} end (elapsed: {elapsed:.1f}s)"


def _render_manifest(mode: str, **overrides: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment.pop("MX_MILES_VERIFY_TENSOR_EQUALITY", None)
    environment.update(
        {
            "KUBE_CONTEXT": "test-context",
            "MILES_BENCH_BLOCK": "2",
            "MILES_BENCH_REPETITION": "3",
            "MILES_BENCH_ROLLOUTS": "11",
            "MILES_BENCH_RUN_LABEL": "matched-run",
            "MILES_WEIGHT_TRANSFER_MODE": mode,
            "MX_SERVER_IMAGE": MX_SERVER_IMAGE,
            "RUN_ID": "sample",
            "RUNTIME_IMAGE": RUNTIME_IMAGE,
        }
    )
    environment.update(overrides)
    return subprocess.run(
        [str(EXAMPLE_DIR / "deploy.sh"), "render"],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )


def test_external_train_args_select_only_external_protocol():
    args = run_miles.build_train_args(
        Path("/models/qwen"),
        Path("/datasets/gsm8k"),
        transfer_mode="external",
        num_rollouts=11,
    )

    assert "--num-rollout 11" in args
    assert "--update-weight-transfer-mode external" in args
    assert (
        "--update-weight-transfer-protocol "
        "modelexpress_rl.collective.integrations.miles_protocol:build_protocol" in args
    )
    assert "--update-weight-transfer-mode broadcast" not in args


def test_broadcast_train_args_have_no_external_protocol():
    args = run_miles.build_train_args(
        Path("/models/qwen"),
        Path("/datasets/gsm8k"),
        transfer_mode="broadcast",
        num_rollouts=11,
    )

    assert "--num-rollout 11" in args
    assert "--update-weight-transfer-mode broadcast" in args
    assert "--update-weight-transfer-protocol" not in args
    assert "modelexpress_rl" not in args


@pytest.mark.parametrize("value", ["", "modelexpress", "p2p", "BROADCAST"])
def test_transfer_mode_rejects_values_outside_matched_arms(value):
    with pytest.raises(ValueError, match="MILES_WEIGHT_TRANSFER_MODE"):
        run_miles.validate_transfer_mode(value)


@pytest.mark.parametrize("value", ["0", "-1", "abc", "1.5"])
def test_rollout_count_must_be_a_positive_integer(value):
    with pytest.raises(ValueError, match="MILES_BENCH_ROLLOUTS"):
        run_miles.validate_rollouts(value)


def test_broadcast_environment_excludes_model_express_and_plugin(monkeypatch):
    for key, value in {
        "LD_LIBRARY_PATH": "/lib",
        "LD_PRELOAD": "/lib/libnccl.so.2",
        "MX_SERVER_ADDRESS": "mx-server:8000",
        "MX_MILES_RUN_ID": "run-1",
        "NCCL_CUMEM_ENABLE": "1",
        "SGLANG_NCCL_SO_PATH": "/lib/libnccl.so.2",
        "SGLANG_PLUGINS": "modelexpress_miles_collective",
    }.items():
        monkeypatch.setenv(key, value)

    forwarded = run_miles.forwarded_environment("broadcast")

    assert forwarded["LD_PRELOAD"] == "/lib/libnccl.so.2"
    assert forwarded["NCCL_CUMEM_ENABLE"] == "1"
    assert "MX_SERVER_ADDRESS" not in forwarded
    assert "MX_MILES_RUN_ID" not in forwarded
    assert "SGLANG_PLUGINS" not in forwarded


def test_parser_emits_complete_ordered_update_records(tmp_path):
    log_path = tmp_path / "actor.log"
    log_path.write_text(
        "\n".join(
            (
                "noise",
                _timer("update_weights_implementation", 9.1),
                _timer("finalize_and_resume_engines", 0.4),
                _timer("update_weights", 10.2),
                _timer("update_weights_implementation", 3.2),
                _timer("finalize_and_resume_engines", 0.3),
                _timer("update_weights", 3.8),
            )
        )
    )

    result = parse_timings.parse_log(log_path, expected_updates=2)

    assert result["observed_update_count"] == 2
    assert result["updates"] == [
        {
            "classification": "cold",
            "finalize_and_resume_engines_s": 0.4,
            "index": 0,
            "update_weights_implementation_s": 9.1,
            "update_weights_s": 10.2,
        },
        {
            "classification": "steady",
            "finalize_and_resume_engines_s": 0.3,
            "index": 1,
            "update_weights_implementation_s": 3.2,
            "update_weights_s": 3.8,
        },
    ]


def test_parser_rejects_incomplete_triplet(tmp_path):
    log_path = tmp_path / "actor.log"
    log_path.write_text(
        "\n".join(
            (
                _timer("update_weights_implementation", 3.2),
                _timer("update_weights", 3.8),
            )
        )
    )

    with pytest.raises(ValueError, match="complete timer triplet"):
        parse_timings.parse_log(log_path, expected_updates=1)


def test_parser_rejects_out_of_order_triplet(tmp_path):
    log_path = tmp_path / "actor.log"
    log_path.write_text(
        "\n".join(
            (
                _timer("finalize_and_resume_engines", 0.3),
                _timer("update_weights_implementation", 3.2),
                _timer("update_weights", 3.8),
            )
        )
    )

    with pytest.raises(ValueError, match="out of order"):
        parse_timings.parse_log(log_path, expected_updates=1)


def test_parser_rejects_unexpected_update_count(tmp_path):
    log_path = tmp_path / "actor.log"
    log_path.write_text(
        "\n".join(
            (
                _timer("update_weights_implementation", 3.2),
                _timer("finalize_and_resume_engines", 0.3),
                _timer("update_weights", 3.8),
            )
        )
    )

    with pytest.raises(ValueError, match="expected 2 complete updates, observed 1"):
        parse_timings.parse_log(log_path, expected_updates=2)


def test_parser_ignores_non_rank_zero_actor_timers(tmp_path):
    log_path = tmp_path / "actor.log"
    log_path.write_text(
        "\n".join(
            (
                _timer("update_weights_implementation", 99.0, rank=1),
                _timer("finalize_and_resume_engines", 99.0, rank=1),
                _timer("update_weights", 99.0, rank=1),
                _timer("update_weights_implementation", 3.2),
                _timer("finalize_and_resume_engines", 0.3),
                _timer("update_weights", 3.8),
            )
        )
    )

    result = parse_timings.parse_log(log_path, expected_updates=1)

    assert result["updates"][0]["update_weights_s"] == 3.8


def test_write_result_is_deterministic(tmp_path):
    output_path = tmp_path / "result.json"
    payload = {
        "updates": [],
        "observed_update_count": 0,
        "expected_update_count": 0,
        "schema_version": 1,
    }

    parse_timings.write_result(output_path, payload)

    assert (
        output_path.read_text() == json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )


@pytest.mark.parametrize(
    ("mode", "expected_kinds", "plugin_value", "server_value"),
    (
        (
            "external",
            ["Namespace", "Service", "Deployment", "Job"],
            "modelexpress_miles_collective",
            "mx-server:8000",
        ),
        ("broadcast", ["Namespace", "Job"], "", ""),
    ),
)
def test_rendered_manifest_carries_matched_benchmark_contract(
    mode, expected_kinds, plugin_value, server_value
):
    result = _render_manifest(mode)

    assert result.returncode == 0, result.stderr
    documents = list(yaml.safe_load_all(result.stdout))
    assert [document["kind"] for document in documents] == expected_kinds
    job = documents[-1]
    environment = {
        item["name"]: item["value"]
        for item in job["spec"]["template"]["spec"]["containers"][0]["env"]
    }
    assert job["metadata"]["name"] == f"miles-m2n-{mode}-sample-b2-r3"
    assert job["metadata"]["labels"]["modelexpress.nvidia.com/benchmark-mode"] == mode
    assert (
        job["metadata"]["labels"]["modelexpress.nvidia.com/benchmark-run"]
        == "matched-run"
    )
    assert environment["MILES_BENCH_ROLLOUTS"] == "11"
    assert environment["SGLANG_PLUGINS"] == plugin_value
    assert environment["MX_SERVER_ADDRESS"] == server_value
    assert environment["MX_MILES_VERIFY_TENSOR_EQUALITY"] == "0"


def test_qualification_must_explicitly_enable_tensor_verification():
    result = _render_manifest("external", MX_MILES_VERIFY_TENSOR_EQUALITY="1")

    assert result.returncode == 0, result.stderr
    job = list(yaml.safe_load_all(result.stdout))[-1]
    environment = {
        item["name"]: item["value"]
        for item in job["spec"]["template"]["spec"]["containers"][0]["env"]
    }
    assert environment["MX_MILES_VERIFY_TENSOR_EQUALITY"] == "1"


@pytest.mark.parametrize("runtime_image", ["runtime:latest", "runtime@sha256:1234"])
def test_runtime_image_must_be_digest_pinned(runtime_image):
    result = _render_manifest("broadcast", RUNTIME_IMAGE=runtime_image)

    assert result.returncode == 2
    assert "RUNTIME_IMAGE must be digest-pinned" in result.stderr


def test_external_server_image_must_be_digest_pinned():
    result = _render_manifest("external", MX_SERVER_IMAGE="server:latest")

    assert result.returncode == 2
    assert "MX_SERVER_IMAGE must be digest-pinned" in result.stderr


def test_broadcast_does_not_require_or_render_server_image():
    result = _render_manifest("broadcast", MX_SERVER_IMAGE="server:latest")

    assert result.returncode == 0, result.stderr
    documents = list(yaml.safe_load_all(result.stdout))
    assert [document["kind"] for document in documents] == ["Namespace", "Job"]
    assert "mx-server" not in result.stdout


def test_broadcast_apply_does_not_wait_for_model_express_server(tmp_path):
    kubectl_log = tmp_path / "kubectl.log"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_kubectl = fake_bin / "kubectl"
    fake_kubectl.write_text(
        '#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "${KUBECTL_LOG}"\n'
    )
    fake_kubectl.chmod(0o755)
    environment = os.environ | {
        "KUBECTL_LOG": str(kubectl_log),
        "KUBE_CONTEXT": "test-context",
        "MILES_WEIGHT_TRANSFER_MODE": "broadcast",
        "MX_SERVER_IMAGE": "server:latest",
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "RUN_ID": "sample",
        "RUNTIME_IMAGE": RUNTIME_IMAGE,
    }

    result = subprocess.run(
        [str(EXAMPLE_DIR / "deploy.sh"), "apply"],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    calls = kubectl_log.read_text()
    assert "apply --dry-run=client" in calls
    assert "rollout status deployment/mx-server" not in calls


def test_substituted_yaml_identifiers_remain_strings():
    result = _render_manifest(
        "broadcast",
        GPU_PRODUCT="off",
        MILES_BENCH_RUN_LABEL="yes",
        NAMESPACE="on",
        RUN_ID="no",
        RUNTIME_PULL_SECRET="null",
    )

    assert result.returncode == 0, result.stderr
    namespace, job = list(yaml.safe_load_all(result.stdout))
    assert namespace["metadata"]["name"] == "on"
    assert job["metadata"]["namespace"] == "on"
    assert job["metadata"]["labels"]["modelexpress.nvidia.com/run-id"] == "no"
    assert job["metadata"]["labels"]["modelexpress.nvidia.com/benchmark-run"] == "yes"
    pod_spec = job["spec"]["template"]["spec"]
    assert pod_spec["imagePullSecrets"][0]["name"] == "null"
    assert pod_spec["nodeSelector"]["nvidia.com/gpu.product"] == "off"


def test_optional_node_name_pins_paired_job_to_same_node():
    pinned = _render_manifest("broadcast", BENCH_NODE_NAME="ip-10-0-0-7")
    unpinned = _render_manifest("broadcast")

    assert pinned.returncode == 0, pinned.stderr
    assert unpinned.returncode == 0, unpinned.stderr
    pinned_job = list(yaml.safe_load_all(pinned.stdout))[-1]
    unpinned_job = list(yaml.safe_load_all(unpinned.stdout))[-1]
    assert (
        pinned_job["spec"]["template"]["spec"]["nodeSelector"]["kubernetes.io/hostname"]
        == "ip-10-0-0-7"
    )
    assert (
        "kubernetes.io/hostname"
        not in unpinned_job["spec"]["template"]["spec"]["nodeSelector"]
    )
