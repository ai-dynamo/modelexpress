# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline contracts: rendered workloads, multi-rank validation, and failure reports."""

import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
from harness import render as prepare
from harness import validation
from harness.report import build_report

PORTABLE_ENV = {
    "context": "test-cluster",
    "namespace": "test-run",
    "region": "test-region-1",
    "bucket": "test-models",
    "service_account": "test-identity",
    "default_paths": "both",
    "extra_worker_resources": {"vpc.amazonaws.com/efa": "1"},
    "worker_env": {"MX_NIXL_BACKEND": "LIBFABRIC"},
    "peer_transfer_marker": "RDMA transfer complete:",
    "worker_node_selector": {"test.example/gpu": "accelerator"},
    "images": {
        "server": "registry.example/server@sha256:" + "a" * 64,
        "runtime": "registry.example/runtime@sha256:" + "b" * 64,
    },
}


@pytest.fixture(autouse=True)
def isolated_runner_environment(monkeypatch):
    for key in [
        "KUBE_CONTEXT",
        "WORKER_CONTEXT",
        "NAMESPACE",
        "KUBECONFIG",
        "MX_CI_S3_REGION",
        "AWS_REGION",
        "AWS_DEFAULT_REGION",
        "MX_CI_S3_BUCKET",
        "AWS_ENDPOINT_URL",
        "SERVER_IMAGE",
        "WORKER_IMAGE",
        "MX_MAIN_SERVER_IMAGE",
        "MX_MAIN_RUNTIME_IMAGE",
    ]:
        monkeypatch.delenv(key, raising=False)


def render(*args, **kwargs):
    kwargs.setdefault("environment", PORTABLE_ENV)
    return prepare.prepare(*args, **kwargs)


@pytest.mark.parametrize(
    "model,paths",
    [("nemotron", "both"), ("nemotron", "s3")],
)
def test_rendered_workloads_share_profile_and_mount_all_runtime_code(
    tmp_path, model, paths
):
    out = tmp_path / "run with spaces"
    config = render(model, out, model + "-test", paths)
    cm = json.loads((out / "harness.json").read_text())
    assert json.loads(cm["data"]["config.json"]) == config
    assert config["scenario"] == "delta"
    assert config["target_version"] == model + "-test-d1"
    assert config["sources"]["s3"] == "OBJECT_STORAGE"
    assert {"harness__scenario.py", "scenarios__delta__scenario.py"} <= cm[
        "data"
    ].keys()
    assert "harness__render.py" not in cm["data"]
    assert "harness__lifecycle.py" not in cm["data"]
    assert "apply_patch.py" not in cm["data"]
    for name, text in cm["data"].items():
        if name.endswith(".py"):
            compile(text, name, "exec")
    control = yaml.safe_load((out / "control.yaml").read_text())
    pod = next(x for x in control["items"] if x["kind"] == "Pod")
    assert any(
        x["mountPath"] == "/opt/benchmark"
        for x in pod["spec"]["containers"][0]["volumeMounts"]
    )
    for role in config["roles"]:
        manifest = yaml.safe_load((out / f"worker-{role}.yaml").read_text())
        pod, service = manifest["items"]
        container = pod["spec"]["containers"][0]
        env = {x["name"]: x["value"] for x in container["env"]}
        assert env["BENCH_MODEL"] == config["model"]
        assert env["BENCH_REVISION"] == config["revision"]
        assert env["BENCH_ROLE"] == role
        assert int(container["resources"]["limits"]["nvidia.com/gpu"]) == config["tp"]
        assert container["resources"]["limits"]["memory"] == config["memory"]
        assert service["spec"]["selector"]["app"] == pod["metadata"]["labels"]["app"]
        assert any(
            v.get("configMap", {}).get("name") == cm["metadata"]["name"]
            for v in pod["spec"]["volumes"]
        )
        anti = pod["spec"]["affinity"]["podAntiAffinity"][
            "requiredDuringSchedulingIgnoredDuringExecution"
        ][0]
        assert anti["labelSelector"]["matchLabels"]["mx-benchmark"] == config["run"]
    assert (out / "worker-peer.yaml").exists() == (paths == "both")
    with pytest.raises(FileExistsError):
        render(model, out, model + "-test", paths)


def test_renderer_rejects_unpinned_image(tmp_path, monkeypatch):
    monkeypatch.setenv("MX_MAIN_RUNTIME_IMAGE", "registry/runtime:latest")
    with pytest.raises(ValueError, match="digest-qualified"):
        render("nemotron", tmp_path / "out", "nemotron-test")


@pytest.mark.parametrize(
    "rows",
    [
        [{"rank": 0, "phase": "init"}],
        [{"rank": 0, "phase": "init"}, {"rank": 0, "phase": "init"}],
        [
            {"rank": 0, "phase": "init"},
            {"rank": 1, "phase": "init-failed", "error": "OOM"},
        ],
    ],
)
def test_partial_or_failed_rank_cannot_pass(rows):
    with pytest.raises(AssertionError):
        validation.ranks(rows, {"tp": 2})


def saved_run(tmp_path):
    config = {
        "key": "nemotron",
        "embedding": "embedding",
        "tp": 2,
        "roles": ["s3", "peer"],
        "run": "nemotron-test",
        "initial_version": "nemotron-test-base",
        "target_version": "nemotron-test-d1",
        "sources": {"s3": "OBJECT_STORAGE", "peer": "GENERATOR"},
        "revision": "revision",
        "expected_tensors_per_rank": None,
        "expected_host_scales_per_rank": None,
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "images.json").write_text("{}")
    (tmp_path / "publication.json").write_text(
        json.dumps(
            {
                "run": "nemotron-test",
                "model_revision": "revision",
                "tensor": "embedding",
                "expected_sha256": "a" * 64,
            }
        )
    )
    lines = []

    def result(role, step, rows):
        lines.append(
            "RESULT "
            + role
            + " "
            + step
            + " "
            + json.dumps(
                {
                    "seconds": 4.0,
                    "http_status": 200,
                    "response": {"ok": True, "result": rows},
                }
            )
        )

    for role in config["roles"]:
        for step, digest in [("base-hashes", "before"), ("updated-hashes", "after")]:
            result(
                role,
                step,
                [
                    {
                        "rank": r,
                        "phase": step,
                        "tensors": {
                            "embedding": {
                                "sha256": digest + str(r),
                                "shape": [2, 4],
                                "dtype": "bfloat16",
                            }
                        },
                    }
                    for r in range(2)
                ],
            )
        result(
            role,
            "refit",
            [
                {
                    "rank": r,
                    "phase": "nemotron-test-d1",
                    "version": "nemotron-test-d1",
                    "serving_version": "nemotron-test-d1",
                    "source": "OBJECT_STORAGE" if role == "s3" else "GENERATOR",
                    "weight_addresses_preserved": True,
                    "stage_seconds": 1.0 + r,
                    "install_seconds": 1.0,
                    "total_seconds": 2.0 + r,
                }
                for r in range(2)
            ],
        )
        result(
            role, "post-refit-inference", [{"token_ids": [3, 4], "logprob_count": 2}]
        )
        log = (
            "(Worker_TP1 pid=9) Model loading took 1 GiB memory and 3.0 seconds\n"
            "(Worker_TP0 pid=8) Model loading took 1 GiB memory and 2.0 seconds\n"
        )
        log += (
            "Streaming weights from s3://bucket/model\n"
            if role == "s3"
            else "RDMA transfer complete: test\n"
        )
        (tmp_path / f"{role}-worker.log").write_text(log)
        (tmp_path / f"{role}-pod.json").write_text(
            json.dumps(
                {
                    "spec": {"nodeName": role},
                    "status": {
                        "containerStatuses": [
                            {"restartCount": 0, "state": {"running": {}}}
                        ]
                    },
                }
            )
        )
    result(
        "s3",
        "verify-checkpoint",
        [
            {
                "rank": r,
                "phase": "checkpoint-verified",
                "version": "nemotron-test-d1",
                "checkpoint_tensor": "embedding",
                "checkpoint_sha256": "a" * 64,
                "verified": True,
            }
            for r in range(2)
        ],
    )
    result(
        "s3",
        "verify-runtime-tensor",
        [
            {
                "rank": rank,
                "phase": "runtime-tensor-verified",
                "version": config["target_version"],
                "checkpoint_tensor": "embedding",
                "expected_sha256": "c" * 64,
                "actual_sha256": "c" * 64,
                "verified": True,
            }
            for rank in range(config["tp"])
        ],
    )
    (tmp_path / "bench-driver.log").write_text("\n".join(lines) + "\nBENCH_PASS\n")
    return config


def test_report_checks_corresponding_tp_ranks_and_rejects_peer_fallback(tmp_path):
    saved_run(tmp_path)
    assert build_report(tmp_path)["status"] == "PASS"
    with (tmp_path / "peer-worker.log").open("a") as f:
        f.write("Trying strategy: model_streamer\n")
    report = build_report(tmp_path)
    assert report["status"] == "FAILED"
    assert "fell back" in report["failure_reason"]


def test_report_rejects_wrong_nonzero_rank_even_with_pass_marker(tmp_path):
    saved_run(tmp_path)
    p = tmp_path / "bench-driver.log"
    lines = p.read_text().splitlines()
    for i, line in enumerate(lines):
        if line.startswith("RESULT peer updated-hashes "):
            record = json.loads(line.split(" ", 3)[3])
            record["response"]["result"][1]["tensors"]["embedding"]["sha256"] = (
                "wrong-rank-1"
            )
            lines[i] = "RESULT peer updated-hashes " + json.dumps(record)
    p.write_text("\n".join(lines) + "\n")
    assert build_report(tmp_path)["status"] == "FAILED"


def test_failed_run_preserves_rejection_evidence(tmp_path):
    saved_run(tmp_path)
    (tmp_path / "bench-driver.log").write_text("")
    failure = {
        "phase": "nemotron-test-d1-failed",
        "rank": 1,
        "error": "refit rejected",
    }
    with (tmp_path / "s3-worker.log").open("a") as f:
        f.write("HOTLOAD_BENCHMARK " + json.dumps(failure) + "\n")
    report = build_report(tmp_path)
    assert report["status"] == "FAILED"
    assert report["workers"]["s3"]["failures"] == [failure]


def test_aws_ci_aliases_native_s3_and_namespace_reach_every_resource(
    tmp_path, monkeypatch
):
    for key, value in {
        "KUBE_CONTEXT": "ci-h100-cluster",
        "NAMESPACE": "mx-ci-123",
        "KUBECONFIG": "/runner/kubeconfig",
        "MX_CI_S3_REGION": "eu-west-1",
        "MX_CI_S3_BUCKET": "ci-snapshots",
        "SERVER_IMAGE": "registry.example/server:" + "a" * 40,
        "WORKER_IMAGE": "registry.example/worker:" + "b" * 40,
    }.items():
        monkeypatch.setenv(key, value)
    out = tmp_path / "aws"
    config = render("nemotron", out, "nemotron-ci", environment="aws-ci")
    assert config["roles"] == ["s3"]
    assert config["storage"] == {
        "endpoint_url": None,
        "region": "eu-west-1",
        "addressing_style": "auto",
    }
    assert config["bucket"] == "ci-snapshots"
    assert config["peer_transfer_marker"]
    for filename in ["control.yaml", "worker-s3.yaml"]:
        manifest = yaml.safe_load((out / filename).read_text())
        assert all(x["metadata"]["namespace"] == "mx-ci-123" for x in manifest["items"])
        pod = next(x for x in manifest["items"] if x["kind"] == "Pod")
        main = pod["spec"]["containers"][0]
        env = {x["name"]: x.get("value") for x in main["env"]}
        assert "AWS_ENDPOINT_URL" not in env
        assert env["AWS_REGION"] == "eu-west-1"
        assert "rdma/ib" not in main["resources"]["limits"]
        assert pod["spec"]["imagePullSecrets"] == [{"name": "nvcr-imagepullsecret"}]
    worker = yaml.safe_load((out / "worker-s3.yaml").read_text())["items"][0]
    assert (
        worker["spec"]["nodeSelector"]["nvidia.com/gpu.product"]
        == "NVIDIA-H100-80GB-HBM3"
    )
    assert (
        json.loads((out / "harness.json").read_text())["metadata"]["namespace"]
        == "mx-ci-123"
    )
    assert "ci-h100-cluster" in (out / "environment.json").read_text()
    assert "/runner/kubeconfig" in (out / "environment.json").read_text()


def test_custom_endpoint_secret_references_and_efa_are_not_assumed(tmp_path):
    env = {
        **PORTABLE_ENV,
        "endpoint_url": "http://minio.example:9000",
        "addressing_style": "path",
        "extra_worker_resources": {"vpc.amazonaws.com/efa": "4"},
        "worker_env": {"MX_NIXL_BACKEND": "LIBFABRIC"},
        "pod_env": [
            {
                "name": "AWS_ACCESS_KEY_ID",
                "valueFrom": {
                    "secretKeyRef": {"name": "object-store", "key": "access-key"}
                },
            }
        ],
        "pod_env_from": [{"secretRef": {"name": "object-store-env"}}],
    }
    out = tmp_path / "custom"
    config = render(
        "nemotron",
        out,
        "nemotron-custom",
        environment=env,
        tp=2,
        cpu="8",
        memory="96Gi",
    )
    assert config["tp"] == 2 and config["storage"]["addressing_style"] == "path"
    for name in ["control.yaml", "worker-s3.yaml", "worker-peer.yaml"]:
        pod = next(
            x
            for x in yaml.safe_load((out / name).read_text())["items"]
            if x["kind"] == "Pod"
        )
        main = pod["spec"]["containers"][0]
        variables = {x["name"]: x for x in main["env"]}
        assert variables["AWS_ENDPOINT_URL"]["value"] == "http://minio.example:9000"
        assert "valueFrom" in variables["AWS_ACCESS_KEY_ID"]
        assert main["envFrom"] == env["pod_env_from"]
        if name != "control.yaml":
            assert variables["MX_NIXL_BACKEND"]["value"] == "LIBFABRIC"
            assert main["resources"]["limits"]["vpc.amazonaws.com/efa"] == "4"
            assert main["resources"]["limits"]["nvidia.com/gpu"] == "2"


def test_aws_ci_renders_kimi_profile_with_eight_gpus(tmp_path):
    out = tmp_path / "kimi-aws"
    config = render("kimi", out, "kimi-ci", environment="aws-ci", **PORTABLE_ENV)
    assert config["model"] == "moonshotai/Kimi-K2.7-Code"
    assert config["revision"] == "74797c9c62378b951a1f6fcf5c4631024e9b8bef"
    assert config["tp"] == 8
    pod = yaml.safe_load((out / "worker-s3.yaml").read_text())["items"][0]
    assert pod["spec"]["containers"][0]["resources"]["limits"]["nvidia.com/gpu"] == "8"
    assert pod["spec"]["containers"][0]["resources"]["limits"]["memory"] == "1Ti"
    assert (
        "apply_patch.py" not in json.loads((out / "harness.json").read_text())["data"]
    )


def test_missing_target_rejected_before_writing_manifests(tmp_path):
    with pytest.raises(ValueError, match="context"):
        render("nemotron", tmp_path / "bad", "nemotron-test", environment="aws-ci")
    assert not (tmp_path / "bad").exists()


def test_kubectl_uses_rendered_target_instead_of_local_default(tmp_path):
    import os
    import subprocess

    out = tmp_path / "rendered"
    render("nemotron", out, "nemotron-test")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    kubectl = bin_dir / "kubectl"
    kubectl.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
    kubectl.chmod(0o755)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; sys.path.insert(0, sys.argv[1]); "
                "from harness.lifecycle import Benchmark; "
                "print(Benchmark(sys.argv[2]).k.call('get', 'pods'), end='')"
            ),
            str(ROOT),
            str(out),
        ],
        env={**os.environ, "PATH": str(bin_dir) + ":" + os.environ["PATH"]},
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.splitlines() == [
        "--context",
        "test-cluster",
        "-n",
        "test-run",
        "get",
        "pods",
    ]


@pytest.mark.parametrize("model", ["unknown", "../kimi", "nvidia/other-model"])
def test_custom_environment_cannot_expand_model_scope(tmp_path, model):
    with pytest.raises(ValueError, match="Unknown model profile"):
        render(
            model,
            tmp_path / "out",
            "nemotron-test",
            environment={**PORTABLE_ENV, "allowed_models": [model]},
        )
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "missing", ["extra_worker_resources", "worker_env", "peer_transfer_marker"]
)
def test_peer_transport_must_be_explicit(tmp_path, missing):
    env = {key: value for key, value in PORTABLE_ENV.items() if key != missing}
    with pytest.raises(ValueError, match="Peer coverage requires explicit"):
        render("nemotron", tmp_path / "out", "nemotron-peer", "both", environment=env)
    assert not (tmp_path / "out").exists()


def test_report_preserves_measured_refit_latency_and_failed_rank(tmp_path):
    saved_run(tmp_path)
    path = tmp_path / "bench-driver.log"
    lines = path.read_text().splitlines()
    for index, line in enumerate(lines):
        if line.startswith("RESULT s3 refit "):
            record = json.loads(line.split(" ", 3)[3])
            record["seconds"] = 4.5
            for row in record["response"]["result"]:
                row.update(stage_seconds=2.0, install_seconds=1.0, total_seconds=3.0)
            lines[index] = "RESULT s3 refit " + json.dumps(record)
    path.write_text("\n".join(lines) + "\n")
    report = build_report(tmp_path)
    assert report["status"] == "PASS"
    assert report["workers"]["s3"]["model_load_seconds"] == [3.0, 2.0]
    assert report["workers"]["s3"]["refit"]["seconds"] == 4.5
    path.write_text(
        path.read_text().replace(
            '"weight_addresses_preserved": true',
            '"weight_addresses_preserved": false',
            1,
        )
    )
    assert build_report(tmp_path)["status"] == "FAILED"


@pytest.mark.parametrize("bad_value", [None, -1, float("nan"), float("inf"), True])
def test_invalid_rank_latency_excludes_run_from_benchmark_summary(tmp_path, bad_value):
    saved_run(tmp_path)
    path = tmp_path / "bench-driver.log"
    lines = path.read_text().splitlines()
    for i, line in enumerate(lines):
        if line.startswith("RESULT peer refit "):
            record = json.loads(line.split(" ", 3)[3])
            record["response"]["result"][1]["install_seconds"] = bad_value
            lines[i] = "RESULT peer refit " + json.dumps(record)
    path.write_text("\n".join(lines) + "\n")
    report = build_report(tmp_path)
    assert report["validation_status"] == "PASS"
    assert report["measurement_status"] == "INVALID"
    assert report["status"] == "FAILED"
    assert report["latency_summary"] == {}
    assert report["workers"]["peer"]["refit"]


def test_report_rejects_publisher_checkpoint_mismatch(tmp_path):
    saved_run(tmp_path)
    path = tmp_path / "publication.json"
    data = json.loads(path.read_text())
    data["expected_sha256"] = "b" * 64
    path.write_text(json.dumps(data))
    report = build_report(tmp_path)
    assert report["validation_status"] == "FAILED"
    assert report["measurement_status"] == "INVALID"
    assert report["latency_summary"] == {}
    assert "published embedding" in report["failure_reason"]


def test_successful_benchmark_summarizes_slowest_rank(tmp_path):
    saved_run(tmp_path)
    report = build_report(tmp_path)
    assert report["validation_status"] == "PASS"
    assert report["measurement_status"] == "VALID"
    assert report["latency_summary"]["s3"]["slowest_rank_total_seconds"] == 3.0
    assert report["latency_summary"]["s3"]["per_rank"][0]["total_seconds"] == 2.0


@pytest.mark.parametrize("field", ["stage_seconds", "install_seconds", "total_seconds"])
def test_missing_latency_field_invalidates_benchmark(tmp_path, field):
    saved_run(tmp_path)
    path = tmp_path / "bench-driver.log"
    lines = path.read_text().splitlines()
    for i, line in enumerate(lines):
        if line.startswith("RESULT s3 refit "):
            record = json.loads(line.split(" ", 3)[3])
            del record["response"]["result"][1][field]
            lines[i] = "RESULT s3 refit " + json.dumps(record)
    path.write_text("\n".join(lines) + "\n")
    report = build_report(tmp_path)
    assert report["validation_status"] == "PASS"
    assert report["measurement_status"] == "INVALID"
    assert report["latency_summary"] == {}


def test_failed_weight_check_excludes_otherwise_complete_measurements(tmp_path):
    saved_run(tmp_path)
    path = tmp_path / "bench-driver.log"
    path.write_text(
        path.read_text().replace(
            '"weight_addresses_preserved": true',
            '"weight_addresses_preserved": false',
            1,
        )
    )
    report = build_report(tmp_path)
    assert report["validation_status"] == "FAILED"
    assert report["measurement_status"] == "INVALID"
    assert report["latency_summary"] == {}
    assert (
        report["workers"]["s3"]["refit"]["response"]["result"][1]["total_seconds"]
        == 3.0
    )


@pytest.mark.parametrize("rank", [0, 1])
def test_checkpoint_digest_mismatch_on_any_rank_fails(tmp_path, rank):
    saved_run(tmp_path)
    path = tmp_path / "bench-driver.log"
    lines = path.read_text().splitlines()
    for index, line in enumerate(lines):
        if line.startswith("RESULT s3 verify-checkpoint "):
            record = json.loads(line.split(" ", 3)[3])
            record["response"]["result"][rank]["checkpoint_sha256"] = "b" * 64
            lines[index] = "RESULT s3 verify-checkpoint " + json.dumps(record)
    path.write_text("\n".join(lines) + "\n")
    report = build_report(tmp_path)
    assert report["validation_status"] == "FAILED"
    assert report["latency_summary"] == {}


def test_runtime_inventory_override_remains_strict(tmp_path):
    config = render(
        "nemotron",
        tmp_path / "override",
        "nemotron-inventory",
        environment={**PORTABLE_ENV, "expected_tensors_per_rank": 2},
    )
    row = {"rank": 0, "phase": "hashes", "tensors": {"a": {}, "b": {}}}
    assert validation.hashes([row], config)[0] == row["tensors"]
    row["tensors"].pop("b")
    with pytest.raises(AssertionError):
        validation.hashes([row], config)
    saved = json.loads((tmp_path / "override" / "config.json").read_text())
    assert saved["expected_tensors_per_rank"] == 2
    assert saved["expected_host_scales_per_rank"] == 18
    default = render("nemotron", tmp_path / "default", "nemotron-default")
    assert default["expected_tensors_per_rank"] == 761


@pytest.mark.parametrize(
    "key", ["expected_tensors_per_rank", "expected_host_scales_per_rank"]
)
@pytest.mark.parametrize("value", [0, -1, True, 1.5, "2"])
def test_invalid_inventory_override_is_rejected(tmp_path, key, value):
    with pytest.raises(ValueError, match=key):
        render(
            "nemotron",
            tmp_path / "invalid",
            "nemotron-invalid",
            environment={**PORTABLE_ENV, key: value},
        )


def test_report_aggregates_load_times_by_rank(tmp_path):
    saved_run(tmp_path)
    report = build_report(tmp_path)
    assert report["status"] == "PASS"
    summary = report["latency_summary"]["s3"]
    assert summary["model_load_seconds"] == 3.0
    assert summary["model_load_seconds_by_rank"] == {0: 2.0, 1: 3.0}


@pytest.mark.parametrize("case", ["missing", "duplicate", "unidentified"])
def test_report_rejects_incomplete_or_repeated_load_timers(tmp_path, case):
    saved_run(tmp_path)
    path = tmp_path / "s3-worker.log"
    text = path.read_text()
    line = text.splitlines()[0]
    if case == "missing":
        text = text.replace(line + "\n", "")
    elif case == "duplicate":
        text += line + "\n"
    else:
        text = text.replace("Worker_TP1", "Worker")
    path.write_text(text)
    report = build_report(tmp_path)
    assert report["validation_status"] == "PASS"
    assert report["measurement_status"] == "INVALID"
    assert report["status"] == "FAILED"
    assert report["latency_summary"] == {}


@pytest.mark.parametrize("line", ['RESULT s3 refit {"response":', "RESULT invalid"])
def test_report_retains_malformed_result_evidence(tmp_path, line):
    saved_run(tmp_path)
    with (tmp_path / "bench-driver.log").open("a") as stream:
        stream.write(line + "\n")
    report = build_report(tmp_path)
    assert report["status"] == "FAILED"
    assert report["measurement_status"] == "INVALID"
    assert report["latency_summary"] == {}
    assert len(report["record_errors"]) == 1
    assert report["record_errors"][0]["line"] > 0
    assert "Malformed RESULT" in report["failure_reason"]


def test_refit_uses_explicit_version_and_source():
    config = {
        "tp": 1,
        "target_version": "update-42",
        "sources": {"receiver": "GENERATOR"},
    }
    row = {
        "rank": 0,
        "phase": "update-42",
        "version": "update-42",
        "serving_version": "update-42",
        "source": "GENERATOR",
        "weight_addresses_preserved": True,
    }
    validation.refit([row], config, "receiver")
    with pytest.raises(AssertionError):
        validation.refit([{**row, "source": "OBJECT_STORAGE"}], config, "receiver")
    with pytest.raises(AssertionError):
        validation.refit([{**row, "serving_version": "old"}], config, "receiver")


@pytest.mark.parametrize("case", ["mismatch", "missing", "failed-rank"])
def test_report_requires_installed_runtime_tensor_evidence(tmp_path, case):
    saved_run(tmp_path)
    path = tmp_path / "bench-driver.log"
    lines = []
    for line in path.read_text().splitlines():
        if line.startswith("RESULT s3 verify-runtime-tensor "):
            if case == "missing":
                continue
            record = json.loads(line.split(" ", 3)[3])
            row = record["response"]["result"][1]
            if case == "mismatch":
                row["actual_sha256"] = "d" * 64
                row["verified"] = False
            else:
                row["phase"] += "-failed"
                row["error"] = "unsupported runtime tensor"
            line = "RESULT s3 verify-runtime-tensor " + json.dumps(record)
        lines.append(line)
    path.write_text("\n".join(lines) + "\n")
    report = build_report(tmp_path)
    assert report["validation_status"] == "FAILED"
    assert report["status"] == "FAILED"
    assert report["latency_summary"] == {}


def test_configmap_projection_imports_packages_without_running_benchmark(tmp_path):
    import subprocess

    rendered = tmp_path / "rendered"
    config = render("nemotron", rendered, "nemotron-packages", "s3")
    manifest = json.loads((rendered / "harness.json").read_text())
    control = yaml.safe_load((rendered / "control.yaml").read_text())
    pod = next(item for item in control["items"] if item["kind"] == "Pod")
    volume = next(
        item for item in pod["spec"]["volumes"] if item["name"] == "benchmark"
    )
    projected = tmp_path / "projected"
    for item in volume["configMap"]["items"]:
        path = projected / item["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(manifest["data"][item["key"]])
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json; from harness.config import load_config; "
                "from harness.runner import RefitRunner; from harness.scenario import load; "
                "from harness.report import build_report; "
                "config = load_config(); assert load(config).config == config; "
                "print(json.dumps(config))"
            ),
        ],
        cwd=projected,
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout) == config
    assert not (projected / "PASS").exists()
    assert not (projected / "FAIL").exists()
    assert not (projected / "harness/render.py").exists()
    assert not (projected / "harness/lifecycle.py").exists()
    worker = yaml.safe_load((rendered / "worker-s3.yaml").read_text())["items"][0]
    worker_volume = next(
        item for item in worker["spec"]["volumes"] if item["name"] == "benchmark"
    )
    assert worker_volume["configMap"]["items"] == volume["configMap"]["items"]
