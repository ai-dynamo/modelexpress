"""Offline contracts: rendered workloads, multi-rank validation, and failure reports."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "common"))
validation = __import__("validation")
build_report = __import__("report").build_report

spec = importlib.util.spec_from_file_location("prepare", ROOT / "scripts/prepare.py")
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


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
        assert anti["labelSelector"]["matchLabels"]["mx-regression"] == config["run"]
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
        "tp": 2,
        "roles": ["s3", "peer"],
        "run": "nemotron-test",
        "revision": "revision",
        "expected_tensors_per_rank": None,
        "expected_host_scales_per_rank": None,
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "images.json").write_text("{}")
    (tmp_path / "publication.json").write_text(
        json.dumps({"run": "nemotron-test", "model_revision": "revision"})
    )
    lines = []

    def result(role, step, rows):
        lines.append(
            "RESULT "
            + role
            + " "
            + step
            + " "
            + json.dumps({"http_status": 200, "response": {"ok": True, "result": rows}})
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
                }
                for r in range(2)
            ],
        )
        result(
            role, "post-refit-inference", [{"token_ids": [3, 4], "logprob_count": 2}]
        )
        log = "Model loading took 1 GiB memory and 2.0 seconds\n"
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
                "verified": True,
                "sha256": str(r),
                "expected_sha256": str(r),
            }
            for r in range(2)
        ],
    )
    (tmp_path / "e2e-driver.log").write_text("\n".join(lines) + "\nE2E_PASS\n")
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
    p = tmp_path / "e2e-driver.log"
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


def test_failed_run_preserves_rejection_and_oom_evidence(tmp_path):
    saved_run(tmp_path)
    (tmp_path / "e2e-driver.log").write_text("")
    failure = {
        "phase": "nemotron-test-d1-failed",
        "rank": 1,
        "error": "refit rejected",
    }
    with (tmp_path / "s3-worker.log").open("a") as f:
        f.write("HOTLOAD_BENCHMARK " + json.dumps(failure) + "\n")
    (tmp_path / "s3-monitor.log").write_text(
        "HOST_MEMORY "
        + json.dumps(
            {
                "memory.stat": "anon 1073741824\nfile 2147483648\n",
                "memory.events": "oom 1\noom_kill 1\n",
            }
        )
        + "\n"
    )
    report = build_report(tmp_path)
    assert report["status"] == "FAILED"
    assert report["workers"]["s3"]["failures"] == [failure]
    assert report["workers"]["s3"]["peak_observed_anon_gib"] == 1
    assert "oom_kill 1" in report["workers"]["s3"]["memory_events"]


@pytest.mark.parametrize("tp", [1, 2])
def test_rendered_nemotron_preserves_model_specific_scale_regression(tmp_path, tp):
    out = tmp_path / "run"
    config = render("nemotron", out, "nemotron-test")
    spec = importlib.util.spec_from_file_location(
        "nemotron_checks", out / "model_checks.py"
    )
    hook = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hook)
    before = [{"module": str(i), "scale": "k", "host": 1.0} for i in range(18)]
    after = [dict(r, host=2.0 if i < 11 else 1.0) for i, r in enumerate(before)]

    def result(role, step):
        if step == "baseline-host-scales":
            return [{"scales": before}]
        if step == "immediate-post-refit-host-scales":
            return [
                {
                    "scales": after,
                    "flashinfer_caches": [
                        {"module": "attention", "values": {"bmm1_scale": None}}
                    ],
                }
            ]
        return [
            {
                "flashinfer_caches": [
                    {
                        "module": "attention",
                        "values": {"bmm1_scale": 1.0 if role == "s3" else 2.0},
                    }
                ]
            }
        ]

    config["tp"] = tp
    original_result = result

    def result(role, step):
        return [
            dict(row, rank=rank, phase=step)
            for rank in range(tp)
            for row in original_result(role, step)
        ]

    report = hook.validate(config, result)
    assert report["changed_peer_host_entries"] == 11
    assert len(report["post_inference_launch_cache_differences"]) == 1
    after[0]["host"] = 1.0
    with pytest.raises(AssertionError, match="expected Nemotron"):
        hook.validate(config, result)


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
    assert "ci-h100-cluster" in (out / "run.env").read_text()
    assert "/runner/kubeconfig" in (out / "run.env").read_text()


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
            "bash",
            "-c",
            'source "$1" "$2"; k get pods',
            "test",
            str(ROOT / "scripts/lib.sh"),
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
    path = tmp_path / "e2e-driver.log"
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
    assert report["workers"]["s3"]["model_load_seconds"] == [2.0]
    assert report["workers"]["s3"]["refit"]["seconds"] == 4.5
    path.write_text(
        path.read_text().replace(
            '"weight_addresses_preserved": true',
            '"weight_addresses_preserved": false',
            1,
        )
    )
    assert build_report(tmp_path)["status"] == "FAILED"


@pytest.mark.parametrize("outside", [False, True])
def test_cleanup_deletes_only_published_objects_in_exact_run(
    tmp_path, monkeypatch, outside
):
    import runpy
    import types

    config = {
        "delta_prefix": "deltas/",
        "bucket": "test-bucket",
        "storage": {"endpoint_url": None, "region": "test", "addressing_style": "auto"},
    }
    key = "deltas/nemotron-other/payload" if outside else "deltas/nemotron-test/payload"
    publication = {"run": "nemotron-test", "objects": [{"key": key, "bytes": 8}]}
    report = tmp_path / "publication.json"
    report.write_text(json.dumps(publication))
    real_read = Path.read_text
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda path, *a, **k: real_read(
            report if str(path) == "/tmp/mx-delta/report.json" else path, *a, **k
        ),
    )
    monkeypatch.setenv("DELTA_RUN", "nemotron-test")
    deleted = []

    class ClientError(Exception):
        def __init__(self):
            self.response = {"Error": {"Code": "404"}}

    class Storage:
        def head_object(self, **kwargs):
            assert kwargs == {"Bucket": "test-bucket", "Key": key}
            if deleted:
                raise ClientError()
            return {"ContentLength": 8}

        def delete_objects(self, **kwargs):
            deleted.append(kwargs)
            return {}

    monkeypatch.setitem(sys.modules, "config", types.SimpleNamespace(CONFIG=config))
    monkeypatch.setitem(
        sys.modules, "boto3", types.SimpleNamespace(client=lambda *a, **k: Storage())
    )
    monkeypatch.setitem(sys.modules, "botocore", types.ModuleType("botocore"))
    monkeypatch.setitem(
        sys.modules, "botocore.config", types.SimpleNamespace(Config=lambda **k: k)
    )
    monkeypatch.setitem(
        sys.modules,
        "botocore.exceptions",
        types.SimpleNamespace(ClientError=ClientError),
    )
    if outside:
        with pytest.raises(AssertionError):
            runpy.run_path(str(ROOT / "common/cleanup_objects.py"))
        assert not deleted
    else:
        runpy.run_path(str(ROOT / "common/cleanup_objects.py"))
        assert deleted == [
            {"Bucket": "test-bucket", "Delete": {"Objects": [{"Key": key}]}}
        ]
