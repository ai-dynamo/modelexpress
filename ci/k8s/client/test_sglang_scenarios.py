# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise inference scenario configuration and rendezvous without GPUs."""

import fnmatch
import json
import os
import re
import shlex
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import yaml

import launch_dynamo

ROOT = Path(__file__).resolve().parents[3]


def documents(path):
    return list(yaml.safe_load_all((ROOT / path).read_text()))


def test_source_accuracy_precedes_target_deployment():
    action = documents(".github/actions/run-mx-p2p-test/action.yml")[0]
    steps = action["runs"]["steps"]
    names = [step["name"] for step in steps]
    capture = names.index("Capture source accuracy before deploying target")
    assert names.index("Wait for source to publish metadata") < capture < names.index("Apply target manifest")
    assert "--capture-source-baseline" in steps[capture]["run"]
    assert steps[capture]["if"] == "inputs.test_script == 'ci/k8s/client/test_p2p_k8s.py'"


@pytest.mark.parametrize("status", [0, 17])
def test_dynamo_discovery_diagnostic_does_not_block_tests(tmp_path, status):
    action = documents(".github/actions/run-mx-dynamo-test/action.yml")[0]
    script = next(step["run"] for step in action["runs"]["steps"] if step["name"] == "Run tests")
    for name, body in {
        "kubectl": f"echo '{{\"items\":[]}}'\nexit {status}\n",
        "pytest": "touch pytest-ran\n",
    }.items():
        path = tmp_path / name
        path.write_text("#!/bin/bash\n" + body)
        path.chmod(0o755)
    result = subprocess.run(
        ["bash", "-e", "-c", script], cwd=tmp_path,
        env={**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}",
             "NAMESPACE": "fixture", "MX_CI_MODEL": "model", "TEST_SCRIPT": "test.py",
             "DGD_NAME": "fixture", "PYTEST_ARGS": "", "FINAL_CR_COUNT": "2"},
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "pytest-ran").exists()
    artifact = json.loads((tmp_path / "inference-artifacts/fixture/worker-discovery.json").read_text())
    assert ("error" in artifact) == (status != 0)


@pytest.mark.parametrize("state", ["starts", "pending", "failed"])
def test_dynamo_container_start_deadline(tmp_path, state):
    action = documents(".github/actions/run-mx-dynamo-test/action.yml")[0]
    steps = action["runs"]["steps"]
    names = [step["name"] for step in steps]
    start = names.index("Wait for scaled-up worker containers to start")
    assert names.index("Scale DGD to replicas=2") < start < names.index(
        "Wait for scaled-up worker to complete P2P transfer"
    )
    pod = {"spec": {"containers": [{"name": "main"}]}, "status": {
        "phase": "Running", "containerStatuses": [{"state": {"running": {}}}],
    }}
    pending = {"spec": pod["spec"], "status": {"phase": "Pending"}}
    failed = {"spec": pod["spec"], "status": {"phase": "Failed"}}
    path = tmp_path / "kubectl"
    path.write_text(
        f"#!{sys.executable}\nimport json\nfrom pathlib import Path\n"
        f"p = Path({str(tmp_path / 'calls')!r})\n"
        "n = int(p.read_text()) + 1 if p.exists() else 1\np.write_text(str(n))\n"
        f"target = {failed!r} if {state!r} == 'failed' else {pending!r}\n"
        f"if {state!r} == 'starts' and n >= 3: target = {pod!r}\n"
        f"print(json.dumps({{'items': [{pod!r}, target]}}))\n"
    )
    path.chmod(0o755)
    script = steps[start]["run"]
    result = subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c",
         "sleep() { SECONDS=$((SECONDS + 300)); }\n" + script], cwd=ROOT,
        env={**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}",
             "NAMESPACE": "fixture", "DGD_NAME": "fixture", "FINAL_CR_COUNT": "2"},
        capture_output=True, text=True, timeout=10,
    )
    if state == "starts":
        assert result.returncode == 0, result.stderr
        assert "beginning the transfer/startup deadline" in result.stdout
    else:
        assert result.returncode != 0
        assert ("did not start within 900s" if state == "pending" else "worker failed") in result.stderr
    assert int((tmp_path / "calls").read_text()) == (1 if state == "failed" else 3)


def test_dynamo_preserves_complete_worker_logs(tmp_path):
    action = documents(".github/actions/run-mx-dynamo-test/action.yml")[0]
    steps = action["runs"]["steps"]
    names = [step["name"] for step in steps]
    assert names.index("Dump pod logs") < names.index("Upload inference comparison artifacts")
    path = tmp_path / "kubectl"
    path.write_text(
        "#!/bin/bash\n"
        "if [[ $1 == logs ]]; then\n"
        "  if [[ \" $* \" != *' --tail=-1 '* ]]; then exit 17; fi\n"
        "  for ((i=0; i<300; i++)); do echo \"startup-line-$i\"; done\n"
        "elif [[ \" $* \" == *' --no-headers '* ]]; then echo worker;\n"
        "elif [[ \" $* \" == *' -o json '* ]]; then\n"
        "  echo '{\"status\":{\"containerStatuses\":[{\"name\":\"main\",\"restartCount\":1}]}}';\n"
        "fi\n"
    )
    path.chmod(0o755)
    script = steps[names.index("Dump pod logs")]["run"]
    result = subprocess.run(
        ["bash", "-e", "-c", script], cwd=tmp_path,
        env={**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}",
             "NAMESPACE": "fixture", "DGD_NAME": "fixture"},
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, result.stderr
    for filename in ["worker.log", "worker-main-previous.log"]:
        lines = (tmp_path / "inference-artifacts/fixture/worker-logs" / filename).read_text().splitlines()
        assert len(lines) == 300
        assert lines[0] == "startup-line-0"
        assert lines[-1] == "startup-line-299"


@pytest.mark.parametrize("ordinal", [0, 1])
def test_sglang_cross_node_rendezvous(tmp_path, ordinal):
    service, statefulset = documents(
        "ci/k8s/client/sgl/manifest-azure-multi-node-tp2.yaml"
    )
    assert service["spec"]["publishNotReadyAddresses"] is True
    assert service["spec"]["clusterIP"] == "None"
    assert statefulset["spec"]["podManagementPolicy"] == "Parallel"
    assert statefulset["spec"]["replicas"] == 2
    pod = statefulset["spec"]["template"]["spec"]
    container = pod["containers"][0]
    assert container["resources"]["limits"]["nvidia.com/gpu"] == "1"
    anti = pod["affinity"]["podAntiAffinity"][
        "requiredDuringSchedulingIgnoredDuringExecution"
    ]
    assert anti[1]["matchLabelKeys"] == ["apps.kubernetes.io/pod-index"]
    python = tmp_path / "python3"
    python.write_text(
        f"#!{sys.executable}\nimport json, sys\nprint(json.dumps(sys.argv[1:]))\n"
    )
    python.chmod(0o755)
    result = subprocess.run(
        ["bash", "-c", container["args"][0]],
        env={
            **os.environ,
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "POD_NAME": f"mx-source-{ordinal}",
            "MX_CI_MODEL": "Qwen/Qwen3.5-0.8B",
            "WORKER_PORT": "8001",
        },
        capture_output=True,
        text=True,
        check=True,
    )
    args = json.loads(result.stdout)
    assert args[:2] == ["-m", "sglang.launch_server"]
    for flag, value in {
        "--tp": "2",
        "--nnodes": "2",
        "--node-rank": str(ordinal),
        "--dist-init-addr": "mx-source-0.mx-source:29500",
        "--load-format": "remote_instance",
    }.items():
        assert args[args.index(flag) + 1] == value


@pytest.mark.parametrize("framework,prefix", [("vllm", "Vllm"), ("sglang", "Sglang")])
@pytest.mark.parametrize("mode", ["aggregated", "disaggregated"])
def test_dynamo_scale_component(tmp_path, framework, prefix, mode):
    action = documents(".github/actions/run-mx-dynamo-test/action.yml")[0]
    script = next(
        step["run"]
        for step in action["runs"]["steps"]
        if step.get("name") == "Resolve mode-specific parameters"
    )
    output = tmp_path / "github-env"
    subprocess.run(
        ["bash", "-e", "-c", script],
        env={**os.environ, "FRAMEWORK": framework, "MODE": mode, "GITHUB_ENV": str(output)},
        check=True,
    )
    values = dict(line.split("=", 1) for line in output.read_text().splitlines())
    directory = "sgl" if framework == "sglang" else "vllm"
    dgd = documents(
        f"ci/k8s/client/{directory}/dynamo/manifest-azure-{mode}.yaml"
    )[0]
    component = prefix + ("Worker" if mode == "aggregated" else "DecodeWorker")
    assert values["SCALE_COMPONENT"] == component
    services = dgd["spec"]["services"]
    assert services[component]["replicas"] == 1
    workers = [name for name in services if name.startswith(prefix)]
    for name in workers:
        assert services[name]["extraPodSpec"]["mainContainer"]["command"] == [
            "python3", "/opt/mx-ci/launch_dynamo.py", framework,
        ]
    assert int(values["INITIAL_CR_COUNT"]) == len(workers)
    assert int(values["FINAL_CR_COUNT"]) == len(workers) + 1
    assert services["Frontend"]["extraPodSpec"]["mainContainer"]["image"].endswith(
        ":1.5.1"
    )


def test_sglang_mtp_revision_and_loaders():
    manifest = documents("ci/k8s/client/sgl/manifest-azure-mtp.yaml")[0]
    container = manifest["spec"]["template"]["spec"]["containers"][0]
    args = container["args"]
    env = {item["name"]: item.get("value") for item in container["env"]}
    assert "--disable-radix-cache" in args
    revision = args[args.index("--revision") + 1]
    assert len(revision) == 40
    assert revision == args[args.index("--speculative-draft-model-revision") + 1]
    assert revision == env["MX_MODEL_REVISION"]
    assert args[args.index("--speculative-draft-model-path") + 1] == env["MX_MODEL_URI"]
    for flag in ("--load-format", "--speculative-draft-load-format"):
        assert args[args.index(flag) + 1] == "remote_instance"


@pytest.mark.parametrize("label,tp,dp,ep", [("DP=2", "1", "2", None), ("EP=2", "2", None, "2")])
def test_sglang_parallel_matrix(label, tp, dp, ep):
    workflow = documents(".github/workflows/modelexpress-inference-gpu-ci.yml")[0]
    job = workflow["jobs"]["test"]
    include = job["strategy"]["matrix"]["include"]
    match = re.search(r"fromJSON\('((?:''|[^'])*)'\)", include)
    partitions = json.loads(match[1].replace("''", "'"))
    case = next(
        entry for entry in partitions["sglang"]
        if entry["name"] == f"P2P {label} test (sglang)"
    )
    manifest = documents(case["manifest"])[0]
    pod = manifest["spec"]["template"]["spec"]
    assert pod["nodeSelector"]["agentpool"] == case["pool"]
    container = pod["containers"][0]
    assert container["resources"]["limits"]["nvidia.com/gpu"] == "2"
    args = container["args"]
    assert args[args.index("--tp") + 1] == tp
    if dp:
        assert args[args.index("--dp-size") + 1] == dp
        assert case["dp_size"] == dp
    if ep:
        assert args[args.index("--ep-size") + 1] == ep
        assert case["model"] == "deepseek-ai/DeepSeek-V2-Lite"
    assert case["tp_size"] == tp


@pytest.mark.parametrize("name", [
    "manifest-azure.yaml", "manifest-azure-mtp.yaml", "manifest-azure-dp2.yaml",
    "manifest-azure-fleet.yaml", "manifest-azure-rolling-update.yaml",
    "manifest-azure-multi-node-tp2.yaml",
])
def test_vllm_qwen_graph_batches_fit_scheduler(name):
    manifest = next(
        doc for doc in documents(f"ci/k8s/client/vllm/{name}")
        if doc["kind"] in {"Job", "Deployment", "StatefulSet"}
    )
    container = manifest["spec"]["template"]["spec"]["containers"][0]
    args = container["args"]
    if container["command"][0] == "/bin/bash":
        args = shlex.split(args[0], comments=True)
    limit = int(args[args.index("--max-num-seqs") + 1])
    config = json.loads(args[args.index("--compilation-config") + 1])
    sizes = config["cudagraph_capture_sizes"]
    assert sizes and min(sizes) > 0
    assert max(sizes) <= limit <= 16
    assert config["cache_dir"]
    assert "--enforce-eager" not in args


@pytest.mark.parametrize("framework,flag", [("vllm", "--model"), ("sglang", "--model-path")])
def test_metadata_inventory_matches_after_cold_and_p2p_load(monkeypatch, tmp_path, framework, flag):
    files = {
        "config.json": "{}", "tokenizer.json": "{}", "tokenizer_config.json": "{}",
        "model.safetensors.index.json": '{"weight_map": {}}',
        "model-00001-of-00001.safetensors": "weights",
        "preprocessor_config.json": "{}", "chat_template.jinja": "template",
    }
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    target.mkdir()
    for folder in (source, target):
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
            (folder / name).write_text(files[name])
    (source / "model.safetensors.index.json").write_text(files["model.safetensors.index.json"])
    inventories = []
    args = [flag, "Qwen/Qwen3.5-0.8B", "--load-format", "modelexpress"]

    for folder in (source, target):
        def download(model, *, ignore_patterns):
            assert model == "Qwen/Qwen3.5-0.8B"
            for name, content in files.items():
                if not any(fnmatch.fnmatch(name, pattern) for pattern in ignore_patterns):
                    (folder / name).write_text(content)

        def execute(executable, argv):
            assert executable == sys.executable
            assert argv == [sys.executable, "-m", f"dynamo.{framework}", *args]
            inventories.append({p.name: p.read_text() for p in folder.iterdir()})

        monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=download))
        monkeypatch.setattr(launch_dynamo.os, "execv", execute)
        launch_dynamo.launch(framework, args)

    assert inventories[0] == inventories[1]
    assert "model.safetensors.index.json" in inventories[0]
    assert "model-00001-of-00001.safetensors" not in inventories[0]
