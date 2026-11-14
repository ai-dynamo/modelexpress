# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import os
import subprocess
import sys
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

import pytest
import safetensors.torch
import torch
import torch.distributed as dist
import yaml

EXAMPLE = Path(__file__).parents[3] / "examples/rl/dynamo_vllm_azure_refit"


@pytest.fixture
def orchestrator():
    return load_example("orchestrate")


@pytest.fixture
def publisher():
    pytest.importorskip("azure.storage.blob")
    return load_example("publish_versions")


def load_example(name):
    spec = importlib.util.spec_from_file_location(name, EXAMPLE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "count,failure",
    [(1, None), (2, None), (2, "pause_generation"), (2, "start_weight_update"),
     (2, "update_weights"), (2, "finish_weight_update"), (2, "verify"), (2, "persist")],
)
def test_update_resumes_only_after_verified_and_persisted_install(
    orchestrator, monkeypatch, count, failure
):
    events = []
    workers = [{"system_url": f"http://worker-{number}"} for number in range(count)]

    def post(url, body, **kwargs):
        worker = url.split("/")[2]
        operation = url.rsplit("/", 1)[-1]
        events.append((worker, operation))
        if operation == "pause_generation":
            assert body == {"mode": "abort", "clear_cache": True}
        elif operation == "update_weights":
            assert body == {"update_info": {"version_id": "v2"}}
        elif operation == "finish_weight_update":
            assert body == {"weight_version": "v2"}
        if worker == "worker-1" and operation == failure:
            raise RuntimeError("injected worker failure")

    def observed(worker):
        name = worker["system_url"].rsplit("/", 1)[-1]
        events.append((name, "verify"))
        return "v1" if name == "worker-1" and failure == "verify" else "v2"

    def persist(version):
        events.append(("catalog", "persist"))
        assert version == "v2"
        if failure == "persist":
            raise RuntimeError("injected persistence failure")

    monkeypatch.setattr(orchestrator, "post", post)
    monkeypatch.setattr(orchestrator, "observed_version", observed)
    monkeypatch.setattr(orchestrator, "set_desired_version", persist)
    names = [f"worker-{number}" for number in range(count)]
    ordered = [(name, "pause_generation") for name in names]
    ordered += [(name, operation) for name in names for operation in (
        "start_weight_update", "update_weights", "finish_weight_update",
    )]
    ordered += [(name, "verify") for name in names] + [("catalog", "persist")]
    ordered += [(name, "resume_generation") for name in names]
    if failure:
        with pytest.raises(RuntimeError):
            orchestrator.install(workers, "v2")
        actor = "catalog" if failure == "persist" else "worker-1"
        assert events == ordered[:ordered.index((actor, failure)) + 1]
    else:
        orchestrator.install(workers, "v2")
        assert events == ordered


def test_http_success_with_worker_error_is_rejected(orchestrator, monkeypatch):
    response = Mock()
    response.json.return_value = {"status": "error"}
    monkeypatch.setattr(orchestrator.requests, "post", lambda *a, **kw: response)
    with pytest.raises(RuntimeError, match="worker operation failed"):
        orchestrator.post("http://worker/engine/update/update_weights", {})


@pytest.mark.parametrize("scenario,failure", [
    ("blob", ""), ("p2p", ""), ("p2p", "no-peer-update"), ("p2p", "fallback"),
    ("p2p", "unexpected-fallback"),
])
def test_runner_checks_sources_and_retries_replacement_lookup(tmp_path, scenario, failure):
    """Run the actual shell workflow with local command doubles, never Kubernetes."""
    commands = r'''
import os, re, sys
from pathlib import Path
from string import Template

root = Path(os.environ["TEST_COMMAND_DIR"])
name = Path(sys.argv[0]).name
args = sys.argv[1:]
if name == "envsubst":
    names = re.findall(r"\$\{(\w+)\}", args[0])
    print(Template(sys.stdin.read()).safe_substitute({n: os.environ[n] for n in names}))
    sys.exit(0)
if name == "sleep":
    sys.exit(0)
with (root / "calls").open("a") as out:
    out.write(" ".join(args) + "\n")
if args[:1] == ["--namespace"]:
    args = args[2:]
count = int(os.environ["EXPECTED_WORKERS"])
failure = os.environ["TEST_FAILURE"]
if args[:2] == ["apply", "-f"]:
    with (root / "manifests").open("a") as out:
        out.write("---\n" + sys.stdin.read())
elif args[:2] == ["create", "configmap"]:
    print("kind: ConfigMap\nmetadata: {name: mx-azure-refit-scripts}")
elif args[:2] == ["get", "configmap"]:
    print("review-v3")
elif args[:2] == ["get", "pods"]:
    if (root / "deleted").exists():
        print("worker-b\ngone\nterminating\nreplacement")
    else:
        print("worker-a" if count == 1 else "worker-a worker-b")
elif args[:2] == ["delete", "pod"]:
    (root / "deleted").touch()
elif args[:2] == ["get", "pod"]:
    if args[2] == "gone":
        print("pod disappeared", file=sys.stderr)
        sys.exit(1)
    print("now|True" if args[2] in {"terminating", "worker-b"} else "|True")
elif args[:1] == ["logs"] and not args[1].startswith("job/"):
    if args[1] == "replacement":
        strategy = "desired_version_p2p" if count == 2 else "desired_version_object_storage"
        print("Trying strategy: " + strategy)
        if failure in {"fallback", "unexpected-fallback"}:
            error = "failed" if failure == "fallback" else "raised unexpected error"
            print("Strategy " + strategy + " " + error + ", trying next")
            print("Trying strategy: desired_version_object_storage")
    elif args[1] == "worker-a":
        for n in (1, 2, 3):
            print(f"active refit version=review-v{n} trying source=OBJECT_STORAGE")
    elif failure != "no-peer-update":
        for n in (1, 2, 3):
            print(f"weight update version=review-v{n} prepared source=GENERATOR method=RuntimeTensorNixlUpdateMethod")
'''
    for name in ("kubectl", "envsubst", "sleep"):
        path = tmp_path / name
        path.write_text(f"#!{sys.executable}\n" + commands)
        path.chmod(0o755)
    result = subprocess.run(
        ["bash", str(EXAMPLE / "run.sh")], capture_output=True, text=True, timeout=20,
        env={**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}",
             "TEST_COMMAND_DIR": str(tmp_path), "TEST_FAILURE": failure,
             "NAMESPACE": "review", "SCENARIO": scenario, "RUN_ID": "review",
             "WORKER_IMAGE": "example/worker:test", "DYNAMO_FRONTEND_IMAGE": "example/frontend:test"},
    )
    calls = (tmp_path / "calls").read_text()
    if failure:
        assert result.returncode != 0
        assert "E2E PASS" not in result.stdout
        assert "wait --for=condition=complete job/mx-azure-refit-restart-verify" not in calls
        return
    assert result.returncode == 0, result.stderr
    assert f"{scenario} replacement recovery" in result.stdout
    assert "get pod gone" in calls and "get pod terminating" in calls
    assert "logs terminating" not in calls and "logs replacement" in calls
    resources = list(yaml.safe_load_all((tmp_path / "manifests").read_text()))
    dgd = next(r for r in resources if r["kind"] == "DynamoGraphDeployment")
    worker = dgd["spec"]["components"][1]
    count = 2 if scenario == "p2p" else 1
    assert worker["replicas"] == count
    engine = worker["podTemplate"]["spec"]["initContainers"][0]
    env = {item["name"]: item.get("value") for item in engine["env"]}
    assert env["MX_P2P_METADATA"] == str(count - 1)
    assert env["MX_GENERATOR_SOURCE_ORDER"] == (
        "GENERATOR,OBJECT_STORAGE" if count == 2 else "OBJECT_STORAGE"
    )
    assert engine["resources"]["limits"]["rdma/ib"] == str(count - 1)
    for job in (r for r in resources if r["kind"] == "Job"):
        container = job["spec"]["template"]["spec"]["containers"][0]
        assert {e["name"]: e.get("value") for e in container["env"]}["EXPECTED_WORKERS"] == str(count)


@pytest.mark.parametrize("upload_fails", [False, True])
def test_publication_marks_azure_version_ready_only_after_upload(publisher, upload_fails):
    control, trainer = Mock(), Mock()
    version = SimpleNamespace(version_id="v1", ref="ref-v1")
    control.create_weight_version.return_value = version
    control.update_weight_version_state.return_value = SimpleNamespace(
        state=publisher.WeightVersionState.READY,
    )

    def upload():
        control.update_weight_version_state.assert_not_called()
        if upload_fails:
            raise RuntimeError("upload failed")

    trainer.stage_shard.return_value.publish.side_effect = upload
    model = torch.nn.Linear(2, 2)
    args = dict(
        model_name="example/model", prefix="az://weights/run", uid="v1",
        payload_format=publisher.WeightPayloadFormat.FULL_HF_CHECKPOINT,
    )
    if upload_fails:
        with pytest.raises(RuntimeError, match="upload failed"):
            publisher.publish(control, trainer, model, **args)
        control.update_weight_version_state.assert_not_called()
    else:
        publisher.publish(control, trainer, model, **args)
        control.update_weight_version_state.assert_called_once_with(
            "v1", publisher.WeightVersionState.READY,
        )
    source = control.create_weight_version.call_args.kwargs["object_storage"]
    assert source.storage_type is publisher.ObjectStorageType.AZURE
    assert source.uri == "az://weights/run/v1/model.safetensors.index.json"
    staged = trainer.stage_shard.call_args.kwargs
    assert staged["version"] == version.ref
    assert [name for batch in staged["tensor_iter"] for name, _ in batch] == [
        "weight", "bias",
    ]


@pytest.mark.skipif(
    not os.environ.get("MX_AZURE_EXAMPLE_TEST_SERVER_URL"),
    reason="requires a Redis-backed ModelExpress server and Azurite connection string",
)
def test_real_azurite_publication_and_replay(publisher, tmp_path):
    """Opt-in CPU check of the example against real Blob and control services."""
    server_url = os.environ["MX_AZURE_EXAMPLE_TEST_SERVER_URL"]
    container = f"mx-example-{uuid4().hex}"
    prefix = f"az://{container}/run"
    model = torch.nn.Linear(4, 4).to(torch.bfloat16)
    seed = tmp_path / "model.safetensors"
    safetensors.torch.save_file(model.state_dict(), seed)
    with ExitStack() as stack:
        blob = stack.enter_context(
            publisher.BlobServiceClient.from_connection_string(
                os.environ["AZURE_STORAGE_CONNECTION_STRING"],
            )
        )
        publisher.ensure_container(container)
        stack.callback(blob.delete_container, container)
        publisher.ensure_container(container)
        dist.init_process_group(
            "gloo", init_method=f"file://{tmp_path}/rendezvous", rank=0, world_size=1,
        )
        stack.callback(dist.destroy_process_group)
        trainer = publisher.ModelExpressTrainerClient.initialize(
            publisher.ModelExpressTrainerConfig(
                model_name="example/model", server_url=server_url,
                staging_mode=publisher.TrainerStagingMode.WRITE_TO_STORAGE,
                payload_format=publisher.WeightPayloadFormat.XOR_DELTA,
                process_group=dist.group.WORLD,
                object_storage=publisher.ObjectStorageConfig(
                    storage_type=publisher.ObjectStorageType.AZURE,
                    uri_prefix=prefix, initial_base_version_id=f"{container}-w1",
                    seed_checkpoint_path=seed,
                ),
            ),
        )
        stack.callback(trainer.close)
        control = stack.enter_context(
            publisher.ModelExpressControlClient.connect(server_url=server_url)
        )
        versions = []
        for number in range(4):
            if number:
                publisher.mutate(model, number)
            versions.append(
                publisher.publish(
                    control, trainer, model, model_name="example/model", prefix=prefix,
                    uid=f"{container}-v{number}" if number else f"{container}-w1",
                    payload_format=(
                        publisher.WeightPayloadFormat.FULL_HF_CHECKPOINT if number < 2
                        else publisher.WeightPayloadFormat.XOR_DELTA
                    ),
                    base=versions[-1].version_id if number > 1 else None,
                )
            )
        publisher.verify_publication(versions[1:], model)
        publisher.mutate(model, 4)
        with pytest.raises(RuntimeError, match="published tensor differs"):
            publisher.verify_publication(versions[1:], model)
