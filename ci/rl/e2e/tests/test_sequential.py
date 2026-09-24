"""Observable two-update protocol and fail-closed evidence contracts."""

import contextlib
import importlib.util
import json
import runpy
import sys
import types
from pathlib import Path

import pytest
from test_harness import ROOT, build_report, render, saved_run


def records(root):
    return {
        tuple(line.split(" ", 3)[1:3]): json.loads(line.split(" ", 3)[3])
        for line in (root / "e2e-driver.log").read_text().splitlines()
        if line.startswith("RESULT ")
    }


def save_records(root, values):
    (root / "e2e-driver.log").write_text(
        "".join(
            f"RESULT {role} {step} {json.dumps(record)}\n"
            for (role, step), record in values.items()
        )
        + "E2E_PASS\n"
    )


@pytest.mark.parametrize("step", ["", "d2-"])
@pytest.mark.parametrize(
    "fault",
    [
        "missing_refit",
        "rank",
        "version",
        "session",
        "trace",
        "unchanged",
        "peer_hash",
        "publisher",
        "checkpoint_hash",
        "checkpoint_rank",
        "inference",
        "layout_missing",
        "layout_partial",
        "layout_size",
        "nan_time",
        "zero_time",
    ],
)
def test_report_rejects_bad_update_even_with_pass_marker(tmp_path, step, fault):
    saved_run(tmp_path)
    values = records(tmp_path)
    refit = values["s3", step + "refit"]
    check = values["s3", step + "verify-checkpoint"]["response"]["result"]
    layout = values["s3", step + "layout-after"]["response"]["result"]
    if fault == "missing_refit":
        del values["s3", step + "refit"]
    elif fault == "rank":
        refit["response"]["result"].pop()
    elif fault == "version":
        refit["response"]["result"][1]["serving_version"] = "wrong"
    elif fault == "session":
        refit["response"]["result"][1]["refit_session"] = "new-client"
    elif fault == "trace":
        refit["response"]["result"][1]["allocation_tracing"] = bool(step)
    elif fault == "unchanged":
        previous = "updated-hashes" if step else "base-hashes"
        values["s3", step + "updated-hashes"] = values["s3", previous]
    elif fault == "peer_hash":
        values["peer", step + "updated-hashes"]["response"]["result"][1]["tensors"][
            "embedding"
        ]["sha256"] = "bad"
    elif fault == "publisher":
        check[0]["publisher_hashes"]["embedding"] = "bad"
    elif fault == "checkpoint_hash":
        check[0]["full_checkpoint_sha256"] = "bad"
    elif fault == "checkpoint_rank":
        check[1]["sha256"] = "bad"
    elif fault == "inference":
        values["peer", step + "post-refit-inference"]["response"]["result"][0][
            "token_ids"
        ] = [8, 9]
    elif fault == "layout_missing":
        layout[1]["files"].pop("shard1")
    elif fault == "layout_partial":
        layout[1]["files"]["shard1"]["inode"] = 999 if step else 11
    elif fault == "layout_size":
        layout[1]["files"]["shard1"]["bytes"] = 999
    elif fault == "nan_time":
        refit["seconds"] = float("nan")
    elif fault == "zero_time":
        refit["seconds"] = 0
    save_records(tmp_path, values)
    assert build_report(tmp_path)["status"] == "FAILED", fault


@pytest.mark.parametrize(
    "fault",
    ["lineage", "missing_trial", "duplicate", "malformed", "reinit", "oom", "restart"],
)
def test_report_rejects_incomplete_or_restarted_run(tmp_path, fault):
    saved_run(tmp_path)
    if fault in ("lineage", "missing_trial"):
        path = tmp_path / "publication.json"
        data = json.loads(path.read_text())
        if fault == "lineage":
            data["trials"][1]["base_version"] = "nemotron-test-base"
        else:
            data["trials"].pop()
        path.write_text(json.dumps(data))
    elif fault in ("duplicate", "malformed", "reinit"):
        path = tmp_path / "e2e-driver.log"
        extra = (
            path.read_text().splitlines()[0]
            if fault == "duplicate"
            else "RESULT s3 broken {"
        )
        if fault == "reinit":
            values = records(tmp_path)
            extra = "RESULT s3 init " + json.dumps(values["s3", "init"])
        path.write_text(path.read_text() + extra + "\n")
    elif fault == "oom":
        (tmp_path / "s3-monitor.log").write_text(
            'HOST_MEMORY {"memory.events": "oom 1\\noom_kill 0\\n"}\n'
        )
    else:
        path = tmp_path / "peer-pod.json"
        path.write_text(
            path.read_text().replace('"restartCount": 0', '"restartCount": 1')
        )
    assert build_report(tmp_path)["status"] == "FAILED"


def test_report_preserves_separate_timings_and_allows_second_trace(tmp_path):
    saved_run(tmp_path)
    report = build_report(tmp_path)
    assert report["status"] == "PASS"
    assert report["s3_savings"] == {"seconds": 2.0, "percent": 50.0}
    assert report["sequential_refits"]["d2"]["peer"]["seconds"] == 2.0
    path = tmp_path / "config.json"
    config = json.loads(path.read_text())
    config["second_update_allocation_tracing"] = True
    path.write_text(json.dumps(config))
    assert build_report(tmp_path)["status"] == "FAILED"
    values = records(tmp_path)
    for role in config["roles"]:
        for row in values[role, "d2-refit"]["response"]["result"]:
            row["allocation_tracing"] = True
    save_records(tmp_path, values)
    assert build_report(tmp_path)["status"] == "PASS"


@pytest.mark.parametrize("fail_second", [False, True])
def test_driver_uses_same_clients_and_resumes_after_each_update(
    tmp_path, monkeypatch, fail_second
):
    config = saved_run(tmp_path)
    config["resource_prefix"] = "mx-test"
    evidence = records(tmp_path)
    output = tmp_path / "driver"
    output.mkdir()
    real_path = Path
    monkeypatch.setitem(sys.modules, "config", types.SimpleNamespace(CONFIG=config))
    calls = []
    steps = {role: 0 for role in config["roles"]}

    class Response:
        status_code = 200

        def __init__(self, result):
            self.result = result

        def json(self):
            return {"ok": True, "result": self.result}

        def raise_for_status(self):
            pass

    def post(url, json, timeout):
        role = "peer" if "-peer:" in url else "s3"
        route = url.rsplit("/", 1)[-1]
        calls.append((role, route, json))
        method = json.get("method")
        kwargs = json.get("kwargs", {})
        if route in ("pause", "resume"):
            return Response({"paused": route == "pause"})
        if route == "generate":
            return Response([{"token_ids": [3, 4], "logprob_count": 2}])
        if method == "diagnostic_start":
            return Response([{"rank": r} for r in range(2)])
        if method == "hotload_init":
            name = "init"
        elif method == "hotload":
            steps[role] += 1
            if fail_second and steps[role] == 2:
                raise RuntimeError("second refit failed")
            assert kwargs["weight_path"] == f"nemotron-test-d{steps[role]}"
            assert kwargs["allocation_tracing"] is (steps[role] == 1)
            name = ("d2-" if steps[role] == 2 else "") + "refit"
        elif method == "checkpoint_layout":
            version = kwargs["version_id"]
            if version.endswith("base"):
                name = "layout-before"
            elif version.endswith("d1"):
                name = "layout-after" if steps[role] == 1 else "d2-layout-before"
            else:
                name = "d2-layout-after"
        elif method == "hotload_verify_checkpoint":
            version = kwargs["version_id"]
            name = (
                version.split(":", 1)[1]
                if version.startswith("hashes:")
                else "verify-checkpoint"
            )
            if steps[role] == 2:
                name = "d2-" + name
        else:
            raise AssertionError(method)
        return Response(evidence[role, name]["response"]["result"])

    monkeypatch.setitem(
        sys.modules,
        "requests",
        types.SimpleNamespace(
            post=post,
            get=lambda *a, **k: Response(None),
            RequestException=RuntimeError,
        ),
    )
    # Redirect only the driver's fixed output/publication paths.
    monkeypatch.setattr(
        "pathlib.Path",
        lambda path: (
            output
            if path == "/tmp/mx-e2e"
            else (
                tmp_path / "publication.json"
                if path == "/tmp/mx-delta/report.json"
                else real_path(path)
            )
        ),
    )
    if fail_second:
        with pytest.raises(RuntimeError, match="second refit failed"):
            runpy.run_path(str(ROOT / "common/run_e2e.py"))
        assert (output / "FAIL").exists() and not (output / "PASS").exists()
    else:
        runpy.run_path(str(ROOT / "common/run_e2e.py"))
        assert (output / "PASS").exists()
        for role in config["roles"]:
            assert (
                sum(
                    r == role and body.get("method") == "hotload_init"
                    for r, _, body in calls
                )
                == 1
            )
            assert (
                sum(
                    r == role and body.get("method") == "hotload"
                    for r, _, body in calls
                )
                == 2
            )
            assert sum(r == role and route == "generate" for r, route, _ in calls) == 3
            assert (output / f"{role}-d2-refit.json").exists()


def test_disabled_allocation_trace_does_not_import_vllm_or_patch_loader(monkeypatch):
    # Load the real context manager without importing CUDA diagnostics dependencies.
    import ast

    source = ast.parse((ROOT / "common/diagnostic_observer.py").read_text())
    function = next(
        n
        for n in source.body
        if isinstance(n, ast.FunctionDef) and n.name == "allocation_trace"
    )
    namespace = {"contextlib": contextlib}
    exec(  # noqa: S102 -- Compile the real diagnostic context manager in isolation.
        compile(
            ast.Module(body=[function], type_ignores=[]),
            "diagnostic_observer.py",
            "exec",
        ),
        namespace,
    )
    monkeypatch.setitem(sys.modules, "vllm", None)
    for rank, enabled in [(0, False), (1, True), (1, False)]:
        with namespace["allocation_trace"](rank, enabled=enabled):
            pass


def test_kimi_hooks_are_rendered_only_for_kimi(tmp_path):
    for model in ("kimi", "nemotron"):
        output = tmp_path / model
        config = render(model, output, model + "-seq")
        code = (output / "model_adapter.py").read_text()
        assert ("get_and_maybe_dequant_weights" in code) == (model == "kimi")
        assert bool(config.get("derived_weight_check")) == (model == "kimi")
        assert config["second_update_allocation_tracing"] is False


def test_kimi_checks_require_both_targeted_derived_weights_on_every_rank():
    spec = importlib.util.spec_from_file_location(
        "kimi_checks", ROOT / "profiles/kimi/model_checks.py"
    )
    hook = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hook)

    def rows(phase):
        return [
            {
                "rank": rank,
                "phase": phase,
                "verified": True,
                "changed": [
                    f"model.layers.0.self_attn.mla.{key}" for key in ("W_UV", "W_UK_T")
                ],
                "tensors": [
                    {
                        "name": f"model.layers.0.self_attn.mla.{key}",
                        "sha256": phase,
                        "pointer": i + 1,
                    }
                    for i, key in enumerate(("W_UV", "W_UK_T"))
                ],
            }
            for rank in range(2)
        ]

    baseline, updated = rows("baseline"), rows("updated")
    result = lambda role, step: baseline if step == "derived-baseline" else updated
    assert hook.validate({"tp": 2, "roles": ["s3", "peer"]}, result)
    updated[1]["tensors"][1]["sha256"] = "baseline"
    with pytest.raises(AssertionError):
        hook.validate({"tp": 2, "roles": ["s3", "peer"]}, result)


@pytest.mark.parametrize("fail_second", [False, True])
def test_publication_chains_versions_with_one_trainer(
    tmp_path, monkeypatch, fail_second
):
    from publication import publish_updates

    monkeypatch.setitem(
        sys.modules,
        "modelexpress_rl",
        types.SimpleNamespace(
            ObjectStorageSource=lambda **kwargs: kwargs,
            ObjectStorageType=types.SimpleNamespace(S3="s3"),
            WeightPayloadFormat=types.SimpleNamespace(XOR_DELTA="xor"),
            WeightVersionState=types.SimpleNamespace(READY="ready"),
        ),
    )
    events = []
    tensors = {"embedding": bytearray([1, 2, 3])}
    snapshots = []

    class Control:
        def create_weight_version(self, **kwargs):
            events.append(("create", kwargs))
            return types.SimpleNamespace(ref=kwargs["uid"])

        def update_weight_version_state(self, version, state):
            events.append(("ready", version))

    class Trainer:
        def stage_shard(self, version, hf_tensor_iter):
            assert hf_tensor_iter == [list(tensors.items())]
            snapshots.append(bytes(tensors["embedding"]))
            events.append(("stage", version))

            def publish():
                if fail_second and version.endswith("d2"):
                    raise RuntimeError("upload failed")
                events.append(("publish", version))

            return types.SimpleNamespace(publish=publish)

        def pop_metrics(self):
            return {"trial": len(snapshots)}

    def mutate():
        tensors["embedding"][0] ^= 7

    def publish():
        import hashlib

        return publish_updates(
            Control(),
            Trainer(),
            run="test",
            model="model",
            uri="s3://bucket/run/",
            tensors=tensors,
            embedding="embedding",
            mutate=mutate,
            digests=lambda: {
                "embedding": hashlib.sha256(tensors["embedding"]).hexdigest()
            },
        )

    if fail_second:
        with pytest.raises(RuntimeError, match="upload failed"):
            publish()
        assert ("ready", "test-d2") not in events
    else:
        trials = publish()
        assert [r["base_version"] for r in trials] == ["test-base", "test-d1"]
        assert [r["publisher_metrics"] for r in trials] == [{"trial": 1}, {"trial": 2}]
        assert trials[0]["expected_sha256"] != trials[1]["expected_sha256"]
        assert [event[0] for event in events] == [
            "create",
            "stage",
            "publish",
            "ready",
        ] * 2
        versions = [value for name, value in events if name == "create"]
        assert [v["object_storage"]["uri"] for v in versions] == [
            "s3://bucket/run/d1/model.safetensors.index.json",
            "s3://bucket/run/d2/model.safetensors.index.json",
        ]
    assert snapshots == [bytes([6, 2, 3]), bytes([1, 2, 3])]


@pytest.mark.parametrize("optimized", [False, True])
def test_report_cli_exits_nonzero_on_invalid_evidence(tmp_path, optimized):
    import subprocess

    saved_run(tmp_path)
    (tmp_path / "publication.json").unlink()
    command = [sys.executable]
    if optimized:
        command.append("-O")
    command.extend([str(ROOT / "common/report.py"), str(tmp_path)])
    process = subprocess.run(command, capture_output=True, text=True, check=False)
    assert process.returncode != 0
    if optimized:
        assert "requires Python assertions" in process.stderr
    else:
        assert json.loads((tmp_path / "report.json").read_text())["status"] == "FAILED"
