"""Run the same pause/refit/verify/resume protocol for the selected model profile."""

import json
import time
import traceback
from pathlib import Path

import model_checks
import requests
import validation
from config import CONFIG as config

root = Path("/tmp/mx-e2e")
root.mkdir(exist_ok=True)
run = config["run"]
urls = {
    role: f"http://{config['resource_prefix']}-{role}:8080/" for role in config["roles"]
}
records = {}
step_prefix = ""


def call(role, name, route, body):
    name = step_prefix + name
    started = time.time()
    t = time.perf_counter()
    response = requests.post(urls[role] + route, json=body, timeout=7200)
    record = {
        "started_unix": started,
        "finished_unix": time.time(),
        "seconds": time.perf_counter() - t,
        "http_status": response.status_code,
        "response": response.json(),
    }
    (root / f"{role}-{name}.json").write_text(json.dumps(record, indent=2))
    print("RESULT", role, name, json.dumps(record), flush=True)
    response.raise_for_status()
    assert record["response"]["ok"], record
    result = record["response"]["result"]
    records[role, name] = result
    return result


def rpc(role, name, method, kwargs):
    rows = call(role, name, "rpc", {"method": method, "kwargs": kwargs})
    validation.ranks(rows, config)
    return rows


def audit(role, name):
    if config["expected_host_scales_per_rank"] is not None:
        validation.scales(
            rpc(role, name, "hotload_verify_checkpoint", {"version_id": "host-scales"}),
            config,
        )


def tensor_hashes(role, name):
    return validation.hashes(
        rpc(role, name, "hotload_verify_checkpoint", {"version_id": "hashes:" + name}),
        config,
    )


try:
    publication = json.loads(Path("/tmp/mx-delta/report.json").read_text())
    trials = validation.trials(publication, config)
    base = {}
    for role, url in urls.items():
        deadline = time.monotonic() + 7200
        while time.monotonic() < deadline:
            try:
                if requests.get(url + "health", timeout=5).status_code == 200:
                    break
            except requests.RequestException:
                pass
            time.sleep(5)
        else:
            raise TimeoutError(role + " readiness")
        validation.inference(
            call(role, "baseline", "generate", {"prompt": "The capital of France is"})
        )
        call(role, "pause", "pause", {})
        # Diagnostic start returns rank records but no benchmark phase.
        started = call(
            role, "diagnostics", "rpc", {"method": "diagnostic_start", "kwargs": {}}
        )
        assert {x["rank"] for x in started} == set(range(config["tp"]))
        audit(role, "baseline-host-scales")
        base[role] = tensor_hashes(role, "base-hashes")
    if "peer" in urls:
        assert base["s3"] == base["peer"], "Cold peer tensors differ per TP rank"
    for role in urls:
        rows = rpc(
            role,
            "init",
            "hotload_init",
            {
                "run_id": run,
                "source": "OBJECT_STORAGE" if role == "s3" else "GENERATOR",
            },
        )
        assert all(x["phase"] == "init" for x in rows)
    sessions = validation.sessions(config, lambda role, name: records[role, name])
    previous_layout = None
    for step, trial in enumerate(trials, 1):
        step_prefix = "" if step == 1 else "d2-"
        version = trial["version"]
        for role in urls:
            if step > 1:
                call(role, "pause", "pause", {})
                audit(role, "baseline-host-scales")
            if config.get("derived_weight_check"):
                rpc(role, "derived-baseline", "derived_verify", {"phase": "baseline"})
        layout = rpc(
            "s3",
            "layout-before",
            "checkpoint_layout",
            {"version_id": trial["base_version"]},
        )
        if previous_layout is not None:
            assert validation.ranks(layout, config) == validation.ranks(
                previous_layout, config
            )
        for role in urls:
            rows = rpc(
                role,
                "refit",
                "hotload",
                {
                    "weight_path": version,
                    "allocation_tracing": True
                    if step == 1
                    else config["second_update_allocation_tracing"],
                },
            )
            validation.refit(rows, config, role, version)
            if role == "s3":
                rpc(
                    role,
                    "verify-checkpoint",
                    "hotload_verify_checkpoint",
                    {"version_id": version},
                )
                previous_layout = rpc(
                    role, "layout-after", "checkpoint_layout", {"version_id": version}
                )
            if config.get("derived_weight_check"):
                rpc(role, "derived-updated", "derived_verify", {"phase": "updated"})
            audit(role, "immediate-post-refit-host-scales")
            tensor_hashes(role, "updated-hashes")

        def result(role, name, step_prefix=step_prefix):
            return records[role, step_prefix + name]

        validation.refitted(config, trial, result, base, sessions, reuse=step == 2)
        if config.get("derived_weight_check"):
            model_checks.validate(config, result)
        for role in urls:
            call(role, "resume", "resume", {})
            validation.inference(
                call(
                    role,
                    "post-refit-inference",
                    "generate",
                    {"prompt": "The capital of France is"},
                )
            )
            call(role, "final-pause", "pause", {})
            audit(role, "post-inference-host-scales")
            call(role, "final-resume", "resume", {})

        base = validation.update(config, trial, result, base, sessions, reuse=step == 2)
        model_checks.validate(config, result)
    (root / "PASS").write_text(
        "Requested paths, all TP ranks, refit and resumed inference verified\n"
    )
    print("E2E_PASS", flush=True)
except Exception:
    (root / "FAIL").write_text(traceback.format_exc())
    traceback.print_exc()
    raise
