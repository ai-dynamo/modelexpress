"""Run the same pause/refit/verify/resume protocol for the selected model profile."""

import json
import time
import traceback
from pathlib import Path

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


def call(role, name, route, body):
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
    base, updated = {}, {}
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
    for role in urls:
        rows = rpc(role, "refit", "hotload", {"weight_path": run + "-d1"})
        validation.refit(rows, config, role)
        if role == "s3":
            verified = rpc(
                role,
                "verify-checkpoint",
                "hotload_verify_checkpoint",
                {"version_id": run + "-d1"},
            )
            assert all(x.get("verified") for x in verified), verified
        audit(role, "immediate-post-refit-host-scales")
        updated[role] = tensor_hashes(role, "updated-hashes")
        assert updated[role] != base[role], "No updated tensors"
    if "peer" in urls:
        assert updated["s3"] == updated["peer"], "Refit peer tensors differ per TP rank"
    for role in urls:
        audit(role, "host-scales")
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
    for role in urls:
        call(role, "final-pause", "pause", {})
        audit(role, "post-inference-host-scales")
        call(role, "final-resume", "resume", {})
    (root / "PASS").write_text(
        "Requested paths, all TP ranks, refit and resumed inference verified\n"
    )
    print("E2E_PASS", flush=True)
except Exception:
    (root / "FAIL").write_text(traceback.format_exc())
    traceback.print_exc()
    raise
