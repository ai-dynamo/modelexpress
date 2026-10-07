# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the same pause/refit/verify/resume protocol for the selected model profile."""

import json
import time
import traceback
from pathlib import Path

import requests
import scenario
import validation
from config import CONFIG as config

root = Path("/tmp/mx-bench")
root.mkdir(exist_ok=True)
case = scenario.load(config)
urls = {
    role: f"http://{config['resource_prefix']}-{role}:8080/" for role in config["roles"]
}


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
    publication = json.loads(Path(case.publication_path).read_text())
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
        audit(role, "baseline-host-scales")
        base[role] = tensor_hashes(role, "base-hashes")
    case.compare_workers(base, config)
    for role in urls:
        rows = rpc(
            role,
            "init",
            "hotload_init",
            {
                "initial_version_id": config["initial_version"],
                "source": config["sources"][role],
            },
        )
        assert all(x["phase"] == "init" for x in rows)
    for role in urls:
        rows = rpc(role, "refit", "hotload", {"weight_path": config["target_version"]})
        validation.refit(rows, config, role)
        case.verify_refit(rpc, role, config, publication)
        audit(role, "immediate-post-refit-host-scales")
        updated[role] = tensor_hashes(role, "updated-hashes")
        case.validate_inventory(base[role], updated[role])
    case.compare_workers(updated, config)
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
    print("BENCH_PASS", flush=True)
except Exception:
    (root / "FAIL").write_text(traceback.format_exc())
    traceback.print_exc()
    raise
