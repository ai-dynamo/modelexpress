# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Apply Azure checkpoints through Dynamo's native RL control routes."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import requests

logger = logging.getLogger("modelexpress.examples.azure_refit.orchestrator")
FRONTEND = "http://mx-azure-refit-frontend-admin:8000"
DISCOVERY = "http://mx-azure-refit-frontend-admin:8001"


def post(url, body, timeout=300):
    response = requests.post(url, json=body, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    if payload.get("status") == "error":
        raise RuntimeError(f"worker operation failed: {url}")
    return payload


def route(worker, group, name):
    return f"{worker['system_url'].rstrip('/')}/engine/{group}/{name}"


def observed_version(worker):
    payload = post(route(worker, "control", "get_weight_version"), {})
    version = payload.get("version", payload.get("weight_version"))
    if not isinstance(version, str) or not version:
        raise RuntimeError("worker returned no serving version")
    return version


def discover_workers(expected, timeout=600):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            response = requests.get(f"{DISCOVERY}/v1/rl/workers", timeout=10)
            response.raise_for_status()
            workers = [w for w in response.json().get("workers", []) if w.get("system_url")]
            if len(workers) == expected:
                return sorted(workers, key=lambda worker: worker["system_url"])
        except requests.RequestException:
            pass
        time.sleep(3)
    raise RuntimeError(f"timed out discovering {expected} Dynamo RL workers")


def inference(model_name):
    payload = post(
        f"{FRONTEND}/v1/chat/completions",
        {
            "model": model_name,
            "messages": [{"role": "user", "content": "Reply with exactly: ready"}],
            "temperature": 0,
            "max_tokens": 16,
        },
        timeout=120,
    )
    return payload["choices"][0]["message"]["content"]


def set_desired_version(version):
    credentials = Path("/var/run/secrets/kubernetes.io/serviceaccount")
    namespace = (credentials / "namespace").read_text().strip()
    token = (credentials / "token").read_text().strip()
    response = requests.patch(
        f"https://kubernetes.default.svc/api/v1/namespaces/{namespace}"
        "/configmaps/mx-azure-refit-desired-version",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/merge-patch+json",
        },
        json={"data": {"desired_version_uid": version}},
        verify=str(credentials / "ca.crt"),
        timeout=30,
    )
    response.raise_for_status()


def initialize_worker(worker, model_name, run_id):
    observed = observed_version(worker)
    info = {
        "model_name": model_name,
        "initial_base_version_id": f"{run_id}-w1",
        "seed_checkpoint_path": "/models",
        "refit_checkpoint_dir": "/var/cache/modelexpress/refit",
        "server_url": "mx-azure-refit-server:8000",
        "object_storage_type": "AZURE",
    }
    if observed != "default":
        info["initial_serving_version_id"] = observed
    post(route(worker, "update", "init_weight_transfer_engine"), {"init_info": info})


def install(workers, version):
    try:
        # Pause all workers before any update; old requests and KV cache expire here.
        for worker in workers:
            post(
                route(worker, "control", "pause_generation"),
                {"mode": "abort", "clear_cache": True},
            )
        for worker in workers:
            post(route(worker, "update", "start_weight_update"), {})
            post(
                route(worker, "update", "update_weights"),
                {"update_info": {"version_id": version}},
                timeout=900,
            )
            post(route(worker, "update", "finish_weight_update"), {"weight_version": version})
        for worker in workers:
            if observed_version(worker) != version:
                raise RuntimeError(f"worker did not install {version}")
        set_desired_version(version)
    except Exception:
        logger.exception("Update failed; paused workers remain paused")
        raise
    for worker in workers:
        post(route(worker, "control", "resume_generation"), {})


def main():
    model_name = os.environ["MODEL_NAME"]
    run_id = os.environ["RUN_ID"]
    mode = os.environ.get("TEST_MODE", "active")
    if mode not in {"active", "verify"}:
        raise ValueError("TEST_MODE must be active or verify")
    expected = int(os.environ.get("EXPECTED_WORKERS", "1"))
    if expected not in (1, 2):
        raise ValueError("EXPECTED_WORKERS must be 1 or 2")
    workers = discover_workers(expected)
    if mode == "verify":
        version = f"{run_id}-v3"
        for worker in workers:
            if observed_version(worker) != version:
                raise RuntimeError(f"worker did not restore {version}")
        logger.info("RESTART PASS: version=%s generation=%r", version, inference(model_name))
        return
    logger.info("Baseline generation=%r", inference(model_name))
    for worker in workers:
        initialize_worker(worker, model_name, run_id)
    for number in (1, 2, 3):
        version = f"{run_id}-v{number}"
        install(workers, version)
        logger.info("Installed %s generation=%r", version, inference(model_name))
    logger.info("ACTIVE REFIT PASS: full checkpoint and two deltas")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    main()
