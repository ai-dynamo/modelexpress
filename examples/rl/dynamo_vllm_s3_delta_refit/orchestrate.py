# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Drive Dynamo RL weight-update routes like an external orchestrator."""

from __future__ import annotations

import os
import time
from typing import Any

import requests

MODEL_NAME = os.environ["MODEL_NAME"]
MODEL_PATH = os.environ.get("MODEL_PATH", "/models")
RUN_ID = os.environ["RUN_ID"]
MODE = os.environ.get("TEST_MODE", "active")
FRONTEND_URL = os.environ.get(
    "DYNAMO_FRONTEND_URL", "http://mx-s3-refit-frontend-admin:8000"
).rstrip("/")
RL_DISCOVERY_URL = os.environ.get(
    "DYNAMO_RL_DISCOVERY_URL", "http://mx-s3-refit-frontend-admin:8001"
).rstrip("/")
MX_SERVER_ADDRESS = os.environ.get(
    "MX_SERVER_ADDRESS", "mx-s3-refit-server:8000"
)
S3_ENDPOINT = os.environ["AWS_ENDPOINT_URL"]
S3_REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
DESIRED_VERSION_CONFIGMAP = os.environ.get(
    "DESIRED_VERSION_CONFIGMAP", "mx-s3-refit-desired-version"
)
EXPECTED_WORKERS = int(os.environ.get("EXPECTED_WORKERS", "2"))


def _version_id(number: int) -> str:
    return f"{RUN_ID}-v{number}"


def _post(url: str, body: dict[str, Any], timeout: int = 300) -> dict[str, Any]:
    response = requests.post(url, json=body, timeout=timeout)
    if not response.ok:
        raise RuntimeError(f"{url} failed ({response.status_code}): {response.text}")
    payload = response.json()
    if payload.get("status") == "error":
        raise RuntimeError(f"{url} failed: {payload}")
    return payload


def _discover_workers(timeout: int = 600) -> list[dict[str, Any]]:
    deadline = time.monotonic() + timeout
    last = ""
    while time.monotonic() < deadline:
        try:
            response = requests.get(f"{RL_DISCOVERY_URL}/v1/rl/workers", timeout=10)
            last = response.text
            if response.ok:
                workers = [
                    worker
                    for worker in response.json().get("workers", [])
                    if worker.get("system_url")
                ]
                if len(workers) == EXPECTED_WORKERS:
                    return sorted(workers, key=lambda worker: worker["system_url"])
                last = f"discovered {len(workers)} of {EXPECTED_WORKERS} workers"
        except requests.RequestException as error:
            last = str(error)
        time.sleep(3)
    raise RuntimeError(f"timed out discovering Dynamo RL workers: {last}")


def _route(worker: dict[str, Any], group: str, name: str) -> str:
    return f"{worker['system_url'].rstrip('/')}/engine/{group}/{name}"


def _observed_version(worker: dict[str, Any]) -> str:
    response = _post(_route(worker, "control", "get_weight_version"), {})
    observed = response.get("version", response.get("weight_version"))
    if not isinstance(observed, str) or not observed:
        raise RuntimeError(f"worker returned invalid serving version: {response}")
    return observed


def _inference() -> str:
    response = requests.post(
        f"{FRONTEND_URL}/v1/chat/completions",
        json={
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": "Reply with exactly: ready"}],
            "temperature": 0,
            "max_tokens": 16,
        },
        timeout=120,
    )
    if not response.ok:
        raise RuntimeError(
            f"inference failed ({response.status_code}): {response.text}"
        )
    return response.json()["choices"][0]["message"]["content"]


def _set_desired_version(version_id: str) -> None:
    namespace = open(
        "/var/run/secrets/kubernetes.io/serviceaccount/namespace",
        encoding="utf-8",
    ).read().strip()
    token = open(
        "/var/run/secrets/kubernetes.io/serviceaccount/token",
        encoding="utf-8",
    ).read().strip()
    url = (
        "https://kubernetes.default.svc/api/v1/namespaces/"
        f"{namespace}/configmaps/{DESIRED_VERSION_CONFIGMAP}"
    )
    response = requests.patch(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/merge-patch+json",
        },
        json={"data": {"desired_version_uid": version_id}},
        verify="/var/run/secrets/kubernetes.io/serviceaccount/ca.crt",
        timeout=30,
    )
    if not response.ok:
        raise RuntimeError(
            f"failed to persist desired version ({response.status_code}): "
            f"{response.text}"
        )


def _initialize_worker(worker: dict[str, Any]) -> None:
    observed = _observed_version(worker)
    init_info: dict[str, Any] = {
        "model_name": MODEL_NAME,
        "initial_base_version_id": f"{RUN_ID}-w1",
        "seed_checkpoint_path": MODEL_PATH,
        "refit_checkpoint_dir": "/var/cache/modelexpress/refit",
        "server_url": MX_SERVER_ADDRESS,
        "object_storage_type": "S3",
        "object_storage_endpoint_url": S3_ENDPOINT,
        "object_storage_region_name": S3_REGION,
    }
    if observed != "default":
        init_info["initial_serving_version_id"] = observed
    _post(
        _route(worker, "update", "init_weight_transfer_engine"),
        {"init_info": init_info},
    )


def _install(workers: list[dict[str, Any]], version_id: str) -> None:
    paused = []
    try:
        for worker in workers:
            _post(
                _route(worker, "control", "pause_generation"),
                {"mode": "keep", "clear_cache": False},
            )
            paused.append(worker)
        for worker in workers:
            _post(_route(worker, "update", "start_weight_update"), {})
            _post(
                _route(worker, "update", "update_weights"),
                {"update_info": {"version_id": version_id}},
                timeout=900,
            )
            _post(
                _route(worker, "update", "finish_weight_update"),
                {"weight_version": version_id},
            )
        for worker in workers:
            observed = _observed_version(worker)
            if observed != version_id:
                raise RuntimeError(
                    f"worker installed {observed!r}, expected {version_id!r}"
                )
        _set_desired_version(version_id)
    finally:
        for worker in paused:
            _post(_route(worker, "control", "resume_generation"), {})


def main() -> None:
    workers = _discover_workers()
    if MODE == "verify":
        expected = _version_id(3)
        for worker in workers:
            observed = _observed_version(worker)
            if observed != expected:
                raise RuntimeError(
                    f"replacement worker serves {observed!r}, expected {expected!r}"
                )
        output = _inference()
        print(
            f"RESTART PASS: version={expected} workers={len(workers)} "
            f"generation={output!r}",
            flush=True,
        )
        return

    baseline = _inference()
    for worker in workers:
        _initialize_worker(worker)
    for number in (1, 2, 3):
        version_id = _version_id(number)
        _install(workers, version_id)
        output = _inference()
        print(
            f"installed version={version_id} generation={output!r}",
            flush=True,
        )
    print(
        f"ACTIVE REFIT PASS: versions=3 workers={len(workers)} "
        f"baseline={baseline!r}",
        flush=True,
    )


if __name__ == "__main__":
    main()
