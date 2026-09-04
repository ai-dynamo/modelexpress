# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Coordinate a full WeightVersion update through Dynamo RL routes."""

from __future__ import annotations

import os
import time
import uuid
from typing import Any

import requests
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM

from modelexpress_rl import (
    FSDPTrainerContext,
    ModelExpressControlClient,
    ModelExpressTrainerClient,
    ModelExpressTrainerConfig,
    WeightPayloadFormat,
    WeightVersionRef,
    WeightVersionState,
)

MODEL_NAME = os.environ["MODEL_NAME"]
FRONTEND_URL = os.environ.get(
    "DYNAMO_FRONTEND_URL", "http://mx-vllm-refit-frontend-admin:8000"
).rstrip("/")
RL_DISCOVERY_URL = os.environ.get(
    "DYNAMO_RL_DISCOVERY_URL", "http://mx-vllm-refit-frontend-admin:8001"
).rstrip("/")
MX_SERVER_ADDRESS = os.environ.get(
    "MX_SERVER_ADDRESS", "mx-vllm-refit-server:8000"
)


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
                if workers:
                    return workers
        except requests.RequestException as error:
            last = str(error)
        time.sleep(3)
    raise RuntimeError(f"timed out discovering Dynamo RL workers: {last}")


def _inference(timeout: int = 120) -> str:
    deadline = time.monotonic() + timeout
    last = ""
    while time.monotonic() < deadline:
        try:
            response = requests.post(
                f"{FRONTEND_URL}/v1/chat/completions",
                json={
                    "model": MODEL_NAME,
                    "messages": [
                        {"role": "user", "content": "Reply with exactly: ready"}
                    ],
                    "temperature": 0,
                    "max_tokens": 8,
                },
                timeout=30,
            )
            if response.ok:
                return response.json()["choices"][0]["message"]["content"]
            last = f"{response.status_code}: {response.text}"
            if response.status_code != 503:
                raise RuntimeError(f"inference failed: {last}")
        except requests.RequestException as error:
            last = str(error)
        time.sleep(2)
    raise RuntimeError(f"timed out waiting for inference routing: {last}")


def _route(worker: dict[str, Any], group: str, name: str) -> str:
    return f"{worker['system_url'].rstrip('/')}/engine/{group}/{name}"


def main() -> None:
    dist.init_process_group(
        "nccl",
        init_method="tcp://127.0.0.1:29500",
        rank=0,
        world_size=1,
    )
    torch.cuda.set_device(0)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
    ).cuda().eval()

    workers = _discover_workers()
    baseline = _inference()
    trainer = ModelExpressTrainerClient.initialize(
        ModelExpressTrainerConfig(
            engine_context=FSDPTrainerContext(),
            device_id=0,
            model_name=MODEL_NAME,
            server_url=MX_SERVER_ADDRESS,
        )
    )
    control = ModelExpressControlClient.connect(server_url=MX_SERVER_ADDRESS)
    version = None
    try:
        source_slot = trainer.bind_tensors(model.state_dict())
        version = control.create_weight_version(
            model_name=MODEL_NAME,
            idempotency_key=f"dynamo-vllm-e2e-{uuid.uuid4().hex}",
            payload_format=WeightPayloadFormat.FULL_TENSOR,
            expected_source_slots=[source_slot],
        )
        trainer.publish_version(version=version.ref)
        ready = control.get_weight_version(version.version_id)
        if ready.state is not WeightVersionState.READY:
            raise RuntimeError(f"published version is not READY: {ready}")

        paused = []
        try:
            for worker in workers:
                _post(
                    _route(worker, "control", "pause_generation"),
                    {"mode": "keep", "clear_cache": False},
                )
                paused.append(worker)
            for worker in workers:
                _post(
                    _route(worker, "update", "init_weight_transfer_engine"),
                    {"init_info": {}},
                )
                _post(_route(worker, "update", "start_weight_update"), {})
                _post(
                    _route(worker, "update", "update_weights"),
                    {"update_info": {"version_id": version.version_id}},
                    timeout=600,
                )
                _post(
                    _route(worker, "update", "finish_weight_update"),
                    {"weight_version": version.version_id},
                )
            for worker in workers:
                installed = _post(
                    _route(worker, "control", "get_weight_version"), {}
                )
                observed = installed.get("version", installed.get("weight_version"))
                if observed != version.version_id:
                    raise RuntimeError(
                        f"worker installed {observed!r}, expected {version.version_id!r}"
                    )
        finally:
            for worker in paused:
                _post(_route(worker, "control", "resume_generation"), {})

        updated = _inference()
        if updated != baseline:
            raise RuntimeError(
                f"deterministic generation changed: before={baseline!r}, after={updated!r}"
            )
        print(
            f"E2E PASS: version={version.version_id} workers={len(workers)} "
            f"generation={updated!r}",
            flush=True,
        )
    finally:
        if version is not None:
            control.delete_weight_version(version.version_id)
            trainer.release_version(version=WeightVersionRef(version.version_id))
        control.close()
        trainer.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
