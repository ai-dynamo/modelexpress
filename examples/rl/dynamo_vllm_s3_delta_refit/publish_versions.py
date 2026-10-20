# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Publish a deterministic full checkpoint followed by XOR deltas."""

from __future__ import annotations

import os
from collections.abc import Iterator

import boto3
import torch
import torch.distributed as dist
from botocore.exceptions import ClientError
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM

from modelexpress_rl import (
    ModelExpressControlClient,
    ModelExpressTrainerClient,
    ModelExpressTrainerConfig,
    ObjectStorageConfig,
    ObjectStorageSource,
    ObjectStorageType,
    TrainerStagingMode,
    WeightPayloadFormat,
    WeightVersionState,
)

MODEL_NAME = os.environ["MODEL_NAME"]
MODEL_PATH = os.environ.get("MODEL_PATH", "/models")
MX_SERVER_ADDRESS = os.environ["MX_SERVER_ADDRESS"]
RUN_ID = os.environ["RUN_ID"]
S3_BUCKET = os.environ.get("S3_BUCKET", "mx-refit")
S3_ENDPOINT = os.environ["AWS_ENDPOINT_URL"]
S3_REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
S3_PREFIX = f"s3://{S3_BUCKET}/{RUN_ID}"
INITIAL_VERSION_ID = f"{RUN_ID}-w1"


def _version_id(number: int) -> str:
    return f"{RUN_ID}-v{number}"


def _tensor_batches(model: torch.nn.Module, size: int = 32) -> Iterator[list]:
    batch = []
    for item in model.named_parameters(remove_duplicate=True):
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def _mutate(model: torch.nn.Module, version: int) -> str:
    candidates = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.is_floating_point()
    ]
    if not candidates:
        raise RuntimeError("model has no floating-point parameter to mutate")
    name, parameter = max(candidates, key=lambda item: item[1].numel())
    count = min(parameter.numel(), 4096)
    with torch.no_grad():
        parameter.reshape(-1)[:count].add_(0.01 * version)
    return name


def _ensure_bucket() -> None:
    client = boto3.client("s3", endpoint_url=S3_ENDPOINT, region_name=S3_REGION)
    try:
        client.head_bucket(Bucket=S3_BUCKET)
    except ClientError as error:
        code = str(error.response.get("Error", {}).get("Code", ""))
        if code not in {"404", "NoSuchBucket", "NotFound"}:
            raise
        client.create_bucket(Bucket=S3_BUCKET)
    finally:
        client.close()


def _resolve_model_path() -> str:
    """Use the shared model cache when populated, otherwise download a seed."""
    if os.path.isfile(os.path.join(MODEL_PATH, "config.json")):
        return MODEL_PATH
    return snapshot_download(
        repo_id=MODEL_NAME,
        local_dir=MODEL_PATH,
    )


def _publish(
    *,
    control: ModelExpressControlClient,
    trainer: ModelExpressTrainerClient,
    model: torch.nn.Module,
    version_id: str,
    artifact_name: str,
    payload_format: WeightPayloadFormat,
    base_version_id: str | None,
) -> None:
    version = control.create_weight_version(
        uid=version_id,
        model_name=MODEL_NAME,
        idempotency_key=f"{RUN_ID}-publish-{artifact_name}",
        payload_format=payload_format,
        base_version_id=base_version_id,
        object_storage=ObjectStorageSource(
            storage_type=ObjectStorageType.S3,
            uri=f"{S3_PREFIX}/{artifact_name}/model.safetensors.index.json",
        ),
    )
    staged = trainer.stage_shard(
        version=version.ref,
        hf_tensor_iter=_tensor_batches(model),
    )
    staged.publish()
    ready = control.update_weight_version_state(
        version.version_id,
        WeightVersionState.READY,
    )
    if ready.state is not WeightVersionState.READY:
        raise RuntimeError(f"version {version_id} did not become READY")
    print(
        f"published version={version_id} format={payload_format.value} "
        f"base={base_version_id!r} metrics={trainer.pop_metrics()}",
        flush=True,
    )


def main() -> None:
    _ensure_bucket()
    model_path = _resolve_model_path()
    dist.init_process_group(
        "gloo",
        init_method="tcp://127.0.0.1:29500",
        rank=0,
        world_size=1,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    ).eval()
    trainer = ModelExpressTrainerClient.initialize(
        ModelExpressTrainerConfig(
            model_name=MODEL_NAME,
            server_url=MX_SERVER_ADDRESS,
            staging_mode=TrainerStagingMode.WRITE_TO_STORAGE,
            payload_format=WeightPayloadFormat.XOR_DELTA,
            process_group=dist.group.WORLD,
            object_storage=ObjectStorageConfig(
                storage_type=ObjectStorageType.S3,
                uri_prefix=S3_PREFIX,
                initial_base_version_id=INITIAL_VERSION_ID,
                seed_checkpoint_path=model_path,
                endpoint_url=S3_ENDPOINT,
                region_name=S3_REGION,
            ),
        )
    )
    control = ModelExpressControlClient.connect(server_url=MX_SERVER_ADDRESS)
    try:
        _publish(
            control=control,
            trainer=trainer,
            model=model,
            version_id=INITIAL_VERSION_ID,
            artifact_name="w1",
            payload_format=WeightPayloadFormat.FULL_HF_CHECKPOINT,
            base_version_id=None,
        )
        changed = _mutate(model, 1)
        print(f"deterministic mutation tensor={changed}", flush=True)
        _publish(
            control=control,
            trainer=trainer,
            model=model,
            version_id=_version_id(1),
            artifact_name="v1",
            payload_format=WeightPayloadFormat.FULL_HF_CHECKPOINT,
            base_version_id=None,
        )
        for number in (2, 3):
            _mutate(model, number)
            _publish(
                control=control,
                trainer=trainer,
                model=model,
                version_id=_version_id(number),
                artifact_name=f"d{number}",
                payload_format=WeightPayloadFormat.XOR_DELTA,
                base_version_id=_version_id(number - 1),
            )
    finally:
        control.close()
        trainer.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
