# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Publish and read back a full checkpoint followed by two Azure XOR deltas."""

from __future__ import annotations

import logging
import os
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory

import safetensors
import torch
import torch.distributed as dist
from azure.core.exceptions import ResourceExistsError
from azure.storage.blob import BlobServiceClient

from modelexpress_rl import (
    ModelExpressControlClient,
    ModelExpressTrainerClient,
    ModelExpressTrainerConfig,
    ObjectStorageConfig,
    ObjectStorageGeneratorConfig,
    ObjectStorageSource,
    ObjectStorageType,
    TrainerStagingMode,
    WeightPayloadFormat,
    WeightVersionState,
)
from modelexpress_rl.inference.methods import CanonicalDeltaUpdateMethod
from modelexpress_rl.inference.plan import ObjectStorageUpdateSource

logger = logging.getLogger("modelexpress.examples.azure_refit.publisher")


def tensor_batches(model, size=32):
    batch = []
    for item in model.named_parameters(remove_duplicate=True):
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def mutate(model, number):
    candidates = [item for item in model.named_parameters() if item[1].is_floating_point()]
    if not candidates:
        raise ValueError("model has no floating-point parameter to mutate")
    name, parameter = max(candidates, key=lambda item: item[1].numel())
    with torch.no_grad():
        parameter.reshape(-1)[:4096].add_(0.01 * number)
    logger.info("Applied synthetic mutation %d to %s", number, name)


def ensure_container(container):
    with BlobServiceClient.from_connection_string(
        os.environ["AZURE_STORAGE_CONNECTION_STRING"]
    ) as client:
        try:
            client.create_container(container)
        except ResourceExistsError as error:
            if error.error_code != "ContainerAlreadyExists":
                raise


def publish(control, trainer, model, *, model_name, prefix, uid, payload_format, base=None):
    version = control.create_weight_version(
        uid=uid,
        model_name=model_name,
        idempotency_key=uid,
        payload_format=payload_format,
        base_version_id=base,
        object_storage=ObjectStorageSource(
            storage_type=ObjectStorageType.AZURE,
            uri=f"{prefix}/{uid}/model.safetensors.index.json",
        ),
    )
    staged = trainer.stage_shard(version=version.ref, tensor_iter=tensor_batches(model))
    staged.publish()
    ready = control.update_weight_version_state(
        version.version_id, WeightVersionState.READY,
    )
    if ready.state is not WeightVersionState.READY:
        raise RuntimeError(f"version {uid} did not become READY")
    logger.info(
        "Published %s (%s), metrics=%s", uid, payload_format.value, trainer.pop_metrics(),
    )
    return ready


def verify_publication(versions, model):
    """Reconstruct from Blob into a fresh cache and compare every canonical tensor."""
    with TemporaryDirectory(prefix="mx-azure-verify-") as directory, ExitStack() as stack:
        receiver = CanonicalDeltaUpdateMethod(
            model_name=versions[0].model_name,
            config=ObjectStorageGeneratorConfig(
                storage_type=ObjectStorageType.AZURE,
                initial_base_version_id=versions[0].version_id,
                seed_checkpoint_path=None,
                refit_checkpoint_dir=directory,
            ),
        )
        stack.callback(receiver.close)
        artifact = receiver.prepare_chain(
            tuple(
                (version, ObjectStorageUpdateSource(
                    storage=version.object_storage,
                    payload_format=version.payload_format,
                ))
                for version in versions
            )
        )
        expected = dict(model.named_parameters(remove_duplicate=True))
        seen = set()
        for shard in artifact.checkpoint.path.glob("*.safetensors"):
            with safetensors.safe_open(shard, framework="pt", device="cpu") as tensors:
                for name in tensors.keys():
                    actual = tensors.get_tensor(name)
                    if name in seen or name not in expected:
                        raise RuntimeError(f"unexpected published tensor: {name}")
                    target = expected[name].detach().cpu()
                    if actual.dtype != target.dtype or not torch.equal(actual, target):
                        raise RuntimeError(f"published tensor differs: {name}")
                    seen.add(name)
        if seen != expected.keys():
            raise RuntimeError("published checkpoint is missing tensors")
        logger.info("Verified %d published tensors through Azure replay", len(seen))


def main():
    from transformers import AutoModelForCausalLM

    model_name = os.environ["MODEL_NAME"]
    model_path = Path(os.environ.get("MODEL_PATH", "/models"))
    run_id = os.environ["RUN_ID"]
    container = "mx-refit"
    prefix = f"az://{container}/{run_id}"
    if not (model_path / "config.json").is_file():
        raise ValueError("MODEL_PATH must contain the complete local Hugging Face checkpoint")
    ensure_container(container)
    with ExitStack() as stack:
        dist.init_process_group(
            "gloo", init_method="tcp://127.0.0.1:29500", rank=0, world_size=1,
        )
        stack.callback(dist.destroy_process_group)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, local_files_only=True,
        ).eval()
        trainer = ModelExpressTrainerClient.initialize(
            ModelExpressTrainerConfig(
                model_name=model_name,
                server_url=os.environ["MX_SERVER_ADDRESS"],
                staging_mode=TrainerStagingMode.WRITE_TO_STORAGE,
                payload_format=WeightPayloadFormat.XOR_DELTA,
                process_group=dist.group.WORLD,
                object_storage=ObjectStorageConfig(
                    storage_type=ObjectStorageType.AZURE,
                    uri_prefix=prefix,
                    initial_base_version_id=f"{run_id}-w1",
                    seed_checkpoint_path=model_path,
                ),
            ),
        )
        stack.callback(trainer.close)
        control = stack.enter_context(
            ModelExpressControlClient.connect(
                server_url=os.environ["MX_SERVER_ADDRESS"],
            )
        )
        versions = []
        for number in range(4):
            if number:
                mutate(model, number)
            uid = f"{run_id}-v{number}" if number else f"{run_id}-w1"
            version = publish(
                control, trainer, model, model_name=model_name, prefix=prefix, uid=uid,
                payload_format=(
                    WeightPayloadFormat.FULL_HF_CHECKPOINT if number < 2
                    else WeightPayloadFormat.XOR_DELTA
                ),
                base=versions[-1].version_id if number > 1 else None,
            )
            versions.append(version)
        verify_publication(versions[1:], model)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    main()
