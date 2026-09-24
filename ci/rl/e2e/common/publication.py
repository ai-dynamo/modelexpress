"""Publish the two-version chain with one trainer and no intermediate reset."""

import time


def publish_updates(
    control, trainer, *, run, model, uri, tensors, embedding, mutate, digests
):
    from modelexpress_rl import (
        ObjectStorageSource,
        ObjectStorageType,
        WeightPayloadFormat,
        WeightVersionState,
    )

    trials = []
    for step in (1, 2):
        version = run + f"-d{step}"
        parent = run + "-base" if step == 1 else run + "-d1"
        mutate()
        expected_hashes = digests()
        v = control.create_weight_version(
            uid=version,
            model_name=model,
            idempotency_key=version,
            payload_format=WeightPayloadFormat.XOR_DELTA,
            base_version_id=parent,
            object_storage=ObjectStorageSource(
                storage_type=ObjectStorageType.S3,
                uri=uri + f"d{step}/model.safetensors.index.json",
            ),
        )
        t = time.perf_counter()
        staged = trainer.stage_shard(
            version=v.ref, hf_tensor_iter=[list(tensors.items())]
        )
        encode_seconds = time.perf_counter() - t
        t = time.perf_counter()
        staged.publish()
        publication_seconds = time.perf_counter() - t
        control.update_weight_version_state(version, WeightVersionState.READY)
        trials.append(
            {
                "version": version,
                "base_version": parent,
                "expected_sha256": expected_hashes[embedding],
                "expected_hashes": expected_hashes,
                "encode_seconds": encode_seconds,
                "publication_seconds": publication_seconds,
                "publisher_metrics": trainer.pop_metrics(),
            }
        )
    return trials
