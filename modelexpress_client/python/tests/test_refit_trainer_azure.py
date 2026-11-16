# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import safetensors.torch
import torch

pytest.importorskip("azure.storage.blob")

from azure.core.exceptions import ResourceExistsError
from modelexpress_rl import (
    ObjectStorageConfig,
    ObjectStorageGeneratorConfig,
    ObjectStorageType,
    WeightVersionRef,
    refit_pb2,
)
from modelexpress_rl.azure import AzureBlobClient, ImmutableAzureConflict
from modelexpress_rl.control import _weight_version
from modelexpress_rl.inference.methods import CanonicalDeltaUpdateMethod
from modelexpress_rl.inference.plan import ObjectStorageUpdateSource
from modelexpress_rl.train.methods import CanonicalDeltaPublicationMethod

from tests.test_refit_azure_reader import sdk
from tests.test_refit_trainer_s3 import _trainer, refit_server


@pytest.fixture
def azure_trainer(monkeypatch, tmp_path, sdk, refit_server):
    service, server_url = refit_server
    service.target.object_storage.storage_type = refit_pb2.OBJECT_STORAGE_TYPE_AZURE
    service.target.object_storage.uri = "az://weights/tests/v1/model.safetensors.index.json"
    monkeypatch.setenv("AZURE_STORAGE_ACCOUNT_NAME", "testaccount")
    backend = sdk.client
    backend.objects = {}
    backend.fail_put_once = None
    get_blob_client = backend.get_blob_client

    def blob_client(*, container, blob):
        key = (container, blob)
        client = get_blob_client(container=container, blob=blob)

        def upload(data, *, overwrite):
            assert overwrite is False
            backend.calls.append(("put", key))
            if key == backend.fail_put_once:
                backend.fail_put_once = None
                raise RuntimeError("injected upload failure")
            if key in backend.objects:
                error = ResourceExistsError("blob already exists")
                error.error_code = "BlobAlreadyExists"
                raise error
            backend.objects[key] = data

        client.upload_blob = upload
        return client

    monkeypatch.setattr(backend, "get_blob_client", blob_client)
    seed = tmp_path / "seed"
    seed.mkdir()
    tensors = {"a": torch.tensor([1.0, 2.0]), "b": torch.tensor([3.0, 4.0])}
    trainer, _ = _trainer(
        monkeypatch, seed, server_url, tensors,
        storage_type=ObjectStorageType.AZURE, storage=AzureBlobClient(),
    )
    try:
        yield SimpleNamespace(
            trainer=trainer, service=service, storage=backend, tensors=tensors,
        )
    finally:
        trainer.close()


@pytest.mark.parametrize("incremental", [False, True])
def test_azure_trainer_publishes_full_and_delta_chain_for_generator(
    azure_trainer, monkeypatch, tmp_path, incremental
):
    ctx = azure_trainer
    monkeypatch.setenv("MX_REFIT_FULL_CHECKPOINT_BATCH_BYTES", "8")
    monkeypatch.setenv("MX_REFIT_UPLOAD_WORKERS", "2")
    versions = []
    for number in (1, 2, 3):
        target = ctx.service.target
        target.uid = f"version-{number}"
        target.object_storage.uri = (
            f"az://weights/tests/V{number}%20 literal/model.safetensors.index.json"
        )
        target.state = refit_pb2.WEIGHT_VERSION_STATE_STAGING
        if number == 1:
            target.payload_format = refit_pb2.WEIGHT_PAYLOAD_FORMAT_FULL_HF_CHECKPOINT
            target.ClearField("base_version_id")
        else:
            target.payload_format = refit_pb2.WEIGHT_PAYLOAD_FORMAT_XOR_DELTA
            target.base_version_id = versions[-1].version_id
        expected = {name: tensor + number for name, tensor in ctx.tensors.items()}
        buckets = [[item] for item in expected.items()]
        call_start = len(ctx.storage.calls)
        if incremental:
            for bucket in buckets:
                staged = ctx.trainer.stage_shard(
                    version=WeightVersionRef(target.uid), tensors=bucket,
                )
        else:
            staged = ctx.trainer.stage_shard(
                version=WeightVersionRef(target.uid), tensor_iter=iter(buckets),
            )
        assert ctx.storage.calls[call_start:] == []
        staged.publish()
        staged.publish()
        index_key = ("weights", f"tests/V{number}%20 literal/model.safetensors.index.json")
        assert ctx.storage.calls[-1] == ("put", index_key)
        assert ctx.trainer._runtime.method.current_base_version_id == target.uid
        # The orchestrator marks a version ready only after publication completes.
        assert target.state == refit_pb2.WEIGHT_VERSION_STATE_STAGING
        target.state = refit_pb2.WEIGHT_VERSION_STATE_READY
        versions.append(_weight_version(target))

    assert ctx.service.registrations == set()
    assert ctx.service.shards == []
    receiver = CanonicalDeltaUpdateMethod(
        model_name="test/model",
        config=ObjectStorageGeneratorConfig(
            storage_type=ObjectStorageType.AZURE,
            initial_base_version_id="base-a",
            seed_checkpoint_path=None,
            refit_checkpoint_dir=tmp_path / "cache",
        ),
    )
    try:
        prepared = receiver.prepare_chain(
            tuple(
                (version, ObjectStorageUpdateSource(
                    storage=version.object_storage,
                    payload_format=version.payload_format,
                ))
                for version in versions
            )
        )
        assert prepared.checkpoint.target_version == versions[-1].version_id
        actual = {}
        for shard in prepared.checkpoint.path.glob("*.safetensors"):
            actual.update(safetensors.torch.load_file(shard))
        assert actual.keys() == expected.keys()
        for name, tensor in expected.items():
            assert torch.equal(actual[name], tensor)
    finally:
        receiver.close()


def test_azure_publication_combines_two_ranks(azure_trainer, monkeypatch):
    ctx = azure_trainer
    first = ctx.trainer._runtime.method
    first._world_size = 2
    service = ctx.trainer._service
    monkeypatch.setattr(torch.distributed, "get_rank", lambda _group=None: 1)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda _group=None: 2)
    second = CanonicalDeltaPublicationMethod(
        config=first._config, model_name="test/model", service=lambda: service,
        rpc_timeout_seconds=30, process_group=None,
        read_seed_tensor=first._read_seed_tensor, writer=AzureBlobClient(),
    )
    methods = (first, second)
    local = threading.local()
    barrier = threading.Barrier(2)
    contributions = [None, None]

    def exchange(value):
        contributions[local.rank] = value
        barrier.wait(timeout=5)
        result = contributions.copy()
        barrier.wait(timeout=5)
        return result

    def gather(value, output, **_kwargs):
        values = exchange(value)
        if output is not None:
            output[:] = values

    monkeypatch.setattr(torch.distributed, "gather_object", gather)
    monkeypatch.setattr(
        torch.distributed, "all_gather_object",
        lambda output, value, **kwargs: gather(value, output, **kwargs),
    )
    try:
        for number in (1, 2):
            target = ctx.service.target
            target.uid = f"version-{number}"
            target.object_storage.uri = (
                f"az://weights/tests/v{number}/model.safetensors.index.json"
            )
            if number == 1:
                target.payload_format = refit_pb2.WEIGHT_PAYLOAD_FORMAT_FULL_HF_CHECKPOINT
                target.ClearField("base_version_id")
            else:
                target.payload_format = refit_pb2.WEIGHT_PAYLOAD_FORMAT_XOR_DELTA
                target.base_version_id = "version-1"

            def publish(rank):
                local.rank = rank
                name = ("a", "b")[rank]
                version = WeightVersionRef(target.uid)
                staged = methods[rank].stage(
                    version=version,
                    tensor_iter=iter([[(name, ctx.tensors[name] + number)]]),
                )
                methods[rank].publish(version=version, staged=staged)

            with ThreadPoolExecutor(max_workers=2) as pool:
                list(pool.map(publish, range(2)))
            index_key = ("weights", f"tests/v{number}/model.safetensors.index.json")
            index = json.loads(ctx.storage.objects[index_key])
            assert set(index["weight_map"]) == {"a", "b"}
            assert len(set(index["weight_map"].values())) == 2
            assert ctx.storage.calls[-1] == ("put", index_key)
            assert all(method.current_base_version_id == target.uid for method in methods)
    finally:
        second.close()


@pytest.mark.parametrize("full", [False, True])
@pytest.mark.parametrize("failure", ["shard", "index", "conflict"])
def test_azure_publication_failure_retains_base_and_staged_payload(
    azure_trainer, monkeypatch, full, failure
):
    ctx = azure_trainer
    # One checksum per shard avoids the shared Safetensors metadata-order retry gap.
    monkeypatch.setenv("MX_REFIT_FULL_CHECKPOINT_BATCH_BYTES", "8")
    if full:
        ctx.service.target.payload_format = refit_pb2.WEIGHT_PAYLOAD_FORMAT_FULL_HF_CHECKPOINT
        ctx.service.target.ClearField("base_version_id")
    staged = ctx.trainer.stage_shard(
        version=WeightVersionRef("target-a"),
        tensor_iter=iter([
            [(name, tensor + (1 if name == "a" else 0))]
            for name, tensor in ctx.tensors.items()
        ]),
    )
    index_key = ("weights", "tests/v1/model.safetensors.index.json")
    shard_key = (
        "weights",
        f"tests/v1/model-{1 if full else 0:05d}-of-{2 if full else 1:05d}.safetensors",
    )
    if failure == "conflict":
        ctx.storage.objects[shard_key] = b"existing immutable data"
        error = ImmutableAzureConflict
    else:
        ctx.storage.fail_put_once = index_key if failure == "index" else shard_key
        error = RuntimeError
    with pytest.raises(error):
        staged.publish()
    method = ctx.trainer._runtime.method
    assert method.current_base_version_id == "base-a"
    assert method._staged is staged._staged
    assert index_key not in ctx.storage.objects
    assert ctx.service.target.state == refit_pb2.WEIGHT_VERSION_STATE_STAGING
    if failure == "conflict":
        assert ctx.storage.objects[shard_key] == b"existing immutable data"
    else:
        staged.publish()
        assert method.current_base_version_id == "target-a"
        assert index_key in ctx.storage.objects


@pytest.mark.parametrize("mismatch", ["provider", "prefix", "base"])
def test_azure_trainer_rejects_incompatible_target_before_upload(azure_trainer, mismatch):
    ctx = azure_trainer
    target = ctx.service.target
    if mismatch == "provider":
        target.object_storage.storage_type = refit_pb2.OBJECT_STORAGE_TYPE_S3
    elif mismatch == "prefix":
        target.object_storage.uri = "az://weights/tests-other/model.safetensors.index.json"
    else:
        target.base_version_id = "other-base"
    with pytest.raises(RuntimeError):
        ctx.trainer.stage_shard(
            version=WeightVersionRef("target-a"),
            tensor_iter=iter([[item] for item in ctx.tensors.items()]),
        )
    assert ctx.storage.calls == []
    assert ctx.trainer._runtime.method.current_base_version_id == "base-a"


@pytest.mark.parametrize("field", ["endpoint_url", "region_name"])
def test_azure_trainer_rejects_s3_connection_fields(tmp_path, field):
    with pytest.raises(ValueError, match=f"{field} is not supported for Azure"):
        ObjectStorageConfig(
            storage_type=ObjectStorageType.AZURE,
            uri_prefix="az://weights/tests",
            initial_base_version_id="base-a",
            seed_checkpoint_path=tmp_path,
            **{field: "s3-setting"},
        )


@pytest.mark.parametrize(
    "prefix", ["s3://weights/tests", "az:///tests", "az://weights/tests?"],
)
def test_azure_trainer_rejects_invalid_blob_prefix(tmp_path, prefix):
    with pytest.raises(ValueError, match="invalid Azure Blob URI"):
        ObjectStorageConfig(
            storage_type=ObjectStorageType.AZURE,
            uri_prefix=prefix,
            initial_base_version_id="base-a",
            seed_checkpoint_path=tmp_path,
        )
