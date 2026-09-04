# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import threading
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import pytest
import safetensors.numpy
import safetensors.torch
import s3transfer.manager
import torch
import zstandard
from safetensors.torch import load_file, save_file

from modelexpress_rl import (
    ObjectStorageGeneratorConfig,
    ObjectStorageSource,
    ObjectStorageType,
    WeightPayloadFormat,
    WeightVersion,
    WeightVersionState,
)
from modelexpress_rl.inference.adapter import GeneratorTransferInputs
from modelexpress_rl.inference import checkpoint_store as checkpoint_store_module
from modelexpress_rl.inference import receiver as receiver_module
from modelexpress_rl.inference.methods import CanonicalDeltaUpdateMethod
import modelexpress_rl.inference.methods.canonical_delta as canonical_delta_module
from modelexpress_rl.inference.plan import (
    ObjectStorageUpdateSource,
)
from modelexpress_rl.s3 import ImmutableS3Conflict, S3Client
from modelexpress_rl.utils import checksum_factory, compress_delta, compute_delta


def _checksum(value):
    checksum = checksum_factory("adler32")
    checksum.update(value)
    return checksum.hexdigest()


class _MemoryS3:
    def __init__(self, objects):
        self.objects = objects
        self.calls = []
        self.fail_key_once = None

    def get(self, uri):
        self.calls.append(uri)
        if uri == self.fail_key_once:
            self.fail_key_once = None
            raise OSError("injected shard download failure")
        return self.objects[uri]

    def size(self, uri):
        return len(self.objects[uri])

    def close(self):
        pass


class _TransferFuture:
    def result(self):
        pass


class _TransferManager:
    def __init__(self, client, config):
        self.client = client
        self.config = config

    def download(self, bucket, key, target):
        response = self.client.get_object(Bucket=bucket, Key=key)
        body = response["Body"]
        try:
            target.write(body.read())
        finally:
            body.close()
        return _TransferFuture()

    def shutdown(self):
        pass


class _Body:
    def __init__(self, data):
        self.data = data

    def read(self):
        return self.data

    def close(self):
        pass


class _S3Error(Exception):
    def __init__(self, code):
        self.response = {"Error": {"Code": code}}


class _S3Backend:
    def __init__(
        self,
        data=b"",
        *,
        put_error=None,
        complete_error=None,
        barrier=None,
    ):
        self.data = data
        self.put_error = put_error
        self.complete_error = complete_error
        self.barrier = barrier
        self.parts = {}
        self.put_request = None
        self.get_request = None
        self.get_requests = []
        self.head_request = None
        self.completed = None
        self.aborted = False

    def put_object(self, **request):
        self.put_request = request
        if self.put_error is not None:
            raise self.put_error

    def get_object(self, **request):
        self.get_request = request
        self.get_requests.append(request)
        return {"Body": _Body(self.data)}

    def head_object(self, **request):
        self.head_request = request
        return {"ContentLength": len(self.data)}

    def create_multipart_upload(self, **request):
        assert request == {"Bucket": "weights", "Key": "delta"}
        return {"UploadId": "upload-1"}

    def upload_part(self, **request):
        if self.barrier is not None and request["PartNumber"] <= 2:
            self.barrier.wait(timeout=2)
        self.parts[request["PartNumber"]] = request["Body"]
        return {"ETag": f"etag-{request['PartNumber']}"}

    def complete_multipart_upload(self, **request):
        if self.complete_error is not None:
            raise self.complete_error
        self.completed = request

    def abort_multipart_upload(self, **_request):
        self.aborted = True

    def close(self):
        pass


@pytest.fixture(autouse=True)
def _transfer_manager(monkeypatch):
    monkeypatch.setattr(s3transfer.manager, "TransferManager", _TransferManager)


class _Adapter:
    def __init__(self, **kwargs):
        self.installed = []
        self._method = CanonicalDeltaUpdateMethod(**kwargs)
        self._checkpoint = self._method._checkpoint
        self._active = None

    def stage_weight(self, inputs):
        version = WeightVersion(
            version_id=inputs.version_id,
            model_name="test/model",
            payload_format=inputs.payload_format,
            base_version_id=inputs.base_version_id,
            object_storage=inputs.object_storage,
            expected_source_slots=(),
            layout_signature=inputs.layout_signature,
            state=WeightVersionState.READY,
            created_at_unix_ms=0,
        )
        self._active = self._method.prepare(
            version=version,
            source=ObjectStorageUpdateSource(
                storage=inputs.object_storage,
                payload_format=inputs.payload_format,
            ),
        )
        return self._active.checkpoint

    def stage_chain(self, inputs):
        chain = []
        for item in inputs:
            version = WeightVersion(
                version_id=item.version_id,
                model_name="test/model",
                payload_format=item.payload_format,
                base_version_id=item.base_version_id,
                object_storage=item.object_storage,
                expected_source_slots=(),
                layout_signature=item.layout_signature,
                state=WeightVersionState.READY,
                created_at_unix_ms=0,
            )
            chain.append(
                (
                    version,
                    ObjectStorageUpdateSource(
                        storage=item.object_storage,
                        payload_format=item.payload_format,
                    ),
                )
            )
        self._active = self._method.prepare_chain(tuple(chain))
        return self._active.checkpoint

    def apply_weight(self, staged):
        assert self._active is not None and self._active.checkpoint is staged
        with self._method.installation_context(self._active):
            self.installed.append(staged.path)
        return {"perf/mx_receive_install_time": 0.0}

    def release_staged_weight(self, staged):
        assert self._active is not None and self._active.checkpoint is staged
        self._method.release(self._active)
        self._active = None

    def close(self):
        self._method.close()


def test_s3_read_parses_uri():
    data = b"canonical-root"
    backend = _S3Backend(data)
    s3 = object.__new__(S3Client)
    s3._client = backend
    s3._download_manager = _TransferManager(backend, None)
    assert s3.get("s3://weights/root.json") == data
    assert backend.get_request == {
        "Bucket": "weights",
        "Key": "root.json",
    }


def test_s3_size_reads_object_content_length():
    backend = _S3Backend(b"canonical-root")
    s3 = object.__new__(S3Client)
    s3._client = backend

    assert s3.size("s3://weights/root.json") == len(b"canonical-root")
    assert backend.head_request == {
        "Bucket": "weights",
        "Key": "root.json",
    }


@pytest.mark.parametrize(
    "uri",
    [
        "s3://weights//root.json",
        "s3://weights/root.json?query",
        "s3://weights/root.json#fragment",
    ],
)
def test_s3_rejects_noncanonical_uri(uri):
    s3 = object.__new__(S3Client)
    with pytest.raises(ValueError, match="invalid S3 URI"):
        s3.get(uri)


def test_s3_write_accepts_only_an_identical_immutable_retry():
    backend = _S3Backend(b"same", put_error=_S3Error("PreconditionFailed"))
    s3 = object.__new__(S3Client)
    s3._client = backend
    s3._multipart_threshold_bytes = 100
    s3._download_manager = _TransferManager(backend, None)
    s3.put(uri="s3://weights/root.json", data=b"same")
    with pytest.raises(ImmutableS3Conflict):
        s3.put(uri="s3://weights/root.json", data=b"different")
    assert backend.put_request["IfNoneMatch"] == "*"


def test_s3_client_uses_transfer_connection_and_retry_settings(monkeypatch):
    import boto3

    captured = {}
    backend = _S3Backend()

    def client(_service, **kwargs):
        captured.update(kwargs)
        return backend

    monkeypatch.setenv("MX_S3_UPLOAD_WORKERS", "2")
    monkeypatch.setenv("MX_S3_DOWNLOAD_WORKERS", "3")
    monkeypatch.setenv("MX_S3_DOWNLOAD_RANGE_THRESHOLD_BYTES", "4096")
    monkeypatch.setenv("MX_S3_DOWNLOAD_RANGE_BYTES", "1024")
    monkeypatch.setenv("MX_S3_DOWNLOAD_IO_CHUNK_BYTES", "512")
    monkeypatch.setenv("MX_S3_DOWNLOAD_MAX_IN_MEMORY_CHUNKS", "7")
    monkeypatch.setenv("MX_S3_MAX_POOL_CONNECTIONS", "17")
    monkeypatch.setenv("MX_S3_MAX_ATTEMPTS", "6")
    monkeypatch.setenv("MX_S3_TCP_KEEPALIVE", "false")
    monkeypatch.setattr(boto3, "client", client)

    s3 = S3Client()
    try:
        config = captured["config"]
        assert config.max_pool_connections == 17
        assert config.retries == {"total_max_attempts": 6, "mode": "standard"}
        assert config.tcp_keepalive is False
        assert s3._upload_pool._max_workers == 2
        transfer = s3._download_manager.config
        assert transfer.multipart_threshold == 4096
        assert transfer.multipart_chunksize == 1024
        assert transfer.max_request_concurrency == 3
        assert transfer.io_chunksize == 512
        assert transfer.max_io_queue_size == 7
        assert transfer.max_in_memory_download_chunks == 7
        assert transfer.num_download_attempts == 6
    finally:
        s3.close()


def test_s3_multipart_uploads_parts_concurrently_and_completes_immutably(
    monkeypatch,
):
    import boto3

    part_bytes = 5 * 1024**2
    barrier = threading.Barrier(2)
    backend = _S3Backend(barrier=barrier)
    monkeypatch.setenv("MX_S3_MULTIPART_THRESHOLD_BYTES", "1")
    monkeypatch.setenv("MX_S3_UPLOAD_PART_BYTES", str(part_bytes))
    monkeypatch.setenv("MX_S3_UPLOAD_WORKERS", "2")
    monkeypatch.setattr(boto3, "client", lambda *_args, **_kwargs: backend)
    data = b"a" * part_bytes + b"b" * part_bytes + b"c"

    s3 = S3Client()
    try:
        s3.put(uri="s3://weights/delta", data=data)
    finally:
        s3.close()

    assert backend.parts == {
        1: data[:part_bytes],
        2: data[part_bytes : 2 * part_bytes],
        3: b"c",
    }
    assert backend.completed["IfNoneMatch"] == "*"
    assert backend.completed["MultipartUpload"] == {
        "Parts": [
            {"ETag": "etag-1", "PartNumber": 1},
            {"ETag": "etag-2", "PartNumber": 2},
            {"ETag": "etag-3", "PartNumber": 3},
        ]
    }
    assert backend.aborted is False


def test_s3_multipart_aborts_and_propagates_conditional_conflict(monkeypatch):
    import boto3

    error = _S3Error("ConditionalRequestConflict")
    backend = _S3Backend(complete_error=error)
    monkeypatch.setenv("MX_S3_MULTIPART_THRESHOLD_BYTES", "1")
    monkeypatch.setenv("MX_S3_UPLOAD_PART_BYTES", str(5 * 1024**2))
    monkeypatch.setattr(boto3, "client", lambda *_args, **_kwargs: backend)

    s3 = S3Client()
    try:
        with pytest.raises(_S3Error) as raised:
            s3.put(uri="s3://weights/delta", data=b"x" * (5 * 1024**2))
    finally:
        s3.close()

    assert raised.value is error
    assert backend.aborted is True


def test_s3_multipart_precondition_failure_accepts_identical_retry(monkeypatch):
    import boto3

    data = b"x" * (5 * 1024**2)
    backend = _S3Backend(
        data,
        complete_error=_S3Error("PreconditionFailed"),
    )
    monkeypatch.setenv("MX_S3_MULTIPART_THRESHOLD_BYTES", "1")
    monkeypatch.setenv("MX_S3_UPLOAD_PART_BYTES", str(5 * 1024**2))
    monkeypatch.setattr(boto3, "client", lambda *_args, **_kwargs: backend)

    s3 = S3Client()
    try:
        s3.put(uri="s3://weights/delta", data=data)
    finally:
        s3.close()

    assert backend.aborted is True


def _artifact(
    base,
    target,
    *,
    checksum=None,
    version="target-a",
    version_label=1,
    base_version="base-a",
):
    delta, _ = compute_delta(target, base)
    assert delta is not None
    shard = safetensors.numpy.save(
        {"weight": compress_delta(delta)},
        metadata={"weight": checksum or _checksum(target)},
    )
    root = json.dumps(
        {
            "metadata": {
                "version": version,
                "base_version": base_version,
                "delta_encoding": "xor",
                "compression_format": "zstd",
                "checksum_format": "adler32",
            },
            "weight_map": {"weight": "model-00000-of-00001.safetensors"},
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    prefix = f"s3://weights/test/v{version_label}"
    return {
        f"{prefix}/model.safetensors.index.json": root,
        f"{prefix}/model-00000-of-00001.safetensors": shard,
    }


def _inputs(
    _root,
    *,
    base_version="base-a",
    version="target-a",
    version_label=1,
    uri=None,
):
    if uri is None:
        uri = f"s3://weights/test/v{version_label}/model.safetensors.index.json"
    return GeneratorTransferInputs(
        version_id=version,
        base_version_id=base_version,
        layout_signature="",
        payload_format=WeightPayloadFormat.XOR_DELTA,
        sources=(),
        object_storage=ObjectStorageSource(
            storage_type=ObjectStorageType.S3,
            uri=uri,
        ),
    )


def _save_full_tensors(tensors):
    checksums = {
        name: _checksum(
            tensor.detach()
            .cpu()
            .contiguous()
            .reshape(-1)
            .view(torch.uint8)
            .numpy()
        )
        for name, tensor in tensors.items()
    }
    return safetensors.torch.save(tensors, metadata=checksums)


def _full_artifact(tensor, *, version_label=2):
    shard_name = "model-00001-of-00001.safetensors"
    shard = _save_full_tensors({"weight": tensor})
    root = json.dumps(
        {
            "metadata": {
                "total_size": tensor.numel() * tensor.element_size(),
                "checksum_format": "adler32",
            },
            "weight_map": {"weight": shard_name},
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    prefix = f"s3://weights/test/v{version_label}"
    return {
        f"{prefix}/model.safetensors.index.json": root,
        f"{prefix}/{shard_name}": shard,
    }


def _full_inputs(*, version="full-a", version_label=2):
    return replace(
        _inputs(
            None,
            version=version,
            version_label=version_label,
        ),
        base_version_id=None,
        payload_format=WeightPayloadFormat.FULL_HF_CHECKPOINT,
    )


@pytest.mark.parametrize(
    "filename",
    ["/tmp/shard", "../shard", "foo/bar", "foo/bar/", ".", ".."],
)
def test_canonical_s3_rejects_unsafe_delta_shard_filenames(
    monkeypatch, tmp_path, filename
):
    objects = _artifact(
        torch.tensor([1.0, 2.0]).view(torch.uint8).numpy(),
        torch.tensor([3.0, 4.0]).view(torch.uint8).numpy(),
    )
    root_uri = "s3://weights/test/v1/model.safetensors.index.json"
    manifest = json.loads(objects[root_uri])
    manifest["weight_map"] = {"weight": filename}
    objects[root_uri] = json.dumps(manifest).encode()
    adapter, storage = _build(monkeypatch, tmp_path, objects)

    with pytest.raises(RuntimeError, match="The index manifest has invalid"):
        adapter.stage_weight(_inputs(objects[root_uri]))

    assert storage.calls == [root_uri]
    assert not adapter._checkpoint.store.delta_path("target-a").exists()
    adapter.close()


def _build(
    monkeypatch,
    tmp_path,
    objects,
    launch_tensors=None,
    checkpoint_max_size_gb=500,
):
    launch = tmp_path / "launch"
    launch.mkdir(exist_ok=True)
    save_file(
        launch_tensors or {"weight": torch.tensor([1.0, 2.0])},
        launch / "model.safetensors",
    )
    storage = _MemoryS3(objects)
    monkeypatch.setattr(canonical_delta_module, "S3Client", lambda **_kwargs: storage)
    adapter = _Adapter(
        model_name="test/model",
        config=ObjectStorageGeneratorConfig(
            storage_type=ObjectStorageType.S3,
            initial_base_version_id="base-a",
            seed_checkpoint_path=launch,
            refit_checkpoint_dir=tmp_path / "cache",
            refit_checkpoint_max_size_gb=checkpoint_max_size_gb,
        ),
    )
    return adapter, storage


@pytest.mark.parametrize("max_size_gb", [0, -1])
def test_object_storage_generator_config_rejects_nonpositive_cache_quota(
    tmp_path, max_size_gb
):
    with pytest.raises(ValueError, match="must be positive"):
        ObjectStorageGeneratorConfig(
            storage_type=ObjectStorageType.S3,
            initial_base_version_id="base-a",
            seed_checkpoint_path=tmp_path / "launch",
            refit_checkpoint_dir=tmp_path / "cache",
            refit_checkpoint_max_size_gb=max_size_gb,
        )


def test_object_storage_generator_config_converts_cache_quota_to_bytes(
    monkeypatch, tmp_path
):
    adapter, _storage = _build(
        monkeypatch,
        tmp_path,
        {},
        checkpoint_max_size_gb=2,
    )

    assert adapter._checkpoint.store.max_size_bytes == 2_000_000_000
    adapter.close()


def test_object_storage_generator_config_defaults_cache_quota(tmp_path):
    config = ObjectStorageGeneratorConfig(
        storage_type=ObjectStorageType.S3,
        initial_base_version_id="base-a",
        seed_checkpoint_path=tmp_path / "launch",
        refit_checkpoint_dir=tmp_path / "cache",
    )

    assert config.refit_checkpoint_max_size_gb == 500


def test_reset_initial_checkpoint_transitions_updating_to_ready(monkeypatch, tmp_path):
    events = []
    write_state = checkpoint_store_module.LocalCheckpointStore.write_state
    copy2 = receiver_module.shutil.copy2

    def track_state(self, *, status, version, checkpoint_paths, source=None):
        events.append(status.value)
        write_state(
            self,
            status=status,
            version=version,
            checkpoint_paths=checkpoint_paths,
            source=source,
        )

    def track_copy(source, target):
        events.append("COPY")
        return copy2(source, target)

    monkeypatch.setattr(
        checkpoint_store_module.LocalCheckpointStore,
        "write_state",
        track_state,
    )
    monkeypatch.setattr(receiver_module.shutil, "copy2", track_copy)
    adapter, _storage = _build(monkeypatch, tmp_path, {})

    assert events == ["UPDATING", "COPY", "READY"]
    state = json.loads(adapter._checkpoint.store.state_path.read_text())
    files = state.pop("files")
    assert files["model.safetensors"][0] > 0
    assert files["model.safetensors"][1] > 0
    assert state == {"status": "READY", "version": "base-a"}
    adapter.close()


def test_canonical_s3_rejects_checkpoint_that_exceeds_cache_quota(
    monkeypatch, tmp_path
):
    objects = _full_artifact(torch.tensor([7.0, 8.0]))
    adapter, _storage = _build(monkeypatch, tmp_path, objects)
    store = adapter._checkpoint.store
    store.max_size_bytes = store.cache_size_bytes()

    with pytest.raises(
        checkpoint_store_module.CheckpointCacheCapacityError,
        match="checkpoint cache quota",
    ):
        adapter.stage_weight(_full_inputs())

    assert store.active_version() == "base-a"
    assert store.full_path("base-a").exists()
    assert not store.full_path("full-a").exists()

    state = store.state()
    assert state is not None
    assert state["status"] == "READY"
    assert state["version"] == "base-a"

    store.max_size_bytes = None
    staged = adapter.stage_weight(_full_inputs())
    assert staged.path == store.full_path("full-a")
    adapter.release_staged_weight(staged)
    adapter.close()


def test_canonical_s3_reseeds_a_modified_ready_checkpoint(monkeypatch, tmp_path):
    first, _storage = _build(monkeypatch, tmp_path, {})
    checkpoint_path = first._checkpoint.local_checkpoint / "model.safetensors"
    save_file({"weight": torch.tensor([9.0, 10.0])}, checkpoint_path)
    first.close()

    second = _Adapter(
        model_name="test/model",
        config=ObjectStorageGeneratorConfig(
            storage_type=ObjectStorageType.S3,
            initial_base_version_id="base-a",
            seed_checkpoint_path=tmp_path / "launch",
            refit_checkpoint_dir=tmp_path / "cache",
        ),
    )

    assert torch.equal(
        load_file(second._checkpoint.local_checkpoint / "model.safetensors")["weight"],
        torch.tensor([1.0, 2.0]),
    )
    second.close()


def test_canonical_s3_rejects_non_s3_source_before_storage_access(
    monkeypatch,
    tmp_path,
):
    adapter, storage = _build(monkeypatch, tmp_path, {})
    inputs = replace(
        _inputs(None),
        object_storage=ObjectStorageSource(
            storage_type=ObjectStorageType.GCS,
            uri="gs://weights/test/v1/model.safetensors.index.json",
        ),
    )

    with pytest.raises(ValueError, match="requires S3 object storage"):
        adapter.stage_weight(inputs)

    assert storage.calls == []


def test_canonical_s3_prepares_then_installs_one_global_index(monkeypatch, tmp_path):
    base = torch.tensor([1.0, 2.0]).view(torch.uint8).numpy()
    target_tensor = torch.tensor([3.0, 4.0])
    target = target_tensor.view(torch.uint8).numpy()
    objects = _artifact(base, target)
    root = objects["s3://weights/test/v1/model.safetensors.index.json"]
    adapter, storage = _build(monkeypatch, tmp_path, objects)

    staged = adapter.stage_weight(_inputs(root))

    state = json.loads(adapter._checkpoint.store.state_path.read_text())
    files = state.pop("files")
    assert files["model.safetensors"][0] > 0
    assert files["model.safetensors"][1] > 0
    assert state == {
        "source": {"uri": "s3://weights/test/v1/model.safetensors.index.json"},
        "status": "READY",
        "version": "target-a",
    }
    assert adapter.installed == []
    assert torch.equal(
        load_file(staged.path / "model.safetensors")["weight"], target_tensor
    )
    assert storage.calls == [
        "s3://weights/test/v1/model.safetensors.index.json",
        "s3://weights/test/v1/model-00000-of-00001.safetensors",
    ]
    assert staged.metrics["perf/mx_receive_delta_download"] >= 0
    assert adapter.apply_weight(staged)["perf/mx_receive_install_time"] >= 0
    assert adapter.installed == [staged.path]
    adapter.release_staged_weight(staged)
    adapter.close()


def test_canonical_s3_replays_multiple_deltas_before_one_install(monkeypatch, tmp_path):
    base = torch.tensor([1.0, 2.0])
    first = torch.tensor([3.0, 4.0])
    target = torch.tensor([5.0, 6.0])
    objects = _artifact(base.view(torch.uint8).numpy(), first.view(torch.uint8).numpy())
    objects.update(
        _artifact(
            first.view(torch.uint8).numpy(),
            target.view(torch.uint8).numpy(),
            version="target-b",
            version_label=2,
            base_version="target-a",
        )
    )
    parse_calls = []
    parse_index_manifest = receiver_module._parse_index_manifest

    def track_parse(data, *, is_delta, version=None):
        if is_delta:
            parse_calls.append(version.version_id)
        return parse_index_manifest(data, is_delta=is_delta, version=version)

    monkeypatch.setattr(receiver_module, "_parse_index_manifest", track_parse)
    adapter, _ = _build(monkeypatch, tmp_path, objects)

    staged = adapter.stage_chain(
        (
            _inputs(None),
            _inputs(
                None,
                version="target-b",
                version_label=2,
                base_version="target-a",
            ),
        )
    )

    assert torch.equal(load_file(staged.path / "model.safetensors")["weight"], target)
    assert json.loads(adapter._checkpoint.store.state_path.read_text())["version"] == (
        "target-b"
    )
    assert parse_calls == ["target-a", "target-b"]
    assert adapter.installed == []
    adapter.apply_weight(staged)
    assert adapter.installed == [staged.path]
    adapter.release_staged_weight(staged)
    adapter.close()


def test_canonical_s3_validates_all_replay_manifests_before_mutation(
    monkeypatch, tmp_path
):
    base_tensor = torch.tensor([1.0, 2.0])
    first = torch.tensor([3.0, 4.0])
    target = torch.tensor([5.0, 6.0])
    objects = _artifact(
        base_tensor.view(torch.uint8).numpy(), first.view(torch.uint8).numpy()
    )
    objects.update(
        _artifact(
            first.view(torch.uint8).numpy(),
            target.view(torch.uint8).numpy(),
            version="target-b",
            version_label=2,
            base_version="target-a",
        )
    )
    second_root = "s3://weights/test/v2/model.safetensors.index.json"
    manifest = json.loads(objects[second_root])
    manifest["metadata"]["base_version"] = "wrong-base"
    objects[second_root] = json.dumps(manifest).encode()
    adapter, _ = _build(monkeypatch, tmp_path, objects)

    with pytest.raises(RuntimeError, match=r"base_version.*target-b"):
        adapter.stage_chain(
            (
                _inputs(None),
                _inputs(
                    None,
                    version="target-b",
                    version_label=2,
                    base_version="target-a",
                ),
            )
        )

    state = json.loads(adapter._checkpoint.store.state_path.read_text())
    state.pop("files")
    assert state == {"status": "READY", "version": "base-a"}
    assert torch.equal(
        load_file(adapter._checkpoint.local_checkpoint / "model.safetensors")["weight"],
        base_tensor,
    )


def test_canonical_s3_keeps_checkpoint_ready_after_install_failure(
    monkeypatch, tmp_path
):
    base = torch.tensor([1.0, 2.0]).view(torch.uint8).numpy()
    target = torch.tensor([3.0, 4.0]).view(torch.uint8).numpy()
    objects = _artifact(base, target)
    adapter, _ = _build(monkeypatch, tmp_path, objects)
    staged = adapter.stage_weight(_inputs(None))

    with pytest.raises(RuntimeError, match="injected install failure"):
        with adapter._method.installation_context(adapter._active):
            raise RuntimeError("injected install failure")
    adapter._method.installation_failed(adapter._active)

    state = json.loads(adapter._checkpoint.store.state_path.read_text())
    assert state["status"] == "READY"
    assert state["version"] == "target-a"
    assert adapter._checkpoint.store.active_version() == "base-a"
    adapter.release_staged_weight(staged)
    cached = adapter.stage_weight(_inputs(None))
    assert cached.path == staged.path
    adapter.release_staged_weight(cached)
    adapter.close()


@pytest.mark.parametrize("target_version", ["v3", "v4", "v5"])
def test_canonical_s3_recovers_from_failed_v2_install_with_disjoint_target(
    monkeypatch, tmp_path, target_version
):
    # The configured base-a checkpoint represents v1 in this lineage.
    tensors = {
        "v1": torch.tensor([1.0, 2.0]),
        "v2": torch.tensor([3.0, 4.0]),
        "v3": torch.tensor([5.0, 6.0]),
        "v4": torch.tensor([7.0, 8.0]),
        "v5": torch.tensor([9.0, 10.0]),
    }
    objects = {}
    for version, base in (("v2", "v1"), ("v3", "v2"), ("v5", "v4")):
        objects.update(
            _artifact(
                tensors[base].view(torch.uint8).numpy(),
                tensors[version].view(torch.uint8).numpy(),
                version=version,
                version_label=int(version[1:]),
                base_version="base-a" if base == "v1" else base,
            )
        )
    objects.update(_full_artifact(tensors["v4"], version_label=4))
    adapter, _storage = _build(monkeypatch, tmp_path, objects)

    def delta(version, base):
        return _inputs(
            None,
            version=version,
            version_label=int(version[1:]),
            base_version="base-a" if base == "v1" else base,
        )

    failed = adapter.stage_weight(delta("v2", "v1"))
    with pytest.raises(RuntimeError, match="injected install failure"):
        with adapter._method.installation_context(adapter._active):
            raise RuntimeError("injected install failure")
    adapter._method.installation_failed(adapter._active)
    adapter.release_staged_weight(failed)
    assert adapter._checkpoint.store.state()["version"] == "v2"
    assert adapter._checkpoint.store.active_version() == "base-a"

    replays = {
        "v3": (delta("v2", "v1"), delta("v3", "v2")),
        "v4": (_full_inputs(version="v4", version_label=4),),
        "v5": (
            _full_inputs(version="v4", version_label=4),
            delta("v5", "v4"),
        ),
    }
    replay = replays[target_version]
    recovered = (
        adapter.stage_weight(replay[0])
        if len(replay) == 1
        else adapter.stage_chain(replay)
    )
    shard = recovered.path / "model.safetensors"
    if not shard.exists():
        shard = recovered.path / "model-00001-of-00001.safetensors"
    assert torch.equal(
        load_file(shard)["weight"],
        tensors[target_version],
    )
    adapter.apply_weight(recovered)
    assert adapter._checkpoint.store.active_version() == target_version
    adapter.release_staged_weight(recovered)
    adapter.close()


def test_canonical_s3_reinstalls_active_checkpoint_after_install_failure(
    monkeypatch, tmp_path
):
    objects = _full_artifact(torch.tensor([7.0, 8.0]))
    objects.update(
        _full_artifact(
            torch.tensor([9.0, 10.0]),
            version_label=3,
        )
    )
    adapter, _storage = _build(monkeypatch, tmp_path, objects)

    active = adapter.stage_weight(_full_inputs())
    adapter.apply_weight(active)
    adapter.release_staged_weight(active)

    failed = adapter.stage_weight(_full_inputs(version="full-b", version_label=3))
    adapter._method.installation_failed(adapter._active)
    adapter.release_staged_weight(failed)

    recovery = adapter.stage_weight(_full_inputs())
    assert recovery.path == adapter._checkpoint.store.full_path("full-a")
    adapter.apply_weight(recovery)
    adapter.release_staged_weight(recovery)
    assert adapter._checkpoint.store.state()["status"] == "READY"
    assert adapter._checkpoint.store.active_version() == "full-a"
    adapter.close()


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("version", "version does not match revision"),
        ("base_version", "base_version does not match revision"),
        ("delta_encoding", "delta_encoding does not match revision"),
        ("compression_format", "unsupported compression format"),
        ("checksum_format", "checksum_format does not match revision"),
    ],
)
def test_canonical_s3_requires_delta_index_metadata_fields(
    monkeypatch,
    tmp_path,
    field,
    message,
):
    base = torch.tensor([1.0, 2.0]).view(torch.uint8).numpy()
    target = torch.tensor([3.0, 4.0]).view(torch.uint8).numpy()
    objects = _artifact(base, target)
    root_uri = "s3://weights/test/v1/model.safetensors.index.json"
    manifest = json.loads(objects[root_uri])
    del manifest["metadata"][field]
    objects[root_uri] = json.dumps(manifest).encode()
    adapter, storage = _build(monkeypatch, tmp_path, objects)

    with pytest.raises(RuntimeError, match=message):
        adapter.stage_weight(_inputs(objects[root_uri]))

    assert storage.calls == [root_uri]
    state = adapter._checkpoint.store.state()
    assert state is not None
    assert state["status"] == "READY"
    assert state["version"] == "base-a"
    adapter.close()


@pytest.mark.parametrize(
    ("field", "value"),
    [("delta_encoding", "replace"), ("checksum_format", "crc32")],
)
def test_canonical_s3_rejects_mismatched_delta_formats(
    monkeypatch,
    tmp_path,
    field,
    value,
):
    base = torch.tensor([1.0, 2.0]).view(torch.uint8).numpy()
    target = torch.tensor([3.0, 4.0]).view(torch.uint8).numpy()
    objects = _artifact(base, target)
    root_uri = "s3://weights/test/v1/model.safetensors.index.json"
    manifest = json.loads(objects[root_uri])
    manifest["metadata"][field] = value
    objects[root_uri] = json.dumps(manifest).encode()
    adapter, storage = _build(monkeypatch, tmp_path, objects)

    with pytest.raises(RuntimeError, match=rf"{field} does not match revision"):
        adapter.stage_weight(_inputs(objects[root_uri]))

    assert storage.calls == [root_uri]
    state = adapter._checkpoint.store.state()
    assert state is not None
    assert state["status"] == "READY"
    assert state["version"] == "base-a"
    adapter.close()


def test_canonical_s3_full_checkpoint_resets_base_for_next_delta(monkeypatch, tmp_path):
    full_tensor = torch.tensor([7.0, 8.0])
    objects = _full_artifact(full_tensor)
    adapter, storage = _build(monkeypatch, tmp_path, objects)

    full = adapter.stage_weight(_full_inputs())
    assert torch.equal(
        load_file(full.path / "model-00001-of-00001.safetensors")["weight"],
        full_tensor,
    )
    assert adapter.apply_weight(full)["perf/mx_receive_install_time"] >= 0
    adapter.release_staged_weight(full)

    next_tensor = torch.tensor([9.0, 10.0])
    objects.update(
        _artifact(
            full_tensor.view(torch.uint8).numpy(),
            next_tensor.view(torch.uint8).numpy(),
            version="delta-after-full",
            version_label=3,
            base_version="full-a",
        )
    )
    delta = adapter.stage_weight(
        _inputs(
            None,
            base_version="full-a",
            version="delta-after-full",
            version_label=3,
        )
    )
    assert torch.equal(
        load_file(delta.path / "model-00001-of-00001.safetensors")["weight"],
        next_tensor,
    )
    assert storage.calls[:2] == [
        "s3://weights/test/v2/model.safetensors.index.json",
        "s3://weights/test/v2/model-00001-of-00001.safetensors",
    ]
    adapter.release_staged_weight(delta)
    adapter.close()


def test_canonical_s3_caches_immutable_lineage_and_activates_after_install(
    monkeypatch, tmp_path
):
    full_tensor = torch.tensor([7.0, 8.0])
    objects = _full_artifact(full_tensor)
    adapter, _storage = _build(monkeypatch, tmp_path, objects)
    cache = adapter._checkpoint.store.cache

    assert json.loads((cache / "active.json").read_text()) == {"version": "base-a"}
    assert (cache / "full" / "base-a" / "model.safetensors").is_file()
    assert json.loads((cache / "chains" / "base-a.json").read_text()) == {
        "deltas": [],
        "full_version": "base-a",
        "version": "base-a",
    }

    full = adapter.stage_weight(_full_inputs())
    assert full.path == cache / "full" / "full-a"
    assert (full.path / "model.safetensors.index.json").is_file()
    assert json.loads((cache / "active.json").read_text()) == {"version": "base-a"}
    assert json.loads((cache / "chains" / "full-a.json").read_text()) == {
        "deltas": [],
        "full_version": "full-a",
        "version": "full-a",
    }

    adapter.apply_weight(full)
    assert json.loads((cache / "active.json").read_text()) == {"version": "full-a"}
    adapter.release_staged_weight(full)

    next_tensor = torch.tensor([9.0, 10.0])
    objects.update(
        _artifact(
            full_tensor.view(torch.uint8).numpy(),
            next_tensor.view(torch.uint8).numpy(),
            version="delta-after-full",
            version_label=3,
            base_version="full-a",
        )
    )
    delta = adapter.stage_weight(
        _inputs(
            None,
            base_version="full-a",
            version="delta-after-full",
            version_label=3,
        )
    )

    assert delta.path == cache / "materialized" / "delta-after-full"
    assert (
        cache
        / "deltas"
        / "delta-after-full"
        / "model.safetensors.index.json"
    ).is_file()
    assert json.loads(
        (cache / "chains" / "delta-after-full.json").read_text()
    ) == {
        "deltas": ["delta-after-full"],
        "full_version": "full-a",
        "version": "delta-after-full",
    }
    assert json.loads((cache / "active.json").read_text()) == {"version": "full-a"}
    assert torch.equal(
        load_file(delta.path / "model-00001-of-00001.safetensors")["weight"],
        next_tensor,
    )
    assert torch.equal(
        load_file(full.path / "model-00001-of-00001.safetensors")["weight"],
        full_tensor,
    )

    adapter.apply_weight(delta)
    assert json.loads((cache / "active.json").read_text()) == {
        "version": "delta-after-full"
    }
    adapter.release_staged_weight(delta)
    adapter.close()


def test_canonical_s3_failed_install_keeps_previous_active_version(
    monkeypatch, tmp_path
):
    objects = _full_artifact(torch.tensor([7.0, 8.0]))
    adapter, _storage = _build(monkeypatch, tmp_path, objects)
    prepared = adapter.stage_weight(_full_inputs())

    with pytest.raises(RuntimeError, match="injected install failure"):
        with adapter._method.installation_context(adapter._active):
            raise RuntimeError("injected install failure")

    assert json.loads(
        (adapter._checkpoint.store.active_path).read_text()
    ) == {
        "version": "base-a"
    }
    adapter.release_staged_weight(prepared)
    adapter.close()


def test_canonical_s3_applies_one_delta_to_the_active_checkpoint(
    monkeypatch, tmp_path
):
    base = torch.tensor([1.0, 2.0])
    middle = torch.tensor([3.0, 4.0])
    target = torch.tensor([5.0, 6.0])
    objects = _artifact(
        base.view(torch.uint8).numpy(),
        middle.view(torch.uint8).numpy(),
    )
    adapter, _storage = _build(monkeypatch, tmp_path, objects)

    first = adapter.stage_weight(_inputs(None))
    adapter.apply_weight(first)
    adapter.release_staged_weight(first)
    first_materialized = adapter._checkpoint.store.materialized_path("target-a")
    apply_shards = adapter._checkpoint._apply_shards
    replace_directory = adapter._checkpoint.store.replace_directory
    apply_calls = 0
    checkpoint_copy_calls = 0

    def track_apply(shards, *, index_metadata):
        nonlocal apply_calls
        apply_calls += 1
        apply_shards(shards, index_metadata=index_metadata)

    @contextmanager
    def track_replace(target, *, copy_from=None):
        nonlocal checkpoint_copy_calls
        if copy_from is not None:
            checkpoint_copy_calls += 1
        with replace_directory(target, copy_from=copy_from) as temporary:
            yield temporary

    monkeypatch.setattr(adapter._checkpoint, "_apply_shards", track_apply)
    monkeypatch.setattr(adapter._checkpoint.store, "replace_directory", track_replace)
    objects.update(
        _artifact(
            middle.view(torch.uint8).numpy(),
            target.view(torch.uint8).numpy(),
            version="target-b",
            version_label=3,
            base_version="target-a",
        )
    )
    second = adapter.stage_weight(
        _inputs(
            None,
            base_version="target-a",
            version="target-b",
            version_label=3,
        )
    )

    assert json.loads(
        (adapter._checkpoint.store.chain_cache / "target-b.json").read_text()
    ) == {
        "deltas": ["target-a", "target-b"],
        "full_version": "base-a",
        "version": "target-b",
    }
    assert torch.equal(
        load_file(second.path / "model.safetensors")["weight"], target
    )
    assert apply_calls == 1
    assert checkpoint_copy_calls == 0
    assert not first_materialized.exists()
    assert torch.equal(
        load_file(
            adapter._checkpoint.store.full_cache
            / "base-a"
            / "model.safetensors"
        )["weight"],
        base,
    )
    adapter.apply_weight(second)

    assert not first_materialized.exists()
    assert second.path.exists()
    adapter.release_staged_weight(second)
    adapter.close()


def test_canonical_s3_in_place_delta_failure_requires_recovery(
    monkeypatch, tmp_path
):
    base = torch.tensor([1.0, 2.0])
    middle = torch.tensor([3.0, 4.0])
    target = torch.tensor([5.0, 6.0])
    objects = _artifact(
        base.view(torch.uint8).numpy(),
        middle.view(torch.uint8).numpy(),
    )
    adapter, _storage = _build(monkeypatch, tmp_path, objects)
    first = adapter.stage_weight(_inputs(None))
    adapter.apply_weight(first)
    adapter.release_staged_weight(first)
    first_path = adapter._checkpoint.store.materialized_path("target-a")
    second_path = adapter._checkpoint.store.materialized_path("target-b")
    objects.update(
        _artifact(
            middle.view(torch.uint8).numpy(),
            target.view(torch.uint8).numpy(),
            version="target-b",
            version_label=3,
            base_version="target-a",
        )
    )

    def fail_apply(_shards, *, index_metadata):
        raise RuntimeError("injected delta failure")

    monkeypatch.setattr(adapter._checkpoint, "_apply_shards", fail_apply)
    with pytest.raises(RuntimeError, match="injected delta failure"):
        adapter.stage_weight(
            _inputs(
                None,
                base_version="target-a",
                version="target-b",
                version_label=3,
            )
        )

    assert not first_path.exists()
    assert second_path.exists()
    assert adapter._checkpoint.store.active_version() == "target-a"
    assert adapter._checkpoint.store.state()["status"] == "UPDATING"
    adapter.close()


def test_installation_fence_blocks_prepare_until_activation(monkeypatch, tmp_path):
    objects = _full_artifact(torch.tensor([7.0, 8.0]))
    objects.update(
        _full_artifact(
            torch.tensor([9.0, 10.0]),
            version_label=3,
        )
    )
    first, storage = _build(monkeypatch, tmp_path, objects)
    second = _Adapter(
        model_name="test/model",
        config=ObjectStorageGeneratorConfig(
            storage_type=ObjectStorageType.S3,
            initial_base_version_id="base-a",
            seed_checkpoint_path=tmp_path / "launch",
            refit_checkpoint_dir=tmp_path / "cache",
        ),
    )
    first_staged = first.stage_weight(_full_inputs())
    original_locked = first._checkpoint.store.locked
    shared_lock_released = threading.Event()
    allow_activation = threading.Event()

    @contextmanager
    def pause_after_shared_lock(*, shared=False):
        with original_locked(shared=shared):
            yield
        if shared:
            shared_lock_released.set()
            assert allow_activation.wait(timeout=5)

    monkeypatch.setattr(first._checkpoint.store, "locked", pause_after_shared_lock)
    prepare_fence_entered = threading.Event()
    installation_locked = second._checkpoint.store.installation_locked

    @contextmanager
    def track_installation_fence(*, shared=False):
        if not shared:
            prepare_fence_entered.set()
        with installation_locked(shared=shared):
            yield

    monkeypatch.setattr(
        second._checkpoint.store,
        "installation_locked",
        track_installation_fence,
    )
    install_errors = []

    def install():
        try:
            with first._method.installation_context(first._active):
                pass
        except BaseException as error:
            install_errors.append(error)

    install_thread = threading.Thread(target=install)
    install_thread.start()
    assert shared_lock_released.wait(timeout=5)

    download_started = threading.Event()
    prepare_errors = []
    second_staged = []
    get = storage.get

    def track_get(uri):
        if "/v3/" in uri:
            download_started.set()
        return get(uri)

    storage.get = track_get

    def prepare():
        try:
            second_staged.append(
                second.stage_weight(
                    _full_inputs(version="full-b", version_label=3)
                )
            )
        except BaseException as error:
            prepare_errors.append(error)

    prepare_thread = threading.Thread(target=prepare)
    prepare_thread.start()
    assert prepare_fence_entered.wait(timeout=5)
    assert not download_started.wait(timeout=0.1)

    allow_activation.set()
    install_thread.join(timeout=5)
    prepare_thread.join(timeout=5)

    assert not install_thread.is_alive()
    assert not prepare_thread.is_alive()
    assert install_errors == []
    assert prepare_errors == []
    assert download_started.is_set()
    first.release_staged_weight(first_staged)
    second.release_staged_weight(second_staged[0])
    first.close()
    second.close()


def test_canonical_s3_reuses_an_immutable_full_artifact_after_restart(
    monkeypatch, tmp_path
):
    objects = _full_artifact(torch.tensor([7.0, 8.0]))
    first, storage = _build(monkeypatch, tmp_path, objects)
    staged = first.stage_weight(_full_inputs())
    first.apply_weight(staged)
    first.close()

    storage.calls.clear()
    restarted = _Adapter(
        model_name="test/model",
        config=ObjectStorageGeneratorConfig(
            storage_type=ObjectStorageType.S3,
            initial_base_version_id="base-a",
            seed_checkpoint_path=tmp_path / "launch",
            refit_checkpoint_dir=tmp_path / "cache",
        ),
    )
    reused = restarted.stage_weight(_full_inputs())

    assert reused.path == restarted._checkpoint.store.full_cache / "full-a"
    assert storage.calls == [
        "s3://weights/test/v2/model.safetensors.index.json"
    ]
    restarted.release_staged_weight(reused)
    restarted.close()


def test_canonical_s3_rejects_a_modified_full_artifact_after_restart(
    monkeypatch, tmp_path
):
    objects = _full_artifact(torch.tensor([7.0, 8.0]))
    first, _storage = _build(monkeypatch, tmp_path, objects)
    staged = first.stage_weight(_full_inputs())
    first.apply_weight(staged)
    first.close()
    save_file(
        {"weight": torch.tensor([9.0, 10.0])},
        staged.path / "model-00001-of-00001.safetensors",
    )

    restarted = _Adapter(
        model_name="test/model",
        config=ObjectStorageGeneratorConfig(
            storage_type=ObjectStorageType.S3,
            initial_base_version_id="base-a",
            seed_checkpoint_path=tmp_path / "launch",
            refit_checkpoint_dir=tmp_path / "cache",
        ),
    )

    with pytest.raises(RuntimeError, match="cached checkpoint artifact changed"):
        restarted.stage_weight(_full_inputs())
    restarted.close()


def test_canonical_s3_rejects_a_corrupt_cached_delta_during_replay(
    monkeypatch, tmp_path
):
    base = torch.tensor([1.0, 2.0])
    middle = torch.tensor([3.0, 4.0])
    target = torch.tensor([5.0, 6.0])
    objects = _artifact(
        base.view(torch.uint8).numpy(),
        middle.view(torch.uint8).numpy(),
    )
    adapter, _storage = _build(monkeypatch, tmp_path, objects)
    first = adapter.stage_weight(_inputs(None))
    adapter.apply_weight(first)
    adapter.release_staged_weight(first)

    corrupt = _artifact(
        base.view(torch.uint8).numpy(),
        torch.tensor([9.0, 10.0]).view(torch.uint8).numpy(),
    )["s3://weights/test/v1/model-00000-of-00001.safetensors"]
    (
        adapter._checkpoint.store.delta_cache
        / "target-a"
        / "model-00000-of-00001.safetensors"
    ).write_bytes(corrupt)
    objects.update(
        _artifact(
            middle.view(torch.uint8).numpy(),
            target.view(torch.uint8).numpy(),
            version="target-b",
            version_label=3,
            base_version="target-a",
        )
    )

    with pytest.raises(RuntimeError, match="cached checkpoint artifact changed"):
        adapter.stage_weight(
            _inputs(
                None,
                base_version="target-a",
                version="target-b",
                version_label=3,
            )
        )
    adapter.close()


def test_canonical_s3_downloads_full_shards_concurrently(
    monkeypatch, tmp_path
):
    target = {
        "a": torch.tensor([3.0, 4.0]),
        "b": torch.tensor([5.0, 6.0]),
    }
    prefix = "s3://weights/test/v2"
    filenames = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
    objects = {
        f"{prefix}/{filenames[0]}": _save_full_tensors({"a": target["a"]}),
        f"{prefix}/{filenames[1]}": _save_full_tensors({"b": target["b"]}),
    }
    objects[f"{prefix}/model.safetensors.index.json"] = json.dumps(
        {
            "metadata": {"total_size": 16},
            "weight_map": {"a": filenames[0], "b": filenames[1]},
        }
    ).encode()
    adapter, storage = _build(
        monkeypatch,
        tmp_path,
        objects,
        launch_tensors={
            "a": torch.tensor([1.0, 1.0]),
            "b": torch.tensor([2.0, 2.0]),
        },
    )
    monkeypatch.setenv("MX_S3_DOWNLOAD_WORKERS", "2")
    download_barrier = threading.Barrier(2)
    write_barrier = threading.Barrier(2)
    download_threads = set()
    write_threads = set()
    get = storage.get
    write_bytes = Path.write_bytes

    def track_get(uri):
        if uri.endswith(".safetensors"):
            download_threads.add(threading.get_ident())
            download_barrier.wait(timeout=5)
        return get(uri)

    storage.get = track_get

    def track_write(path, data):
        if path.name in filenames:
            write_threads.add(threading.get_ident())
            write_barrier.wait(timeout=5)
        return write_bytes(path, data)

    monkeypatch.setattr(Path, "write_bytes", track_write)

    staged = adapter.stage_weight(_full_inputs())

    assert len(download_threads) == 2
    assert len(write_threads) == 2
    assert torch.equal(load_file(staged.path / filenames[0])["a"], target["a"])
    assert torch.equal(load_file(staged.path / filenames[1])["b"], target["b"])
    assert not (adapter._checkpoint.store.full_cache / "full-a.tmp").exists()
    adapter.release_staged_weight(staged)
    adapter.close()


def test_canonical_s3_streams_scalar_full_tensor(monkeypatch, tmp_path):
    objects = _full_artifact(torch.tensor(2.0))
    adapter, _storage = _build(
        monkeypatch,
        tmp_path,
        objects,
        launch_tensors={"weight": torch.tensor(1.0)},
    )

    staged = adapter.stage_weight(_full_inputs())

    assert torch.equal(
        load_file(staged.path / "model-00001-of-00001.safetensors")["weight"],
        torch.tensor(2.0),
    )
    adapter.release_staged_weight(staged)
    adapter.close()


@pytest.mark.parametrize(
    "metadata",
    [
        None,
        {"format": "pt"},
        {"format": "pt", "creator": "test", "weight": "trained-v2"},
    ],
    ids=["absent", "unrelated", "multiple-with-tensor-name-collision"],
)
def test_canonical_s3_accepts_full_checkpoint_without_checksums(
    monkeypatch,
    tmp_path,
    metadata,
):
    target = torch.tensor([7.0, 8.0])
    objects = _full_artifact(target)
    root_uri = "s3://weights/test/v2/model.safetensors.index.json"
    index = json.loads(objects[root_uri])
    del index["metadata"]["checksum_format"]
    objects[root_uri] = json.dumps(index).encode()
    shard_uri = "s3://weights/test/v2/model-00001-of-00001.safetensors"
    objects[shard_uri] = safetensors.torch.save(
        {"weight": target},
        metadata=metadata,
    )
    adapter, _storage = _build(monkeypatch, tmp_path, objects)

    staged = adapter.stage_weight(_full_inputs())

    assert torch.equal(
        load_file(staged.path / "model-00001-of-00001.safetensors")["weight"],
        target,
    )
    adapter.release_staged_weight(staged)
    adapter.close()


def test_canonical_s3_requires_declared_full_checkpoint_checksums(
    monkeypatch,
    tmp_path,
):
    target = torch.tensor([7.0, 8.0])
    objects = _full_artifact(target)
    shard_uri = "s3://weights/test/v2/model-00001-of-00001.safetensors"
    objects[shard_uri] = safetensors.torch.save(
        {"weight": target},
        metadata={"format": "pt"},
    )
    adapter, _storage = _build(monkeypatch, tmp_path, objects)

    with pytest.raises(RuntimeError, match="missing checksum for tensor 'weight'"):
        adapter.stage_weight(_full_inputs())

    adapter.close()


@pytest.mark.parametrize("checksum_format", [None, "crc32"])
def test_canonical_s3_rejects_unsupported_full_checkpoint_checksum_format(
    monkeypatch,
    tmp_path,
    checksum_format,
):
    objects = _full_artifact(torch.tensor([7.0, 8.0]))
    root_uri = "s3://weights/test/v2/model.safetensors.index.json"
    index = json.loads(objects[root_uri])
    index["metadata"]["checksum_format"] = checksum_format
    objects[root_uri] = json.dumps(index).encode()
    adapter, storage = _build(monkeypatch, tmp_path, objects)

    with pytest.raises(RuntimeError, match="unsupported checksum format"):
        adapter.stage_weight(_full_inputs())

    assert storage.calls == [root_uri]
    adapter.close()


def test_canonical_s3_rejects_corrupt_full_tensor_bytes(monkeypatch, tmp_path):
    objects = _full_artifact(torch.tensor([7.0, 8.0]))
    shard_uri = "s3://weights/test/v2/model-00001-of-00001.safetensors"
    shard = bytearray(objects[shard_uri])
    shard[-1] ^= 1
    objects[shard_uri] = bytes(shard)
    adapter, _storage = _build(monkeypatch, tmp_path, objects)

    with pytest.raises(RuntimeError, match="checksum differs"):
        adapter.stage_weight(_full_inputs())

    assert torch.equal(
        load_file(adapter._checkpoint.local_checkpoint / "model.safetensors")[
            "weight"
        ],
        torch.tensor([1.0, 2.0]),
    )
    adapter.close()


@pytest.mark.parametrize(
    ("launch_tensors", "remote_tensors"),
    [
        (
            {"a": torch.tensor([1.0]), "b": torch.tensor([2.0])},
            {"a": torch.tensor([3.0])},
        ),
        (
            {"a": torch.tensor([1.0])},
            {"a": torch.tensor([3.0]), "b": torch.tensor([4.0])},
        ),
    ],
    ids=["missing-tensor", "extra-tensor"],
)
def test_canonical_s3_requires_exact_full_checkpoint_tensor_set(
    monkeypatch,
    tmp_path,
    launch_tensors,
    remote_tensors,
):
    prefix = "s3://weights/test/v2"
    shard_name = "model-00001-of-00001.safetensors"
    objects = {
        f"{prefix}/model.safetensors.index.json": json.dumps(
            {
                "metadata": {"total_size": 0},
                "weight_map": dict.fromkeys(remote_tensors, shard_name),
            }
        ).encode(),
        f"{prefix}/{shard_name}": _save_full_tensors(remote_tensors),
    }
    adapter, storage = _build(
        monkeypatch,
        tmp_path,
        objects,
        launch_tensors=launch_tensors,
    )

    with pytest.raises(RuntimeError, match="tensor set differs"):
        adapter.stage_weight(_full_inputs())

    assert storage.calls == [f"{prefix}/model.safetensors.index.json"]
    checkpoint = load_file(adapter._checkpoint.local_checkpoint / "model.safetensors")
    assert checkpoint.keys() == launch_tensors.keys()
    assert all(
        torch.equal(checkpoint[name], tensor)
        for name, tensor in launch_tensors.items()
    )
    state = json.loads(adapter._checkpoint.store.state_path.read_text())
    assert state["status"] == "UPDATING"
    adapter.close()


@pytest.mark.parametrize(
    ("local_tensor", "remote_tensor"),
    [
        (
            torch.tensor([1.0, 2.0]),
            torch.tensor([3, 4], dtype=torch.int32),
        ),
        (
            torch.tensor([1.0, 2.0]),
            torch.tensor([[3.0, 4.0]]),
        ),
    ],
    ids=["dtype", "shape"],
)
def test_canonical_s3_rejects_incompatible_full_checkpoint_tensor_metadata(
    monkeypatch,
    tmp_path,
    local_tensor,
    remote_tensor,
):
    objects = _full_artifact(remote_tensor)
    adapter, _storage = _build(
        monkeypatch,
        tmp_path,
        objects,
        launch_tensors={"weight": local_tensor},
    )

    with pytest.raises(RuntimeError, match="metadata differs"):
        adapter.stage_weight(_full_inputs())

    assert torch.equal(
        load_file(adapter._checkpoint.local_checkpoint / "model.safetensors")[
            "weight"
        ],
        local_tensor,
    )
    state = json.loads(adapter._checkpoint.store.state_path.read_text())
    assert state["status"] == "UPDATING"
    adapter.close()


def test_canonical_s3_rejects_incompatible_full_checkpoint_byte_size(
    monkeypatch,
    tmp_path,
):
    local_tensor = torch.tensor([1.0, 2.0])
    objects = _full_artifact(torch.tensor([3.0, 4.0]))
    shard_uri = "s3://weights/test/v2/model-00001-of-00001.safetensors"
    shard = bytearray(objects[shard_uri])
    header_size = int.from_bytes(shard[:8], "little")
    header = json.loads(shard[8 : 8 + header_size])
    header["weight"]["data_offsets"][1] = 1
    encoded_header = json.dumps(header, separators=(",", ":")).encode()
    assert len(encoded_header) <= header_size
    shard[8 : 8 + header_size] = encoded_header.ljust(header_size, b" ")
    objects[shard_uri] = bytes(shard)
    adapter, _storage = _build(
        monkeypatch,
        tmp_path,
        objects,
        launch_tensors={"weight": local_tensor},
    )

    with pytest.raises(RuntimeError, match="metadata differs"):
        adapter.stage_weight(_full_inputs())

    assert torch.equal(
        load_file(adapter._checkpoint.local_checkpoint / "model.safetensors")[
            "weight"
        ],
        local_tensor,
    )
    adapter.close()


def test_canonical_s3_full_download_failure_leaves_checkpoint_updating(
    monkeypatch, tmp_path
):
    objects = _full_artifact(torch.tensor([7.0, 8.0]))
    adapter, storage = _build(monkeypatch, tmp_path, objects)
    shard_uri = "s3://weights/test/v2/model-00001-of-00001.safetensors"
    storage.fail_key_once = shard_uri

    with pytest.raises(RuntimeError, match="full HF checkpoint download failed"):
        adapter.stage_weight(_full_inputs())

    state = json.loads(adapter._checkpoint.store.state_path.read_text())
    assert state["status"] == "UPDATING"
    assert state["version"] == "base-a"
    assert torch.equal(
        load_file(adapter._checkpoint.local_checkpoint / "model.safetensors")["weight"],
        torch.tensor([1.0, 2.0]),
    )
    assert not (adapter._checkpoint.store.full_cache / "full-a.tmp").exists()

    with pytest.raises(RuntimeError, match="update is incomplete"):
        adapter.stage_weight(_full_inputs())
    adapter.close()


def test_canonical_s3_midstream_failure_leaves_cache_failed_closed(
    monkeypatch, tmp_path
):
    target = {
        "a": torch.tensor([3.0]),
        "b": torch.tensor([4.0]),
    }
    filenames = [
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ]
    prefix = "s3://weights/test/v2"
    objects = {
        f"{prefix}/{filenames[0]}": _save_full_tensors({"a": target["a"]}),
        f"{prefix}/{filenames[1]}": _save_full_tensors({"b": target["b"]}),
        f"{prefix}/model.safetensors.index.json": json.dumps(
            {
                "metadata": {"total_size": 8},
                "weight_map": {"a": filenames[0], "b": filenames[1]},
            }
        ).encode(),
    }
    adapter, storage = _build(
        monkeypatch,
        tmp_path,
        objects,
        launch_tensors={
            "a": torch.tensor([1.0]),
            "b": torch.tensor([2.0]),
        },
    )

    monkeypatch.setenv("MX_S3_DOWNLOAD_WORKERS", "2")
    get = storage.get

    def fail_second_download(uri):
        if uri.endswith(filenames[1]):
            raise RuntimeError("injected batch download failure")
        return get(uri)

    storage.get = fail_second_download

    with pytest.raises(RuntimeError, match="full HF checkpoint download failed"):
        adapter.stage_weight(_full_inputs())

    state = json.loads(adapter._checkpoint.store.state_path.read_text())
    assert state["status"] == "UPDATING"
    assert state["version"] == "base-a"
    checkpoint = load_file(adapter._checkpoint.local_checkpoint / "model.safetensors")
    assert torch.equal(checkpoint["a"], torch.tensor([1.0]))
    assert torch.equal(checkpoint["b"], torch.tensor([2.0]))
    assert not (adapter._checkpoint.store.full_cache / "full-a").exists()
    with pytest.raises(RuntimeError, match="update is incomplete"):
        adapter.stage_weight(_full_inputs())
    adapter.close()


def test_canonical_s3_final_state_failure_leaves_checkpoint_updating(
    monkeypatch, tmp_path
):
    objects = _full_artifact(torch.tensor([7.0, 8.0]))
    adapter, _storage = _build(monkeypatch, tmp_path, objects)

    write_state = adapter._checkpoint.store.write_state

    def fail_ready_state(*, status, version, checkpoint_paths, source=None):
        if status == checkpoint_store_module.CheckpointState.READY:
            raise OSError("state failed")
        write_state(
            status=status,
            version=version,
            checkpoint_paths=checkpoint_paths,
            source=source,
        )

    monkeypatch.setattr(adapter._checkpoint.store, "write_state", fail_ready_state)

    with pytest.raises(OSError, match="state failed"):
        adapter.stage_weight(_full_inputs())

    state = json.loads(adapter._checkpoint.store.state_path.read_text())
    assert state["status"] == "UPDATING"
    assert state["version"] == "base-a"
    with pytest.raises(RuntimeError, match="update is incomplete"):
        adapter.stage_weight(_full_inputs())
    assert torch.equal(
        load_file(
            adapter._checkpoint.local_checkpoint
            / "model-00001-of-00001.safetensors"
        )["weight"],
        torch.tensor([7.0, 8.0]),
    )
    assert not (adapter._checkpoint.store.full_cache / "full-a.tmp").exists()
    adapter.close()


def test_canonical_s3_install_rejects_updating_checkpoint(monkeypatch, tmp_path):
    objects = _full_artifact(torch.tensor([7.0, 8.0]))
    adapter, _storage = _build(monkeypatch, tmp_path, objects)
    staged = adapter.stage_weight(_full_inputs())
    adapter._checkpoint.store.write_state(
        status=checkpoint_store_module.CheckpointState.UPDATING,
        version="full-a",
        checkpoint_paths=adapter._checkpoint.checkpoint_paths,
    )

    with pytest.raises(RuntimeError, match="prepared checkpoint changed"):
        adapter.apply_weight(staged)

    adapter.release_staged_weight(staged)
    adapter.close()


def test_canonical_s3_install_rejects_modified_checkpoint(monkeypatch, tmp_path):
    objects = _full_artifact(torch.tensor([7.0, 8.0]))
    adapter, _storage = _build(monkeypatch, tmp_path, objects)
    staged = adapter.stage_weight(_full_inputs())
    save_file(
        {"weight": torch.tensor([9.0, 10.0])},
        staged.path / "model-00001-of-00001.safetensors",
    )

    with pytest.raises(RuntimeError, match="prepared checkpoint changed"):
        adapter.apply_weight(staged)

    adapter.release_staged_weight(staged)
    adapter.close()


def test_canonical_s3_shared_cache_observes_full_update(
    monkeypatch, tmp_path
):
    full_tensor = torch.tensor([7.0, 8.0])
    objects = _full_artifact(full_tensor)
    first, storage = _build(monkeypatch, tmp_path, objects)
    second = _Adapter(
        model_name="test/model",
        config=ObjectStorageGeneratorConfig(
            storage_type=ObjectStorageType.S3,
            initial_base_version_id="base-a",
            seed_checkpoint_path=tmp_path / "launch",
            refit_checkpoint_dir=tmp_path / "cache",
        ),
    )

    first_staged = first.stage_weight(_full_inputs())
    second_staged = second.stage_weight(_full_inputs())

    assert torch.equal(
        load_file(second_staged.path / "model-00001-of-00001.safetensors")[
            "weight"
        ],
        full_tensor,
    )
    assert storage.calls == [
        "s3://weights/test/v2/model.safetensors.index.json",
        "s3://weights/test/v2/model-00001-of-00001.safetensors",
    ]
    first.release_staged_weight(first_staged)
    second.release_staged_weight(second_staged)
    first.close()
    second.close()


def test_canonical_s3_accepts_an_empty_delta(monkeypatch, tmp_path):
    key = "s3://weights/test/v1/model.safetensors.index.json"
    root = json.dumps(
        {
            "metadata": {
                "version": "target-a",
                "base_version": "base-a",
                "delta_encoding": "xor",
                "compression_format": "zstd",
                "checksum_format": "adler32",
            },
            "weight_map": {},
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    adapter, storage = _build(monkeypatch, tmp_path, {key: root})

    staged = adapter.stage_weight(_inputs(root))

    assert storage.calls == [key]
    assert torch.equal(
        load_file(staged.path / "model.safetensors")["weight"],
        torch.tensor([1.0, 2.0]),
    )


def test_canonical_s3_rejects_unsupported_compression(monkeypatch, tmp_path):
    base = torch.tensor([1.0, 2.0]).view(torch.uint8).numpy()
    target = torch.tensor([3.0, 4.0]).view(torch.uint8).numpy()
    objects = _artifact(base, target)
    key = "s3://weights/test/v1/model.safetensors.index.json"
    manifest = json.loads(objects[key])
    manifest["metadata"]["compression_format"] = "gzip"
    objects[key] = json.dumps(manifest).encode()
    adapter, storage = _build(monkeypatch, tmp_path, objects)

    with pytest.raises(RuntimeError, match="unsupported compression format 'gzip'"):
        adapter.stage_weight(_inputs(objects[key]))

    assert storage.calls == [key]
    state = adapter._checkpoint.store.state()
    assert state is not None
    assert state["status"] == "READY"
    assert state["version"] == "base-a"


def test_canonical_s3_rejects_invalid_manifest_json(monkeypatch, tmp_path):
    key = "s3://weights/test/v1/model.safetensors.index.json"
    adapter, storage = _build(monkeypatch, tmp_path, {key: b"not-json"})

    with pytest.raises(RuntimeError, match="not valid JSON"):
        adapter.stage_weight(_inputs(None))

    assert storage.calls == [key]
    state = adapter._checkpoint.store.state()
    assert state is not None
    assert state["status"] == "READY"
    assert state["version"] == "base-a"


def test_canonical_s3_downloads_unique_shards_concurrently(monkeypatch, tmp_path):
    adapter, storage = _build(monkeypatch, tmp_path, {})
    barrier = threading.Barrier(2)

    def get(uri):
        barrier.wait(timeout=2)
        return uri.encode()

    storage.get = get
    root = "s3://weights/prefix/model.safetensors.index.json"

    shards = adapter._checkpoint._download_deltas(
        {
            "weight_a": "model-00000-of-00002.safetensors",
            "weight_b": "model-00000-of-00002.safetensors",
            "weight_c": "model-00001-of-00002.safetensors",
        },
        root,
    )

    assert shards == {
        "model-00000-of-00002.safetensors": (
            b"s3://weights/prefix/model-00000-of-00002.safetensors",
            ["weight_a", "weight_b"],
        ),
        "model-00001-of-00002.safetensors": (
            b"s3://weights/prefix/model-00001-of-00002.safetensors",
            ["weight_c"],
        ),
    }


def test_canonical_s3_applies_delta_tensors_concurrently(monkeypatch, tmp_path):
    launch = tmp_path / "launch"
    launch.mkdir()
    tensor_size = (2 << 20) + 1
    base = {
        "weight_a": torch.zeros(tensor_size, dtype=torch.uint8),
        "weight_b": torch.ones(tensor_size, dtype=torch.uint8),
    }
    target = {
        "weight_a": torch.full((tensor_size,), 2, dtype=torch.uint8),
        "weight_b": torch.full((tensor_size,), 3, dtype=torch.uint8),
    }
    save_file(base, launch / "model.safetensors")
    checkpoint = receiver_module._LocalCheckpoint(
        model_name="test/model",
        config=ObjectStorageGeneratorConfig(
            storage_type=ObjectStorageType.S3,
            initial_base_version_id="base-a",
            seed_checkpoint_path=launch,
            refit_checkpoint_dir=tmp_path / "cache",
        ),
        s3=_MemoryS3({}),
    )
    checkpoint.initialize()

    encoded = {}
    checksums = {}
    for name in target:
        current = target[name].view(torch.uint8).numpy()
        delta, _ = compute_delta(current, base[name].view(torch.uint8).numpy())
        assert delta is not None
        encoded[name] = compress_delta(delta)
        checksums[name] = _checksum(current)
    shard = safetensors.numpy.save(encoded, metadata=checksums)

    barrier = threading.Barrier(2)
    decompressor = zstandard.ZstdDecompressor
    read_counts = {}
    read_sizes = []
    read_lock = threading.Lock()

    class TrackingReader:
        def __init__(self, reader):
            self.reader = reader

        def read(self, size):
            block = self.reader.read(size)
            with read_lock:
                read_sizes.append(size)
                if block:
                    thread_id = threading.get_ident()
                    read_counts[thread_id] = read_counts.get(thread_id, 0) + 1
            return block

        def close(self):
            self.reader.close()

    class StreamingDecompressor:
        def stream_reader(self, compressed):
            barrier.wait(timeout=2)
            return TrackingReader(decompressor().stream_reader(compressed))

        def decompress(self, *_args, **_kwargs):
            raise AssertionError("full-tensor decompression should not be used")

    monkeypatch.setenv("MX_REFIT_DELTA_WORKERS", "2")
    monkeypatch.setattr(
        zstandard,
        "ZstdDecompressor",
        StreamingDecompressor,
    )

    checkpoint._apply_shards(
        index_metadata={
            "compression_format": "zstd",
            "checksum_format": "adler32",
        },
        shards={"delta.safetensors": (shard, list(target))},
    )

    loaded = load_file(checkpoint.local_checkpoint / "model.safetensors")
    assert all(torch.equal(loaded[name], value) for name, value in target.items())
    assert sorted(read_counts.values()) == [2, 2]
    assert max(read_sizes) == 2 << 20
    assert all(0 < size <= 2 << 20 for size in read_sizes)


def test_reconstructed_checksum_failure_propagates(monkeypatch, tmp_path):
    base = torch.tensor([1.0, 2.0]).view(torch.uint8).numpy()
    target = torch.tensor([3.0, 4.0]).view(torch.uint8).numpy()
    objects = _artifact(base, target, checksum="00000000")
    root = objects["s3://weights/test/v1/model.safetensors.index.json"]
    adapter, _ = _build(monkeypatch, tmp_path, objects)
    with pytest.raises(RuntimeError, match="target checksum differs"):
        adapter.stage_weight(_inputs(root))

    failed_state = json.loads(adapter._checkpoint.store.state_path.read_text())
    assert failed_state["status"] == "UPDATING"
    assert failed_state["version"] == "base-a"

    recovered, _ = _build(monkeypatch, tmp_path, objects)
    state = json.loads(recovered._checkpoint.store.state_path.read_text())
    assert state["status"] == "READY"
    assert state["version"] == "base-a"
    assert torch.equal(
        load_file(recovered._checkpoint.local_checkpoint / "model.safetensors")[
            "weight"
        ],
        torch.tensor([1.0, 2.0]),
    )


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("missing_metadata", "missing checksum metadata"),
        ("missing_tensor", "missing tensor 'weight'"),
        ("missing_checksum", "missing checksum for tensor 'weight'"),
        ("missing_local_tensor", "tensor 'other'.*absent from the local checkpoint"),
    ],
)
def test_malformed_delta_shard_reports_context(
    monkeypatch,
    tmp_path,
    case,
    message,
):
    base = torch.tensor([1.0, 2.0]).view(torch.uint8).numpy()
    target = torch.tensor([3.0, 4.0]).view(torch.uint8).numpy()
    objects = _artifact(base, target)
    root_key = "s3://weights/test/v1/model.safetensors.index.json"
    filename = "model-00000-of-00001.safetensors"
    shard_key = f"s3://weights/test/v1/{filename}"
    delta, _ = compute_delta(target, base)
    assert delta is not None
    encoded = compress_delta(delta)
    checksum = _checksum(target)

    if case == "missing_metadata":
        objects[shard_key] = safetensors.numpy.save({"weight": encoded})
    elif case == "missing_tensor":
        objects[shard_key] = safetensors.numpy.save(
            {"other": encoded}, metadata={"other": checksum}
        )
    elif case == "missing_checksum":
        objects[shard_key] = safetensors.numpy.save(
            {"weight": encoded}, metadata={"other": checksum}
        )
    else:
        manifest = json.loads(objects[root_key])
        manifest["weight_map"] = {"other": filename}
        objects[root_key] = json.dumps(manifest).encode()
        objects[shard_key] = safetensors.numpy.save(
            {"other": encoded}, metadata={"other": checksum}
        )

    adapter, _ = _build(monkeypatch, tmp_path, objects)
    with pytest.raises(RuntimeError, match=message) as raised:
        adapter.stage_weight(_inputs(objects[root_key]))

    assert filename in str(raised.value)
    state = json.loads(adapter._checkpoint.store.state_path.read_text())
    assert state["status"] == "UPDATING"
    assert state["version"] == "base-a"


def test_wrong_base_fails_before_s3_download(monkeypatch, tmp_path):
    base = torch.tensor([1.0, 2.0]).view(torch.uint8).numpy()
    target = torch.tensor([3.0, 4.0]).view(torch.uint8).numpy()
    objects = _artifact(base, target)
    root = objects["s3://weights/test/v1/model.safetensors.index.json"]
    adapter, storage = _build(monkeypatch, tmp_path, objects)

    with pytest.raises(RuntimeError, match="exact base"):
        adapter.stage_weight(_inputs(root, base_version="other-base"))

    assert storage.calls == []


def test_child_download_failure_leaves_checkpoint_updating(
    monkeypatch,
    tmp_path,
):
    base = torch.tensor([1.0, 2.0]).view(torch.uint8).numpy()
    target = torch.tensor([3.0, 4.0]).view(torch.uint8).numpy()
    objects = _artifact(base, target)
    root_key = "s3://weights/test/v1/model.safetensors.index.json"
    shard_key = "s3://weights/test/v1/model-00000-of-00001.safetensors"
    adapter, storage = _build(monkeypatch, tmp_path, objects)
    storage.fail_key_once = shard_key

    with pytest.raises(RuntimeError, match="injected shard download failure"):
        adapter.stage_weight(_inputs(objects[root_key]))

    state = json.loads(adapter._checkpoint.store.state_path.read_text())
    assert state["status"] == "UPDATING"
    assert state["version"] == "base-a"
    with pytest.raises(RuntimeError, match="update is incomplete"):
        adapter.stage_weight(_inputs(objects[root_key]))


def test_corrupt_zstd_fails_before_checkpoint_mutation(monkeypatch, tmp_path):
    base = torch.tensor([1.0, 2.0]).view(torch.uint8).numpy()
    target = torch.tensor([3.0, 4.0]).view(torch.uint8).numpy()
    objects = _artifact(base, target)
    root_key = "s3://weights/test/v1/model.safetensors.index.json"
    shard_key = "s3://weights/test/v1/model-00000-of-00001.safetensors"
    shard = bytearray(objects[shard_key])
    header_size = int.from_bytes(shard[:8], "little")
    data_start = 8 + header_size
    shard[data_start:] = b"\x28\xb5\x2f\xfd" + bytes(len(shard) - data_start - 4)
    objects[shard_key] = bytes(shard)
    adapter, _ = _build(monkeypatch, tmp_path, objects)

    with pytest.raises(RuntimeError, match="delta byte size differs"):
        adapter.stage_weight(_inputs(objects[root_key]))

    state = json.loads(adapter._checkpoint.store.state_path.read_text())
    assert state["status"] == "UPDATING"
    assert state["version"] == "base-a"


def test_cached_target_requires_the_same_source_identity(monkeypatch, tmp_path):
    base = torch.tensor([1.0, 2.0]).view(torch.uint8).numpy()
    target = torch.tensor([3.0, 4.0]).view(torch.uint8).numpy()
    objects = _artifact(base, target)
    root_key = "s3://weights/test/v1/model.safetensors.index.json"
    alternate_key = "s3://weights/test/alternate/v1/model.safetensors.index.json"
    objects[alternate_key] = objects[root_key]
    adapter, _ = _build(monkeypatch, tmp_path, objects)
    staged = adapter.stage_weight(_inputs(objects[root_key]))
    adapter.apply_weight(staged)
    adapter.release_staged_weight(staged)

    with pytest.raises(RuntimeError, match="different source identity"):
        adapter.stage_weight(_inputs(objects[alternate_key], uri=alternate_key))


def test_installed_target_becomes_the_next_exact_base(monkeypatch, tmp_path):
    base = torch.tensor([1.0, 2.0]).view(torch.uint8).numpy()
    first_tensor = torch.tensor([3.0, 4.0])
    first = first_tensor.view(torch.uint8).numpy()
    second_tensor = torch.tensor([5.0, 6.0])
    second = second_tensor.view(torch.uint8).numpy()
    objects = _artifact(base, first)
    objects.update(
        _artifact(
            first,
            second,
            version="target-b",
            version_label=2,
            base_version="target-a",
        )
    )
    adapter, _ = _build(monkeypatch, tmp_path, objects)
    first_root = objects["s3://weights/test/v1/model.safetensors.index.json"]
    staged = adapter.stage_weight(_inputs(first_root))
    adapter.apply_weight(staged)
    adapter.release_staged_weight(staged)

    second_root = objects["s3://weights/test/v2/model.safetensors.index.json"]
    staged = adapter.stage_weight(
        _inputs(
            second_root,
            version="target-b",
            version_label=2,
            base_version="target-a",
        )
    )

    assert torch.equal(
        load_file(staged.path / "model.safetensors")["weight"],
        second_tensor,
    )
