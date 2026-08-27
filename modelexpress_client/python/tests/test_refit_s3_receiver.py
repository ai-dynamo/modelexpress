# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import threading
from dataclasses import replace

import pytest
import safetensors.numpy
import s3transfer.manager
import torch
import zstandard
from safetensors.torch import load_file, save_file

from modelexpress_rl import (
    ObjectStorageGeneratorConfig,
    ObjectStorageSource,
    ObjectStorageType,
    WeightPayloadFormat,
)
from modelexpress_rl.inference.adapter import GeneratorTransferInputs
from modelexpress_rl.inference import receiver as receiver_module
from modelexpress_rl.inference.receiver import CanonicalS3GeneratorAdapter
from modelexpress_rl.s3 import ImmutableS3Conflict, S3Client
from modelexpress_rl.utils import adler32_checksum, compress_delta, compute_delta


class _MemoryS3:
    def __init__(self, objects):
        self.objects = objects
        self.calls = []
        self.fail_key_once = None

    def get(self, uri):
        self.calls.append(uri)
        if uri == self.fail_key_once:
            self.fail_key_once = None
            raise RuntimeError("injected shard download failure")
        return self.objects[uri]

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


class _Adapter(CanonicalS3GeneratorAdapter):
    def __init__(self, **kwargs):
        self.installed = []
        super().__init__(**kwargs)

    def install_prepared_checkpoint(self, prepared):
        self.installed.append(prepared.path)


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
    version_number=1,
    base_version="base-a",
):
    delta, _ = compute_delta(target, base)
    assert delta is not None
    shard = safetensors.numpy.save(
        {"weight": compress_delta(delta)},
        metadata={"weight": checksum or adler32_checksum(target)},
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
    prefix = f"s3://weights/test/v{version_number}"
    return {
        f"{prefix}/model.safetensors.index.json": root,
        f"{prefix}/model-00000-of-00001.safetensors": shard,
    }


def _inputs(
    _root,
    *,
    base_version="base-a",
    version="target-a",
    version_number=1,
    uri=None,
):
    if uri is None:
        uri = f"s3://weights/test/v{version_number}/model.safetensors.index.json"
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


def _build(monkeypatch, tmp_path, objects):
    launch = tmp_path / "launch"
    launch.mkdir(exist_ok=True)
    save_file({"weight": torch.tensor([1.0, 2.0])}, launch / "model.safetensors")
    storage = _MemoryS3(objects)
    monkeypatch.setattr(receiver_module, "S3Client", lambda **_kwargs: storage)
    adapter = _Adapter(
        model_name="test/model",
        config=ObjectStorageGeneratorConfig(
            storage_type=ObjectStorageType.S3,
            initial_base_version_id="base-a",
            launch_checkpoint=launch,
            preparation_cache_dir=tmp_path / "cache",
        ),
    )
    return adapter, storage


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

    with pytest.raises(RuntimeError, match="unsupported.*compression"):
        adapter.stage_weight(_inputs(objects[key]))

    assert storage.calls == [key]


def test_canonical_s3_rejects_invalid_manifest_json(monkeypatch, tmp_path):
    key = "s3://weights/test/v1/model.safetensors.index.json"
    adapter, storage = _build(monkeypatch, tmp_path, {key: b"not-json"})

    with pytest.raises(RuntimeError, match="not valid JSON"):
        adapter.stage_weight(_inputs(None))

    assert storage.calls == [key]


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
            launch_checkpoint=launch,
            preparation_cache_dir=tmp_path / "cache",
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
        checksums[name] = adler32_checksum(current)
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
        receiver_module.zstandard,
        "ZstdDecompressor",
        StreamingDecompressor,
    )

    checkpoint.decompressor = receiver_module._DECOMPRESSORS["zstd"]
    checkpoint._apply_shards({"delta.safetensors": (shard, list(target))})

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

    recovered, _ = _build(monkeypatch, tmp_path, objects)
    state = json.loads(recovered._checkpoint.state_path.read_text())
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
    checksum = adler32_checksum(target)

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
    state = json.loads(adapter._checkpoint.state_path.read_text())
    assert state["version"] == "base-a"


def test_wrong_base_fails_before_s3_download(monkeypatch, tmp_path):
    base = torch.tensor([1.0, 2.0]).view(torch.uint8).numpy()
    target = torch.tensor([3.0, 4.0]).view(torch.uint8).numpy()
    objects = _artifact(base, target)
    root = objects["s3://weights/test/v1/model.safetensors.index.json"]
    adapter, storage = _build(monkeypatch, tmp_path, objects)

    with pytest.raises(ValueError, match="exact local base"):
        adapter.stage_weight(_inputs(root, base_version="other-base"))

    assert storage.calls == []


def test_child_download_failure_is_retryable(
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

    with pytest.raises(RuntimeError, match="canonical delta download failed"):
        adapter.stage_weight(_inputs(objects[root_key]))

    state = json.loads(adapter._checkpoint.state_path.read_text())
    assert state["version"] == "base-a"
    assert adapter.stage_weight(_inputs(objects[root_key])).target_version == "target-a"


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

    state = json.loads(adapter._checkpoint.state_path.read_text())
    assert state["version"] == "base-a"


def test_cached_target_requires_the_same_canonical_root(monkeypatch, tmp_path):
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

    with pytest.raises(RuntimeError, match="different canonical root"):
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
            version_number=2,
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
            version_number=2,
            base_version="target-a",
        )
    )

    assert torch.equal(
        load_file(staged.path / "model.safetensors")["weight"],
        second_tensor,
    )
