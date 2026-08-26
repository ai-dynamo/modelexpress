# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import struct
import threading
import zlib
from concurrent import futures

import grpc
import numpy as np
import pytest
import safetensors.numpy
import torch
import zstandard
from modelexpress_rl import (
    ModelExpressTrainerClient,
    ModelExpressTrainerConfig,
    ObjectStorageConfig,
    ObjectStorageType,
    TrainerStagingMode,
    WeightPayloadFormat,
    WeightVersionRef,
    refit_pb2,
    refit_pb2_grpc,
)
from modelexpress_rl.s3 import S3Client
from modelexpress_rl.train import client as trainer_client_module


class _MemoryS3:
    def __init__(self) -> None:
        self.objects = {}
        self.fail_next = False

    def put(self, *, uri, data):
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("injected upload failure")
        existing = self.objects.setdefault(uri, data)
        if existing != data:
            raise RuntimeError("immutable object differs")

    def close(self):
        pass


class _S3Error(Exception):
    def __init__(self, code):
        self.response = {"Error": {"Code": code}}


class _FailingAbortS3:
    def __init__(self, complete_error):
        self.complete_error = complete_error
        self.abort_calls = 0

    def create_multipart_upload(self, **_request):
        return {"UploadId": "upload-1"}

    def upload_part(self, **request):
        return {"ETag": "etag", "PartNumber": request["PartNumber"]}

    def complete_multipart_upload(self, **_request):
        raise self.complete_error

    def abort_multipart_upload(self, **_request):
        self.abort_calls += 1
        raise RuntimeError("abort failed")


class _RefitService(refit_pb2_grpc.RefitServiceServicer):
    def __init__(self) -> None:
        self.registrations = set()
        self.shards = []
        self.deleted_shards = []
        self.target = refit_pb2.WeightVersion(
            uid="target-a",
            model_name="test/model",
            version_number=1,
            payload_format=refit_pb2.WEIGHT_PAYLOAD_FORMAT_XOR_DELTA,
            base_version_id="base-a",
            object_storage=refit_pb2.ObjectStorageSource(
                uri="s3://weights/tests/v1/model.safetensors.index.json",
                storage_type=refit_pb2.OBJECT_STORAGE_TYPE_S3,
            ),
            state=refit_pb2.WEIGHT_VERSION_STATE_STAGING,
        )

    def RegisterWorker(self, request, _context):
        self.registrations.add(request.worker.worker_id)
        return refit_pb2.RegisterWorkerResponse(worker=request.worker)

    def GetWeightVersion(self, request, context):
        if request.uid != self.target.uid:
            context.abort(grpc.StatusCode.NOT_FOUND, "version not found")
        return refit_pb2.GetWeightVersionResponse(version=self.target)

    def CreateWeightVersionShard(self, request, _context):
        self.shards.append(request.shard)
        return refit_pb2.CreateWeightVersionShardResponse(
            shard=request.shard,
            version=self.target,
        )

    def DeleteWeightVersionShard(self, request, context):
        if any(
            deleted.version_id == request.version_id
            and deleted.source_slot_id == request.source_slot_id
            and deleted.worker_id == request.worker_id
            for deleted in self.deleted_shards
        ):
            context.abort(grpc.StatusCode.NOT_FOUND, "shard already deleted")
        self.deleted_shards.append(request)
        return refit_pb2.DeleteWeightVersionShardResponse(deleted=True)


def _write_safetensors(path, tensors):
    header = {}
    payload = bytearray()
    for name, tensor in tensors.items():
        data = tensor.contiguous().view(torch.uint8).numpy().tobytes()
        header[name] = {
            "dtype": "F32",
            "shape": list(tensor.shape),
            "data_offsets": [len(payload), len(payload) + len(data)],
        }
        payload.extend(data)
    encoded_header = json.dumps(header, separators=(",", ":")).encode()
    path.write_bytes(struct.pack("<Q", len(encoded_header)) + encoded_header + payload)


@pytest.fixture
def refit_server():
    service = _RefitService()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    refit_pb2_grpc.add_RefitServiceServicer_to_server(service, server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    try:
        yield service, f"127.0.0.1:{port}"
    finally:
        server.stop(grace=None).wait()


def _trainer(
    monkeypatch,
    tmp_path,
    server_url,
    launch_tensors=None,
    process_group=None,
    prepare_base=True,
):
    launch = tmp_path / "model.safetensors"
    launch_tensors = launch_tensors or {"weight": torch.tensor([1.0, 2.0])}
    _write_safetensors(launch, launch_tensors)
    storage = _MemoryS3()
    monkeypatch.setattr(trainer_client_module, "S3Client", lambda **_kwargs: storage)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda _group=None: 0)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda _group=None: 1)
    monkeypatch.setattr(
        torch.distributed,
        "all_gather_object",
        lambda output, value, group=None: output.__setitem__(0, value),
    )
    monkeypatch.setattr(
        torch.distributed,
        "gather_object",
        lambda value, output, dst=0, group=None: (
            output.__setitem__(0, value) if output is not None else None
        ),
    )
    monkeypatch.setattr(
        torch.distributed,
        "all_reduce",
        lambda value, op=None, group=None: None,
    )
    trainer = ModelExpressTrainerClient.initialize(
        ModelExpressTrainerConfig(
            model_name="test/model",
            worker_id="trainer-a",
            server_url=server_url,
            staging_mode=TrainerStagingMode.WRITE_TO_STORAGE,
            payload_format=WeightPayloadFormat.XOR_DELTA,
            registration_ttl_seconds=60,
            process_group=process_group,
            object_storage=ObjectStorageConfig(
                storage_type=ObjectStorageType.S3,
                uri_prefix="s3://weights/tests",
                initial_base_version_id="base-a",
                launch_checkpoint=launch,
            ),
        )
    )
    if prepare_base:
        trainer.prepare_delta_base(
            hf_tensor_iter=iter([list(launch_tensors.items())]),
        )
    return trainer, storage


@pytest.mark.parametrize("code", ["InternalError", "PreconditionFailed"])
def test_s3_multipart_abort_failure_preserves_upload_outcome(code, caplog):
    data = b"delta"
    error = _S3Error(code)
    backend = _FailingAbortS3(error)
    s3 = object.__new__(S3Client)
    s3._client = backend
    s3._multipart_threshold_bytes = 1
    s3._upload_part_bytes = 5 * 1024**2
    s3._upload_pool = futures.ThreadPoolExecutor(max_workers=1)
    reads = []
    s3.get = lambda uri: reads.append(uri) or data
    caplog.set_level(logging.WARNING)

    try:
        if code == "PreconditionFailed":
            s3.put(uri="s3://weights/delta", data=data)
        else:
            with pytest.raises(_S3Error) as raised:
                s3.put(uri="s3://weights/delta", data=data)
            assert raised.value is error
    finally:
        s3._upload_pool.shutdown()

    assert backend.abort_calls == 1
    assert reads == (["s3://weights/delta"] if code == "PreconditionFailed" else [])
    assert "Failed to abort multipart upload" in caplog.text


def test_s3_prepare_delta_base_uses_owned_launch_tensors(
    monkeypatch, tmp_path, refit_server, caplog
):
    _service, server_url = refit_server
    launch_tensors = {
        "a": torch.tensor([1.0]),
        "b": torch.tensor([2.0]),
    }
    trainer, _storage = _trainer(
        monkeypatch,
        tmp_path,
        server_url,
        launch_tensors,
        prepare_base=False,
    )
    try:
        with caplog.at_level(logging.INFO, logger=trainer_client_module.__name__):
            trainer.prepare_delta_base(
                hf_tensor_iter=iter([[("a", torch.tensor([9.0]))]]),
            )

        assert set(trainer._snapshot) == {"a"}
        assert np.array_equal(
            trainer._snapshot["a"],
            launch_tensors["a"].view(torch.uint8).numpy(),
        )
        assert (
            "ModelExpress prepare_delta_base: rank=0 tensors=1 duration=" in caplog.text
        )
    finally:
        trainer.close()


def test_s3_prepare_delta_base_reads_framework_buckets_concurrently(
    monkeypatch, tmp_path, refit_server
):
    _service, server_url = refit_server
    monkeypatch.setenv("MX_REFIT_DELTA_WORKERS", "2")
    trainer, _storage = _trainer(
        monkeypatch,
        tmp_path,
        server_url,
        {
            "a": torch.tensor([1.0]),
            "b": torch.tensor([2.0]),
        },
        prepare_base=False,
    )
    reader = trainer._read_launch_tensor
    assert reader is not None
    barrier = threading.Barrier(2)
    threads = set()

    def read(name):
        threads.add(threading.get_ident())
        barrier.wait(timeout=2)
        return reader(name)

    trainer._read_launch_tensor = read
    try:
        trainer.prepare_delta_base(
            hf_tensor_iter=iter(
                [
                    [("a", torch.tensor([9.0]))],
                    [("b", torch.tensor([9.0]))],
                ]
            ),
        )

        assert len(threads) == 2
        assert set(trainer._snapshot) == {"a", "b"}
    finally:
        trainer.close()


def test_s3_close_releases_delta_base(monkeypatch, tmp_path, refit_server):
    _service, server_url = refit_server
    trainer, _storage = _trainer(monkeypatch, tmp_path, server_url)
    trainer._metric_delta = object()

    trainer.close()

    assert trainer._snapshot == {}
    assert trainer._read_launch_tensor is None
    assert trainer._metric_delta is None


def test_s3_process_group_belongs_to_trainer_config(
    monkeypatch, tmp_path, refit_server
):
    _service, server_url = refit_server
    process_group = object()
    trainer, _storage = _trainer(
        monkeypatch,
        tmp_path,
        server_url,
        process_group=process_group,
    )

    try:
        assert trainer._process_group is process_group
    finally:
        trainer.close()


def test_s3_stage_is_local_then_publish_uploads_version_root(
    monkeypatch, tmp_path, refit_server
):
    service, server_url = refit_server
    trainer, storage = _trainer(monkeypatch, tmp_path, server_url)
    current = torch.tensor([1.0, 3.0])

    try:
        with pytest.raises(RuntimeError, match="does not use source slots"):
            _ = trainer.source_slot_id
        staged = trainer.stage_shard(
            version=WeightVersionRef("target-a"),
            hf_tensor_iter=iter([[("weight", current)]]),
        )
        assert storage.objects == {}

        staged.publish()
        staged.publish()
    finally:
        trainer.close()

    assert service.registrations == set()
    assert service.shards == []
    root_uri = "s3://weights/tests/v1/model.safetensors.index.json"
    index = json.loads(storage.objects[root_uri])
    assert index["metadata"] == {
        "base_version": "base-a",
        "checksum_format": "adler32",
        "compression_format": "zstd",
        "delta_encoding": "xor",
        "version": 1,
    }
    assert index["weight_map"] == {"weight": "model-00000-of-00001.safetensors"}
    removed_digests = {"base_digest", "target_digest", "format_digest"}
    assert removed_digests.isdisjoint(index)
    shard_key = next(
        uri for uri in storage.objects if uri.endswith(".safetensors")
    )
    assert shard_key == "s3://weights/tests/v1/model-00000-of-00001.safetensors"
    blob = storage.objects[shard_key]
    (header_size,) = struct.unpack("<Q", blob[:8])
    header = json.loads(blob[8 : 8 + header_size])
    expected_checksum = f"{zlib.adler32(current.view(torch.uint8).numpy()):08x}"
    assert header["__metadata__"] == {"weight": expected_checksum}
    assert removed_digests.isdisjoint(header)
    encoded = safetensors.numpy.load(blob)["weight"]
    assert header["weight"] == {
        "data_offsets": [0, len(encoded)],
        "dtype": "U8",
        "shape": [len(encoded)],
    }
    assert encoded.dtype == np.uint8
    assert encoded.ndim == 1
    assert encoded.tobytes().startswith(bytes.fromhex("28b52ffd"))
    decoded = zstandard.ZstdDecompressor().decompress(encoded)
    expected_delta = torch.bitwise_xor(
        torch.tensor([1.0, 2.0]).view(torch.uint8), current.view(torch.uint8)
    )
    assert decoded == expected_delta.numpy().tobytes()


def test_s3_stage_requires_version_number(monkeypatch, tmp_path, refit_server):
    service, server_url = refit_server
    service.target.ClearField("version_number")
    trainer, storage = _trainer(monkeypatch, tmp_path, server_url)

    try:
        with pytest.raises(RuntimeError, match="must have a version number"):
            trainer.stage_shard(
                version=WeightVersionRef("target-a"),
                hf_tensor_iter=iter([[("weight", torch.tensor([1.0, 3.0]))]]),
            )
    finally:
        trainer.close()

    assert storage.objects == {}


def test_s3_stage_requires_version_uri_under_configured_prefix(
    monkeypatch, tmp_path, refit_server
):
    service, server_url = refit_server
    service.target.object_storage.uri = (
        "s3://weights/other/v1/model.safetensors.index.json"
    )
    trainer, storage = _trainer(monkeypatch, tmp_path, server_url)

    try:
        with pytest.raises(RuntimeError, match="does not match the configured prefix"):
            trainer.stage_shard(
                version=WeightVersionRef("target-a"),
                hf_tensor_iter=iter([[("weight", torch.tensor([1.0, 3.0]))]]),
            )
    finally:
        trainer.close()

    assert storage.objects == {}


def test_s3_publish_failure_keeps_handle_retryable(monkeypatch, tmp_path, refit_server):
    service, server_url = refit_server
    trainer, storage = _trainer(monkeypatch, tmp_path, server_url)

    try:
        staged = trainer.stage_shard(
            version=WeightVersionRef("target-a"),
            hf_tensor_iter=iter([[("weight", torch.tensor([4.0, 5.0]))]]),
        )
        encoded = {
            name: value.tobytes()
            for name, value in staged._staged.encoded_deltas.items()
        }
        storage.fail_next = True
        with pytest.raises(RuntimeError, match="injected upload failure"):
            staged.publish()
        assert trainer._current_base_version_id == "base-a"
        assert service.shards == []
        assert {
            name: value.tobytes()
            for name, value in staged._staged.encoded_deltas.items()
        } == encoded

        staged.publish()
        assert trainer._current_base_version_id == "target-a"
        assert staged._staged.encoded_deltas == {}
        assert staged._staged.checksums == {}
    finally:
        trainer.close()

    assert service.shards == []


def test_s3_processes_buckets_concurrently_and_uploads_one_shard(
    monkeypatch, tmp_path, refit_server
):
    _service, server_url = refit_server
    monkeypatch.setenv("MX_REFIT_DELTA_WORKERS", "2")
    trainer, storage = _trainer(
        monkeypatch,
        tmp_path,
        server_url,
        {
            "a": torch.tensor([1.0, 2.0]),
            "b": torch.tensor([3.0, 4.0]),
        },
    )
    process_barrier = threading.Barrier(2)
    process_threads = set()
    process_bucket = trainer._process_delta_bucket
    save = safetensors.numpy.save
    save_calls = 0

    def track_process(bucket):
        process_threads.add(threading.get_ident())
        process_barrier.wait(timeout=5)
        return process_bucket(bucket)

    def track_save(*args, **kwargs):
        nonlocal save_calls
        save_calls += 1
        return save(*args, **kwargs)

    trainer._process_delta_bucket = track_process
    monkeypatch.setattr(safetensors.numpy, "save", track_save)
    try:
        staged = trainer.stage_shard(
            version=WeightVersionRef("target-a"),
            hf_tensor_iter=iter(
                [
                    [("a", torch.tensor([2.0, 3.0]))],
                    [("b", torch.tensor([4.0, 5.0]))],
                ]
            ),
        )
        assert len(process_threads) == 2
        assert save_calls == 0
        monkeypatch.setattr(
            trainer_client_module,
            "compress_delta",
            lambda _delta: pytest.fail("publication recompressed a staged delta"),
        )
        staged.publish()
        assert save_calls == 1
    finally:
        trainer.close()

    filenames = sorted(
        uri.rsplit("/", 1)[-1]
        for uri in storage.objects
        if uri.endswith(".safetensors")
    )
    assert filenames == ["model-00000-of-00001.safetensors"]
    index = json.loads(
        storage.objects["s3://weights/tests/v1/model.safetensors.index.json"]
    )
    assert set(index["weight_map"]) == {"a", "b"}
    assert set(index["weight_map"].values()) == {filenames[0]}


def test_s3_preserves_framework_bucket_boundaries(monkeypatch, tmp_path, refit_server):
    _service, server_url = refit_server
    monkeypatch.setenv("MX_REFIT_DELTA_WORKERS", "1")
    trainer, _storage = _trainer(
        monkeypatch,
        tmp_path,
        server_url,
        {
            "a": torch.tensor([1.0]),
            "b": torch.tensor([2.0]),
            "c": torch.tensor([3.0]),
        },
    )
    buckets = [
        [
            ("a", torch.tensor([2.0])),
            ("b", torch.tensor([3.0])),
        ],
        [("c", torch.tensor([4.0]))],
    ]
    processed = []
    process_bucket = trainer._process_delta_bucket

    def track_process(bucket):
        processed.append(bucket)
        return process_bucket(bucket)

    trainer._process_delta_bucket = track_process
    try:
        trainer.stage_shard(
            version=WeightVersionRef("target-a"),
            hf_tensor_iter=iter(buckets),
        )
    finally:
        trainer.close()

    assert processed[0] is buckets[0]
    assert processed[1] is buckets[1]


def test_s3_exposes_local_metrics_after_publication(
    monkeypatch, tmp_path, refit_server
):
    _service, server_url = refit_server
    trainer, storage = _trainer(monkeypatch, tmp_path, server_url)
    current = torch.tensor([1.0, 3.0])
    clock = iter([10.0, 12.0, 20.0, 23.0, 30.0, 34.0])
    gather_calls = 0
    all_gather = torch.distributed.all_gather_object

    def track_gather(output, value, group=None):
        nonlocal gather_calls
        gather_calls += 1
        return all_gather(output, value, group=group)

    monkeypatch.setattr(torch.distributed, "all_gather_object", track_gather)
    monkeypatch.setattr(
        trainer_client_module,
        "perf_counter",
        lambda: next(clock),
    )
    try:
        staged = trainer.stage_shard(
            version=WeightVersionRef("target-a"),
            hf_tensor_iter=iter([[("weight", current)]]),
        )
        expected_delta = torch.bitwise_xor(
            torch.tensor([1.0, 2.0]).view(torch.uint8),
            current.view(torch.uint8),
        )
        assert gather_calls == 0
        assert staged._staged.changed_bytes == int(torch.count_nonzero(expected_delta))
        assert staged._staged.total_bytes == current.numel() * current.element_size()
        assert staged._staged.wire_bytes == 0

        staged.publish()
        assert gather_calls == 1
        shard = next(
            data
            for uri, data in storage.objects.items()
            if uri.endswith(".safetensors")
        )
        assert staged._staged.wire_bytes == len(shard)

        assert trainer.pop_metrics() == {
            "changed_bytes": int(torch.count_nonzero(expected_delta)),
            "total_bytes": expected_delta.numel(),
            "wire_bytes": len(shard),
            "stage_delta_time": 2.0,
            "publish_object_storage_time": 3.0,
        }
        assert trainer.pop_metrics() == {}
    finally:
        trainer.close()


def test_s3_clean_update_still_publishes_root_index(
    monkeypatch, tmp_path, refit_server
):
    service, server_url = refit_server
    trainer, storage = _trainer(monkeypatch, tmp_path, server_url)

    try:
        staged = trainer.stage_shard(
            version=WeightVersionRef("target-a"),
            hf_tensor_iter=iter([[("weight", torch.tensor([1.0, 2.0]))]]),
        )
        assert staged._staged.encoded_deltas == {}
        assert staged._staged.checksums == {}
        assert staged._staged.wire_bytes == 0
        staged.publish()
        metrics = trainer.pop_metrics()
    finally:
        trainer.close()

    assert len(storage.objects) == 1
    assert service.shards == []
    index = json.loads(
        storage.objects["s3://weights/tests/v1/model.safetensors.index.json"]
    )
    assert index["weight_map"] == {}
    assert metrics["changed_bytes"] == 0
    assert metrics["total_bytes"] > 0
    assert metrics["wire_bytes"] == 0
    assert metrics["stage_delta_time"] >= 0
    assert metrics["publish_object_storage_time"] >= 0


def test_s3_chains_from_published_base_and_keeps_previous_advertisement(
    monkeypatch, tmp_path, refit_server
):
    service, server_url = refit_server
    trainer, storage = _trainer(monkeypatch, tmp_path, server_url)

    try:
        first = trainer.stage_shard(
            version=WeightVersionRef("target-a"),
            hf_tensor_iter=iter([[("weight", torch.tensor([1.0, 3.0]))]]),
        )
        first.publish()
        assert first._staged.encoded_deltas == {}
        assert first._staged.checksums == {}
        assert first._staged.candidate_snapshot == {}

        service.target = refit_pb2.WeightVersion(
            uid="target-b",
            model_name="test/model",
            version_number=2,
            payload_format=refit_pb2.WEIGHT_PAYLOAD_FORMAT_XOR_DELTA,
            base_version_id="target-a",
            object_storage=refit_pb2.ObjectStorageSource(
                uri="s3://weights/tests/v2/model.safetensors.index.json",
                storage_type=refit_pb2.OBJECT_STORAGE_TYPE_S3,
            ),
            state=refit_pb2.WEIGHT_VERSION_STATE_STAGING,
        )
        second = trainer.stage_shard(
            version=WeightVersionRef("target-b"),
            hf_tensor_iter=iter([[("weight", torch.tensor([2.0, 4.0]))]]),
        )
        second.publish()
        assert second._staged.encoded_deltas == {}
        assert second._staged.checksums == {}
        assert second._staged.candidate_snapshot == {}
        trainer.release_version(version=WeightVersionRef("target-a"))
        assert trainer._published_shards == {}
    finally:
        trainer.close()

    second_shard = next(
        data
        for uri, data in storage.objects.items()
        if "/v2/" in uri and uri.endswith(".safetensors")
    )
    encoded = safetensors.numpy.load(second_shard)["weight"]
    decoded = zstandard.ZstdDecompressor().decompress(encoded)
    expected = torch.bitwise_xor(
        torch.tensor([1.0, 3.0]).view(torch.uint8),
        torch.tensor([2.0, 4.0]).view(torch.uint8),
    )
    assert decoded == expected.numpy().tobytes()
    assert service.deleted_shards == []


def test_s3_release_keeps_canonical_advertisement(monkeypatch, tmp_path, refit_server):
    service, server_url = refit_server
    trainer, _storage = _trainer(monkeypatch, tmp_path, server_url)

    try:
        trainer.stage_shard(
            version=WeightVersionRef("target-a"),
            hf_tensor_iter=iter([[("weight", torch.tensor([1.0, 3.0]))]]),
        ).publish()
        service.target = refit_pb2.WeightVersion(
            uid="target-b",
            model_name="test/model",
            version_number=2,
            payload_format=refit_pb2.WEIGHT_PAYLOAD_FORMAT_XOR_DELTA,
            base_version_id="target-a",
            object_storage=refit_pb2.ObjectStorageSource(
                uri="s3://weights/tests/v2/model.safetensors.index.json",
                storage_type=refit_pb2.OBJECT_STORAGE_TYPE_S3,
            ),
            state=refit_pb2.WEIGHT_VERSION_STATE_STAGING,
        )
        trainer.stage_shard(
            version=WeightVersionRef("target-b"),
            hf_tensor_iter=iter([[("weight", torch.tensor([2.0, 4.0]))]]),
        ).publish()

        trainer.release_version(version=WeightVersionRef("target-a"))
        assert trainer._published_shards == {}
    finally:
        trainer.close()

    assert service.shards == []
    assert service.deleted_shards == []


def test_s3_propagates_tensor_processing_error(monkeypatch, tmp_path, refit_server):
    _service, server_url = refit_server
    trainer, storage = _trainer(monkeypatch, tmp_path, server_url)

    try:
        with pytest.raises(KeyError, match="missing"):
            trainer.stage_shard(
                version=WeightVersionRef("target-a"),
                hf_tensor_iter=iter([[("missing", torch.tensor([1.0]))]]),
            )
    finally:
        trainer.close()

    assert storage.objects == {}


def test_object_storage_config_requires_storage_delta_pair(tmp_path):
    launch = tmp_path / "model.safetensors"
    _write_safetensors(launch, {"weight": torch.tensor([1.0])})
    object_storage = ObjectStorageConfig(
        storage_type=ObjectStorageType.S3,
        uri_prefix="s3://weights/tests",
        initial_base_version_id="base-a",
        launch_checkpoint=launch,
    )
    assert object_storage.root_uri(42) == (
        "s3://weights/tests/v42/model.safetensors.index.json"
    )
    with pytest.raises(ValueError, match="WRITE_TO_STORAGE and XOR_DELTA"):
        ModelExpressTrainerClient.initialize(
            ModelExpressTrainerConfig(
                model_name="test/model",
                staging_mode=TrainerStagingMode.IN_PLACE,
                payload_format=WeightPayloadFormat.XOR_DELTA,
                object_storage=object_storage,
            )
        )


def test_object_storage_config_rejects_unsupported_provider(tmp_path):
    launch = tmp_path / "model.safetensors"
    _write_safetensors(launch, {"weight": torch.tensor([1.0])})

    with pytest.raises(ValueError, match="only S3 object storage"):
        ModelExpressTrainerClient.initialize(
            ModelExpressTrainerConfig(
                model_name="test/model",
                staging_mode=TrainerStagingMode.WRITE_TO_STORAGE,
                payload_format=WeightPayloadFormat.XOR_DELTA,
                object_storage=ObjectStorageConfig(
                    storage_type=ObjectStorageType.GCS,
                    uri_prefix="gs://weights/tests",
                    initial_base_version_id="base-a",
                    launch_checkpoint=launch,
                ),
            )
        )
