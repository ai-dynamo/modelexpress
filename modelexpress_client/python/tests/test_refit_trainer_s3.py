# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
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
    S3Config,
    TrainerStagingMode,
    WeightPayloadFormat,
    WeightVersionRef,
    refit_pb2,
    refit_pb2_grpc,
)
from modelexpress_rl.s3 import S3Object
from modelexpress_rl.train import client as trainer_client_module


class _MemoryS3:
    def __init__(self) -> None:
        self.objects = {}
        self.fail_next = False

    def put(self, *, bucket, key, data):
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("injected upload failure")
        existing = self.objects.setdefault((bucket, key), data)
        if existing != data:
            raise RuntimeError("immutable object differs")
        return S3Object(
            bucket=bucket,
            key=key,
            checksum=f"crc32c:{len(data):08x}",
        )

    def close(self):
        pass


class _RefitService(refit_pb2_grpc.RefitServiceServicer):
    def __init__(self) -> None:
        self.registrations = set()
        self.shards = []
        self.deleted_shards = []
        self.fail_delete_response_once = False
        self.target = refit_pb2.WeightVersion(
            uid="target-a",
            model_name="test/model",
            payload_format=refit_pb2.WEIGHT_PAYLOAD_FORMAT_XOR_DELTA,
            base_version_id="base-a",
            expected_source_slots=["canonical.delta.root"],
            state=refit_pb2.WEIGHT_VERSION_STATE_STAGING,
        )

    def RegisterWorker(self, request, _context):
        self.registrations.add(request.worker.worker_id)
        return request.worker

    def GetWeightVersion(self, request, context):
        if request.uid != self.target.uid:
            context.abort(grpc.StatusCode.NOT_FOUND, "version not found")
        return self.target

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
        if self.fail_delete_response_once:
            self.fail_delete_response_once = False
            context.abort(grpc.StatusCode.UNAVAILABLE, "delete response lost")
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
            s3=S3Config(
                bucket="weights",
                initial_base_version_id="base-a",
                launch_checkpoint=launch,
                prefix="tests",
            ),
        )
    )
    if prepare_base:
        trainer.prepare_delta_base(
            hf_tensor_iter=iter([list(launch_tensors.items())]),
        )
    return trainer, storage


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


def test_s3_stage_is_local_then_publish_advertises_one_canonical_root(
    monkeypatch, tmp_path, refit_server
):
    service, server_url = refit_server
    trainer, storage = _trainer(monkeypatch, tmp_path, server_url)
    current = torch.tensor([1.0, 3.0])

    try:
        assert trainer.source_slot_id == "canonical.delta.root"
        staged = trainer.stage_shard(
            version=WeightVersionRef("target-a"),
            hf_tensor_iter=iter([[("weight", current)]]),
        )
        assert storage.objects == {}

        staged.publish()
        staged.publish()
    finally:
        trainer.close()

    assert len(service.shards) == 1
    shard = service.shards[0]
    assert shard.source_slot_id == "canonical.delta.root"
    assert shard.WhichOneof("transport") == "s3"
    assert shard.s3.bucket == "weights"
    assert shard.s3.key.endswith("/model.safetensors.index.json")
    index = json.loads(storage.objects[("weights", shard.s3.key)])
    assert index["metadata"] == {
        "base_version": "base-a",
        "checksum_format": "adler32",
        "compression_format": "zstd",
        "delta_encoding": "xor",
        "version": "target-a",
    }
    assert index["weight_map"] == {"weight": "model-00000-of-00001.safetensors"}
    removed_digests = {"base_digest", "target_digest", "format_digest"}
    assert removed_digests.isdisjoint(index)
    shard_key = next(
        key for bucket, key in storage.objects if key.endswith(".safetensors")
    )
    blob = storage.objects[("weights", shard_key)]
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
    assert (
        shard.manifest_digest
        == hashlib.sha256(storage.objects[("weights", shard.s3.key)]).hexdigest()
    )


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

    assert len(service.shards) == 1


def test_s3_processes_buckets_concurrently_and_uploads_one_shard(
    monkeypatch, tmp_path, refit_server
):
    service, server_url = refit_server
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
        key.rsplit("/", 1)[-1]
        for bucket, key in storage.objects
        if bucket == "weights" and key.endswith(".safetensors")
    )
    assert filenames == ["model-00000-of-00001.safetensors"]
    root = service.shards[0]
    index = json.loads(storage.objects[("weights", root.s3.key)])
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


def test_s3_collects_byte_metrics_after_publication(
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
            for (_bucket, key), data in storage.objects.items()
            if key.endswith(".safetensors")
        )
        assert staged._staged.wire_bytes == len(shard)

        reductions = []
        process_group = object()
        trainer._process_group = process_group

        def all_reduce(value, op=None, group=None):
            reductions.append((value.tolist(), op, group))

        monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
        trainer.collect_metrics()
        trainer.collect_metrics()
        assert reductions == [
            (
                [
                    int(torch.count_nonzero(expected_delta)),
                    expected_delta.numel(),
                    len(shard),
                ],
                None,
                process_group,
            ),
            (
                [2.0, 3.0, 4.0],
                torch.distributed.ReduceOp.MAX,
                process_group,
            ),
        ]
        assert trainer.pop_metrics() == {
            "perf/update_weights_density": int(torch.count_nonzero(expected_delta))
            / expected_delta.numel(),
            "perf/update_weights_wire_bytes": len(shard),
            "perf/mx_stage_delta_time": 2.0,
            "perf/mx_publish_s3_time": 3.0,
            "perf/mx_publish_server": 4.0,
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
        trainer.collect_metrics()
        metrics = trainer.pop_metrics()
    finally:
        trainer.close()

    assert len(storage.objects) == 1
    shard = service.shards[0]
    index = json.loads(storage.objects[("weights", shard.s3.key)])
    assert index["weight_map"] == {}
    assert metrics["perf/update_weights_density"] == 0.0
    assert metrics["perf/update_weights_wire_bytes"] == 0
    assert metrics["perf/mx_stage_delta_time"] >= 0
    assert metrics["perf/mx_publish_s3_time"] >= 0
    assert metrics["perf/mx_publish_server"] >= 0


def test_s3_chains_from_published_base_and_releases_previous(
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

        service.target = refit_pb2.WeightVersion(
            uid="target-b",
            model_name="test/model",
            payload_format=refit_pb2.WEIGHT_PAYLOAD_FORMAT_XOR_DELTA,
            base_version_id="target-a",
            expected_source_slots=["canonical.delta.root"],
            state=refit_pb2.WEIGHT_VERSION_STATE_STAGING,
        )
        second = trainer.stage_shard(
            version=WeightVersionRef("target-b"),
            hf_tensor_iter=iter([[("weight", torch.tensor([2.0, 4.0]))]]),
        )
        second.publish()
        assert second._staged.encoded_deltas == {}
        assert second._staged.checksums == {}
        trainer.release_version(version=WeightVersionRef("target-a"))
    finally:
        trainer.close()

    second_shard = next(
        data
        for (bucket, key), data in storage.objects.items()
        if bucket == "weights" and "target-b" in key and key.endswith(".safetensors")
    )
    encoded = safetensors.numpy.load(second_shard)["weight"]
    decoded = zstandard.ZstdDecompressor().decompress(encoded)
    expected = torch.bitwise_xor(
        torch.tensor([1.0, 3.0]).view(torch.uint8),
        torch.tensor([2.0, 4.0]).view(torch.uint8),
    )
    assert decoded == expected.numpy().tobytes()
    assert service.deleted_shards[0].version_id == "target-a"


def test_s3_release_retry_accepts_not_found_after_lost_delete_response(
    monkeypatch, tmp_path, refit_server
):
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
            payload_format=refit_pb2.WEIGHT_PAYLOAD_FORMAT_XOR_DELTA,
            base_version_id="target-a",
            expected_source_slots=["canonical.delta.root"],
            state=refit_pb2.WEIGHT_VERSION_STATE_STAGING,
        )
        trainer.stage_shard(
            version=WeightVersionRef("target-b"),
            hf_tensor_iter=iter([[("weight", torch.tensor([2.0, 4.0]))]]),
        ).publish()

        service.fail_delete_response_once = True
        with pytest.raises(grpc.RpcError) as raised:
            trainer.release_version(version=WeightVersionRef("target-a"))
        assert raised.value.code() == grpc.StatusCode.UNAVAILABLE
        assert raised.value.details() == "delete response lost"
        assert "target-a" in trainer._published_shards

        trainer.release_version(version=WeightVersionRef("target-a"))
        assert "target-a" not in trainer._published_shards
    finally:
        trainer.close()

    assert len(service.deleted_shards) == 1


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


def test_s3_config_requires_storage_delta_pair(tmp_path):
    launch = tmp_path / "model.safetensors"
    _write_safetensors(launch, {"weight": torch.tensor([1.0])})
    s3 = S3Config(
        bucket="weights",
        initial_base_version_id="base-a",
        launch_checkpoint=launch,
    )
    with pytest.raises(ValueError, match="WRITE_TO_STORAGE and XOR_DELTA"):
        ModelExpressTrainerClient.initialize(
            ModelExpressTrainerConfig(
                model_name="test/model",
                staging_mode=TrainerStagingMode.IN_PLACE,
                payload_format=WeightPayloadFormat.XOR_DELTA,
                s3=s3,
            )
        )
