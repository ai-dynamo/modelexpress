# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import hashlib
import json
import threading

import google_crc32c
import pytest
import safetensors.numpy
import torch
import zstandard
from safetensors.torch import load_file, save_file

from modelexpress_rl import S3GeneratorConfig, WeightPayloadFormat
from modelexpress_rl.inference.adapter import (
    GeneratorSource,
    GeneratorTransferInputs,
    S3GeneratorSource,
)
from modelexpress_rl.inference import receiver as receiver_module
from modelexpress_rl.inference.receiver import (
    CANONICAL_DELTA_SOURCE_SLOT,
    CanonicalS3GeneratorAdapter,
)
from modelexpress_rl.s3 import S3Client, S3Object
from modelexpress_rl.utils import adler32_checksum, compress_delta, compute_delta


class _MemoryS3:
    def __init__(self, objects):
        self.objects = objects
        self.calls = []
        self.fail_key_once = None

    def get(self, location):
        self.calls.append(location.key)
        data = self.objects[location.key]
        assert location.checksum == _crc32c(data)
        return data

    def get_key(self, *, bucket, key):
        assert bucket == "weights"
        self.calls.append(key)
        if key == self.fail_key_once:
            self.fail_key_once = None
            raise RuntimeError("injected shard download failure")
        return self.objects[key]

    def close(self):
        pass


class _Adapter(CanonicalS3GeneratorAdapter):
    def __init__(self, **kwargs):
        self.installed = []
        super().__init__(**kwargs)

    def install_prepared_checkpoint(self, prepared):
        self.installed.append(prepared.path)


def _crc32c(data):
    return f"crc32c:{google_crc32c.value(data):08x}"


def test_s3_read_honors_object_version_and_verifies_crc32c():
    data = b"canonical-root"

    class Body:
        def read(self):
            return data

        def close(self):
            pass

    class Client:
        def __init__(self):
            self.request = None

        def get_object(self, **request):
            self.request = request
            return {"Body": Body(), "ChecksumCRC32C": _encoded_crc32c(data)}

    backend = Client()
    s3 = object.__new__(S3Client)
    s3._client = backend
    location = S3Object(
        bucket="weights",
        key="root.json",
        checksum=_crc32c(data),
        object_version="object-a",
    )

    assert s3.get(location) == data
    assert backend.request == {
        "Bucket": "weights",
        "Key": "root.json",
        "VersionId": "object-a",
        "ChecksumMode": "ENABLED",
    }
    with pytest.raises(ValueError, match="S3 checksum mismatch"):
        s3.get(
            S3Object(
                bucket="weights",
                key="root.json",
                checksum="crc32c:00000000",
            )
        )


def _encoded_crc32c(data):
    value = google_crc32c.value(data)
    return base64.b64encode(value.to_bytes(4, "big")).decode()


def _artifact(
    base, target, *, checksum=None, version="target-a", base_version="base-a"
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
    prefix = f"test/weights_v{version}"
    return {
        f"{prefix}/model.safetensors.index.json": root,
        f"{prefix}/model-00000-of-00001.safetensors": shard,
    }


def _inputs(
    root,
    *,
    digest=None,
    base_version="base-a",
    version="target-a",
    key=None,
):
    if key is None:
        key = f"test/weights_v{version}/model.safetensors.index.json"
    return GeneratorTransferInputs(
        version_id=version,
        base_version_id=base_version,
        layout_signature="",
        payload_format=WeightPayloadFormat.XOR_DELTA,
        sources=(
            GeneratorSource(
                source_slot_id=CANONICAL_DELTA_SOURCE_SLOT,
                worker_id="trainer-0",
                manifest_digest=digest or hashlib.sha256(root).hexdigest(),
                transport=S3GeneratorSource(
                    S3Object(
                        bucket="weights",
                        key=key,
                        checksum=_crc32c(root),
                    )
                ),
            ),
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
        config=S3GeneratorConfig(
            initial_base_version_id="base-a",
            launch_checkpoint=launch,
            preparation_cache_dir=tmp_path / "cache",
        ),
    )
    return adapter, storage


def test_canonical_s3_prepares_then_installs_one_global_index(monkeypatch, tmp_path):
    base = torch.tensor([1.0, 2.0]).view(torch.uint8).numpy()
    target_tensor = torch.tensor([3.0, 4.0])
    target = target_tensor.view(torch.uint8).numpy()
    objects = _artifact(base, target)
    root = objects[
        "test/weights_vtarget-a/model.safetensors.index.json"
    ]
    adapter, storage = _build(monkeypatch, tmp_path, objects)

    staged = adapter.stage_weight(_inputs(root))

    assert adapter.installed == []
    assert torch.equal(
        load_file(staged.path / "model.safetensors")["weight"], target_tensor
    )
    assert storage.calls == [
        "test/weights_vtarget-a/model.safetensors.index.json",
        "test/weights_vtarget-a/model-00000-of-00001.safetensors",
    ]
    assert staged.metrics["perf/mx_receive_delta_download"] >= 0
    assert adapter.apply_weight(staged)["perf/mx_receive_install_time"] >= 0
    assert adapter.installed == [staged.path]
    adapter.release_staged_weight(staged)
    adapter.close()


def test_canonical_s3_accepts_an_empty_delta(monkeypatch, tmp_path):
    key = "test/weights_vtarget-a/model.safetensors.index.json"
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


def test_canonical_s3_uses_weight_map_without_index_validation(monkeypatch, tmp_path):
    base = torch.tensor([1.0, 2.0]).view(torch.uint8).numpy()
    target_tensor = torch.tensor([3.0, 4.0])
    objects = _artifact(base, target_tensor.view(torch.uint8).numpy())
    key = "test/weights_vtarget-a/model.safetensors.index.json"
    root = json.dumps(
        {"weight_map": {"weight": "model-00000-of-00001.safetensors"}}
    ).encode()
    objects[key] = root
    adapter, _storage = _build(monkeypatch, tmp_path, objects)

    staged = adapter.stage_weight(_inputs(root))

    assert torch.equal(
        load_file(staged.path / "model.safetensors")["weight"], target_tensor
    )


def test_canonical_s3_downloads_unique_shards_concurrently(monkeypatch, tmp_path):
    adapter, storage = _build(monkeypatch, tmp_path, {})
    barrier = threading.Barrier(2)

    def get_key(*, bucket, key):
        assert bucket == "weights"
        barrier.wait(timeout=2)
        return key.encode()

    storage.get_key = get_key
    root = S3Object(
        bucket="weights",
        key="prefix/model.safetensors.index.json",
        checksum="",
    )

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
            b"prefix/model-00000-of-00002.safetensors",
            ["weight_a", "weight_b"],
        ),
        "model-00001-of-00002.safetensors": (
            b"prefix/model-00001-of-00002.safetensors",
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
        config=S3GeneratorConfig(
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

    checkpoint._apply_shards(
        {"delta.safetensors": (shard, list(target))}
    )

    loaded = load_file(checkpoint.local_checkpoint / "model.safetensors")
    assert all(torch.equal(loaded[name], value) for name, value in target.items())
    assert sorted(read_counts.values()) == [2, 2]
    assert max(read_sizes) == 2 << 20
    assert all(0 < size <= 2 << 20 for size in read_sizes)


def test_manifest_mismatch_fails_before_checkpoint_mutation(monkeypatch, tmp_path):
    base = torch.tensor([1.0, 2.0]).view(torch.uint8).numpy()
    target = torch.tensor([3.0, 4.0]).view(torch.uint8).numpy()
    objects = _artifact(base, target)
    root = objects[
        "test/weights_vtarget-a/model.safetensors.index.json"
    ]
    adapter, _ = _build(monkeypatch, tmp_path, objects)

    with pytest.raises(RuntimeError, match="manifest digest mismatch"):
        adapter.stage_weight(_inputs(root, digest="0" * 64))

    state = json.loads(adapter._checkpoint.state_path.read_text())
    assert state["version"] == "base-a"


def test_reconstructed_checksum_failure_propagates(monkeypatch, tmp_path):
    base = torch.tensor([1.0, 2.0]).view(torch.uint8).numpy()
    target = torch.tensor([3.0, 4.0]).view(torch.uint8).numpy()
    objects = _artifact(base, target, checksum="00000000")
    root = objects[
        "test/weights_vtarget-a/model.safetensors.index.json"
    ]
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


def test_wrong_base_fails_before_s3_download(monkeypatch, tmp_path):
    base = torch.tensor([1.0, 2.0]).view(torch.uint8).numpy()
    target = torch.tensor([3.0, 4.0]).view(torch.uint8).numpy()
    objects = _artifact(base, target)
    root = objects[
        "test/weights_vtarget-a/model.safetensors.index.json"
    ]
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
    root_key = "test/weights_vtarget-a/model.safetensors.index.json"
    shard_key = (
        "test/weights_vtarget-a/model-00000-of-00001.safetensors"
    )
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
    root_key = "test/weights_vtarget-a/model.safetensors.index.json"
    shard_key = (
        "test/weights_vtarget-a/model-00000-of-00001.safetensors"
    )
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
    root_key = "test/weights_vtarget-a/model.safetensors.index.json"
    alternate_key = (
        "test/alternate/weights_vtarget-a/model.safetensors.index.json"
    )
    objects[alternate_key] = objects[root_key]
    adapter, _ = _build(monkeypatch, tmp_path, objects)
    staged = adapter.stage_weight(_inputs(objects[root_key]))
    adapter.apply_weight(staged)
    adapter.release_staged_weight(staged)

    with pytest.raises(RuntimeError, match="different canonical root"):
        adapter.stage_weight(_inputs(objects[alternate_key], key=alternate_key))


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
            base_version="target-a",
        )
    )
    adapter, _ = _build(monkeypatch, tmp_path, objects)
    first_root = objects[
        "test/weights_vtarget-a/model.safetensors.index.json"
    ]
    staged = adapter.stage_weight(_inputs(first_root))
    adapter.apply_weight(staged)
    adapter.release_staged_weight(staged)

    second_root = objects[
        "test/weights_vtarget-b/model.safetensors.index.json"
    ]
    staged = adapter.stage_weight(
        _inputs(
            second_root,
            version="target-b",
            base_version="target-a",
        )
    )

    assert torch.equal(
        load_file(staged.path / "model.safetensors")["weight"],
        second_tensor,
    )
