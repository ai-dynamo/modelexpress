# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Delta snapshots in a fake S3, host checkpoint caches, staged deltas, and fake modules."""

import json
import sys
from types import ModuleType

import numpy as np
import pytest
import safetensors.numpy
import torch
from modelexpress_rl import (
    ObjectStorageGeneratorConfig,
    ObjectStorageType,
    WeightPayloadFormat,
)
from modelexpress_rl.inference import receiver as receiver_module
from modelexpress_rl.inference.staging import (
    StagedDelta,
    join_nonzero,
    nonzero_bytes,
    write_staging,
)
from modelexpress_rl.utils import checksum_factory, compress_delta, compute_delta
from safetensors.torch import load_file, save_file

PREFIX = "s3://weights/hotload"
INDEX = "model.safetensors.index.json"
SHARD = "model-00000-of-00001.safetensors"

# Only the C++ checker reads a tensor's changed bytes alone.
_needs_native_check = pytest.mark.skipif(
    getattr(receiver_module, "_delta_ext", None) is None,
    reason="built without the C++ delta checker",
)


def _bytes(values):
    return torch.tensor(values).contiguous().reshape(-1).view(torch.uint8).numpy()


def _checksum(value):
    checksum = checksum_factory("adler32")
    checksum.update(value)
    return checksum.hexdigest()


def _uri(identity):
    return f"{PREFIX}/{identity}/{INDEX}"


def _tensors_delta_snapshot(base, target, *, identity, base_identity, version=None):
    """The objects of an XOR delta from ``base`` to ``target``, both ``{name: values}``.

    Tensors that do not change are left out.
    """
    tensors, checksums = {}, {}
    for name, values in target.items():
        delta, changed = compute_delta(_bytes(values), _bytes(base[name]))
        if changed:
            tensors[name] = compress_delta(delta)
            checksums[name] = _checksum(_bytes(values))
    shard = safetensors.numpy.save(tensors, metadata=checksums)
    index = json.dumps(
        {
            "metadata": {
                "version": version or identity,
                "base_version": base_identity,
                "delta_encoding": "xor",
                "compression_format": "zstd",
                "checksum_format": "adler32",
            },
            "weight_map": {name: SHARD for name in tensors},
        }
    ).encode()
    return {_uri(identity): index, f"{PREFIX}/{identity}/{SHARD}": shard}


def _delta_snapshot(base, target, *, identity, base_identity, version=None):
    """The objects of a one-tensor XOR delta from ``base`` to ``target``."""
    return _tensors_delta_snapshot(
        {"weight": base},
        {"weight": target},
        identity=identity,
        base_identity=base_identity,
        version=version,
    )


def _delta(version_id, *, base, identity):
    return receiver_module._ObjectStorageVersion(
        version_id=version_id,
        base_version_id=base,
        payload_format=WeightPayloadFormat.XOR_DELTA,
        uri=_uri(identity),
    )


class _FakeS3:
    def __init__(self, objects):
        self.objects = dict(objects)
        self.gets = []

    def get(self, uri):
        self.gets.append(uri)
        return self.objects[uri]

    def size(self, uri):
        return len(self.objects[uri])

    def exists(self, uri):
        return uri in self.objects

    def list_children(self, prefix_uri, *, start_after=None):
        prefix = prefix_uri.rstrip("/") + "/"
        names = {
            key[len(prefix) :].split("/", 1)[0]
            for key in self.objects
            if key.startswith(prefix) and "/" in key[len(prefix) :]
        }
        return sorted(
            name for name in names if start_after is None or name > start_after
        )

    def close(self):
        pass


def _save(path, tensors):
    """Write ``tensors`` as the checkpoint in ``path``."""
    path.mkdir(parents=True, exist_ok=True)
    save_file(
        {name: tensor.contiguous() for name, tensor in tensors.items()},
        path / "model.safetensors",
    )
    return path


def _config(tmp_path, tensors=None):
    """A cache config seeded from a launch checkpoint of ``tensors``, ``{name: values}``."""
    launch = tmp_path / "launch"
    if not launch.exists():
        tensors = tensors or {"weight": [1.0, 2.0]}
        _save(launch, {name: torch.tensor(values) for name, values in tensors.items()})
    return ObjectStorageGeneratorConfig(
        storage_type=ObjectStorageType.S3,
        initial_base_version_id="base",
        seed_checkpoint_path=launch,
        refit_checkpoint_dir=tmp_path / "cache",
        refit_checkpoint_max_size_gb=1,
    )


def _checkpoint(tmp_path, s3, *, initialize=True, tensors=None, stage_deltas=False):
    checkpoint = receiver_module._LocalCheckpoint(
        model_name="test/model",
        config=_config(tmp_path, tensors),
        reader=s3,
        stage_deltas=stage_deltas,
    )
    if initialize:
        checkpoint.initialize()
    return checkpoint


def _files(checkpoint):
    """The tensors the cache head's checkpoint files hold."""
    state = checkpoint.store.state()
    path = checkpoint.store.checkpoint_path(state.files_version) / "model.safetensors"
    return {name: tensor.tolist() for name, tensor in load_file(path).items()}


def _sparse_bytes(rng, size, density):
    """Random bytes, each nonzero with probability ``density``."""
    return np.where(rng.random(size) < density, rng.integers(1, 256, size), 0).astype(
        np.uint8
    )


def _staged_scene(directory, rng, sizes, *, density):
    """Random checkpoint files of tensors of ``sizes`` bytes, and two deltas staged on them.

    Returns a checkpoint that reads the files, ``directory / "model.bin"``, each
    tensor's bytes after the staged deltas, and the staged deltas.
    """
    path = directory / "model.bin"
    files = rng.integers(0, 256, sum(sizes.values()), dtype=np.uint8)
    files.tofile(path)
    locations, offset = {}, 0
    for name, size in sizes.items():
        locations[name] = (path, offset, size)
        offset += size
    head = {
        name: files[start : start + size].copy()
        for name, (_p, start, size) in locations.items()
    }
    layers = []
    for index in range(2):
        tensors = {}
        for name, size in sizes.items():
            xor = _sparse_bytes(rng, size, density)
            head[name] ^= xor
            tensors[name] = (size, *join_nonzero(size, [nonzero_bytes(xor, 0)]))
        staging = directory / f"layer{index}"
        staging.mkdir()
        write_staging(staging, base_version=f"v{index}", tensors=tensors)
        layers.append(StagedDelta(staging))
    checkpoint = receiver_module._LocalCheckpoint.__new__(
        receiver_module._LocalCheckpoint
    )
    checkpoint.locations = locations
    return checkpoint, head, layers


def _check_passes(head, layers, delta, bases, frames=None):
    """Passes that check each tensor's ``delta`` on ``head``, the files under the staged ``layers``.

    ``bases`` holds the checksums of ``head`` that the pass knows.
    """
    frames = frames or {
        name: compress_delta(xor).tobytes() for name, xor in delta.items()
    }
    return [
        receiver_module._TensorPass(
            name,
            check=memoryview(frames[name]),
            checksum=_checksum(head[name] ^ delta[name]),
            overlay=tuple(layer.tensor(name) for layer in layers),
            base_checksum=bases.get(name),
        )
        for name in delta
    ]


def _fake_modules(monkeypatch, modules):
    """Put modules of ``{name: attributes}`` and empty parent packages in ``sys.modules``."""
    parents = set()
    for name in modules:
        parts = name.split(".")
        parents.update(".".join(parts[:end]) for end in range(1, len(parts)))
    for name in parents - modules.keys():
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    for name, attributes in modules.items():
        module = ModuleType(name)
        vars(module).update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
