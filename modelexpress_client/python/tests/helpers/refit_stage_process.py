# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""File-backed object storage and synthetic checkpoints for stage-process tests."""

import hashlib
import json
import os
import time
import uuid
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np

# Directory that stands in for object storage, in the tests and in workers.
READER_ROOT_ENV = "MX_TEST_STAGE_READER_ROOT"
# Set to make a worker started by the tests never become ready.
STALL_ENV = "MX_TEST_STAGE_WORKER_STALL"
INDEX_NAME = "model.safetensors.index.json"
# Longest a held read waits, so an aborted test cannot leave a worker behind.
_HOLD_LIMIT_SECONDS = 60.0


def object_path(root, uri: str) -> Path:
    assert uri.startswith("s3://"), uri
    return Path(root) / "objects" / uri[len("s3://") :]


class FileReader:
    """ObjectStorageReader over files, shared by processes through the filesystem.

    Every read is appended to ``<root>/reads.log`` as ``<pid> <kind> <uri>``.
    While ``<root>/hold`` exists, a shard read creates ``<root>/holding`` and
    blocks: the caller is then in the middle of a preparation. ``fail_reads``
    makes chosen reads fail.
    """

    def __init__(self, root) -> None:
        self.root = Path(root)
        self.reads: list[tuple[str, str]] = []

    def _record(self, kind: str, uri: str) -> None:
        self.reads.append((kind, uri))
        with open(self.root / "reads.log", "a") as log:
            log.write(f"{os.getpid()} {kind} {uri}\n")

    def get(self, uri: str) -> bytes:
        self._record("get", uri)
        hold = self.root / "hold"
        if uri.endswith(".safetensors") and hold.exists():
            (self.root / "holding").touch()
            deadline = time.monotonic() + _HOLD_LIMIT_SECONDS
            while hold.exists():
                if time.monotonic() > deadline:
                    raise TimeoutError(f"read of {uri} was held too long")
                time.sleep(0.01)
        fault = self._take_fault(uri)
        if fault is None:
            return object_path(self.root, uri).read_bytes()
        if fault["remove"] is not None:
            os.unlink(fault["remove"])
        if fault["corrupt"]:
            data = object_path(self.root, uri).read_bytes()
            return data[:-1] + bytes([data[-1] ^ 0xFF])
        raise OSError(f"injected failure reading {uri}")

    def _take_fault(self, uri: str) -> dict | None:
        """Consume one fault that ``fail_reads`` left for ``uri``, if any."""
        faults = self.root / "faults"
        if not faults.is_dir():
            return None
        for token in sorted(faults.iterdir()):
            try:
                fault = json.loads(token.read_text())
                if fault["uri"] not in uri:
                    continue
                # Removing the token claims it; readers race for it.
                token.unlink()
            except (OSError, ValueError):
                continue
            return fault
        return None

    def size(self, uri: str) -> int:
        self._record("size", uri)
        return object_path(self.root, uri).stat().st_size

    def close(self) -> None:
        pass


def fail_reads(
    root, uri_part: str, *, times: int = 1, corrupt: bool = False, remove=None
) -> None:
    """Make the next ``times`` reads of a URI that contains ``uri_part`` fail.

    A failed read raises. With ``corrupt`` it returns the object with its last
    byte changed instead, which a reader notices only when it decodes it. With
    ``remove`` it first deletes that file, as another process might meanwhile.
    """
    faults = Path(root) / "faults"
    faults.mkdir(exist_ok=True)
    fault = {
        "uri": uri_part,
        "corrupt": corrupt,
        "remove": None if remove is None else str(remove),
    }
    for _ in range(times):
        (faults / uuid.uuid4().hex).write_text(json.dumps(fault))


def reads_by_pid(root) -> dict[int, list[tuple[str, str]]]:
    """Return every logged read, grouped by the process that made it."""
    reads: dict[int, list[tuple[str, str]]] = defaultdict(list)
    log = Path(root) / "reads.log"
    if log.exists():
        for line in log.read_text().splitlines():
            pid, kind, uri = line.split(" ", 2)
            reads[int(pid)].append((kind, uri))
    return reads


def _adler32(array: np.ndarray) -> str:
    data = np.ascontiguousarray(array).view(np.uint8).reshape(-1)
    return f"{zlib.adler32(data):08x}"


def digest_tensors(tensors: dict[str, np.ndarray]) -> str:
    sha = hashlib.sha256()
    for name in sorted(tensors):
        sha.update(name.encode())
        sha.update(np.ascontiguousarray(tensors[name]).view(np.uint8).tobytes())
    return sha.hexdigest()


def digest_checkpoint(path) -> str:
    import safetensors.numpy

    tensors = {}
    for file in sorted(Path(path).glob("*.safetensors")):
        tensors.update(safetensors.numpy.load_file(str(file)))
    return digest_tensors(tensors)


def layout(shards: int = 2, tensors_per_shard: int = 2) -> dict[str, str]:
    weight_map = {}
    for shard in range(shards):
        filename = f"model-{shard + 1:05d}-of-{shards:05d}.safetensors"
        for index in range(tensors_per_shard):
            weight_map[f"layers.{shard}.w{index}"] = filename
    return weight_map


def mutate(tensors: dict[str, np.ndarray], rng) -> dict[str, np.ndarray]:
    """Return a copy with every other tensor perturbed in a few places."""
    changed = {name: value.copy() for name, value in tensors.items()}
    for name in sorted(changed)[::2]:
        flat = changed[name].reshape(-1)
        where = rng.choice(flat.size, size=max(1, flat.size // 16), replace=False)
        flat[where] += rng.standard_normal(where.size).astype(flat.dtype)
    return changed


def write_checkpoint(directory, tensors, weight_map, *, checksums: bool) -> None:
    import safetensors.numpy

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    by_shard = defaultdict(dict)
    for name, filename in weight_map.items():
        by_shard[filename][name] = tensors[name]
    for filename, shard in by_shard.items():
        metadata = (
            {name: _adler32(value) for name, value in shard.items()}
            if checksums
            else None
        )
        safetensors.numpy.save_file(shard, str(directory / filename), metadata=metadata)
    index_metadata = {"total_size": sum(int(t.nbytes) for t in tensors.values())}
    if checksums:
        index_metadata["checksum_format"] = "adler32"
    (directory / INDEX_NAME).write_text(
        json.dumps({"metadata": index_metadata, "weight_map": weight_map})
    )


def publish_full(root, prefix: str, tensors, weight_map) -> str:
    """Write a full checkpoint under ``s3://<prefix>`` and return its index URI."""
    uri = f"s3://{prefix}/{INDEX_NAME}"
    write_checkpoint(object_path(root, uri).parent, tensors, weight_map, checksums=True)
    return uri


def publish_delta(root, prefix: str, base, target, weight_map) -> str:
    """Write a canonical XOR delta under ``s3://<prefix>`` and return its index URI."""
    import safetensors.numpy

    from modelexpress_rl.utils import compress_delta, compute_delta

    uri = f"s3://{prefix}/{INDEX_NAME}"
    directory = object_path(root, uri).parent
    directory.mkdir(parents=True, exist_ok=True)
    shards = defaultdict(dict)
    checksums = defaultdict(dict)
    delta_map = {}
    for name, filename in weight_map.items():
        new = np.ascontiguousarray(target[name]).view(np.uint8).reshape(-1)
        old = np.ascontiguousarray(base[name]).view(np.uint8).reshape(-1)
        delta, _changed = compute_delta(new, old)
        if delta is None:
            continue
        shards[filename][name] = compress_delta(delta)
        checksums[filename][name] = _adler32(new)
        delta_map[name] = filename
    for filename, shard in shards.items():
        safetensors.numpy.save_file(
            shard, str(directory / filename), metadata=checksums[filename]
        )
    (directory / INDEX_NAME).write_text(
        json.dumps(
            {
                "metadata": {
                    "delta_encoding": "xor",
                    "compression_format": "zstd",
                    "checksum_format": "adler32",
                },
                "weight_map": delta_map,
            }
        )
    )
    return uri
