# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the NIXL OBJ (S3-over-RDMA) safetensors loader.

A fake NIXL agent copies object bytes into registered CPU tensors with
``ctypes.memmove``, so the full scheduler runs without GPUs, NIXL, or S3.
"""

from __future__ import annotations

import ctypes
import json
import random

import pytest
import torch
from safetensors.torch import load as st_load
from safetensors.torch import save as st_save

from modelexpress import s3_rdma_transfer
from modelexpress.s3_rdma_loader import (
    MxS3RdmaLoader,
    S3RdmaIntegrityError,
    S3RdmaLoadConfig,
    S3RdmaTransferTimeout,
    S3ObjectReader,
    TensorPlan,
    parse_s3_prefix,
    plan_windows,
    resolve_checkpoint,
)
from modelexpress.s3_rdma_transfer import S3RdmaBackendUnavailable, S3RdmaEndpoint, S3RdmaTransferError

BUCKET = "models"
ENDPOINT = S3RdmaEndpoint(
    bucket=BUCKET,
    endpoint_url="http://aistor:9100",
    region="us-east-1",
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeReader(S3ObjectReader):
    """In-memory S3 with top-level listing semantics (Delimiter='/')."""

    def __init__(self, objects: dict[str, bytes], *, change_etag_after_lists: int | None = None):
        super().__init__(client=None)
        self.objects = objects
        self.lists = 0
        self.range_calls = 0
        self._change_after = change_etag_after_lists

    def list_objects(self, bucket, prefix):
        assert bucket == BUCKET
        self.lists += 1
        changed = self._change_after is not None and self.lists > self._change_after
        return {
            key: (len(data), "changed" if changed else f"etag-{len(data)}")
            for key, data in self.objects.items()
            if key.startswith(prefix) and "/" not in key[len(prefix):]
        }

    def get_object(self, bucket, key):
        return self.objects[key]

    def get_range(self, bucket, key, start, end):
        self.range_calls += 1
        return self.objects[key][start:end]


class FakeHandle:
    def __init__(self, idx, addr, length, key, offset):
        self.idx = idx
        self.addr = addr
        self.length = length
        self.key = key
        self.offset = offset
        self.polls = 0
        self.polls_needed = 1
        self.released = False


class FakeS3RdmaAgent:
    """Loopback OBJ agent that validates descriptors like NIXL would."""

    def __init__(
        self,
        name,
        store,
        *,
        plugins=("OBJ",),
        mem_types=("DRAM_SEG", "VRAM_SEG", "OBJ_SEG"),
        create_error=None,
        vram_register_error=None,
        fail=(),
        short=(),
        skip=(),
        stuck=(),
        corrupt=(),
        seed=0,
        max_polls=4,
    ):
        self.name = name
        self.store = store
        self.plugins = list(plugins)
        self.mem_types = list(mem_types)
        self.create_error = create_error
        self.vram_register_error = vram_register_error
        self.fail, self.short, self.skip, self.stuck = set(fail), set(short), set(skip), set(stuck)
        self.corrupt = set(corrupt)
        self.rng = random.Random(seed)
        self.max_polls = max_polls
        self.backend_params = None
        self.obj_keys: dict[int, str] = {}
        self.slot_regs: dict[int, int] = {}
        self.inflight: set[FakeHandle] = set()
        self.handles: list[FakeHandle] = []
        self.posted = 0
        self.events: list[tuple] = []
        self.local_types: list[tuple[str, int]] = []

    def get_plugin_list(self):
        return self.plugins

    def create_backend(self, backend, params):
        assert backend == "OBJ"
        self.backend_params = dict(params)
        if self.create_error is not None:
            raise self.create_error

    def get_backend_mem_types(self, backend):
        return self.mem_types

    def register_memory(self, regions, mem_type, backends=()):
        assert list(backends) == ["OBJ"]
        assert all(len(region) == 4 for region in regions)
        if mem_type == "VRAM" and self.vram_register_error is not None:
            raise self.vram_register_error
        if mem_type == "OBJ":
            for offset, size, dev_id, key in regions:
                assert offset == 0
                assert dev_id not in self.obj_keys, "duplicate OBJ devId"
                assert len(self.store[key]) == size
                self.obj_keys[dev_id] = key
        else:
            for addr, length, _dev, meta in regions:
                assert meta == ""
                assert addr not in self.slot_regs
                self.slot_regs[addr] = length
        self.events.append(("register", mem_type, len(regions)))
        return (mem_type, tuple(regions))

    def deregister_memory(self, reg, backends=()):
        assert not self.inflight, "memory deregistered while reads are in flight"
        mem_type, regions = reg
        if mem_type == "OBJ":
            for region in regions:
                del self.obj_keys[region[2]]
        else:
            for region in regions:
                del self.slot_regs[region[0]]
        self.events.append(("deregister", mem_type))

    def get_xfer_descs(self, descs, mem_type):
        return (mem_type, list(descs))

    def initialize_xfer(self, op, local, remote, remote_agent, backends=()):
        assert op == "READ" and remote_agent == self.name and list(backends) == ["OBJ"]
        (local_type, local_descs), (remote_type, remote_descs) = local, remote
        assert local_type in ("VRAM", "DRAM") and remote_type == "OBJ"
        assert len(local_descs) == 1 and len(remote_descs) == 1
        (addr, length, local_dev), (offset, remote_len, dev_id) = local_descs[0], remote_descs[0]
        self.local_types.append((local_type, local_dev))
        assert length == remote_len
        assert addr in self.slot_regs, "local descriptor must start at a registration base"
        assert length <= self.slot_regs[addr]
        key = self.obj_keys[dev_id]
        assert offset + length <= len(self.store[key])
        handle = FakeHandle(self.posted, addr, length, key, offset)
        self.handles.append(handle)
        return handle

    def transfer(self, handle):
        self.posted += 1
        handle.polls_needed = self.rng.randint(1, self.max_polls)
        self.inflight.add(handle)
        return "PROC"

    def check_xfer_state(self, handle):
        if handle.idx in self.stuck:
            return "PROC"
        handle.polls += 1
        if handle.polls < handle.polls_needed:
            return "PROC"
        self.inflight.discard(handle)
        if handle.idx in self.fail:
            raise RuntimeError("nixlBackendError: injected failure")
        data = self.store[handle.key][handle.offset : handle.offset + handle.length]
        if handle.idx in self.short:
            data = data[: handle.length // 2]
        if handle.idx in self.corrupt:
            data = data[:-1] + bytes([data[-1] ^ 0xFF])
        if handle.idx not in self.skip and data:
            ctypes.memmove(handle.addr, data, len(data))
        return "DONE"

    def release_xfer_handle(self, handle):
        assert handle not in self.inflight, "released a running request"
        assert not handle.released, "released a request twice"
        handle.released = True

    def unreleased(self):
        return [handle for handle in self.handles if not handle.released]


def install_fake_nixl(monkeypatch):
    """Install FakeS3RdmaAgent as s3_rdma_transfer.NixlAgent; returns a configurator."""
    state = {"kwargs": {}, "agents": []}

    def factory(name, _config):
        agent = FakeS3RdmaAgent(name, state["store"], **state["kwargs"])
        state["agents"].append(agent)
        return agent

    monkeypatch.setattr(s3_rdma_transfer, "NIXL_AVAILABLE", True)
    monkeypatch.setattr(s3_rdma_transfer, "NixlAgent", factory)
    monkeypatch.setattr(s3_rdma_transfer, "NixlAgentConfig", lambda **kwargs: kwargs)
    monkeypatch.setattr(s3_rdma_transfer, "_QUARANTINE", [])

    def configure(store, **kwargs):
        state["store"] = store
        state["kwargs"] = kwargs
        return state

    return configure


@pytest.fixture
def fake_nixl(monkeypatch):
    return install_fake_nixl(monkeypatch)


def make_checkpoint(shards: list[dict[str, torch.Tensor]], prefix="llama/", index=True) -> dict[str, bytes]:
    objects = {}
    weight_map = {}
    for i, tensors in enumerate(shards):
        filename = f"model-{i + 1:05d}-of-{len(shards):05d}.safetensors" if index else "model.safetensors"
        objects[prefix + filename] = st_save(tensors)
        weight_map.update({name: filename for name in tensors})
    if index:
        objects[prefix + "model.safetensors.index.json"] = json.dumps({"weight_map": weight_map}).encode()
    objects[prefix + "config.json"] = b"{}"
    return objects


def sample_shards() -> list[dict[str, torch.Tensor]]:
    gen = torch.Generator().manual_seed(7)
    return [
        {
            "embed.weight": torch.randn(64, 33, generator=gen),
            "norm.weight": torch.randn(33, generator=gen).to(torch.bfloat16),
            "scalar": torch.tensor(3.5),
            "empty": torch.empty(0, 4),
        },
        {
            "proj.weight": torch.randn(17, 9, generator=gen).to(torch.float16),
            "mask": torch.rand(13, generator=gen) > 0.5,
            "ids": torch.randint(-100, 100, (11,), generator=gen, dtype=torch.int8),
            "fp8": torch.randn(5, 7, generator=gen).to(torch.float8_e4m3fn),
            "big": torch.randn(700, generator=gen),
        },
        {"tiny": torch.tensor([1], dtype=torch.int64)},
    ]


def as_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def reference(objects: dict[str, bytes]) -> dict[str, torch.Tensor]:
    expected = {}
    for key, data in objects.items():
        if key.endswith(".safetensors"):
            expected.update(st_load(data))
    return expected


def make_loader(
    backend, objects, *, window=64, inflight=2, verify="tail", timeout=5.0, reader=None, staging="auto"
):
    config = S3RdmaLoadConfig(
        window_bytes=window, inflight=inflight, verify=verify, timeout_s=timeout, staging=staging
    )
    return MxS3RdmaLoader(backend, 0, config, reader=reader or FakeReader(objects), endpoint=ENDPOINT)


def load_all(loader, uri="s3://models/llama"):
    loader.prepare(uri)
    try:
        return list(loader.iter_tensors())
    finally:
        loader.release()


# ---------------------------------------------------------------------------
# URI and checkpoint resolution
# ---------------------------------------------------------------------------


class TestParseS3Prefix:
    @pytest.mark.parametrize(
        ("uri", "expected"),
        [
            ("s3://models", ("models", "")),
            ("s3://models/", ("models", "")),
            ("s3://models/llama", ("models", "llama/")),
            ("S3://models/a/b/", ("models", "a/b/")),
        ],
    )
    def test_valid(self, uri, expected):
        assert parse_s3_prefix(uri) == expected

    @pytest.mark.parametrize("uri", ["gs://models/x", "s3:///x", "/local/path", "models/x"])
    def test_invalid(self, uri):
        with pytest.raises(ValueError):
            parse_s3_prefix(uri)


class TestResolveCheckpoint:
    def test_uses_index_weight_map(self):
        objects = make_checkpoint(sample_shards())
        objects["llama/extra.safetensors"] = st_save({"stray": torch.zeros(1)})
        shards = resolve_checkpoint(FakeReader(objects), BUCKET, "llama/")
        assert [shard.key for shard in shards] == [
            "llama/model-00001-of-00003.safetensors",
            "llama/model-00002-of-00003.safetensors",
            "llama/model-00003-of-00003.safetensors",
        ]

    def test_index_referencing_missing_object(self):
        objects = make_checkpoint(sample_shards())
        del objects["llama/model-00002-of-00003.safetensors"]
        with pytest.raises(FileNotFoundError, match="missing objects"):
            resolve_checkpoint(FakeReader(objects), BUCKET, "llama/")

    def test_single_file(self):
        objects = make_checkpoint([sample_shards()[0]], index=False)
        objects["llama/other.safetensors"] = st_save({"stray": torch.zeros(1)})
        shards = resolve_checkpoint(FakeReader(objects), BUCKET, "llama/")
        assert [shard.key for shard in shards] == ["llama/model.safetensors"]

    def test_listing_ignores_subdirectories(self):
        objects = {
            "llama/b.safetensors": st_save({"b": torch.zeros(2)}),
            "llama/a.safetensors": st_save({"a": torch.zeros(2)}),
            "llama/original/c.safetensors": st_save({"c": torch.zeros(2)}),
        }
        shards = resolve_checkpoint(FakeReader(objects), BUCKET, "llama/")
        assert [shard.key for shard in shards] == ["llama/a.safetensors", "llama/b.safetensors"]

    def test_no_safetensors(self):
        with pytest.raises(FileNotFoundError):
            resolve_checkpoint(FakeReader({"llama/config.json": b"{}"}), BUCKET, "llama/")

    def test_duplicate_tensor_names(self):
        objects = {
            "llama/a.safetensors": st_save({"w": torch.zeros(2)}),
            "llama/b.safetensors": st_save({"w": torch.zeros(2)}),
        }
        with pytest.raises(ValueError, match="appears in both"):
            resolve_checkpoint(FakeReader(objects), BUCKET, "llama/")

    def test_header_longer_than_probe(self, monkeypatch):
        monkeypatch.setattr("modelexpress.s3_rdma_loader._HEADER_PROBE_BYTES", 16)
        objects = make_checkpoint(sample_shards())
        reader = FakeReader(objects)
        shards = resolve_checkpoint(reader, BUCKET, "llama/")
        assert {entry.name for entry in shards[0].header.tensors} == set(sample_shards()[0])


# ---------------------------------------------------------------------------
# Window planner
# ---------------------------------------------------------------------------


def _random_layout(rng):
    tensors, sizes = [], []
    for obj_idx in range(rng.randint(1, 4)):
        offset = rng.randint(8, 40)
        for t in range(rng.randint(1, 8)):
            nbytes = rng.choice([0, 1, 3, rng.randint(1, 50), rng.randint(50, 400)])
            tensors.append(
                TensorPlan(f"o{obj_idx}.t{t}", torch.uint8, (nbytes,), nbytes, obj_idx, offset)
            )
            offset += nbytes + rng.choice([0, 0, 0, rng.randint(1, 5)])
        sizes.append(offset + rng.randint(0, 10))
    return tensors, sizes


class TestPlanWindows:
    @pytest.mark.parametrize("seed", range(40))
    def test_random_layouts_cover_each_tensor_once(self, seed):
        rng = random.Random(seed)
        tensors, sizes = _random_layout(rng)
        window = rng.choice([1, 2, 7, 16, 64, 1000])
        keep = None
        if seed % 3 == 0:
            skipped = {t.name for t in tensors if rng.random() < 0.3}
            keep = lambda t: t.name not in skipped  # noqa: E731
        windows = plan_windows(sizes, tensors, window, tensor_filter=keep)

        covered: dict[int, list[tuple[int, int]]] = {}
        for win in windows:
            assert 0 < win.length <= window
            assert win.offset + win.length <= sizes[win.obj_idx]
            for piece in win.pieces:
                tensor = tensors[piece.tensor_idx]
                assert tensor.obj_idx == win.obj_idx
                assert win.offset + piece.win_off == tensor.start + piece.dst_off
                assert piece.win_off + piece.nbytes <= win.length
                covered.setdefault(piece.tensor_idx, []).append((piece.dst_off, piece.nbytes))
        for idx, tensor in enumerate(tensors):
            wanted = tensor.nbytes > 0 and (keep is None or keep(tensor))
            if not wanted:
                assert idx not in covered
                continue
            spans = sorted(covered[idx])
            cursor = 0
            for dst_off, nbytes in spans:
                assert dst_off == cursor
                cursor += nbytes
            assert cursor == tensor.nbytes

    def test_last_window_of_object_is_short(self):
        tensors = [TensorPlan("a", torch.uint8, (10,), 10, 0, 8)]
        windows = plan_windows([18], tensors, 4)
        assert [(w.offset, w.length) for w in windows] == [(8, 4), (12, 4), (16, 2)]

    def test_windows_never_cross_objects(self):
        tensors = [
            TensorPlan("a", torch.uint8, (6,), 6, 0, 8),
            TensorPlan("b", torch.uint8, (6,), 6, 1, 8),
        ]
        windows = plan_windows([14, 14], tensors, 100)
        assert [(w.obj_idx, w.offset, w.length) for w in windows] == [(0, 8, 6), (1, 8, 6)]


# ---------------------------------------------------------------------------
# End-to-end loads through the fake agent
# ---------------------------------------------------------------------------


class TestLoad:
    @pytest.mark.parametrize("window", [7, 64, 1000, 1 << 20])
    @pytest.mark.parametrize("inflight", [1, 2, 4])
    @pytest.mark.parametrize("verify", ["tail", "off", "full"])
    def test_byte_exact(self, fake_nixl, mock_accelerator_backend_cls, window, inflight, verify):
        objects = make_checkpoint(sample_shards())
        fake_nixl(objects, seed=window * 31 + inflight)
        backend = mock_accelerator_backend_cls(s3_rdma=True)
        loaded = load_all(make_loader(backend, objects, window=window, inflight=inflight, verify=verify))

        expected = reference(objects)
        names = [name for name, _ in loaded]
        assert sorted(names) == sorted(expected)
        assert len(names) == len(set(names))
        for name, tensor in loaded:
            want = expected[name]
            assert tensor.dtype == want.dtype
            assert tensor.shape == want.shape
            assert as_bytes(tensor) == as_bytes(want)

    def test_yielded_tensors_do_not_alias_slots(self, fake_nixl, mock_accelerator_backend_cls):
        objects = make_checkpoint(sample_shards())
        fake_nixl(objects)
        loader = make_loader(mock_accelerator_backend_cls(s3_rdma=True), objects, window=32, inflight=2)
        loader.prepare("s3://models/llama")
        slot_ranges = [(s.data_ptr(), s.data_ptr() + s.numel()) for s in loader._slots]
        loaded = list(loader.iter_tensors())
        loader.release()
        bases = set()
        for _name, tensor in loaded:
            if tensor.numel() == 0:
                continue
            base = tensor.untyped_storage().data_ptr()
            assert base not in bases
            bases.add(base)
            assert not any(lo <= base < hi for lo, hi in slot_ranges)

    def test_backend_params_and_teardown(self, fake_nixl, mock_accelerator_backend_cls):
        objects = make_checkpoint(sample_shards())
        state = fake_nixl(objects)
        backend = mock_accelerator_backend_cls(s3_rdma=True)
        load_all(make_loader(backend, objects, inflight=3))
        agent = state["agents"][0]
        assert agent.backend_params["accelerated"] == "true"
        assert agent.backend_params["bucket"] == BUCKET
        assert agent.backend_params["num_threads"] == "3"
        assert agent.events[-2:] == [("deregister", "VRAM"), ("deregister", "OBJ")]
        assert not agent.slot_regs and not agent.obj_keys
        assert not agent.unreleased()
        assert backend.set_device_calls[0] == 0
        assert backend.empty_cache_calls == 1

    def test_tensor_larger_than_pool(self, fake_nixl, mock_accelerator_backend_cls):
        objects = make_checkpoint([{"huge": torch.arange(4096, dtype=torch.float32)}])
        fake_nixl(objects)
        loaded = dict(load_all(make_loader(mock_accelerator_backend_cls(s3_rdma=True), objects, window=100, inflight=2)))
        assert torch.equal(loaded["huge"], torch.arange(4096, dtype=torch.float32))

    def test_auto_staging_prefers_device_memory(self, fake_nixl, mock_accelerator_backend_cls):
        objects = make_checkpoint(sample_shards())
        state = fake_nixl(objects)
        load_all(make_loader(mock_accelerator_backend_cls(s3_rdma=True), objects))
        assert {local for local, _dev in state["agents"][0].local_types} == {"VRAM"}

    @pytest.mark.parametrize("staging", ["auto", "host"])
    def test_host_staging_is_byte_exact(self, fake_nixl, mock_accelerator_backend_cls, staging):
        objects = make_checkpoint(sample_shards())
        state = fake_nixl(objects, vram_register_error=RuntimeError("cuMemObjGetDescriptor failed"))
        backend = mock_accelerator_backend_cls(s3_rdma=True)
        loaded = load_all(make_loader(backend, objects, inflight=3, staging=staging))

        expected = reference(objects)
        assert sorted(name for name, _ in loaded) == sorted(expected)
        for name, tensor in loaded:
            assert tensor.device == backend.torch_device(0)
            assert as_bytes(tensor) == as_bytes(expected[name])
        agent = state["agents"][0]
        assert set(agent.local_types) == {("DRAM", 0)}
        assert ("register", "DRAM", 3) in agent.events
        assert agent.events[-2:] == [("deregister", "DRAM"), ("deregister", "OBJ")]
        assert not agent.slot_regs and not agent.unreleased()

    def test_host_staging_fences_copies_before_poisoning_a_slot(self, fake_nixl, mock_accelerator_backend_cls):
        objects = make_checkpoint(sample_shards())
        fake_nixl(objects)
        backend = mock_accelerator_backend_cls(s3_rdma=True)
        loader = make_loader(backend, objects, window=32, inflight=2, staging="host")
        events = []
        scatter, poison = loader._scatter, loader._poison_tail
        loader._scatter = lambda *a: events.append("scatter") or scatter(*a)
        loader._poison_tail = lambda *a: events.append("poison") or poison(*a)
        sync = backend.synchronize
        backend.synchronize = lambda *a: events.append("sync") or sync(*a)
        load_all(loader)
        assert "scatter" in events and "poison" in events[events.index("scatter") :]
        last_scatter = None
        for idx, event in enumerate(events):
            if event == "scatter":
                last_scatter = idx
            elif event == "poison" and last_scatter is not None:
                assert "sync" in events[last_scatter:idx]

    def test_device_staging_does_not_fall_back(self, fake_nixl, mock_accelerator_backend_cls):
        objects = make_checkpoint(sample_shards())
        state = fake_nixl(objects, vram_register_error=RuntimeError("cuMemObjGetDescriptor failed"))
        loader = make_loader(mock_accelerator_backend_cls(s3_rdma=True), objects, staging="device")
        with pytest.raises(S3RdmaTransferError, match="could not register VRAM staging slots"):
            loader.prepare("s3://models/llama")
        loader.release()
        assert not state["agents"][0].local_types


class TestFailures:
    def _loader(self, fake_nixl, backend, objects, agent_kwargs, **loader_kwargs):
        state = fake_nixl(objects, **agent_kwargs)
        return make_loader(backend, objects, **loader_kwargs), state

    def test_priming_failure_raises_from_prepare(self, fake_nixl, mock_accelerator_backend_cls):
        objects = make_checkpoint(sample_shards())
        loader, state = self._loader(fake_nixl, mock_accelerator_backend_cls(s3_rdma=True), objects, {"fail": {0}}, inflight=1)
        with pytest.raises(S3RdmaTransferError):
            loader.prepare("s3://models/llama")
        loader.release()
        assert not state["agents"][0].slot_regs
        assert not state["agents"][0].unreleased()

    def test_mid_stream_failure_drains_before_free(self, fake_nixl, mock_accelerator_backend_cls):
        objects = make_checkpoint(sample_shards())
        loader, state = self._loader(
            fake_nixl, mock_accelerator_backend_cls(s3_rdma=True), objects, {"fail": {5}}, window=16, inflight=3
        )
        loader.prepare("s3://models/llama")
        with pytest.raises(S3RdmaTransferError):
            list(loader.iter_tensors())
        assert not loader.finished
        assert not state["agents"][0].inflight
        loader.release()
        assert not state["agents"][0].unreleased()

    def test_tail_canary_catches_short_read(self, fake_nixl, mock_accelerator_backend_cls):
        objects = make_checkpoint(sample_shards())
        loader, _ = self._loader(
            fake_nixl, mock_accelerator_backend_cls(s3_rdma=True), objects, {"short": {3}}, window=16, inflight=2
        )
        with pytest.raises(S3RdmaIntegrityError, match="did not deliver"):
            load_all(loader)

    def test_tail_canary_catches_missing_write(self, fake_nixl, mock_accelerator_backend_cls):
        objects = make_checkpoint(sample_shards())
        loader, _ = self._loader(
            fake_nixl, mock_accelerator_backend_cls(s3_rdma=True), objects, {"skip": {2}}, window=16, inflight=2
        )
        with pytest.raises(S3RdmaIntegrityError):
            load_all(loader)

    def test_tail_canary_catches_corrupt_tail(self, fake_nixl, mock_accelerator_backend_cls):
        objects = make_checkpoint(sample_shards())
        loader, _ = self._loader(
            fake_nixl, mock_accelerator_backend_cls(s3_rdma=True), objects, {"corrupt": {1}}, window=16, inflight=2
        )
        with pytest.raises(S3RdmaIntegrityError):
            load_all(loader)

    def test_verify_off_does_not_detect_corruption(self, fake_nixl, mock_accelerator_backend_cls):
        objects = make_checkpoint(sample_shards())
        loader, _ = self._loader(
            fake_nixl,
            mock_accelerator_backend_cls(s3_rdma=True),
            objects,
            {"corrupt": {0}},
            window=1 << 20,
            inflight=1,
            verify="off",
        )
        loaded = dict(load_all(loader))
        expected = reference(objects)
        assert any(as_bytes(loaded[name]) != as_bytes(expected[name]) for name in loaded)

    def test_stuck_read_times_out_and_quarantines(self, fake_nixl, mock_accelerator_backend_cls):
        objects = make_checkpoint(sample_shards())
        loader, _ = self._loader(
            fake_nixl,
            mock_accelerator_backend_cls(s3_rdma=True),
            objects,
            {"stuck": {1}},
            window=16,
            inflight=2,
            timeout=0.05,
        )
        loader.prepare("s3://models/llama")
        with pytest.raises(S3RdmaTransferTimeout):
            list(loader.iter_tensors())
        loader.release()
        assert len(s3_rdma_transfer._QUARANTINE) == 1

    def test_early_close_drains(self, fake_nixl, mock_accelerator_backend_cls):
        objects = make_checkpoint(sample_shards())
        loader, state = self._loader(
            fake_nixl, mock_accelerator_backend_cls(s3_rdma=True), objects, {}, window=8, inflight=4
        )
        loader.prepare("s3://models/llama")
        weights = loader.iter_tensors()
        next(weights)
        next(weights)
        weights.close()
        assert not loader.finished
        assert not state["agents"][0].inflight
        loader.release()
        assert not state["agents"][0].unreleased()

    def test_object_changed_during_load(self, fake_nixl, mock_accelerator_backend_cls):
        objects = make_checkpoint(sample_shards())
        fake_nixl(objects)
        reader = FakeReader(objects, change_etag_after_lists=1)
        loader = make_loader(mock_accelerator_backend_cls(s3_rdma=True), objects, reader=reader)
        with pytest.raises(S3RdmaIntegrityError, match="changed while"):
            load_all(loader)

    @pytest.mark.parametrize(
        "agent_kwargs",
        [
            {"mem_types": ("DRAM_SEG", "OBJ_SEG")},
            {"plugins": ("UCX",)},
            {"create_error": RuntimeError("no RDMA device")},
        ],
    )
    def test_backend_unavailable(self, fake_nixl, mock_accelerator_backend_cls, agent_kwargs):
        objects = make_checkpoint(sample_shards())
        fake_nixl(objects, **agent_kwargs)
        reader = FakeReader(objects)
        loader = make_loader(mock_accelerator_backend_cls(s3_rdma=True), objects, reader=reader)
        with pytest.raises(S3RdmaBackendUnavailable):
            loader.prepare("s3://models/llama")
        loader.release()
        assert (reader.lists, reader.range_calls) == (0, 0)

    def test_bucket_mismatch(self, fake_nixl, mock_accelerator_backend_cls):
        objects = make_checkpoint(sample_shards())
        loader, _ = self._loader(fake_nixl, mock_accelerator_backend_cls(s3_rdma=True), objects, {})
        with pytest.raises(ValueError, match="bucket"):
            loader.prepare("s3://other/llama")


# ---------------------------------------------------------------------------
# Config and endpoint discovery
# ---------------------------------------------------------------------------


class TestS3RdmaLoadConfig:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"window_bytes": 0},
            {"inflight": 0},
            {"inflight": 65},
            {"threads": -1},
            {"verify": "sometimes"},
            {"staging": "pinned"},
        ],
    )
    def test_rejects_invalid(self, kwargs):
        with pytest.raises(ValueError):
            S3RdmaLoadConfig(**kwargs)

    def test_accepts_windows_past_4_gib(self):
        assert S3RdmaLoadConfig(window_bytes=8 * 1024**3).window_bytes == 8 * 1024**3

    def test_from_env(self, monkeypatch):
        monkeypatch.setenv("MX_S3_RDMA_WINDOW_MB", "64")
        monkeypatch.setenv("MX_S3_RDMA_INFLIGHT", "12")
        monkeypatch.delenv("MX_S3_RDMA_THREADS", raising=False)
        monkeypatch.setenv("MX_S3_RDMA_VERIFY", "off")
        monkeypatch.setenv("MX_S3_RDMA_STAGING", " Host ")
        config = S3RdmaLoadConfig.from_env()
        assert config.window_bytes == 64 * 1024 * 1024
        assert config.num_threads == 12
        assert config.verify == "off"
        assert config.staging == "host"


class TestReaderFromSession:
    @pytest.fixture(autouse=True)
    def _isolated_aws(self, monkeypatch, tmp_path):
        for name in (
            "AWS_ENDPOINT_URL",
            "AWS_ENDPOINT_URL_S3",
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_SESSION_TOKEN",
            "AWS_PROFILE",
            "AWS_CA_BUNDLE",
        ):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "config"))
        monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "credentials"))
        monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
        monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")

    def test_endpoint_and_credentials(self, monkeypatch):
        monkeypatch.setenv("AWS_ENDPOINT_URL", "http://aistor1:9100")
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "ak")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "sk")
        monkeypatch.setenv("AWS_SESSION_TOKEN", "tok")
        _reader, endpoint = S3ObjectReader.from_session("models")
        assert endpoint.endpoint_url == "http://aistor1:9100"
        assert endpoint.bucket == "models"
        params = endpoint.backend_params(4)
        assert not {"access_key", "secret_key", "session_token"} & params.keys()

    def test_rejects_aws_s3(self, monkeypatch):
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "ak")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "sk")
        with pytest.raises(S3RdmaTransferError, match="does not serve S3-over-RDMA"):
            S3ObjectReader.from_session("models")

    def test_requires_credentials(self, monkeypatch):
        monkeypatch.setenv("AWS_ENDPOINT_URL", "http://aistor1:9100")
        with pytest.raises(S3RdmaTransferError, match="credentials"):
            S3ObjectReader.from_session("models")
