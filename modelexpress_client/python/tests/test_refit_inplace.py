# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Staging of nonzero XOR bytes and in-place installation of deltas."""

import copy
import dataclasses
import errno
import functools
import json
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors.torch import load_file, save_file

from modelexpress_rl.inference import receiver as receiver_module
from modelexpress_rl.inference.engines.vllm import inplace as inplace_module
from modelexpress_rl.inference.engines.vllm import installer as installer_module
from modelexpress_rl.inference.plan import PreparedCheckpointArtifact
from modelexpress_rl.inference.receiver import PreparedCheckpoint
from modelexpress_rl.inference.staging import (
    POSITIONS_FILE,
    VALUES_FILE,
    StagedDelta,
    join_nonzero,
    nonzero_bytes,
    write_staging,
)
from modelexpress_rl.utils import index_checkpoint_tensors
from tests import refit_fixtures
from tests.refit_fixtures import (
    _bytes,
    _checksum,
    _delta,
    _delta_snapshot,
    _fake_modules,
    _FakeS3,
    _save,
    _sparse_bytes,
    _tensors_delta_snapshot,
)


def _stage(directory, tensors, base="base"):
    """Stage {name: (base bytes, target bytes)} as their XOR."""
    staged = {}
    for name, (before, after) in tensors.items():
        xor = np.bitwise_xor(before, after)
        staged[name] = (xor.size, *join_nonzero(xor.size, [nonzero_bytes(xor, 0)]))
    directory.mkdir(parents=True, exist_ok=True)
    write_staging(directory, base_version=base, tensors=staged)
    return StagedDelta(directory)


def test_staged_delta_round_trips_nonzero_bytes(tmp_path):
    base = np.zeros(10, dtype=np.uint8)
    target = base.copy()
    target[[1, 7]] = [5, 9]
    staged = _stage(tmp_path / "staging", {"a": (base, target), "b": (base, base)})

    assert StagedDelta.base_of(tmp_path / "staging") == "base"
    assert "a" in staged and "missing" not in staged
    tensor = staged.tensor("a")
    assert tensor.nbytes == 10
    assert tensor.positions.tolist() == [1, 7]
    assert tensor.values.tolist() == [5, 9]
    assert staged.tensor("b").positions.size == 0
    staged.close()


@pytest.mark.parametrize("density", [0.0, 0.01, 0.3, 1.0])
@pytest.mark.parametrize("size", [0, 7, 8, 17, 4097])
def test_nonzero_bytes_match_a_byte_scan_at_any_alignment(size, density):
    rng = np.random.default_rng(size)
    block = _sparse_bytes(rng, size, density)
    # Start one byte into a buffer, so 8-byte words are unaligned.
    shifted = np.frombuffer(b"\0" + block.tobytes(), dtype=np.uint8)[1:]

    positions, values = nonzero_bytes(shifted, 11)

    assert positions.tolist() == (np.flatnonzero(block) + 11).tolist()
    assert values.tolist() == block[block != 0].tolist()


def test_staged_base_is_read_without_the_index(tmp_path):
    staged = _stage(
        tmp_path / "staging",
        {"a": (np.zeros(4, dtype=np.uint8), np.arange(4, dtype=np.uint8))},
        base="v7",
    )
    staged.close()
    (tmp_path / "staging" / "index.json").unlink()

    assert StagedDelta.base_of(tmp_path / "staging") == "v7"


def test_staging_rejects_offsets_of_the_wrong_width(tmp_path):
    tensors = {"a": (4, np.arange(2, dtype=np.uint64), np.ones(2, dtype=np.uint8))}

    with pytest.raises(ValueError, match="offsets"):
        write_staging(tmp_path, base_version="base", tensors=tensors)


def test_checkpoint_index_reuses_unchanged_headers(tmp_path):
    first, second = tmp_path / "v1", tmp_path / "v2"
    for directory in (first, second):
        directory.mkdir()
        save_file(
            {"a": torch.zeros(2), "lm_head.weight": torch.ones(3)},
            directory / "model.safetensors",
        )

    _, locations, metadata = index_checkpoint_tensors(first)
    _, again, same = index_checkpoint_tensors(first)
    _, moved, reused = index_checkpoint_tensors(second)

    assert same is metadata and again is locations
    # A new version directory with the same headers keeps the parsed metadata.
    assert reused is metadata
    assert moved["a"] == (second / "model.safetensors", *locations["a"][1:])

    (second / "config.json").write_text(json.dumps({"tie_word_embeddings": True}))
    _, _, tied = index_checkpoint_tensors(second)
    assert "lm_head.weight" not in tied and "lm_head.weight" in metadata

    save_file({"a": torch.zeros(4)}, second / "model.safetensors")
    _, changed, fresh = index_checkpoint_tensors(second)
    assert fresh is not metadata and fresh["a"]["shape"] == [4]
    assert list(changed) == ["a"]


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "mps",
            marks=pytest.mark.skipif(
                not torch.backends.mps.is_available(), reason="no MPS"
            ),
        ),
    ],
)
def test_staged_bytes_rebuild_the_dense_delta(tmp_path, device):
    after = np.random.default_rng(1).integers(0, 256, 64, dtype=np.uint8)
    after[::3] = 0
    staged = _stage(tmp_path / "staging", {"a": (np.zeros(64, dtype=np.uint8), after)})

    with inplace_module._quiet_readonly_buffers():
        data = inplace_module.staged_bytes(
            64, staged.tensor("a"), torch.device(device), chunk=5
        )

    assert data.device.type == device
    assert data.cpu().numpy().tolist() == after.tolist()
    assert not inplace_module.staged_bytes(8, None, torch.device(device)).any()
    staged.close()


def test_input_budget_builds_inputs_on_the_host_while_held_ones_fill_the_room():
    device = torch.device("meta")
    budget = inplace_module.InputBudget(device, 100)

    first = torch.empty(60, dtype=torch.uint8, device=budget.device(60))
    budget.hold(first)
    assert first.device == device
    assert budget.device(60) == torch.device("cpu")
    # A loader may hold only a view of an input.
    view = first.view(torch.int16).view(5, 6)
    del first
    assert budget.device(60) == torch.device("cpu")
    del view
    assert budget.device(60) == device
    assert budget.host_bytes == 120


def test_large_tensors_stage_wide_offsets():
    small, _ = join_nonzero(16, [(np.array([3]), np.array([1], dtype=np.uint8))])
    large, _ = join_nonzero(
        1 << 33, [(np.array([1 << 32]), np.array([1], dtype=np.uint8))]
    )

    assert small.dtype == np.uint32
    assert large.dtype == np.uint64 and large.tolist() == [1 << 32]


# Caches stage each delta's changed bytes, as for an engine that installs in place.
_checkpoint = functools.partial(refit_fixtures._checkpoint, stage_deltas=True)


def test_delta_preparation_stages_nothing_for_an_engine_that_reloads(tmp_path):
    s3 = _FakeS3(
        _delta_snapshot([1.0, 2.0], [1.0, 3.0], identity="v1", base_identity="base")
    )
    checkpoint = _checkpoint(tmp_path, s3, stage_deltas=False)

    prepared = checkpoint.prepare_chain((_delta("v1", base="base", identity="v1"),))

    assert prepared.staging is None and prepared.staging_base is None
    assert not checkpoint.store.staging_path("v1").exists()


def test_delta_preparation_stages_the_changed_bytes(tmp_path):
    s3 = _FakeS3(
        _delta_snapshot([1.0, 2.0], [1.0, 3.0], identity="v1", base_identity="base")
    )
    checkpoint = _checkpoint(tmp_path, s3)

    prepared = checkpoint.prepare_chain((_delta("v1", base="base", identity="v1"),))

    assert prepared.staging == checkpoint.store.staging_path("v1")
    assert prepared.staging_base == "base"
    staged = StagedDelta(prepared.staging)
    xor = np.bitwise_xor(_bytes([1.0, 2.0]), _bytes([1.0, 3.0]))
    assert staged.tensor("weight").positions.tolist() == np.flatnonzero(xor).tolist()
    assert staged.tensor("weight").values.tolist() == xor[xor != 0].tolist()
    staged.close()

    # A co-located rank reusing the prepared head sees the same staging.
    reused = _checkpoint(tmp_path, s3, initialize=False).prepare_chain(
        (_delta("v1", base="base", identity="v1"),)
    )
    assert reused.staging == prepared.staging and reused.staging_base == "base"


def test_evicting_a_version_removes_its_staging(tmp_path):
    s3 = _FakeS3(
        _delta_snapshot([1.0, 2.0], [1.0, 3.0], identity="v1", base_identity="base")
    )
    checkpoint = _checkpoint(tmp_path, s3)
    checkpoint.prepare_chain((_delta("v1", base="base", identity="v1"),))
    store = checkpoint.store
    store.activate("base")

    store.ensure_capacity(0, protected_versions=())
    assert store.staging_path("v1").exists()
    store.max_size_bytes = 0
    with pytest.raises(receiver_module.CheckpointCacheCapacityError):
        store.ensure_capacity(1, protected_versions=())
    assert not store.staging_path("v1").exists()


def test_a_delta_without_room_for_its_staging_still_prepares(monkeypatch, tmp_path):
    s3 = _FakeS3(
        _delta_snapshot([1.0, 2.0], [1.0, 3.0], identity="v1", base_identity="base")
    )
    checkpoint = _checkpoint(tmp_path, s3)

    def full(*_args, **_kwargs):
        raise receiver_module.CheckpointCacheCapacityError("checkpoint cache is full")

    monkeypatch.setattr(checkpoint, "_write_staging", full)
    prepared = checkpoint.prepare_chain((_delta("v1", base="base", identity="v1"),))

    # The files already hold the target, which installs by reloading instead.
    assert prepared.staging is None
    assert checkpoint.store.state().version == "v1"
    assert load_file(prepared.path / "model.safetensors")["weight"].tolist() == [
        1.0,
        3.0,
    ]


# Each delta changes some tensors of a two-tensor checkpoint.
_LINEAGE = {
    "base": {"a": [1.0, 2.0], "b": [3.0, 4.0]},
    "v1": {"a": [1.0, 5.0], "b": [3.0, 4.0]},
    "v2": {"a": [9.0, 5.0], "b": [6.0, 4.0]},
    "v3": {"a": [7.0, 5.0], "b": [6.0, 4.0]},
    "other": {"a": [8.0, 5.0], "b": [3.0, 4.0]},
}


def _lineage_snapshot(version, base):
    """The delta from ``base`` to ``version`` of every tensor that changes."""
    return _tensors_delta_snapshot(
        _LINEAGE[base], _LINEAGE[version], identity=version, base_identity=base
    )


def _lineage_checkpoint(tmp_path, monkeypatch, *, layers=None):
    if layers is not None:
        monkeypatch.setenv("MX_REFIT_STAGED_LAYERS", str(layers))
    objects = {}
    for version, base in (("v1", "base"), ("v2", "v1"), ("v3", "v2")):
        objects.update(_lineage_snapshot(version, base))
    s3 = _FakeS3(objects)
    return _checkpoint(tmp_path, s3, tensors=_LINEAGE["base"]), s3


def _lineage_delta(version):
    base = {"v1": "base", "v2": "v1", "v3": "v2"}[version]
    return _delta(version, base=base, identity=version)


def _staged_target(prepared):
    """The target a preparation describes: its files XOR each staged delta since."""
    tensors = load_file(prepared.path / "model.safetensors")
    for staging in prepared.overlay:
        staged = StagedDelta(staging)
        for name in list(tensors):
            if name in staged:
                data = tensors[name].reshape(-1).view(torch.uint8)
                delta = staged.tensor(name)
                data[torch.from_numpy(delta.positions.astype(np.int64))] ^= (
                    torch.from_numpy(np.array(delta.values))
                )
        staged.close()
    return {name: tensor.tolist() for name, tensor in tensors.items()}


def _target_checksums(version):
    return {
        name: _checksum(_bytes(values)) for name, values in _LINEAGE[version].items()
    }


def _sparse(prepared):
    return (
        prepared.metrics["perf/mx_receive_delta_sparse_checks"],
        prepared.metrics["perf/mx_receive_delta_sparse_fallbacks"],
    )


def test_staged_delta_reads_row_ranges_and_checks_file_sizes(tmp_path):
    tensors = {
        "a": (16, np.array([1, 3], dtype=np.uint32), np.array([7, 8], dtype=np.uint8)),
        "b": (16, np.array([2], dtype=np.uint32), np.array([9], dtype=np.uint8)),
    }
    write_staging(tmp_path, base_version="base", tensors=tensors)
    staged = StagedDelta(tmp_path)

    assert staged.row("b") == 1
    assert staged.byte_ranges(0, 2) == ((0, 12), (0, 3))
    assert staged.byte_ranges(1, 2) == ((8, 12), (2, 3))
    out = bytearray(4)
    staged.read_into(POSITIONS_FILE, 8, memoryview(out))
    assert np.frombuffer(bytes(out), dtype=np.uint32).tolist() == [2]
    staged.close()

    with (tmp_path / VALUES_FILE).open("ab") as handle:
        handle.write(b"\0")
    with pytest.raises(ValueError, match="does not match"):
        StagedDelta(tmp_path)
    installer = inplace_module.InPlaceInstaller(
        torch.nn.Module(), device=torch.device("cpu"), generic_reload=lambda load: None
    )
    with pytest.raises(inplace_module.InPlaceUnsupported, match="does not match"):
        installer.install(tmp_path / "checkpoint", tmp_path)


def test_staged_runs_split_by_size_and_offset_width(tmp_path):
    tensors = {
        "a": (16, np.arange(4, dtype=np.uint32), np.ones(4, dtype=np.uint8)),
        "b": (16, np.arange(3, dtype=np.uint32), np.ones(3, dtype=np.uint8)),
        "c": (
            1 << 33,
            np.array([1 << 32], dtype=np.uint64),
            np.ones(1, dtype=np.uint8),
        ),
        "d": (16, np.arange(5, dtype=np.uint32), np.ones(5, dtype=np.uint8)),
    }
    write_staging(tmp_path, base_version="base", tensors=tensors)
    staged = StagedDelta(tmp_path)

    assert inplace_module.staged_runs(staged, 0, 4, limit=8) == [(0, 2), (2, 3), (3, 4)]
    assert inplace_module.staged_runs(staged, 0, 2, limit=5) == [(0, 1), (1, 2)]
    # Rows larger than the limit form runs of their own.
    assert inplace_module.staged_runs(staged, 1, 4, limit=2) == [(1, 2), (2, 3), (3, 4)]
    assert inplace_module.staged_runs(staged, 1, 1) == []
    staged.close()


class _Exchange:
    """Ranks simulated by threads that all-gather through a barrier.

    A rank left waiting for the others fails after ``timeout`` seconds.
    """

    def __init__(self, size, timeout=10.0):
        self.size = size
        self._barrier = threading.Barrier(size, timeout=timeout)
        self._parts = [None] * size
        # Bytes each rank sent in each of its gathers of staged runs.
        self.gathered = [[] for _ in range(size)]

    def peers(self, rank):
        def exchange(tensor):
            self._parts[rank] = tensor
            self._barrier.wait()
            gathered = torch.cat(self._parts)
            self._barrier.wait()
            return gathered

        def all_gather(tensor):
            gathered = exchange(tensor)
            self.gathered[rank].append(tensor.numel())
            return gathered

        def agree(flags):
            return exchange(flags).view(self.size, -1).amin(0)

        return inplace_module.Peers(rank, self.size, all_gather, agree)


@pytest.mark.parametrize("size", [2, 3])
def test_peers_split_reads_and_deliver_whole_runs(tmp_path, size):
    rng = np.random.default_rng(5)
    tensors = {}
    for index, nbytes in enumerate((64, 9, 33, 128)):
        after = rng.integers(1, 256, nbytes, dtype=np.uint8)
        after[rng.random(nbytes) < 0.6] = 0
        tensors[f"t{index}"] = (np.zeros(nbytes, dtype=np.uint8), after)
    # A row with no staged bytes forms an empty run.
    tensors["t4"] = (np.zeros(8, dtype=np.uint8), np.zeros(8, dtype=np.uint8))
    staged = _stage(tmp_path / "staging", tensors)
    runs = inplace_module.staged_runs(staged, 0, 5, limit=20)
    assert runs[-1] == (4, 5)
    cpu = torch.device("cpu")
    reader = inplace_module.StagedRunReader(staged, runs, cpu, piece=8, depth=2)
    expected = list(reader)
    reader.close()
    exchange = _Exchange(size)
    delivered = [None] * size

    def read(rank):
        peer = inplace_module.StagedRunReader(
            staged, runs, cpu, peers=exchange.peers(rank), piece=8, depth=2
        )
        delivered[rank] = list(peer)
        peer.close()

    threads = [threading.Thread(target=read, args=(rank,)) for rank in range(size)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    for run in expected:
        rows = [staged.tensor(f"t{row}") for row in range(run.begin, run.end)]
        offsets = np.concatenate([row.positions for row in rows]).astype(np.int64)
        values = np.concatenate([row.values for row in rows])
        assert run.offsets.to(torch.int64).tolist() == offsets.tolist()
        assert run.values.tolist() == values.tolist()
    for runs_of_rank in delivered:
        assert [(run.begin, run.end) for run in runs_of_rank] == runs
        for got, want in zip(runs_of_rank, expected):
            assert torch.equal(got.offsets, want.offsets)
            assert torch.equal(got.values, want.values)
    staged.close()


def test_runs_place_narrow_and_wide_offsets(tmp_path):
    tensors = {
        "a": (16, np.array([2, 5], dtype=np.uint32), np.array([1, 2], dtype=np.uint8)),
        "b": (
            1 << 33,
            np.array([1, (1 << 32) + 3], dtype=np.uint64),
            np.array([3, 4], dtype=np.uint8),
        ),
    }
    write_staging(tmp_path, base_version="base", tensors=tensors)
    staged = StagedDelta(tmp_path)
    runs = inplace_module.staged_runs(staged, 0, 2)
    assert runs == [(0, 1), (1, 2)]
    cpu = torch.device("cpu")
    flat = torch.zeros(32 + inplace_module._DISCARD_BYTES, dtype=torch.uint8)
    # a: two-byte columns from byte 1 of each four-byte row; b: its first 16
    # bytes as one range at 16, so its byte past 4 GiB is discarded.
    rules = {0: (0, 4, 2, 1), 1: (16, inplace_module._ONE_ROW, 16, 0)}
    placement = inplace_module.StagedPlacement(staged, (0, 2), rules, 32, cpu)

    reader = inplace_module.StagedRunReader(staged, runs, cpu)
    for run in reader:
        placement.place(flat, run, window=1)
    reader.close()

    expected = torch.zeros(32, dtype=torch.uint8)
    expected[1], expected[2], expected[17] = 1, 2, 3
    assert torch.equal(flat[:32], expected)
    staged.close()


def _fake_layerwise(monkeypatch, meta):
    layerwise = {
        "get_layerwise_info": lambda _layer: SimpleNamespace(
            restore_metadata=(meta, {})
        )
    }
    _fake_modules(
        monkeypatch, {"vllm.model_executor.model_loader.reload.layerwise": layerwise}
    )


@pytest.mark.parametrize("tp_size,tp_rank", [(1, 0), (2, 1)])
@pytest.mark.parametrize("transposed", [True, False])
def test_expert_layout_follows_the_weight_loader_rules(
    monkeypatch, tmp_path, tp_size, tp_rank, transposed
):
    experts, out, inner = 2, 4, 6  # gate/up: [out, inner]; down: [inner, out]
    generator = torch.Generator().manual_seed(3)
    dense, tensors, names, metadata = {}, {}, {}, {}
    for expert in range(experts):
        for proj, shape in (
            ("gate", (out, inner)),
            ("up", (out, inner)),
            ("down", (inner, out)),
        ):
            name = f"model.layers.5.mlp.experts.{expert}.{proj}_proj.weight"
            value = torch.randint(
                -(2**31), 2**31 - 1, shape, generator=generator, dtype=torch.int32
            )
            dense[(expert, proj)] = value
            tensors[name] = (
                np.zeros(value.numel() * 4, dtype=np.uint8),
                value.reshape(-1).view(torch.uint8).numpy(),
            )
            names[(expert, proj, "weight")] = name
            metadata[name] = {
                "shape": list(shape),
                "dtype": "I32",
                "byte_size": value.numel() * 4,
            }
    staged = _stage(tmp_path / "staging", tensors)

    def loader_layout(proj):
        # Reference: the routed-expert loader's transpose, then its TP shard.
        stacked = []
        for expert in range(experts):
            value = dense[(expert, proj)]
            dim = 0 if proj != "down" else 1
            size = value.shape[dim] // tp_size
            value = value.narrow(dim, tp_rank * size, size)
            stacked.append(value.t() if transposed else value)
        return torch.stack(stacked)

    w13 = torch.cat(
        [loader_layout("gate"), loader_layout("up")], dim=2 if transposed else 1
    )
    w2 = loader_layout("down")
    meta = {}
    for name, tensor in (("w13_weight", w13), ("w2_weight", w2)):
        meta_tensor = torch.empty(tensor.shape, dtype=tensor.dtype, device="meta")
        meta_tensor.is_transposed = transposed
        meta[name] = meta_tensor
    _fake_layerwise(monkeypatch, meta)
    layer = SimpleNamespace(
        moe_config=SimpleNamespace(
            moe_parallel_config=SimpleNamespace(tp_size=tp_size, ep_size=1),
            tp_rank=tp_rank,
        )
    )

    arranged = inplace_module.arrange_expert_delta(
        layer, staged, names, metadata, device=torch.device("cpu")
    )

    assert torch.equal(arranged["w13_weight"], w13)
    assert torch.equal(arranged["w2_weight"], w2)
    staged.close()


def test_packed_experts_may_load_into_plain_weight_parameters(monkeypatch, tmp_path):
    experts, rows, cols = 2, 4, 3
    dense, tensors, names, metadata = {}, {}, {}, {}
    for expert in range(experts):
        for proj, shape in (
            ("w1", (rows, cols)),
            ("w3", (rows, cols)),
            ("w2", (cols, rows)),
        ):
            name = f"language_model.model.layers.1.block_sparse_moe.experts.{expert}.{proj}.weight_packed"
            value = torch.randint(0, 255, shape, dtype=torch.uint8)
            dense[(expert, proj)] = value
            tensors[name] = (
                np.zeros(value.numel(), dtype=np.uint8),
                value.reshape(-1).numpy(),
            )
            metadata[name] = {
                "shape": list(shape),
                "dtype": "U8",
                "byte_size": value.numel(),
            }
            match = inplace_module._EXPERT.search(name)
            names[(expert, inplace_module._PROJECTIONS[match[3]], match[4])] = name
    staged = _stage(tmp_path / "staging", tensors)
    meta = {
        "w13_weight": torch.empty(
            (experts, 2 * rows, cols), dtype=torch.uint8, device="meta"
        ),
        "w2_weight": torch.empty(
            (experts, cols, rows), dtype=torch.uint8, device="meta"
        ),
    }
    _fake_layerwise(monkeypatch, meta)
    layer = SimpleNamespace(
        moe_config=SimpleNamespace(
            moe_parallel_config=SimpleNamespace(tp_size=1, ep_size=1), tp_rank=0
        )
    )

    arranged = inplace_module.arrange_expert_delta(
        layer, staged, names, metadata, device=torch.device("cpu")
    )

    expected = torch.stack(
        [torch.cat([dense[(e, "w1")], dense[(e, "w3")]]) for e in range(experts)]
    )
    assert torch.equal(arranged["w13_weight"], expected)
    assert torch.equal(
        arranged["w2_weight"], torch.stack([dense[(e, "w2")] for e in range(experts)])
    )
    staged.close()


def _expert_layers(monkeypatch, tmp_path):
    """Two routed-expert layers and a staged delta that spans several runs of each."""
    experts, out, inner = 2, 4, 6
    generator = torch.Generator().manual_seed(11)

    def random(shape):
        return torch.randint(
            -(2**31), 2**31 - 1, shape, generator=generator, dtype=torch.int32
        )

    class RoutedExperts(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w13_weight = torch.nn.Parameter(
                random((experts, 2 * out, inner)), requires_grad=False
            )
            self.w2_weight = torch.nn.Parameter(
                random((experts, inner, out)), requires_grad=False
            )
            self.moe_config = SimpleNamespace(
                moe_parallel_config=SimpleNamespace(tp_size=1, ep_size=1), tp_rank=0
            )
            self.quant_method = SimpleNamespace()

    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([torch.nn.Module() for _ in range(2)])
    for layer in model.layers:
        layer.mlp = torch.nn.Module()
        layer.mlp.experts = RoutedExperts()
    monkeypatch.setattr(inplace_module, "_routed_experts_class", lambda: RoutedExperts)
    _fake_layerwise(
        monkeypatch,
        {
            "w13_weight": torch.empty(
                (experts, 2 * out, inner), dtype=torch.int32, device="meta"
            ),
            "w2_weight": torch.empty(
                (experts, inner, out), dtype=torch.int32, device="meta"
            ),
        },
    )
    # Processing that keeps the loaded layout.
    monkeypatch.setattr(
        inplace_module, "process_like_weights", lambda _layer, arranged: arranged
    )
    # Runs of a few rows each, so layers span several runs of one reader.
    monkeypatch.setattr(
        inplace_module,
        "staged_runs",
        functools.partial(inplace_module.staged_runs, limit=64),
    )
    deltas, checkpoint_tensors, tensors = {}, {}, {}
    for index in range(2):
        for expert in range(experts):
            for proj, shape in (
                ("gate", (out, inner)),
                ("up", (out, inner)),
                ("down", (inner, out)),
            ):
                name = f"model.layers.{index}.mlp.experts.{expert}.{proj}_proj.weight"
                delta = random(shape)
                delta[torch.rand(shape, generator=generator) < 0.5] = 0
                deltas[(index, expert, proj)] = delta
                checkpoint_tensors[name] = torch.zeros(shape, dtype=torch.int32)
                tensors[name] = (
                    np.zeros(delta.numel() * 4, dtype=np.uint8),
                    delta.reshape(-1).view(torch.uint8).numpy(),
                )
    checkpoint = _save(tmp_path / "checkpoint", checkpoint_tensors)
    staged = _stage(tmp_path / "staging", tensors)
    return model, checkpoint, staged, deltas


def test_expert_layers_install_from_one_reader(monkeypatch, tmp_path):
    experts = 2
    model, checkpoint, staged, deltas = _expert_layers(monkeypatch, tmp_path)
    before = [
        (layer.mlp.experts.w13_weight.clone(), layer.mlp.experts.w2_weight.clone())
        for layer in model.layers
    ]
    installer = inplace_module.InPlaceInstaller(
        model, device=torch.device("cpu"), generic_reload=lambda load: None
    )

    metrics = installer._install(checkpoint, staged)

    assert metrics["perf/mx_receive_inplace_expert_layers"] == 2
    for index, layer in enumerate(model.layers):
        w13, w2 = before[index]
        gate_up = torch.stack(
            [
                torch.cat([deltas[(index, e, "gate")], deltas[(index, e, "up")]])
                for e in range(experts)
            ]
        )
        down = torch.stack([deltas[(index, e, "down")] for e in range(experts)])
        assert torch.equal(layer.mlp.experts.w13_weight, w13 ^ gate_up)
        assert torch.equal(layer.mlp.experts.w2_weight, w2 ^ down)
    staged.close()


@pytest.mark.parametrize(
    ("failure", "expected"),
    [
        ("layer", inplace_module.InPlaceUnsupported),
        ("read", OSError),
        ("gather", torch.OutOfMemoryError),
    ],
)
def test_a_rank_that_fails_partway_still_gathers_every_run(
    monkeypatch, tmp_path, failure, expected
):
    model, checkpoint, staged, _deltas = _expert_layers(monkeypatch, tmp_path)
    exchange = _Exchange(2)
    peers = [exchange.peers(rank) for rank in range(2)]
    stageds = [staged, StagedDelta(tmp_path / "staging")]
    read_into, gather, calls = stageds[1].read_into, peers[1].all_gather, []
    # Rank 1 cannot read its share of the values from row 2 on.
    unreadable = staged.byte_ranges(2, 3)[1][0]

    def fail_layer(_layer, _staged, _names, _metadata, runs):
        next(runs)
        raise inplace_module.InPlaceUnsupported("layout differs")

    def fail_read(file, offset, out):
        if file == VALUES_FILE and offset >= unreadable:
            raise OSError(errno.EIO, "Input/output error")
        read_into(file, offset, out)

    def fail_gather(tensor):
        # Out of memory before its third gather starts.
        calls.append(tensor.numel())
        if len(calls) == 3:
            raise torch.OutOfMemoryError("CUDA out of memory")
        return gather(tensor)

    if failure == "read":
        monkeypatch.setattr(stageds[1], "read_into", fail_read)
    if failure == "gather":
        peers[1] = dataclasses.replace(peers[1], all_gather=fail_gather)
    installers = [
        inplace_module.InPlaceInstaller(
            replica,
            device=torch.device("cpu"),
            generic_reload=lambda load: None,
            peers=peer,
        )
        for replica, peer in zip((model, copy.deepcopy(model)), peers)
    ]
    if failure == "layer":
        monkeypatch.setattr(installers[1], "_install_experts", fail_layer)
    outcomes = [None, None]

    def install(rank):
        try:
            outcomes[rank] = installers[rank]._install(checkpoint, stageds[rank])
        except Exception as error:
            outcomes[rank] = error

    threads = [
        threading.Thread(target=install, args=(rank,), daemon=True) for rank in range(2)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)

    # Neither rank is left waiting, and only the failing one raises, with its own error.
    assert not any(thread.is_alive() for thread in threads)
    assert isinstance(outcomes[0], dict)
    assert isinstance(outcomes[1], expected)
    # The ranks gather every run together, in the same order.
    runs = inplace_module.staged_runs(staged, 0, 6) + inplace_module.staged_runs(
        staged, 6, 12
    )
    assert len(runs) > 2 and len(exchange.gathered[0]) == len(runs)
    assert exchange.gathered[1] == exchange.gathered[0]
    stageds[1].close()
    staged.close()


def test_peers_write_an_expert_layer_sparsely_only_if_all_of_them_can():
    # A peer whose shard of layer 5 has no sparse layout.
    installer = SimpleNamespace(
        _peers=SimpleNamespace(
            agree=lambda flags: flags * torch.tensor([1, 0, 1], dtype=torch.int32)
        )
    )

    layouts = inplace_module.InPlaceInstaller._agreed_layouts(
        installer, [3, 5, 7], {3: "three", 5: "five"}
    )

    assert layouts == {3: "three"}


class PPMissingLayer(torch.nn.Module):
    """Named like vLLM's placeholder for a layer of another pipeline stage."""


@pytest.mark.parametrize(
    "layers",
    [
        # Layer 1 belongs to another pipeline stage.
        (torch.nn.Module, PPMissingLayer),
        # Layer 1 is a speculative-decoding layer this model does not build.
        (torch.nn.Module,),
    ],
    ids=["other_stage", "not_built"],
)
def test_checkpoint_layers_the_rank_does_not_serve_are_skipped(
    monkeypatch, tmp_path, layers
):
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([layer() for layer in layers])
    monkeypatch.setattr(inplace_module, "_routed_experts_class", lambda: None)
    installer = inplace_module.InPlaceInstaller(
        model, device=torch.device("cpu"), generic_reload=lambda load: None
    )
    names = [
        "model.layers.1.mlp.experts.0.gate_proj.weight",
        "model.layers.1.eh_proj.weight",
    ]
    checkpoint = _save(
        tmp_path / "checkpoint",
        {name: torch.zeros(2, dtype=torch.int32) for name in names},
    )
    staged = _stage(
        tmp_path / "staging",
        {
            name: (np.zeros(8, dtype=np.uint8), np.arange(8, dtype=np.uint8))
            for name in names
        },
    )

    metrics = installer._install(checkpoint, staged)

    assert metrics["perf/mx_receive_inplace_expert_layers"] == 0
    assert metrics["perf/mx_receive_inplace_generic_tensors"] == 0
    staged.close()


def test_generic_install_handles_loaders_that_write_live_storage(monkeypatch, tmp_path):
    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(
                torch.tensor([1, 2, 3, 4], dtype=torch.int32), requires_grad=False
            )
            self.register_buffer(
                "mirror",
                torch.tensor([9, 8, 7, 6], dtype=torch.int32),
                persistent=False,
            )

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([Layer()])

        def load_weights(self, weights):
            layer = self.layers[0]
            for name, tensor in weights:
                if name.endswith("attn.weight"):
                    # XOR restore: processed value XORed into live storage.
                    layer.weight.data.bitwise_xor_(tensor)
                    # A loader mirror writes the loaded value directly.
                    layer.mirror.copy_(tensor)

    monkeypatch.setattr(inplace_module, "_routed_experts_class", lambda: None)
    model = Model()
    installer = inplace_module.InPlaceInstaller(
        model, device=torch.device("cpu"), generic_reload=lambda load: load(None)
    )
    name = "model.layers.0.attn.weight"
    checkpoint = _save(
        tmp_path / "checkpoint", {name: torch.zeros(4, dtype=torch.int32)}
    )
    delta = torch.tensor([0, 5, 0, 1], dtype=torch.int32)
    staged = _stage(
        tmp_path / "staging",
        {name: (np.zeros(16, dtype=np.uint8), delta.view(torch.uint8).numpy())},
    )

    metrics = installer._install(checkpoint, staged)

    layer = model.layers[0]
    assert metrics["perf/mx_receive_inplace_generic_tensors"] == 1
    # Both reloads build the 16-byte input in host memory on a CPU device.
    assert metrics["perf/mx_receive_inplace_host_input_bytes"] == 32
    assert layer.weight.tolist() == [1, 2 ^ 5, 3, 4 ^ 1]
    assert layer.mirror.tolist() == [9, 8 ^ 5, 7, 6 ^ 1]
    staged.close()


class _ShardModel(torch.nn.Module):
    """Loads a rank's shard of its one 4x2 input along ``dim`` by XORing it into live storage."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        live = [[1, 2], [3, 4]] if dim == 0 else [[1], [2], [3], [4]]
        self.layers = torch.nn.ModuleList([torch.nn.Module()])
        self.layers[0].weight = torch.nn.Parameter(
            torch.tensor(live, dtype=torch.int32), requires_grad=False
        )
        self.inputs = []

    def load_weights(self, weights):
        for _name, tensor in weights:
            self.inputs.append(tensor)
            shard = tensor.narrow(0, 2, 2) if self.dim == 0 else tensor.narrow(1, 1, 1)
            self.layers[0].weight.data.bitwise_xor_(shard)


def _shard_install(monkeypatch, tmp_path, dim):
    monkeypatch.setattr(inplace_module, "_routed_experts_class", lambda: None)
    model = _ShardModel(dim)
    installer = inplace_module.InPlaceInstaller(
        model, device=torch.device("cpu"), generic_reload=lambda load: load(None)
    )
    name = "model.layers.0.proj.weight"
    checkpoint = _save(
        tmp_path / "checkpoint", {name: torch.zeros(4, 2, dtype=torch.int32)}
    )
    # Nonzero staged bytes at offsets 0, 12, 16 and 28.
    delta = torch.tensor([[7, 0], [0, 7], [5, 0], [0, 1]], dtype=torch.int32)
    staged = _stage(
        tmp_path / "staging",
        {
            name: (
                np.zeros(32, dtype=np.uint8),
                delta.view(torch.uint8).numpy().reshape(-1),
            )
        },
    )
    return model, installer, name, checkpoint, staged


def test_unrecognized_expert_tensors_are_not_installed_in_place(monkeypatch, tmp_path):
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([torch.nn.Module()])
    monkeypatch.setattr(inplace_module, "_routed_experts_class", lambda: None)
    installer = inplace_module.InPlaceInstaller(
        model, device=torch.device("cpu"), generic_reload=lambda load: None
    )
    name = "model.layers.0.mlp.experts.0.gate_proj.weight_scale_2"
    checkpoint = _save(
        tmp_path / "checkpoint", {name: torch.zeros(2, dtype=torch.int32)}
    )
    staged = _stage(
        tmp_path / "staging",
        {name: (np.zeros(8, dtype=np.uint8), np.arange(8, dtype=np.uint8))},
    )

    with pytest.raises(inplace_module.InPlaceUnsupported, match="unrecognized"):
        installer._install(checkpoint, staged)
    staged.close()


def test_checkpoint_catalog_groups_names_by_layer():
    names = [
        "model.embed_tokens.weight",
        "model.layers.3.mlp.experts.1.down_proj.weight_packed",
        "model.layers.3.self_attn.o_proj.weight",
        "model.layers.3.mlp.experts.0.w9.weight",
    ]

    catalog = inplace_module.CheckpointCatalog.of({name: {} for name in names})

    assert catalog.layers == {names[0]: None, names[1]: 3, names[2]: 3, names[3]: 3}
    assert catalog.experts == {3: {(1, "down", "weight_packed"): names[1]}}
    assert catalog.per_expert == {names[1], names[3]}
    assert catalog.unmatched_experts == {names[3]}
    assert catalog.generic == (names[0], names[2])


def test_processing_a_delta_leaves_the_kernel_on_live_storage(monkeypatch):
    class Info:
        def __init__(self):
            self.kernel_tensors = None
            self.kernel_non_persistent_buffers = set()

        def reset(self):
            self.kernel_tensors = None

    info = Info()

    def params_buffers(layer):
        return dict(layer._parameters), dict(layer._buffers)

    def restore_on_meta(layer, _info):
        for name, param in list(layer._parameters.items()):
            delattr(layer, name)
            layer.register_parameter(
                name, torch.nn.Parameter(torch.empty_like(param), requires_grad=False)
            )

    def place(layer, info):
        for name in list(layer._parameters):
            delattr(layer, name)
        for name, param in info.kernel_tensors[0].items():
            layer.register_parameter(name, param)

    _fake_modules(
        monkeypatch,
        {
            "vllm.model_executor.model_loader.reload.layerwise": {
                "_place_kernel_tensors": place,
                "get_layerwise_info": lambda _layer: info,
            },
            "vllm.model_executor.model_loader.reload.meta": {
                "SKIP_TENSORS": set(),
                "materialize_layer": lambda _l, _i: None,
                "restore_layer_on_meta": restore_on_meta,
            },
            "vllm.model_executor.model_loader.reload.utils": {
                "get_layer_params_buffers": params_buffers,
                "get_layer_tensors": lambda layer: dict(layer._parameters),
            },
        },
    )

    class Method:
        def process_weights_after_loading(self, layer):
            # Like a MoE kernel setup, capture the tensor being processed.
            layer.w13_weight.data.add_(1)
            self.kernel_weight = layer.w13_weight
            layer.extra = "set during processing"

    layer = torch.nn.Module()
    layer.register_parameter(
        "w13_weight", torch.nn.Parameter(torch.tensor([5, 5]), requires_grad=False)
    )
    layer.quant_method = Method()
    live = layer.w13_weight
    layer.quant_method.kernel_weight = live

    processed = inplace_module.process_like_weights(
        layer, {"w13_weight": torch.tensor([1, 2])}
    )

    assert processed["w13_weight"].tolist() == [2, 3]
    assert layer.w13_weight is live and live.tolist() == [5, 5]
    assert layer.quant_method.kernel_weight is live
    assert not hasattr(layer, "extra")


def test_xor_restore_leaves_elements_no_loader_writes_unchanged(monkeypatch):
    # Reload materializes tensors uninitialized, here as all-ones memory.
    def materialize(meta_tensor):
        return torch.full(meta_tensor.shape, -1, dtype=meta_tensor.dtype)

    def place(layer, info):
        delattr(layer, "weight")
        layer.register_parameter("weight", info.kernel_tensors[0]["weight"])

    _fake_modules(
        monkeypatch,
        {
            "vllm.model_executor.model_loader.reload.layerwise": {
                "_copy_and_restore_kernel_tensors": None,
                "_place_kernel_tensors": place,
            },
            "vllm.model_executor.model_loader.reload.meta": {
                "materialize_meta_tensor": materialize
            },
        },
    )
    layerwise = sys.modules["vllm.model_executor.model_loader.reload.layerwise"]
    meta = sys.modules["vllm.model_executor.model_loader.reload.meta"]
    live = torch.nn.Parameter(
        torch.tensor([1, 2, 3, 0], dtype=torch.int32), requires_grad=False
    )
    layer = torch.nn.Module()
    layer.register_parameter("weight", live)
    info = SimpleNamespace(
        kernel_tensors=({"weight": live}, {}),
        kernel_non_persistent_buffers=set(),
        loaded_weights=[("weight", None)],
    )

    with inplace_module.xor_restore():
        # Like vLLM's layer processing: materialize the layer, load every row but
        # the padding one, then restore the live tensors.
        weight = meta.materialize_meta_tensor(
            torch.empty(4, dtype=torch.int32, device="meta")
        )
        weight[:3] = torch.tensor([4, 5, 6], dtype=torch.int32)
        delattr(layer, "weight")
        layer.register_parameter(
            "weight", torch.nn.Parameter(weight, requires_grad=False)
        )
        layerwise._copy_and_restore_kernel_tensors(layer, info)

    assert layer.weight is live
    assert live.tolist() == [1 ^ 4, 2 ^ 5, 3 ^ 6, 0]
    assert meta.materialize_meta_tensor is materialize


def test_xor_into_rejects_mismatched_storage():
    target = torch.zeros(4, dtype=torch.int32)
    inplace_module.xor_into(target, torch.ones(4, dtype=torch.int32))
    assert target.tolist() == [1, 1, 1, 1]
    with pytest.raises(inplace_module.InPlaceUnsupported):
        inplace_module.xor_into(target, torch.ones(3, dtype=torch.int32))


def test_xor_into_handles_strided_live_tensors():
    storage = torch.zeros(4, 4, dtype=torch.bfloat16)
    target = storage.t()
    value = torch.arange(16, dtype=torch.int16).view(torch.bfloat16).view(4, 4)

    inplace_module.xor_into(target, value)

    assert torch.equal(inplace_module.bits(target), inplace_module.bits(value))
    assert torch.equal(inplace_module.bits(storage), inplace_module.bits(value).t())


def _installer():
    return installer_module._VllmInstaller(
        model=torch.nn.Module(),
        vllm_config=SimpleNamespace(
            load_config=SimpleNamespace(load_format="modelexpress"), quant_config=None
        ),
        model_config=SimpleNamespace(model="/launch", revision=None),
        device=torch.device("cpu"),
    )


class _InPlace:
    calls = []
    error = None

    def __init__(self, model, *, device, generic_reload, peers):
        pass

    def install(self, path, staging, *, overlay=(), reload_values=None):
        _InPlace.calls.append((path, staging))
        if _InPlace.error is not None:
            raise _InPlace.error
        return {
            "perf/mx_receive_inplace_time": 0.5,
            "perf/mx_receive_inplace_expert_layers": 2.0,
            "perf/mx_receive_inplace_sparse_expert_layers": 1.0,
            "perf/mx_receive_inplace_generic_tensors": 3.0,
            "perf/mx_receive_inplace_value_tensors": 0.0,
        }


@pytest.fixture
def staged_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(installer_module, "InPlaceInstaller", _InPlace)
    _InPlace.calls, _InPlace.error = [], None
    return PreparedCheckpoint(
        "v2", tmp_path, {}, staging=tmp_path / "staging", staging_base="v1"
    )


def test_in_place_install_needs_the_staged_base_to_be_live(staged_checkpoint):
    installer = _installer()

    assert installer._install_in_place(staged_checkpoint, live_version=None) is None
    assert installer._install_in_place(staged_checkpoint, live_version="v0") is None
    metrics = installer._install_in_place(staged_checkpoint, live_version="v1")

    assert metrics["perf/mx_receive_inplace_time"] == 0.5
    assert _InPlace.calls == [(staged_checkpoint.path, staged_checkpoint.staging)]


@pytest.mark.parametrize("materialized", [True])
def test_unsupported_in_place_install_falls_back_to_a_reload(
    staged_checkpoint, monkeypatch, materialized
):
    installer = _installer()
    _InPlace.error = inplace_module.InPlaceUnsupported("no routed experts")
    reloaded = []
    monkeypatch.setattr(
        installer,
        "install_checkpoint",
        lambda path, overlay=(): reloaded.append((path, overlay)),
    )
    installer._live_version = "v1"
    overlay = () if materialized else (staged_checkpoint.staging,)
    checkpoint = dataclasses.replace(staged_checkpoint, overlay=overlay)

    installer.install(PreparedCheckpointArtifact(checkpoint))

    # Files that hold an earlier version reload through the staged deltas since.
    assert reloaded == [(checkpoint.path, overlay)]
    assert installer._live_version == "v2"


def test_in_place_install_replaces_the_reload(staged_checkpoint, monkeypatch):
    installer = _installer()
    reloaded = []
    monkeypatch.setattr(
        installer, "install_checkpoint", lambda path, overlay=(): reloaded.append(path)
    )
    installer._live_version = "v1"

    metrics = installer.install(PreparedCheckpointArtifact(staged_checkpoint))

    assert reloaded == []
    assert metrics["perf/mx_receive_inplace_expert_layers"] == 2.0
    assert installer._live_version == "v2"
