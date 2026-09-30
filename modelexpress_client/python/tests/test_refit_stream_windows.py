# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from modelexpress_rl.inference.engines.vllm.stream_windows import (
    layer_windows,
    windowed_weights,
)


def test_layer_windows_group_decoder_layers_after_other_tensors():
    names = [
        "model.layers.3.mlp.experts.0.w",
        "lm_head.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.1.mlp.gate.weight",
        "vision_tower.blocks.0.wo.weight",
        "model.layers.2.input_layernorm.weight",
        "model.layers.10.mlp.gate.weight",
    ]

    assert layer_windows(names, 2) == [
        ["lm_head.weight", "vision_tower.blocks.0.wo.weight"],
        ["model.layers.0.self_attn.o_proj.weight", "model.layers.1.mlp.gate.weight"],
        ["model.layers.2.input_layernorm.weight", "model.layers.3.mlp.experts.0.w"],
        ["model.layers.10.mlp.gate.weight"],
    ]


def test_layer_windows_zero_disables_windowing():
    names = ["model.layers.1.w", "embed.weight", "model.layers.0.w"]
    assert layer_windows(names, 0) == [sorted(names)]


def test_layer_windows_rejects_negative_size():
    with pytest.raises(ValueError):
        layer_windows(["a"], -1)


class _Metadata:
    def __init__(self, name, value):
        self.name = name
        self.value = value


class _Chunks:
    def __init__(self, id, path, offsets, sizes):
        self.id, self.path, self.offsets, self.sizes = id, path, offsets, sizes

    @staticmethod
    def contiguous(id, path, offset, sizes):
        offsets = []
        for size in sizes:
            offsets.append(offset)
            offset += size
        return _Chunks(id, path, offsets, sizes)


def _install_fake_streamer(monkeypatch, files):
    events = []

    class Streamer:
        def __enter__(self):
            events.append("enter")
            return self

        def __exit__(self, *exc):
            events.append("exit")

        def stream_files(self, requests, credentials, device, is_distributed):
            events.append(
                (
                    "stream",
                    [(r.id, r.path, list(r.offsets), list(r.sizes)) for r in requests],
                    device,
                    is_distributed,
                )
            )
            self._pending = [
                (r.id, i, (r.path, offset))
                for r in requests
                for i, offset in enumerate(r.offsets)
            ]

        def get_chunks(self):
            yield from reversed(self._pending)

    def prepare_request(fs, paths, credentials):
        events.append(("metadata", list(paths)))
        result = []
        for path in paths:
            header, tensors = files[path]
            result.append(
                (
                    header,
                    [_Metadata(name, value) for name, _, value in tensors],
                    [size for _, size, _ in tensors],
                )
            )
        return result

    by_location = {}
    for path, (header, tensors) in files.items():
        offset = header
        for name, size, value in tensors:
            by_location[(path, offset)] = (name, value)
            offset += size

    def create_torch_tensor(buffer, metadata):
        name, value = by_location[buffer]
        assert name == metadata.name
        return torch.tensor([value])

    root = ModuleType("runai_model_streamer")
    root.DistributedStreamer = Streamer
    root.FileChunks = _Chunks
    package = ModuleType("runai_model_streamer.safetensors_streamer")
    pytorch = ModuleType(
        "runai_model_streamer.safetensors_streamer.safetensors_pytorch"
    )
    pytorch.prepare_request = prepare_request
    pytorch.create_torch_tensor = create_torch_tensor
    package.safetensors_pytorch = pytorch
    root.safetensors_streamer = package
    monkeypatch.setitem(sys.modules, "runai_model_streamer", root)
    monkeypatch.setitem(
        sys.modules, "runai_model_streamer.safetensors_streamer", package
    )
    monkeypatch.setitem(
        sys.modules,
        "runai_model_streamer.safetensors_streamer.safetensors_pytorch",
        pytorch,
    )
    platforms = ModuleType("vllm.platforms")
    platforms.current_platform = SimpleNamespace(
        is_cuda_alike=lambda: True, current_device=lambda: 3
    )
    monkeypatch.setitem(sys.modules, "vllm.platforms", platforms)
    return events


def test_windowed_weights_stream_each_window_once_with_exact_ranges(monkeypatch):
    files = {
        "s3://b/a.safetensors": (
            100,
            [
                ("model.layers.0.w", 10, 0.0),
                ("model.layers.1.w", 20, 1.0),
                ("embed.weight", 5, 9.0),
            ],
        ),
        "s3://b/b.safetensors": (
            50,
            [("model.layers.1.v", 7, 1.5), ("model.layers.2.w", 3, 2.0)],
        ),
    }
    events = _install_fake_streamer(monkeypatch, files)
    windows = [
        ["embed.weight"],
        ["model.layers.0.w", "model.layers.1.v", "model.layers.1.w"],
        ["model.layers.2.w"],
    ]

    loaded = list(windowed_weights(list(files), windows, is_distributed=True))

    assert [name for name, _ in loaded] == [
        "embed.weight",
        "model.layers.1.v",
        "model.layers.1.w",
        "model.layers.0.w",
        "model.layers.2.w",
    ]
    assert {n: t.item() for n, t in loaded}["model.layers.1.v"] == 1.5
    assert events == [
        "enter",
        ("metadata", list(files)),
        ("stream", [(0, "s3://b/a.safetensors", [130], [5])], "cuda:3", True),
        (
            "stream",
            [
                (0, "s3://b/a.safetensors", [100, 110], [10, 20]),
                (1, "s3://b/b.safetensors", [50], [7]),
            ],
            "cuda:3",
            True,
        ),
        ("stream", [(1, "s3://b/b.safetensors", [57], [3])], "cuda:3", True),
        "exit",
    ]


def test_windowed_weights_use_cpu_when_not_distributed(monkeypatch):
    files = {"s3://b/a.safetensors": (8, [("model.layers.0.w", 4, 1.0)])}
    events = _install_fake_streamer(monkeypatch, files)

    list(windowed_weights(list(files), [["model.layers.0.w"]], is_distributed=False))

    assert events[2][2:] == ("cpu", False)


def test_windowed_weights_reject_windows_that_do_not_cover_checkpoint(monkeypatch):
    files = {
        "s3://b/a.safetensors": (
            8,
            [("model.layers.0.w", 4, 1.0), ("model.layers.1.w", 4, 2.0)],
        )
    }
    events = _install_fake_streamer(monkeypatch, files)

    with pytest.raises(ValueError, match="windows"):
        list(windowed_weights(list(files), [["model.layers.0.w"]], is_distributed=True))
    assert not any(isinstance(e, tuple) and e[0] == "stream" for e in events)
