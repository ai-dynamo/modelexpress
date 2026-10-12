# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from modelexpress.refit.reshard.types import IncompleteRefit
from modelexpress_rl.inference.engines.vllm.installer import _VllmInstaller
from modelexpress_rl.inference.plan import PreparedStreamingTensors, SourceReplayBatch
from tests.test_refit_vllm_installer import _install_fake_vllm


def _transfer_and_install(installer, sources, captured, layout, bounded):
    names = frozenset(layout)
    last = {copy.param_name: index for index, copy in enumerate(captured.copies)}
    writes = tuple(
        (index, copy, sources[copy.src_name])
        for index, copy in enumerate(captured.copies)
    )
    if bounded:
        def batches():
            for index, recorded, source in writes:
                borrowed = source.clone()
                yield SourceReplayBatch(
                    ((index, recorded, borrowed),), names, last,
                    len(writes), captured.generation,
                )
                # The iterator may overwrite its receive arena immediately.
                borrowed.fill_(97)

        installer.install_streaming(
            PreparedStreamingTensors(batches, names, {})
        )
    else:
        installer.install_tensors(
            SourceReplayBatch(writes, names, last, len(writes), captured.generation)
        )


def test_missing_replay_api_fails_before_initialization(monkeypatch):
    initialized = []
    _install_fake_vllm(monkeypatch, initialized.append)
    del sys.modules["vllm.model_executor.model_loader.reload.meta"].materialize_layer
    model = nn.Linear(2, 2, bias=False)
    installer = _VllmInstaller(
        model=model,
        vllm_config=object(),
        model_config=object(),
        device=torch.device("cpu"),
    )
    with pytest.raises(RuntimeError, match="layerwise reload APIs"):
        installer.install_tensors({"weight": torch.ones(2, 2)})
    assert not initialized


@pytest.mark.parametrize("registry", ["parameter", "buffer"])
def test_persistent_only_owner_skips_materialization(monkeypatch, registry):
    from types import ModuleType

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            if registry == "parameter":
                self.bias = nn.Parameter(torch.zeros(2), requires_grad=False)
            else:
                self.register_buffer("bias", torch.zeros(2))

        def load_weights(self, weights):
            for _, tensor in weights:
                self.bias.copy_(tensor)

    model = Model()
    original = model.bias
    reset = []

    def initialize(target):
        pair = (
            ({"bias": original}, {})
            if registry == "parameter"
            else ({}, {"bias": original})
        )
        layerwise.LAYERWISE_INFO[target] = SimpleNamespace(
            kernel_tensors=pair, loaded_weights=(), reset=lambda: reset.append(True)
        )

    _install_fake_vllm(monkeypatch, initialize)
    layerwise = sys.modules["vllm.model_executor.model_loader.reload.layerwise"]
    layerwise._get_original_loader = lambda tensor: getattr(
        tensor, "weight_loader", None
    )
    layerwise._place_kernel_tensors = lambda layer, info: None
    utils = ModuleType("vllm.model_executor.model_loader.weight_utils")
    utils.default_weight_loader = lambda parameter, value: parameter.copy_(value)
    monkeypatch.setitem(sys.modules, utils.__name__, utils)

    def materialize(*args):
        pytest.fail("a persistent-only owner must not materialize unrelated tensors")

    sys.modules[
        "vllm.model_executor.model_loader.reload.meta"
    ].materialize_layer = materialize
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    installer = _VllmInstaller(
        model=model,
        vllm_config=object(),
        model_config=object(),
        device=torch.device("cpu"),
    )
    captured, layout = installer.capture([("bias", torch.float32, (2,))])
    assert len(captured.copies) == 1 and set(layout) == {"bias"}
    installer.install_streaming(
        PreparedStreamingTensors(
            lambda: iter([{"bias": torch.tensor([3.0, 7.0])}]), frozenset(layout), {}
        )
    )
    assert model.bias is original and torch.equal(model.bias, torch.tensor([3.0, 7.0]))


@pytest.mark.parametrize("streaming", [False, True])
def test_replaced_pending_slot_fails_before_its_copy(monkeypatch, streaming):
    from modelexpress.refit.reshard.types import IncompleteRefit

    _install_fake_vllm(monkeypatch, lambda model: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    base = sys.modules[
        "vllm.model_executor.layers.quantization.base_config"
    ].QuantizeMethodBase
    model = nn.Module()
    model.a = nn.Linear(2, 2, bias=False)
    model.b = nn.Linear(2, 2, bias=False)
    replacement = nn.Parameter(torch.zeros(2, 2))

    class Mutate(base):
        def process_weights_after_loading(self, owner):
            model.b.weight = replacement

    model.a.quant_method = Mutate()
    installer = _VllmInstaller(
        model=model,
        vllm_config=object(),
        model_config=object(),
        device=torch.device("cpu"),
    )
    tensors = {name: torch.ones_like(value) for name, value in model.named_parameters()}
    with pytest.raises(IncompleteRefit, match="pending.*slot"):
        if streaming:
            installer.install_streaming(
                PreparedStreamingTensors(
                    lambda: iter([{name: value} for name, value in tensors.items()]),
                    frozenset(tensors),
                    {},
                )
            )
        else:
            installer.install_tensors(tensors)
    assert torch.equal(replacement, torch.zeros(2, 2))


def _persistent_installer(monkeypatch, model):
    from contextlib import nullcontext
    from types import ModuleType

    import modelexpress_rl.inference.nixl_staged_transfer as transfer_module

    _install_fake_vllm(monkeypatch, lambda model: None)
    layerwise = sys.modules["vllm.model_executor.model_loader.reload.layerwise"]
    layerwise._get_original_loader = lambda tensor: getattr(
        tensor, "weight_loader", None
    )
    layerwise._place_kernel_tensors = lambda layer, info: None
    utils = ModuleType("vllm.model_executor.model_loader.weight_utils")
    utils.default_weight_loader = lambda parameter, value: parameter.copy_(value)
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    monkeypatch.setenv("MX_RESHARD_PUBLISH_DIGEST", "0")
    monkeypatch.setattr(transfer_module, "classic_cuda_alloc", nullcontext)
    return _VllmInstaller(
        model=model,
        vllm_config=object(),
        model_config=object(),
        device=torch.device("cpu"),
    )


@pytest.mark.parametrize("registry", ["parameter", "buffer"])
@pytest.mark.parametrize("bounded", [False, True])
@pytest.mark.parametrize("sliced", [False, True])
def test_transposed_persistent_destination_reload(
    monkeypatch, registry, bounded, sliced
):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            value = torch.zeros(2, 7)[:, 1::2].t() if sliced else torch.zeros(2, 3).t()
            if registry == "parameter":
                self.bias = nn.Parameter(value, requires_grad=False)
            else:
                self.register_buffer("bias", value)

        def load_weights(self, weights):
            for _, source in weights:
                target = self.bias[1:] if sliced else self.bias
                target.copy_(source)

    model = Model()
    original, pointer = model.bias, model.bias.data_ptr()
    installer = _persistent_installer(monkeypatch, model)
    stride = model.bias.stride()
    source = (
        torch.arange(4.0).reshape(2, 2) if sliced else torch.arange(6.0).reshape(3, 2)
    )
    native = Model()
    native.load_weights([("bias", source)])
    captured, layout = installer.capture([("bias", source.dtype, tuple(source.shape))])
    _transfer_and_install(installer, {"bias": source}, captured, layout, bounded)
    assert model.bias is original and model.bias.data_ptr() == pointer
    assert model.bias.stride() == stride
    assert torch.equal(model.bias, native.bias)


@pytest.mark.parametrize("bounded", [False, True])
@pytest.mark.parametrize("view", [False, True])
@pytest.mark.parametrize("interleaved", [False, True])
def test_shared_storage_buffer_destinations_remain_distinct(
    monkeypatch, bounded, view, interleaved
):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            storage = torch.zeros(4)
            self.register_buffer("a", storage[::2] if interleaved else storage[:2])
            self.register_buffer("b", storage[1::2] if interleaved else storage[2:])

        def load_weights(self, weights):
            for name, source in weights:
                target = getattr(self, name)
                (target[:] if view else target).copy_(source)

    model = Model()
    originals = {
        name: (tensor, tensor.data_ptr()) for name, tensor in model.named_buffers()
    }
    installer = _persistent_installer(monkeypatch, model)
    sources = {"a": torch.tensor([1.0, 2.0]), "b": torch.tensor([7.0, 9.0])}
    captured, layout = installer.capture(
        [(name, value.dtype, tuple(value.shape)) for name, value in sources.items()]
    )
    assert [copy.param_name for copy in captured.copies] == ["a", "b"]
    assert all(copy.dest_offset == 0 for copy in captured.copies)
    _transfer_and_install(installer, sources, captured, layout, bounded)
    for name, source in sources.items():
        tensor = getattr(model, name)
        assert tensor is originals[name][0] and tensor.data_ptr() == originals[name][1]
        assert torch.equal(tensor, source)
    assert model.a.untyped_storage()._cdata == model.b.untyped_storage()._cdata


@pytest.mark.parametrize("bounded", [False, True])
@pytest.mark.parametrize("digest", [False, True])
def test_mixed_replay_and_direct_overwrite_preserves_capture_order(
    monkeypatch, bounded, digest
):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("bias", torch.zeros(2))

        def load_weights(self, weights):
            for name, source in weights:
                if name == "a":
                    self.bias.copy_(source.float())
                else:
                    self.bias[:1].copy_(source)

    sources = {
        "a": torch.tensor([3.0, 7.0], dtype=torch.float16),
        "b": torch.tensor([11.0]),
    }
    native = Model()
    native.load_weights(list(sources.items()))
    model = Model()
    installer = _persistent_installer(monkeypatch, model)
    monkeypatch.setenv("MX_RESHARD_PUBLISH_DIGEST", "1" if digest else "0")
    captured, layout = installer.capture(
        [(name, value.dtype, tuple(value.shape)) for name, value in sources.items()]
    )
    _transfer_and_install(installer, sources, captured, layout, bounded)
    assert torch.equal(model.bias, native.bias)


@pytest.mark.parametrize("bounded", [False, True])
@pytest.mark.parametrize("stamped", [False, True])
def test_overlapping_registered_destinations_replay_in_recorded_order(
    monkeypatch, bounded, stamped
):

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.backing = torch.full((4,), -5.0)
            self.register_buffer("a", self.backing[:3])
            self.register_buffer("b", self.backing[1:])

        def load_weights(self, weights):
            for name, source in weights:
                target = self.b if name == "a_in" else self.a
                if stamped:
                    target.weight_loader(target[:], source)
                else:
                    target.copy_(source)

    model = Model()
    installer = _persistent_installer(monkeypatch, model)
    sources = {
        "a_in": torch.tensor([7.0, 8.0, 9.0]),
        "b_in": torch.tensor([1.0, 2.0, 3.0]),
    }
    native = Model()
    for tensor in native.buffers():
        tensor.weight_loader = lambda target, source: target.copy_(source)
    native.load_weights(list(sources.items()))
    assert torch.equal(native.backing, torch.tensor([1.0, 2.0, 3.0, 9.0]))
    originals = [(tensor, tensor.data_ptr()) for tensor in model.buffers()]
    captured, layout = installer.capture(
        [(name, source.dtype, tuple(source.shape)) for name, source in sources.items()]
    )
    _transfer_and_install(installer, sources, captured, layout, bounded)
    assert torch.equal(model.backing, native.backing)
    for tensor, (original, pointer) in zip(model.buffers(), originals):
        assert tensor is original and tensor.data_ptr() == pointer


@pytest.mark.parametrize("bounded", [False, True])
def test_identical_persistent_buffer_aliases_remain_supported(monkeypatch, bounded):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            tensor = torch.full((2,), 13.0)
            self.register_buffer("a", tensor)
            self.register_buffer("b", tensor)

        def load_weights(self, weights):
            for _, source in weights:
                self.b.copy_(source)

    model = Model()
    original = model.a
    source = torch.tensor([3.0, 7.0])
    installer = _persistent_installer(monkeypatch, model)
    captured, layout = installer.capture(
        [("source", source.dtype, tuple(source.shape))]
    )
    assert not captured.unsupported
    _transfer_and_install(installer, {"source": source}, captured, layout, bounded)
    assert model.a is model.b is original
    assert torch.equal(model.a, source)


@pytest.mark.parametrize("bounded", [False, True])
def test_interleaved_owners_replay_across_source_batches(monkeypatch, bounded):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.backing = torch.full((6,), -17.0)
            self.a = nn.Module()
            self.b = nn.Module()
            self.a.register_buffer("weight", self.backing[1:5])
            self.b.register_buffer("weight", self.backing[2:6])

        def load_weights(self, weights):
            for name, value in weights:
                target = self.b.weight if name == "1" else self.a.weight
                target.copy_(value)

    source = {str(i): torch.full((4,), float(i + 1)) for i in range(3)}
    native, model = Model(), Model()
    native.load_weights(source.items())
    installer = _persistent_installer(monkeypatch, model)
    captured, layout = installer.capture(
        [(name, tensor.dtype, tuple(tensor.shape)) for name, tensor in source.items()]
    )
    assert [copy.param_name for copy in captured.copies] == [
        "a.weight",
        "b.weight",
        "a.weight",
    ]
    _transfer_and_install(installer, source, captured, layout, bounded)
    assert torch.equal(model.backing, native.backing)


@pytest.mark.parametrize("bounded", [False, True])
def test_materialized_overlapping_owners_keep_recorded_effects(monkeypatch, bounded):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.backing = torch.full((6,), -17.0)
            self.a, self.b = nn.Module(), nn.Module()
            self.a.weight = nn.Parameter(self.backing[1:5], requires_grad=False)
            self.b.weight = nn.Parameter(self.backing[2:6], requires_grad=False)
            self.register_buffer("untouched_alias", self.backing[2:4], persistent=False)

        def load_weights(self, weights):
            for name, value in weights:
                (self.b if name == "1" else self.a).weight.copy_(value)

    model = Model()
    installer = _persistent_installer(monkeypatch, model)
    layerwise = sys.modules["vllm.model_executor.model_loader.reload.layerwise"]
    originals = {owner: owner.weight for owner in (model.a, model.b)}

    class Info:
        loaded_weights = ()

        def __init__(self, tensor):
            self.kernel_tensors = ({"weight": tensor}, {})

        def reset(self):
            self.kernel_tensors = None

    def initialize(target):
        for owner in (target.a, target.b):
            layerwise.LAYERWISE_INFO[owner] = Info(originals[owner])
            owner.weight = nn.Parameter(
                torch.empty(4, device="meta"), requires_grad=False
            )

    def place(owner, info):
        owner.weight = info.kernel_tensors[0]["weight"]

    def restore(owner, info):
        original = info.kernel_tensors[0]["weight"]
        original.copy_(owner.weight)
        owner.weight = original

    layerwise.initialize_layerwise_reload = initialize
    layerwise._place_kernel_tensors = place
    layerwise._copy_and_restore_kernel_tensors = restore
    pointers = [tensor.data_ptr() for tensor in originals.values()]
    for update in (0, 10):
        source = {str(i): torch.full((4,), float(update + i + 1)) for i in range(3)}
        native = Model()
        native.load_weights(source.items())
        captured, layout = installer.capture(
            [
                (name, tensor.dtype, tuple(tensor.shape))
                for name, tensor in source.items()
            ]
        )
        _transfer_and_install(installer, source, captured, layout, bounded)
        assert torch.equal(model.backing, native.backing)
        assert torch.equal(model.untouched_alias, native.untouched_alias)
        assert [owner.weight for owner in originals] == list(originals.values())
        assert [owner.weight.data_ptr() for owner in originals] == pointers




def test_online_fp8_streaming_rejected_before_reload_or_source(monkeypatch):
    class OnlineFp8:
        pass

    OnlineFp8.__module__ = "vllm.model_executor.layers.quantization.online.fp8"
    model = nn.Linear(2, 2, bias=False)
    model.quant_method = OnlineFp8()
    installer = _persistent_installer(monkeypatch, model)
    assert PreparedStreamingTensors not in installer.capabilities.artifact_types
    monkeypatch.setattr(installer, "_reload", lambda callback: pytest.fail("reload started"))
    prepared = PreparedStreamingTensors(
        lambda: pytest.fail("source read started"), frozenset({"weight"}), {}
    )
    with pytest.raises(IncompleteRefit, match="online block-FP8"):
        installer.install_streaming(prepared)


@pytest.mark.parametrize("invalid", ["reordered", "incomplete", "generation"])
def test_replay_rejects_invalid_update(monkeypatch, invalid):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("weight", torch.zeros(2))

        def load_weights(self, weights):
            for _, value in weights:
                self.weight.copy_(value)

    model = Model()
    installer = _persistent_installer(monkeypatch, model)
    capture, layout = installer.capture([("a", torch.float32, (2,)), ("b", torch.float32, (2,))])
    writes = tuple((i, recorded, torch.ones(2)) for i, recorded in enumerate(capture.copies))
    if invalid == "reordered":
        writes = tuple(reversed(writes))
    if invalid == "incomplete":
        writes = writes[:1]
    batch = SourceReplayBatch(
        writes, frozenset(layout), {"weight": 1}, 2,
        1 if invalid == "generation" else 0,
    )
    with pytest.raises(IncompleteRefit, match="recorded order|before every|invalidated"):
        installer.install_tensors(batch)
    if invalid != "incomplete":
        assert torch.equal(model.weight, torch.zeros(2))


@pytest.mark.parametrize("cast", [False, True])
def test_quantized_replay_reuses_capture_and_restores_renamed_storage(monkeypatch, cast):
    from types import ModuleType

    monkeypatch.setenv("MX_RESHARD_PUBLISH_DIGEST", "0")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    sources = {
        f"layer{i}.weight": torch.linspace(0.1, 2.0, 256).reshape(16, 16) + i
        for i in range(6)
    }

    class Model(nn.Module):
        capture_calls = 0

        def load_weights(self, weights):
            self.capture_calls += 1
            for name, tensor in weights:
                target = self.get_submodule(name.rpartition(".")[0]).weight
                target.copy_(tensor.bfloat16().float() if cast else tensor)

    model = Model()
    originals = {}
    for name in sources:
        owner = nn.Module()
        owner.packed = nn.Parameter(torch.zeros(16, 16), requires_grad=False)
        model.add_module(name.split(".")[0], owner)
        originals[owner] = owner.packed

    class Info:
        loaded_weights = ()

        def __init__(self, tensor):
            self.kernel_tensors = ({"packed": tensor}, {})

        def reset(self):
            self.kernel_tensors = None

    def initialize(target):
        for owner in target.children():
            layerwise.LAYERWISE_INFO[owner] = Info(originals[owner])
            del owner.packed
            owner.weight = nn.Parameter(torch.empty(16, 16, device="meta"))

    _install_fake_vllm(monkeypatch, initialize)
    layerwise = sys.modules["vllm.model_executor.model_loader.reload.layerwise"]
    layerwise._get_original_loader = lambda tensor: getattr(
        tensor, "weight_loader", None
    )

    def place(owner, info):
        del owner.weight
        owner.packed = info.kernel_tensors[0]["packed"]

    layerwise._place_kernel_tensors = place
    utils = ModuleType("vllm.model_executor.model_loader.weight_utils")
    utils.default_weight_loader = lambda target, value: target.copy_(value)
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    quant_base = sys.modules[
        "vllm.model_executor.layers.quantization.base_config"
    ].QuantizeMethodBase

    class Quant(quant_base):
        def process_weights_after_loading(self, owner):
            owner.packed = nn.Parameter(owner.weight * 2, requires_grad=False)
            del owner.weight

    def restore(owner, info):
        original = info.kernel_tensors[0]["packed"]
        original.copy_(owner.packed)
        owner.packed = original

    layerwise._copy_and_restore_kernel_tensors = restore
    for owner in model.children():
        owner.quant_method = Quant()
    manifest = [
        (name, tensor.dtype, tuple(tensor.shape)) for name, tensor in sources.items()
    ]
    installer = _VllmInstaller(
        model=model,
        vllm_config=SimpleNamespace(quant_config=object()),
        model_config=object(),
        device=torch.device("cpu"),
    )
    captured, layout = installer.capture(manifest)
    assert model.capture_calls == 1
    assert set(layout) == set(sources)
    assert all(owner.packed is originals[owner] for owner in model.children())
    assert PreparedStreamingTensors in installer.capabilities.artifact_types
    pointer_by_owner = {owner: tensor.data_ptr() for owner, tensor in originals.items()}
    for update in range(2):
        if update:
            for source in sources.values():
                source.add_(3)
        captured_again, layout_again = installer.capture(manifest)
        assert model.capture_calls == 1
        assert captured_again == captured and layout_again == layout
        _transfer_and_install(installer, sources, captured_again, layout_again, True)
        for name, source in sources.items():
            owner = getattr(model, name.split(".")[0])
            expected = source.bfloat16().float() if cast else source
            assert not hasattr(owner, "weight")
            assert owner.packed is originals[owner]
            assert owner.packed.data_ptr() == pointer_by_owner[owner]
            assert torch.equal(owner.packed, expected * 2)


@pytest.mark.parametrize("mutation", ["zero", "fill", "copy", "view"])
def test_capture_rejects_live_destination_mutation_before_changing_values(
    monkeypatch, mutation
):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("weight", torch.full((4,), 7.0))

        def load_weights(self, weights):
            if mutation == "zero":
                self.weight.zero_()
            elif mutation == "fill":
                self.weight.fill_(3)
            elif mutation == "copy":
                self.weight.copy_(torch.zeros(4))
            else:
                self.weight[1:].add_(3)
            for _, source in weights:
                self.weight[:2].copy_(source)

    model = Model()
    installer = _persistent_installer(monkeypatch, model)
    captured, _ = installer.capture([("weight", torch.float32, (2,))])
    assert torch.equal(model.weight, torch.full((4,), 7.0))
    assert captured.unsupported == ["weight"]
    assert "live destination mutation" in captured.unsupported_reasons["weight"]
