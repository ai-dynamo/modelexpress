# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Alias restoration and local tensor materialization safety."""

import gc
import weakref

import pytest
import torch
from modelexpress_rl.inference.engines.vllm import installer as api


def _fixture():
    model = torch.nn.Module()
    model.branch = torch.nn.Linear(2, 2)
    model.alias = model.branch
    installer = api._VllmInstaller(
        model=model,
        vllm_config=object(),
        model_config=object(),
        device=torch.device("cpu"),
    )
    return model, installer


def test_local_alias_views_decline_custom_ancestor_lookup(monkeypatch):
    class Root(torch.nn.Module):
        pass

    model = Root()
    model.branch = model.alias = torch.nn.Linear(2, 2)
    installer = api._VllmInstaller(
        model=model,
        vllm_config=object(),
        model_config=object(),
        device=torch.device("cpu"),
    )
    aliases = installer._parameter_aliases(model)
    assert api._ordinary_alias_lookups(aliases)

    def lookup(self, name):
        return torch.nn.Module.__getattr__(self, name)

    monkeypatch.setattr(Root, "__getattr__", lookup)
    assert not api._ordinary_alias_lookups(aliases)


def test_local_alias_views_accept_ordinary_mixin_dictionaries():
    class Mixin:
        pass

    class Root(Mixin, torch.nn.Module):
        pass

    model = Root()
    model.branch = model.alias = torch.nn.Linear(2, 2)
    installer = api._VllmInstaller(
        model=model,
        vllm_config=object(),
        model_config=object(),
        device=torch.device("cpu"),
    )
    aliases = installer._parameter_aliases(model)
    assert api._ordinary_alias_lookups(aliases)

    class CustomRoot(Root):
        @property
        def __dict__(self):
            raise RuntimeError("custom dictionary callback")

    model.__class__ = CustomRoot
    assert not api._ordinary_alias_lookups(aliases)


def test_materialization_accepts_ordinary_mixin_dictionaries():
    from types import SimpleNamespace

    class Mixin:
        pass

    class Layer(Mixin, torch.nn.Linear):
        pass

    layer = Layer(2, 2)
    info = SimpleNamespace(restore_device=torch.device("cpu"))
    assert api._materialization_is_local(layer, info)

    class CustomLayer(Layer):
        @property
        def __dict__(self):
            raise RuntimeError("custom dictionary callback")

    layer.__class__ = CustomLayer
    assert not api._materialization_is_local(layer, info)


def test_python_writer_check_reads_replaced_registries_and_instance_hooks():
    model, installer = _fixture()
    aliases = installer._parameter_aliases(model)
    assert api._standard_parameter_writes(aliases)
    model.branch._parameters = dict(model.branch._parameters)
    model.branch.weight = torch.nn.Parameter(torch.ones(2, 2))
    assert api._standard_parameter_writes(aliases)
    model.branch.register_parameter = lambda name, value: None
    assert not api._standard_parameter_writes(aliases)


def test_python_writer_check_does_not_repeat_custom_hook_truth(monkeypatch):
    model, installer = _fixture()
    aliases = installer._parameter_aliases(model)
    calls = []

    class Hooks(dict):
        def __bool__(self):
            calls.append("bool")
            return len(calls) > 1

    monkeypatch.setattr(
        torch.nn.modules.module, "_global_parameter_registration_hooks", Hooks()
    )
    assert api._standard_parameter_writes(aliases)
    assert calls == ["bool"]


def test_retained_alias_metadata_does_not_pin_replaced_parameters() -> None:
    model, installer = _fixture()
    aliases = installer._parameter_aliases(model)
    original = weakref.ref(model.branch.weight)
    model.branch.weight = torch.nn.Parameter(torch.ones(2, 2))
    gc.collect()
    assert original() is None
    assert aliases.groups


def test_cached_alias_owner_view_restores_complete_ties() -> None:
    model, installer = _fixture()
    model.other = torch.nn.Linear(2, 2)
    model.other.weight = model.branch.weight
    plan = installer._parameter_aliases(model)
    views = plan.by_owner
    # Native materialization replaces the canonical parameter object. A view
    # for either owner must reconnect the entire tie, including other owners.
    replacement = torch.nn.Parameter(torch.ones(2, 2))
    model.branch.weight = replacement
    installer._restore_parameter_aliases(views[model.other])
    assert model.other.weight is replacement
    assert model.alias.weight is replacement


@pytest.mark.parametrize(
    "custom",
    [
        "setter",
        "parameter_hook",
        "buffer_hook",
        "hook_dictionary",
        "registry",
        "module_dictionary",
        "tensor",
        "tensor_instance",
        "tensor_dictionary",
        "context",
    ],
)
def test_materialization_locality_declines_user_callbacks(monkeypatch, custom):
    from types import SimpleNamespace

    class Layer(torch.nn.Linear):
        pass

    layer = Layer(2, 2)
    info = SimpleNamespace(restore_device=torch.device("cpu"))
    assert api._materialization_is_local(layer, info)
    if custom == "setter":
        monkeypatch.setattr(
            Layer,
            "__setattr__",
            lambda self, name, value: torch.nn.Module.__setattr__(self, name, value),
        )
    elif custom in ("parameter_hook", "buffer_hook"):
        name = (
            "_global_parameter_registration_hooks"
            if custom == "parameter_hook"
            else "_global_buffer_registration_hooks"
        )
        monkeypatch.setitem(
            getattr(torch.nn.modules.module, name),
            "locality-test",
            lambda module, name, value: value,
        )
    elif custom == "registry":

        class Registry(dict):
            pass

        layer._parameters = Registry(layer._parameters)
    elif custom == "hook_dictionary":

        class Hooks(dict):
            def __bool__(self):
                raise RuntimeError("hook metadata callback")

        monkeypatch.setattr(
            torch.nn.modules.module, "_global_parameter_registration_hooks", Hooks()
        )
    elif custom == "context":
        info.restore_device = object()
    elif custom == "module_dictionary":

        class ModuleAttributes(dict):
            def get(self, *args):
                raise RuntimeError("module metadata callback")

        layer.__dict__ = ModuleAttributes(layer.__dict__)
    elif custom == "tensor_instance":
        layer.weight.size = lambda: torch.Size((2, 2))
    elif custom == "tensor_dictionary":

        class TensorAttributes(dict):
            def copy(self):
                raise RuntimeError("tensor metadata callback")

        layer.weight.__dict__ = TensorAttributes(layer.weight.__dict__)
    else:

        class Parameter(torch.nn.Parameter):
            def size(self, *args):
                return super().size(*args)

        layer.weight = Parameter(torch.zeros(2, 2))
    assert not api._materialization_is_local(layer, info)


def test_materialization_locality_declines_torch_function_mode():
    from torch.overrides import TorchFunctionMode
    from types import SimpleNamespace

    class Observe(TorchFunctionMode):
        def __torch_function__(self, function, types, args=(), kwargs=None):
            return function(*args, **(kwargs or {}))

    layer = torch.nn.Linear(2, 2)
    info = SimpleNamespace(restore_device=torch.device("cpu"))
    with Observe():
        assert not api._materialization_is_local(layer, info)


def test_materialization_accepts_native_parameter_forwarding_and_rejects_overrides(
    monkeypatch,
):
    from types import SimpleNamespace

    class EngineParameter(torch.nn.Parameter):
        @classmethod
        def __torch_function__(cls, function, types, args=(), kwargs=None):
            return super().__torch_function__(function, types, args, kwargs or {})

    class Weight(EngineParameter):
        pass

    layer = torch.nn.Linear(2, 2)
    layer.weight = Weight(torch.ones(2, 2))
    info = SimpleNamespace(restore_device=torch.device("cpu"))
    function = EngineParameter.__torch_function__.__func__
    native = (EngineParameter, function, function.__code__, EngineParameter.__mro__)
    assert not api._materialization_is_local(layer, info)
    assert api._materialization_is_local(layer, info, native)

    class CustomWeight(Weight):
        @classmethod
        def __torch_function__(cls, function, types, args=(), kwargs=None):
            return super().__torch_function__(function, types, args, kwargs or {})

    layer.weight = CustomWeight(torch.ones(2, 2))
    assert not api._materialization_is_local(layer, info, native)
    layer.weight = Weight(torch.ones(2, 2))
    monkeypatch.setattr(function, "__code__", function.__code__.replace())
    assert not api._materialization_is_local(layer, info, native)


def test_materialization_declines_changed_native_parameter_inheritance():
    from types import SimpleNamespace

    class EngineParameter(torch.nn.Parameter):
        @classmethod
        def __torch_function__(cls, function, types, args=(), kwargs=None):
            return super().__torch_function__(function, types, args, kwargs or {})

    class AdditionalBase:
        pass

    class Weight(EngineParameter, AdditionalBase):
        pass

    function = EngineParameter.__torch_function__.__func__
    native = (EngineParameter, function, function.__code__, EngineParameter.__mro__)
    layer = torch.nn.Linear(2, 2)
    layer.weight = Weight(torch.ones(2, 2))
    info = SimpleNamespace(restore_device=torch.device("cpu"))
    assert not api._materialization_is_local(layer, info, native)


def test_materialization_accepts_ordinary_buffers_but_not_custom_tensor_dispatch():
    from types import SimpleNamespace

    layer = torch.nn.Linear(2, 2)
    layer.register_buffer("scale", torch.ones(2))
    info = SimpleNamespace(restore_device=torch.device("cpu"))
    assert api._materialization_is_local(layer, info)

    class CustomTensor(torch.Tensor):
        @classmethod
        def __torch_function__(cls, function, types, args=(), kwargs=None):
            return super().__torch_function__(function, types, args, kwargs or {})

    layer.scale = torch.ones(2).as_subclass(CustomTensor)
    assert not api._materialization_is_local(layer, info)
