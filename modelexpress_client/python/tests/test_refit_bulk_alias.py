# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Python alias validation preserves live ownership and callback ordering."""

import gc

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


@pytest.mark.parametrize("mutate_before_capture", [False, True])
def test_python_owner_check_observes_mutated_getattr_code(
    monkeypatch, mutate_before_capture
):
    model, installer = _fixture()
    aliases = None if mutate_before_capture else installer._parameter_aliases(model)

    def changed_getattr(self, name):
        raise RuntimeError("changed module lookup")

    monkeypatch.setattr(api._MODULE_GETATTR, "__code__", changed_getattr.__code__)
    if aliases is None:
        aliases = installer._parameter_aliases(model)
    with pytest.raises(RuntimeError, match="changed module lookup"):
        installer._validate_alias_owners(aliases)


def test_python_owner_check_rejects_replaced_module():
    model, installer = _fixture()
    aliases = installer._parameter_aliases(model)
    installer._validate_alias_owners(aliases)
    model.branch = torch.nn.Linear(2, 2)
    with pytest.raises(api.IncompleteRefit, match="replaced during refit"):
        installer._validate_alias_owners(aliases)


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


def _select_alias_plan(plan, model):
    groups, modules = {}, {}
    for path, module in model.named_modules(remove_duplicate=False):
        modules[path] = module
        for name, parameter in module._parameters.items():
            if parameter is not None:
                groups.setdefault(id(parameter), []).append((path, module, name))
    return api._select_parameter_aliases(model, groups.values(), modules, plan)


def test_fresh_alias_plan_does_not_retain_replaced_parameters():
    import weakref

    model, installer = _fixture()
    first = _select_alias_plan(None, model)
    original = weakref.ref(model.branch.weight)
    model.branch.weight = torch.nn.Parameter(torch.ones(2, 2))
    gc.collect()
    assert original() is None
    assert _select_alias_plan(first, model) is first
    installer._restore_parameter_aliases(first)


@pytest.mark.parametrize(
    "mutation", ["split", "join", "add", "remove", "reorder", "class"]
)
def test_alias_plan_recompiles_changed_structure(mutation):
    model, _ = _fixture()
    model.other = torch.nn.Linear(2, 2)
    if mutation == "split":
        model.other.weight = model.branch.weight
    first = _select_alias_plan(None, model)
    if mutation == "split":
        model.other.weight = torch.nn.Parameter(torch.zeros(2, 2))
    elif mutation == "join":
        model.other.weight = model.branch.weight
    elif mutation == "add":
        model.other.extra = model.branch.weight
    elif mutation == "remove":
        del model.alias
    elif mutation == "reorder":
        model._modules = dict(reversed(tuple(model._modules.items())))
    else:

        class Different(torch.nn.Module):
            pass

        model.__class__ = Different
    assert _select_alias_plan(first, model) is not first


def test_alias_plan_declines_changed_custom_compile_behavior():
    calls = []

    class Meta(type):
        pass

    class Owner(torch.nn.Linear, metaclass=Meta):
        pass

    model, _ = _fixture()
    model.branch = model.alias = Owner(2, 2)
    first = _select_alias_plan(None, model)

    def changed_hash(cls):
        calls.append(cls.__name__)
        return type.__hash__(cls)

    Meta.__hash__ = changed_hash
    _select_alias_plan(None, model)
    expected = list(calls)
    calls.clear()
    assert _select_alias_plan(first, model) is not first
    assert calls == expected and calls


@pytest.mark.parametrize("kind", ["path", "leaf"])
def test_alias_plan_does_not_retain_nonstandard_names(kind):
    class Name(str):
        pass

    model, _ = _fixture()
    first = _select_alias_plan(None, model)
    modules = dict(model.named_modules(remove_duplicate=False))
    groups = [
        [
            (
                Name("branch") if kind == "path" else "branch",
                model.branch,
                Name("weight") if kind == "leaf" else "weight",
            ),
            ("alias", model.alias, "weight"),
        ]
    ]
    assert api._select_parameter_aliases(model, groups, modules, first) is not first


def test_parameter_reordering_reuses_owner_views_without_reordering_callbacks():
    model, _ = _fixture()
    model.other = torch.nn.Linear(2, 2)
    model.other_alias = model.other
    initial = _select_alias_plan(None, model)
    initial_views = initial.by_owner

    # Native reload leaves skipped parameters in place and re-registers others.
    model.branch._parameters = dict(reversed(tuple(model.branch._parameters.items())))
    reordered = _select_alias_plan(initial, model)
    fresh = _select_alias_plan(None, model)
    assert reordered is not initial
    assert reordered.groups == fresh.groups != initial.groups
    assert reordered.owners == fresh.owners
    assert reordered.by_owner[model.other] is initial_views[model.other]
    assert reordered.by_owner[model.branch] is not initial_views[model.branch]
    next_batch = _select_alias_plan(initial, model)
    assert next_batch.by_owner[model.branch] is reordered.by_owner[model.branch]


def test_new_alias_owners_do_not_enter_the_initial_owner_view_cache():
    model, _ = _fixture()
    initial = _select_alias_plan(None, model)
    initial.by_owner
    cached = dict(initial._owner_plans)
    model.other = torch.nn.Linear(2, 2)
    model.other_alias = model.other
    later = _select_alias_plan(initial, model)
    later.by_owner
    assert later._owner_plans is not initial._owner_plans
    assert initial._owner_plans == cached


def test_captured_alias_plan_rejects_changed_live_owner():
    model, installer = _fixture()
    first = _select_alias_plan(None, model)
    assert _select_alias_plan(first, model) is first
    model.branch = torch.nn.Linear(2, 2)
    with pytest.raises(api.IncompleteRefit, match="replaced during refit"):
        installer._restore_parameter_aliases(first)


def test_alias_plan_does_not_retain_newly_aliased_modules():
    import weakref

    model, _ = _fixture()
    first = _select_alias_plan(None, model)
    model.extra = model.extra_alias = torch.nn.Linear(2, 2)
    module = weakref.ref(model.extra)
    temporary = _select_alias_plan(first, model)
    assert temporary is not first
    del temporary, model.extra, model.extra_alias
    gc.collect()
    assert module() is None


def test_alias_plan_rechecks_mro_after_initial_compilation():
    class Base(torch.nn.Linear):
        pass

    class Alternative(torch.nn.Linear):
        pass

    class Owner(Base):
        pass

    model, _ = _fixture()
    model.branch = model.alias = Owner(2, 2)
    first = _select_alias_plan(None, model)
    Owner.__bases__ = (Alternative,)
    assert _select_alias_plan(first, model) is not first


def test_alias_plan_rechecks_class_changed_during_initial_compilation(monkeypatch):
    class Changed(torch.nn.Module):
        pass

    model, _ = _fixture()
    original = api._compile_parameter_aliases
    changed = []

    def compile_then_change(*args):
        plan = original(*args)
        if not changed:
            changed.append(True)
            model.__class__ = Changed
        return plan

    monkeypatch.setattr(api, "_compile_parameter_aliases", compile_then_change)
    first = _select_alias_plan(None, model)
    assert _select_alias_plan(first, model) is not first


def _reordered_alias_fixture(*, tie_weight=False):
    model = torch.nn.Module()
    model.primary = model.alias = torch.nn.LayerNorm(4)
    model.other = torch.nn.LayerNorm(4)
    model.other.bias = model.primary.bias
    if tie_weight:
        model.other.weight = model.primary.weight
    installer = api._VllmInstaller(
        model=model,
        vllm_config=object(),
        model_config=object(),
        device=torch.device("cpu"),
    )
    initial = _select_alias_plan(None, model)
    # Re-registering a materialized weight after a retained bias changes only
    # parameter iteration order, as an engine reload can do.
    for module in (model.primary, model.other):
        weight = module._parameters.pop("weight")
        module.register_parameter("weight", weight)
    return model, installer, initial


def _alias_rows(model):
    groups, modules = {}, {}
    for path, module in model.named_modules(remove_duplicate=False):
        modules[path] = module
        for leaf, parameter in module._parameters.items():
            if parameter is not None:
                groups.setdefault(id(parameter), []).append((path, module, leaf))
    return list(groups.values()), modules


def test_fresh_alias_plan_preserves_true_tie_geometry_callback_order():
    from torch.overrides import TorchFunctionMode

    model, installer, initial = _reordered_alias_fixture(tie_weight=True)
    fresh = _select_alias_plan(None, model)
    aligned = _select_alias_plan(initial, model)
    names = {id(model.primary.weight): "weight", id(model.primary.bias): "bias"}

    class Observe(TorchFunctionMode):
        def __init__(self):
            self.calls = []

        def __torch_function__(self, function, types, args=(), kwargs=None):
            if args and id(args[0]) in names:
                self.calls.append((function.__name__, names[id(args[0])]))
            return function(*args, **(kwargs or {}))

    def trace(plan):
        with Observe() as mode:
            installer._restore_parameter_aliases(plan)
        return mode.calls

    expected = trace(fresh)
    assert expected and expected[0][1] == "bias"
    assert trace(aligned) == expected
    assert trace(initial) != expected


def test_fresh_alias_plan_preserves_later_custom_lookup_and_setter_order(monkeypatch):
    model, installer, initial = _reordered_alias_fixture(tie_weight=True)
    aligned = _select_alias_plan(initial, model)
    fresh = _select_alias_plan(None, model)
    calls = []
    original_lookup = model.get_submodule
    original_set = torch.nn.LayerNorm.__setattr__

    def lookup(path):
        calls.append(("lookup", path))
        return original_lookup(path)

    def assign(owner, name, value):
        calls.append(("set", id(owner), name))
        return original_set(owner, name, value)

    monkeypatch.setattr(model, "get_submodule", lookup)
    monkeypatch.setattr(torch.nn.LayerNorm, "__setattr__", assign)
    installer._restore_parameter_aliases(fresh)
    expected = list(calls)
    calls.clear()
    installer._restore_parameter_aliases(aligned)
    assert calls == expected and calls


@pytest.mark.parametrize(
    "change",
    ["split", "join", "duplicate", "member-order", "owner-order", "ancestor", "class"],
)
def test_fresh_alias_plan_recompiles_changed_structure(change):
    model, _installer, initial = _reordered_alias_fixture()
    if change == "split":
        model.other.bias = torch.nn.Parameter(torch.zeros(4))
    elif change == "join":
        model.other.weight = model.primary.weight
    elif change == "owner-order":
        model._modules = dict(reversed(tuple(model._modules.items())))
    elif change == "class":

        class Changed(torch.nn.Module):
            pass

        model.__class__ = Changed
    groups, modules = _alias_rows(model)
    if change == "duplicate":
        groups.append(groups[0])
    elif change == "member-order":
        groups[0] = list(reversed(groups[0]))
    elif change == "ancestor":
        modules["alias"] = torch.nn.LayerNorm(4)
    selected = api._select_parameter_aliases(model, groups, modules, initial)
    assert selected is not initial


def test_fresh_alias_plan_declines_custom_metaclass_with_original_compile_trace():
    calls = []

    class Meta(type):
        pass

    class Owner(torch.nn.LayerNorm, metaclass=Meta):
        pass

    model = torch.nn.Module()
    model.primary = model.alias = Owner(4)
    initial = _select_alias_plan(None, model)
    model.primary._parameters = dict(reversed(tuple(model.primary._parameters.items())))

    def changed_hash(cls):
        calls.append(cls.__name__)
        return type.__hash__(cls)

    Meta.__hash__ = changed_hash
    _select_alias_plan(None, model)
    expected = list(calls)
    calls.clear()
    selected = _select_alias_plan(initial, model)
    assert selected is not initial
    assert calls == expected and calls


def test_reused_plan_keeps_complete_ties_in_owner_views():
    model, installer = _fixture()
    model.other = torch.nn.Linear(2, 2)
    model.other.weight = model.branch.weight
    plan = _select_alias_plan(None, model)
    views = plan.by_owner
    assert _select_alias_plan(plan, model).by_owner is views
    # Native materialization replaces the canonical parameter object. A view
    # for either owner must reconnect the entire tie, including other owners.
    replacement = torch.nn.Parameter(torch.ones(2, 2))
    model.branch.weight = replacement
    installer._restore_parameter_aliases(views[model.other])
    assert model.other.weight is replacement
    assert model.alias.weight is replacement
    model.other = torch.nn.Linear(2, 2)
    with pytest.raises(api.IncompleteRefit, match="replaced during refit"):
        installer._restore_parameter_aliases(views[model.branch])


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
