# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Python alias validation preserves live ownership and callback ordering."""

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
def test_python_owner_check_observes_replaced_getattr(
    monkeypatch, mutate_before_capture
):
    model, installer = _fixture()
    aliases = None if mutate_before_capture else installer._parameter_aliases(model)

    def changed_getattr(self, name):
        raise RuntimeError("changed module lookup")

    monkeypatch.setattr(torch.nn.Module, "__getattr__", changed_getattr)
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



def test_fresh_aliases_release_replaced_parameters():
    import gc
    import weakref

    model, installer = _fixture()
    aliases = installer._parameter_aliases(model)
    original = weakref.ref(model.branch.weight)
    model.branch.weight = torch.nn.Parameter(torch.ones(2, 2))
    gc.collect()
    assert original() is None
    installer._restore_parameter_aliases(aliases)


@pytest.mark.parametrize("change", ["split", "join", "add", "remove", "reorder"])
def test_fresh_aliases_follow_current_tie_structure(change):
    model, installer = _fixture()
    model.other = torch.nn.Linear(2, 2)
    model.other.weight = model.branch.weight
    if change == "split":
        model.other.weight = torch.nn.Parameter(torch.zeros(2, 2))
    elif change == "join":
        model.other.bias = model.branch.bias
    elif change == "add":
        model.other.extra = model.branch.weight
    elif change == "remove":
        del model.alias
    else:
        model._modules = dict(reversed(tuple(model._modules.items())))
    aliases = installer._parameter_aliases(model)
    expected = {}
    for path, module in model.named_modules(remove_duplicate=False):
        for name, parameter in module._parameters.items():
            expected.setdefault(id(parameter), []).append((module, name))
    assert aliases.groups == tuple(tuple(group) for group in expected.values() if len(group) > 1)
    installer._restore_parameter_aliases(aliases)


def test_fresh_aliases_reconnect_complete_ties():
    model, installer = _fixture()
    model.other = torch.nn.Linear(2, 2)
    model.other.weight = model.branch.weight
    aliases = installer._parameter_aliases(model)
    replacement = torch.nn.Parameter(torch.ones(2, 2))
    model.branch.weight = replacement
    installer._restore_parameter_aliases(aliases)
    assert model.other.weight is replacement
    assert model.alias.weight is replacement


def test_alias_validation_observes_custom_ancestor_lookup(monkeypatch):
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
    installer._validate_alias_owners(aliases)

    calls = []

    def lookup(self, name):
        calls.append(name)
        return torch.nn.Module.__getattr__(self, name)

    monkeypatch.setattr(Root, "__getattr__", lookup)
    installer._validate_alias_owners(aliases)
    assert "branch" in calls and "alias" in calls


def test_alias_validation_accepts_mixins_and_observes_custom_dictionary():
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
    installer._validate_alias_owners(aliases)

    class CustomRoot(Root):
        @property
        def __dict__(self):
            raise RuntimeError("custom dictionary callback")

    model.__class__ = CustomRoot
    with pytest.raises(RuntimeError, match="custom dictionary callback"):
        installer._validate_alias_owners(aliases)


@pytest.mark.parametrize("custom", ["hash", "equality", "neither"])
def test_owner_views_preserve_ties_with_custom_hashing(custom):
    class Owner(torch.nn.Linear):
        pass

    if custom == "hash":
        Owner.__hash__ = lambda self: object.__hash__(self)
    elif custom == "equality":
        Owner.__eq__ = lambda self, other: self is other

    model, _ = _fixture()
    model.branch = model.alias = Owner(2, 2)
    model.other = model.other_alias = Owner(2, 2)
    model.other.weight = model.branch.weight
    aliases = api._VllmInstaller._parameter_aliases(model)
    views = aliases.by_owner
    assert set(views) == {model.branch, model.other}
    assert aliases.by_owner is views
    replacement = torch.nn.Parameter(torch.ones(2, 2))
    model.branch.weight = replacement
    installer = api._VllmInstaller(
        model=model, vllm_config=object(), model_config=object(),
        device=torch.device("cpu"),
    )
    installer._restore_parameter_aliases(views[model.other])
    assert model.other.weight is model.branch.weight is replacement


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
    initial = api._VllmInstaller._parameter_aliases(model)
    # Re-registering a materialized weight after a retained bias changes only
    # parameter iteration order, as an engine reload can do.
    for module in (model.primary, model.other):
        weight = module._parameters.pop("weight")
        module.register_parameter("weight", weight)
    return model, installer, initial


def test_fresh_alias_plan_preserves_true_tie_geometry_callback_order():
    from torch.overrides import TorchFunctionMode

    model, installer, initial = _reordered_alias_fixture(tie_weight=True)
    fresh = api._VllmInstaller._parameter_aliases(model)
    aligned = api._VllmInstaller._parameter_aliases(model)
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
    aligned = api._VllmInstaller._parameter_aliases(model)
    fresh = api._VllmInstaller._parameter_aliases(model)
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


def test_reused_plan_keeps_complete_ties_in_owner_views():
    model, installer = _fixture()
    model.other = torch.nn.Linear(2, 2)
    model.other.weight = model.branch.weight
    plan = api._VllmInstaller._parameter_aliases(model)
    views = plan.by_owner
    assert plan.by_owner is views
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
