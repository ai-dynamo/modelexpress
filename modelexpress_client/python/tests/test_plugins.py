# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch.nn as nn

import modelexpress.load_strategy as chain
from modelexpress.adapter import EngineAdapter, StrategyFailed, StrategyRecoveryError
from modelexpress.load_strategy import LoadResult, LoadStrategy


@pytest.fixture
def registry(monkeypatch, tmp_path):
    module = importlib.import_module("modelexpress.plugins")
    monkeypatch.setattr(module, "plugins", {})
    monkeypatch.setattr(module, "_loaded_directories", {})
    monkeypatch.setenv("MX_PLUGIN_DIR", str(tmp_path))
    monkeypatch.setenv("MX_LOAD_PLUGIN", "example")
    monkeypatch.delenv("MX_LOAD_PLUGIN_BEFORE", raising=False)
    return module


def write_plugin(path, name="example", body="return result", prefix="", extra=""):
    path.joinpath(f"mx_plugin_{name}.py").write_text(
        "from modelexpress.load_strategy import LoadStrategy, LoadResult, StrategyFailed, StrategyRecoveryError\n"
        + prefix
        + f"\nclass Custom(LoadStrategy):\n    name = {name!r}\n"
        + f"    def load(self, result, ctx):\n        {body}\n"
        + extra
        + "\ndef register_plugin():\n    return Custom\n"
    )


def load_plugin(registry, name):
    registry.initialize_plugins()
    return registry.get_plugin(name)


def test_get_plugin_does_not_discover_entries(registry, tmp_path):
    write_plugin(tmp_path, prefix="raise AssertionError('must not import')\n")
    with pytest.raises(ValueError, match="example"):
        registry.get_plugin("example")
    assert registry.plugins == {}
    assert registry._loaded_directories == {}


@pytest.mark.parametrize("selected", [None, ""])
def test_disabled_never_initializes_or_imports_external_plugins(monkeypatch, tmp_path, selected):
    write_plugin(tmp_path, prefix="raise RuntimeError('must not import')\n")
    monkeypatch.setenv("MX_PLUGIN_DIR", str(tmp_path))
    monkeypatch.setenv("MX_LOAD_PLUGIN_BEFORE", "invalid")
    if selected is None:
        monkeypatch.delenv("MX_LOAD_PLUGIN", raising=False)
    else:
        monkeypatch.setenv("MX_LOAD_PLUGIN", selected)
    from modelexpress import plugins

    initialize = Mock(side_effect=AssertionError("must not initialize plugins"))
    select = Mock(side_effect=AssertionError("must not select a plugin"))
    monkeypatch.setattr(plugins, "initialize_plugins", initialize)
    monkeypatch.setattr(plugins, "get_plugin", select)
    execute = Mock(return_value="loaded")
    monkeypatch.setattr(chain, "execute_load_strategies", execute)
    assert chain.LoadStrategyChain.run(nn.Linear(1, 1), None) == "loaded"
    assert len(execute.call_args.args[2]) == 6
    initialize.assert_not_called()
    select.assert_not_called()


def test_selected_inserted_and_instance_fresh(registry, monkeypatch, tmp_path):
    write_plugin(tmp_path)
    write_plugin(tmp_path, "unused", extra="    def __init__(self):\n        raise AssertionError('unused constructed')\n")
    execute = Mock()
    monkeypatch.setattr(chain, "execute_load_strategies", execute)
    chain.LoadStrategyChain.run(nn.Linear(1, 1), None)
    first = execute.call_args.args[2]
    chain.LoadStrategyChain.run(nn.Linear(1, 1), None)
    second = execute.call_args.args[2]
    assert [s.name for s in first] == ["rdma", "example", "server-cache", "instant_tensor", "model_streamer", "gds", "default"]
    assert first[1] is not second[1]
    assert set(registry.plugins) == {"example", "unused"}


@pytest.mark.parametrize("before", ["rdma", "server-cache", "instant_tensor", "model_streamer", "gds", "default"])
def test_plugin_runs_before_configured_builtin(registry, monkeypatch, tmp_path, before):
    write_plugin(tmp_path)
    monkeypatch.setenv("MX_LOAD_PLUGIN_BEFORE", before)
    execute = Mock()
    monkeypatch.setattr(chain, "execute_load_strategies", execute)
    chain.LoadStrategyChain.run(nn.Linear(1, 1), None)
    names = [strategy.name for strategy in execute.call_args.args[2]]
    assert names[names.index(before) - 1] == "example"
    assert [name for name in names if name != "example"] == ["rdma", "server-cache", "instant_tensor", "model_streamer", "gds", "default"]


@pytest.mark.parametrize("before", ["missing", "example", ""])
def test_invalid_plugin_position_aborts_before_discovery(registry, monkeypatch, tmp_path, before):
    write_plugin(tmp_path, prefix="raise AssertionError('must not import')\n")
    monkeypatch.setenv("MX_LOAD_PLUGIN_BEFORE", before)
    execute = Mock()
    monkeypatch.setattr(chain, "execute_load_strategies", execute)
    with pytest.raises(ValueError, match="MX_LOAD_PLUGIN_BEFORE"):
        chain.LoadStrategyChain.run(nn.Linear(1, 1), None)
    execute.assert_not_called()
    assert registry.plugins == {}


def test_cache_relative_imports_and_concurrent_initialization(registry, tmp_path, caplog):
    caplog.set_level("INFO", logger="modelexpress.plugins")
    tmp_path.joinpath("helper.py").write_text("VALUE = 42\n")
    marker = tmp_path / "imports"
    write_plugin(tmp_path, prefix=f"from .helper import VALUE\nwith open({str(marker)!r}, 'a') as f: f.write('x')\n")
    before = list(sys.path)
    with ThreadPoolExecutor(max_workers=8) as pool:
        instances = list(pool.map(lambda _: load_plugin(registry, "example"), range(16)))
    assert marker.read_text() == "x"
    assert len({id(instance) for instance in instances}) == 16
    assert sys.path == before
    cls = registry.plugins["example"]
    assert sys.modules[cls.__module__].VALUE == 42
    messages = [record.getMessage() for record in caplog.records if record.name == "modelexpress.plugins"]
    assert messages == [f"Loaded loading strategy plugin 'example' from {tmp_path / 'mx_plugin_example.py'}"]


@pytest.mark.parametrize("source", [
    "def register_plugin(): return object\n",
    "from modelexpress.load_strategy import LoadStrategy\ndef register_plugin(): return LoadStrategy\n",
    "def register_plugin(required): return None\n",
    "register_plugin = 1\n",
    "from modelexpress.load_strategy import LoadStrategy\nclass Bad(LoadStrategy):\n name='wrong'\n def load(self,result,ctx): return result\ndef register_plugin(): return Bad\n",
    "from modelexpress.load_strategy import LoadStrategy\nclass Bad(LoadStrategy):\n name='example'\n def __init__(self, required): pass\n def load(self,result,ctx): return result\ndef register_plugin(): return Bad\n",
])
def test_invalid_entry_is_fatal_and_cleaned(registry, tmp_path, source):
    tmp_path.joinpath("mx_plugin_example.py").write_text(source)
    before = set(sys.modules)
    with pytest.raises((ValueError, TypeError, RuntimeError)):
        load_plugin(registry, "example")
    assert registry.plugins == {}
    assert not any(name.startswith("_mx_plugins_") for name in set(sys.modules) - before)
    write_plugin(tmp_path)
    assert load_plugin(registry, "example").name == "example"


def test_missing_directory_and_missing_name_are_fatal(registry, monkeypatch, tmp_path):
    monkeypatch.setenv("MX_PLUGIN_DIR", str(tmp_path / "absent"))
    with pytest.raises(ValueError, match="example"):
        load_plugin(registry, "example")


def test_registration_rejects_duplicates_and_chain_validates_reserved_names(registry):
    class Custom(LoadStrategy):
        name = "custom"
        def load(self, result, ctx):
            return result
    registry.register_plugin("custom", Custom)
    with pytest.raises(ValueError, match="already"):
        registry.register_plugin("custom", Custom)
    with pytest.raises(ValueError, match="built-in"):
        registry.validate_plugin_names({"custom"})


@pytest.mark.parametrize("failure,mutated", [
    ("raise StrategyFailed('miss')", False),
    ("raise StrategyFailed('dirty', mutated=True)", True),
    ("raise RuntimeError('unknown')", True),
    ("return None", True),
    ("return LoadResult(value=None)", True),
    ("return LoadResult(value=result.value, model='invalid')", True),
    ("return LoadResult(value=result.value, publishable='yes')", True),
    ("return LoadResult(value=result.value, metadata=None)", True),
])
def test_failure_rolls_back_and_reinitializes_before_fallback(registry, monkeypatch, tmp_path, failure, mutated):
    write_plugin(tmp_path, body=failure, extra="    def rollback(self, ctx):\n        ctx.events.append('rollback')\n")
    old, restored = nn.Linear(1, 1), nn.Linear(1, 1)
    events = []
    def reinit(result):
        assert result.model is old
        events.append("reinit")
        return LoadResult(value=restored, model=restored)
    ctx = SimpleNamespace(global_rank=0, identity=SimpleNamespace(model_name="test"), events=events, adapter=SimpleNamespace(reinit_for_retry=reinit))
    class Fallback(LoadStrategy):
        name = "fallback"
        def load(self, result, ctx):
            assert result.model is (restored if mutated else old)
            events.append("fallback")
            return result
    publish = Mock()
    monkeypatch.setattr(chain, "publish_source_if_supported", publish)
    assert chain.execute_load_strategies(old, ctx, [load_plugin(registry, "example"), Fallback()]) is (restored if mutated else old)
    assert events == (["rollback", "reinit", "fallback"] if mutated else ["rollback", "fallback"])
    publish.assert_called_once()


def test_availability_delegated_and_rdma_success_skips_plugin_load(registry, monkeypatch, tmp_path):
    write_plugin(tmp_path, body="raise AssertionError('load should be skipped')", extra="    def is_available(self, ctx):\n        ctx.events.append('available')\n        return ctx.enabled\n")
    model = nn.Linear(1, 1)
    ctx = SimpleNamespace(global_rank=0, identity=SimpleNamespace(model_name="test"), events=[], enabled=True)
    class Rdma(LoadStrategy):
        name = "rdma"
        def load(self, result, ctx):
            return result
    publish = Mock()
    monkeypatch.setattr(chain, "publish_source_if_supported", publish)
    assert chain.execute_load_strategies(model, ctx, [Rdma(), load_plugin(registry, "example")]) is model
    assert ctx.events == ["available"]
    publish.assert_called_once()
    ctx.enabled = False
    assert load_plugin(registry, "example").is_available(ctx) is False


def test_recovery_error_aborts_and_rolls_back(registry, monkeypatch, tmp_path):
    write_plugin(tmp_path, body="raise StrategyRecoveryError('recovery')", extra="    def rollback(self, ctx):\n        ctx.events.append('rollback')\n")
    fallback = Mock()
    fallback.name = "fallback"
    ctx = SimpleNamespace(global_rank=0, identity=SimpleNamespace(model_name="test"), events=[])
    with pytest.raises(StrategyRecoveryError):
        chain.execute_load_strategies(nn.Linear(1, 1), ctx, [load_plugin(registry, "example"), fallback])
    assert ctx.events == ["rollback"]
    fallback.load.assert_not_called()


def test_plugin_success_publishes_once(registry, monkeypatch, tmp_path):
    write_plugin(tmp_path)
    model = nn.Linear(1, 1)
    fallback = Mock()
    fallback.name = "fallback"
    publish = Mock()
    monkeypatch.setattr(chain, "publish_source_if_supported", publish)
    ctx = SimpleNamespace(global_rank=0, identity=SimpleNamespace(model_name="test"))
    assert chain.execute_load_strategies(model, ctx, [load_plugin(registry, "example"), fallback]) is model
    publish.assert_called_once()
    fallback.load.assert_not_called()


def test_requires_capability_is_preserved(registry, tmp_path):
    write_plugin(tmp_path, prefix="from modelexpress.adapter import EngineAdapter\n", extra="    requires = (EngineAdapter.load_via_native,)\n")
    strategy = load_plugin(registry, "example")
    assert strategy.is_available(SimpleNamespace(adapter=EngineAdapter())) is False
    class Native(EngineAdapter):
        def load_via_native(self, result):
            return result
    assert strategy.is_available(SimpleNamespace(adapter=Native())) is True


def test_directory_namespaces_isolate_relative_helpers(registry, monkeypatch, tmp_path):
    for directory, name, value in [(tmp_path, "example", 1), (tmp_path / "other", "other", 2)]:
        directory.mkdir(exist_ok=True)
        directory.joinpath("helper.py").write_text(f"VALUE = {value}\n")
        write_plugin(directory, name, prefix="from .helper import VALUE\n")
        monkeypatch.setenv("MX_PLUGIN_DIR", str(directory))
        load_plugin(registry, name)
    assert sys.modules[registry.plugins["example"].__module__].VALUE == 1
    assert sys.modules[registry.plugins["other"].__module__].VALUE == 2


@pytest.mark.parametrize("stage", ["availability", "rollback", "reinit", "construction"])
def test_fatal_errors_do_not_reach_fallback(registry, monkeypatch, tmp_path, stage):
    extras = {
        "availability": "    def is_available(self, ctx):\n        raise RuntimeError('fatal')\n",
        "rollback": "    def rollback(self, ctx):\n        raise RuntimeError('fatal')\n",
        "construction": "    def __init__(self):\n        raise RuntimeError('fatal')\n",
        "reinit": "",
    }
    write_plugin(tmp_path, body="raise StrategyFailed('dirty', mutated=True)", extra=extras[stage])
    def reinit(result):
        raise RuntimeError("fatal")
    ctx = SimpleNamespace(global_rank=0, identity=SimpleNamespace(model_name="test"), adapter=SimpleNamespace(reinit_for_retry=reinit))
    fallback = Mock()
    fallback.name = "fallback"
    with pytest.raises(RuntimeError, match="fatal"):
        chain.execute_load_strategies(nn.Linear(1, 1), ctx, [load_plugin(registry, "example"), fallback])
    fallback.load.assert_not_called()


def test_rl_bypasses_selected_plugin(monkeypatch, tmp_path):
    write_plugin(tmp_path, prefix="raise RuntimeError('must not import')\n")
    monkeypatch.setenv("MX_PLUGIN_DIR", str(tmp_path))
    monkeypatch.setenv("MX_LOAD_PLUGIN", "example")
    monkeypatch.setenv("MX_LOAD_STRATEGY_CHAIN", "RL")
    monkeypatch.setenv("MX_LOAD_PLUGIN_BEFORE", "invalid")
    from modelexpress_rl.inference.load_strategy import RLLoadStrategyChain
    run = Mock(return_value="rl")
    monkeypatch.setattr(RLLoadStrategyChain, "run", run)
    assert chain.run_load_strategy_chain(nn.Linear(1, 1), None) == "rl"


def test_example_requires_method_implementation(registry, monkeypatch):
    root = Path(__file__).resolve().parents[3]
    monkeypatch.setenv("MX_PLUGIN_DIR", str(root / "examples" / "plugins"))
    registry.initialize_plugins()
    strategy = registry.plugins["example"]()
    for method in ("is_available", "load", "rollback"):
        args = (LoadResult(value=nn.Linear(1, 1)), None) if method == "load" else (None,)
        with pytest.raises(NotImplementedError, match=method):
            getattr(strategy, method)(*args)


def test_engine_specific_non_module_value_is_valid(registry, tmp_path):
    write_plugin(tmp_path, body="return LoadResult(value='engine-handle', publishable=False)")
    loaded = load_plugin(registry, "example").load(LoadResult(value=object()), None)
    assert loaded.value == "engine-handle"


@pytest.mark.parametrize("selected", ["rdma", "example"])
def test_builtin_collision_aborts_before_selection_or_execution(registry, monkeypatch, tmp_path, selected):
    write_plugin(tmp_path, "rdma")
    write_plugin(tmp_path)
    monkeypatch.setenv("MX_LOAD_PLUGIN", selected)
    select = Mock()
    monkeypatch.setattr(registry, "get_plugin", select)
    execute = Mock()
    monkeypatch.setattr(chain, "execute_load_strategies", execute)
    with pytest.raises(ValueError, match="built-in"):
        chain.LoadStrategyChain.run(nn.Linear(1, 1), None)
    execute.assert_not_called()
    select.assert_not_called()
    assert set(registry.plugins) == {"rdma", "example"}


def test_discovery_failure_removes_earlier_registrations_and_helpers(registry, tmp_path, caplog):
    caplog.set_level("INFO", logger="modelexpress.plugins")
    tmp_path.joinpath("helper.py").write_text("VALUE = 1\n")
    write_plugin(tmp_path, "aaa", prefix="from .helper import VALUE\n")
    write_plugin(tmp_path, "zzz", prefix="raise RuntimeError('broken')\n")
    before = set(sys.modules)
    with pytest.raises(RuntimeError, match="broken"):
        load_plugin(registry, "aaa")
    assert registry.plugins == {}
    assert not any(name.startswith("_mx_plugins_") for name in set(sys.modules) - before)
    assert not any(record.name == "modelexpress.plugins" for record in caplog.records)


def test_selected_name_missing_from_existing_directory(registry, tmp_path):
    write_plugin(tmp_path, "other")
    with pytest.raises(ValueError, match="example"):
        load_plugin(registry, "example")
