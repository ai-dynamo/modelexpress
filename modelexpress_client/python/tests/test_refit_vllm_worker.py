# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest


_WORKER_MODULE = "modelexpress_rl.inference.engines.vllm.worker"


@pytest.fixture(autouse=True)
def _clear_worker_module():
    sys.modules.pop(_WORKER_MODULE, None)
    yield
    sys.modules.pop(_WORKER_MODULE, None)


def test_vllm_bootstrap_device_id_uses_vllm_platform_mapping(
    monkeypatch,
):
    calls = []
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(assigned_physical_gpu_ids=[3, 7])
    )
    platforms = ModuleType("vllm.platforms")
    platforms.current_platform = SimpleNamespace(
        logical_device_id_to_visible_device_id=lambda local_rank: (
            calls.append(("resolve", local_rank)) or 1
        )
    )
    interface = ModuleType("vllm.platforms.interface")
    interface.set_assigned_physical_gpu_ids = lambda device_ids: calls.append(
        ("assign", device_ids)
    )
    monkeypatch.setitem(sys.modules, "vllm.platforms", platforms)
    monkeypatch.setitem(sys.modules, "vllm.platforms.interface", interface)

    worker_module = _import_worker_module(monkeypatch, object)

    assert worker_module._get_vllm_bootstrap_device_id(config, 1) == 1
    assert calls == [("assign", [3, 7]), ("resolve", 1)]


def _import_worker_module(monkeypatch, worker_cls):
    packages = ["vllm", "vllm.v1", "vllm.v1.worker"]
    for name in packages:
        package = ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    gpu_worker = ModuleType("vllm.v1.worker.gpu_worker")
    gpu_worker.Worker = worker_cls
    monkeypatch.setitem(sys.modules, "vllm.v1.worker.gpu_worker", gpu_worker)

    return importlib.import_module(_WORKER_MODULE)


def test_vllm_worker_bootstraps_before_base_initialization(monkeypatch):
    calls = []

    class Worker:
        def __init__(self, config, local_rank, *args, **kwargs):
            calls.append(("vllm", config, local_rank, args, kwargs))

    worker_module = _import_worker_module(monkeypatch, Worker)
    bootstrap = object()
    monkeypatch.setattr(
        worker_module,
        "VllmGeneratorBootstrap",
        lambda config, local_rank: (
            calls.append(("bootstrap", config, local_rank)) or bootstrap
        ),
    )
    config = object()

    worker = worker_module.ModelExpressVllmWorker(config, 2, "rank", ready=True)

    assert calls == [
        ("bootstrap", config, 2),
        ("vllm", config, 2, ("rank",), {"ready": True}),
    ]
    assert worker._model_express_bootstrap is bootstrap
