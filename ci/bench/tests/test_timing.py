# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Refit latency excludes tracing setup, logging, and tensor verification."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


def test_refit_timers_exclude_recording_and_pointer_checks(monkeypatch):
    clock = [0.0]
    records = []
    released = []

    def advance(seconds, result=None):
        clock[0] += seconds
        return result

    staged = SimpleNamespace(metrics={}, release=lambda: released.append(True))
    client = SimpleNamespace(
        stage_weight=lambda **kwargs: advance(2, staged),
        apply_weight=lambda weight: advance(3),
        _serving_version_id="run-d1",
    )
    cuda = SimpleNamespace(
        synchronize=lambda device: advance(0.5),
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    monkeypatch.setitem(sys.modules, "config", SimpleNamespace(CONFIG={}))
    monkeypatch.setitem(sys.modules, "modelexpress_rl", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "modelexpress_rl.version",
        SimpleNamespace(WeightVersionRef=lambda value: value),
    )
    monkeypatch.setitem(sys.modules, "modelexpress", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "modelexpress.tensor_utils",
        SimpleNamespace(
            collect_module_tensors=lambda model: advance(
                100, {"weight": SimpleNamespace(data_ptr=lambda: 1)}
            )
        ),
    )
    spec = importlib.util.spec_from_file_location(
        "timed_worker",
        Path(__file__).resolve().parents[1] / "common/bench_worker.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.time, "perf_counter", lambda: clock[0])

    def record(worker, phase, body):
        records.append((phase, body))
        advance(100)

    monkeypatch.setattr(module.BenchWorkerExtension, "_record", record)
    worker = SimpleNamespace(
        device="cuda",
        rank=0,
        _hotload_client=client,
        _hotload_source="OBJECT_STORAGE",
        _initial_weight_ptrs={"weight": 1},
        model_runner=SimpleNamespace(get_model=lambda: object()),
    )
    module.BenchWorkerExtension._hotload_impl(worker, "run-d1")
    phase, result = records[-1]
    assert phase == "run-d1"
    assert result["stage_seconds"] == 2.5
    assert result["install_seconds"] == 3.5
    assert result["total_seconds"] == 6.0
    assert released == [True]
