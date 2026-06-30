# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the elastic-scaling benchmark harness.

We exercise the orchestrator's CPU smoke path + the result aggregation
logic without touching MX, NIXL, or torch.cuda. The "live" subprocess
path is tested by running it against a real MX server (out of scope
for the unit test suite — see ``bench_elastic_scaling.py --mode=live``
for the integration smoke).
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from argparse import Namespace
from pathlib import Path

import pytest


_HERE = Path(__file__).resolve().parent
_BENCH = _HERE.parent / "benchmarks" / "bench_elastic_scaling.py"


@pytest.fixture(scope="module")
def bench():
    """Load the benchmark script as a module."""
    spec = importlib.util.spec_from_file_location("bench_es", _BENCH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["bench_es"] = mod
    spec.loader.exec_module(mod)
    return mod


def _args(**overrides):
    """Build a Namespace with defaults that match the CLI."""
    defaults = dict(
        scenario="elastic_scale",
        mode="cpu",
        mx_server_url="localhost:8001",
        model_name="bench/m",
        num_receivers=3,
        num_tensors=4,
        tensor_bytes=1024 * 1024,
        steps=2,
        step_interval=0.0,
        join_interval=0.5,
        trainer_warmup=0.0,
        poll_interval=0.1,
        cycle_timeout=5.0,
        deadline=10.0,
        trainer_compile_target="cutlass_fp8",
        tmpdir="/tmp/mx_bench_test",
        output=None,
        verbose=False,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


def test_cpu_smoke_produces_consistent_result(bench):
    """CPU smoke mode runs end-to-end and produces a well-formed BenchResult."""
    result = bench.run_cpu_smoke(_args(num_receivers=2, steps=3))
    assert result.scenario == "elastic_scale_cpu_smoke"
    assert result.trainer is not None
    assert result.trainer.published_versions == [1, 2, 3]
    assert len(result.receivers) == 2
    for r in result.receivers:
        assert r.join_latency_seconds == 0.05
        assert len(r.cycles) == 3
        for c in r.cycles:
            assert c.bytes_received == 4 * 1024 * 1024
            assert c.source_worker_rank == 0


def test_receiver_bandwidth_math(bench):
    """avg_bandwidth_gbps math: total_bits / total_rdma_seconds / 1e9."""
    r = bench.ReceiverResult(
        receiver_id="r", worker_rank=0, started_at=0.0
    )
    r.cycles.append(bench.ReceiverCycleResult(
        version=1, bytes_received=1_250_000_000, rdma_seconds=1.0
    ))
    # 1.25 GB * 8 = 10 Gbits, over 1s → 10 Gbps
    assert r.avg_bandwidth_gbps() == pytest.approx(10.0)


def test_trainer_egress_under_no_tree_equals_total(bench):
    """Without tree fan-out, all receivers pull from the trainer (rank 0),
    so trainer_egress_bytes should equal total_delivered."""
    result = bench.run_cpu_smoke(_args(num_receivers=4, steps=2))
    total = sum(r.total_bytes() for r in result.receivers)
    assert result.trainer_egress_bytes() == total


def test_trainer_egress_under_tree_fanout_can_be_less(bench):
    """If receivers report source_worker_rank != 0, those bytes are NOT
    counted against the trainer's egress. This is how the harness
    measures the tree-fan-out savings."""
    # Hand-build a result where 2 of the 3 receivers pulled from a replica
    result = bench.BenchResult(
        scenario="tree_fanout",
        config={},
        trainer=bench.TrainerResult(
            worker_id="t", mx_source_id="sid", started_at=0.0,
            published_versions=[1], compile_target="cutlass_fp8",
            total_published_bytes=1_000_000,
        ),
        receivers=[
            bench.ReceiverResult(
                receiver_id=f"r{i}", worker_rank=0, started_at=0.0,
                cycles=[bench.ReceiverCycleResult(
                    version=1, bytes_received=1_000_000,
                    source_worker_rank=(0 if i == 0 else 1),
                )]
            )
            for i in range(3)
        ],
        started_at=0.0, finished_at=1.0,
    )
    assert result.trainer_egress_bytes() == 1_000_000  # only r0 went to trainer
    total = sum(r.total_bytes() for r in result.receivers)
    assert total == 3_000_000


def test_compile_target_verdicts(bench):
    """Compile-target scenario derivation: receivers with any successful
    cycle are 'accepted'; receivers with only error cycles are 'rejected'."""
    result = bench.BenchResult(
        scenario="compile_target", config={},
        trainer=bench.TrainerResult(
            worker_id="t", mx_source_id="sid", started_at=0.0,
            published_versions=[1], compile_target="cutlass_fp8",
            total_published_bytes=1_000_000,
        ),
        receivers=[
            bench.ReceiverResult(
                receiver_id="recv-match", worker_rank=0, started_at=0.0,
                compile_target_filter=["cutlass_fp8"],
                cycles=[bench.ReceiverCycleResult(version=1, bytes_received=1_000_000)],
            ),
            bench.ReceiverResult(
                receiver_id="recv-mismatch", worker_rank=0, started_at=0.0,
                compile_target_filter=["deep_gemm_fp8"],
                cycles=[bench.ReceiverCycleResult(
                    version=1, error="no source matches filters"
                )],
            ),
            bench.ReceiverResult(
                receiver_id="recv-no-filter", worker_rank=0, started_at=0.0,
                compile_target_filter=None,
                cycles=[bench.ReceiverCycleResult(version=1, bytes_received=1_000_000)],
            ),
        ],
        started_at=0.0, finished_at=1.0,
    )
    derived = bench._scenario_derived(result)
    assert derived["verdicts"] == {
        "recv-match": "accepted",
        "recv-mismatch": "rejected",
        "recv-no-filter": "accepted",
    }


def test_p99_p50_join_latency(bench):
    """Elastic-scale percentile helpers."""
    assert bench._p([], 0.5) is None
    assert bench._p([1.0], 0.5) == 1.0
    assert bench._p([1, 2, 3, 4, 5], 0.5) == 3
    assert bench._p([1, 2, 3, 4, 5], 0.99) == 5


def test_summary_table_includes_fanout_factor_for_tree_scenario(bench):
    """The human-readable summary should call out the fan-out factor on
    the tree_fanout scenario only."""
    result = bench.BenchResult(
        scenario="tree_fanout", config={},
        trainer=bench.TrainerResult(
            worker_id="t", mx_source_id="sid", started_at=0.0,
            published_versions=[1], compile_target="cutlass_fp8",
            total_published_bytes=1_000_000,
        ),
        receivers=[
            bench.ReceiverResult(
                receiver_id="r0", worker_rank=0, started_at=0.0,
                cycles=[bench.ReceiverCycleResult(
                    version=1, bytes_received=1_000_000, source_worker_rank=0
                )]
            ),
            bench.ReceiverResult(
                receiver_id="r1", worker_rank=0, started_at=0.0,
                cycles=[bench.ReceiverCycleResult(
                    version=1, bytes_received=1_000_000, source_worker_rank=1
                )]
            ),
        ],
        started_at=0.0, finished_at=1.0,
    )
    table = result.to_summary_table()
    assert "fanout_factor" in table
    assert "Tree fan-out" in table

    # The elastic_scale variant should NOT mention fanout_factor
    result.scenario = "elastic_scale"
    table = result.to_summary_table()
    assert "fanout_factor" not in table


def test_argparse_defaults_round_trip(bench):
    """The CLI parser produces a Namespace the orchestrator can consume."""
    ns = bench._parse_args(["--scenario=compile_target", "--num-receivers=5"])
    assert ns.scenario == "compile_target"
    assert ns.num_receivers == 5


def test_main_cpu_mode_writes_output(bench, tmp_path):
    """End-to-end CLI: --mode=cpu --output writes a JSON file with the
    expected shape."""
    out = tmp_path / "result.json"
    rc = bench.main([
        "--mode=cpu",
        "--scenario=tree_fanout",
        "--num-receivers=2",
        "--steps=2",
        "--output", str(out),
    ])
    assert rc == 0
    data = json.loads(out.read_text())
    assert data["scenario"] == "tree_fanout_cpu_smoke"
    assert data["derived"]["scenario_specific"]["fanout_factor"] is not None
    assert len(data["receivers"]) == 2
