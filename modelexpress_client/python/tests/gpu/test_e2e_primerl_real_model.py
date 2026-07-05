# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real-model end-to-end PrimeRL weight-transfer tests.

Loads Qwen/Qwen2.5-0.5B (290 parameters, 988 MB bfloat16) on two GPUs and
validates the full weight-transfer pipeline:

  Trainer (GPU 0) --bake/resolve/route--> RDMA descriptors --cudaMemcpy--> Inference (GPU 1)

Three scenarios are exercised:

  1. Baseline (FSDP/DP=2, row sharding only)
       Standard two-rank FSDP trainer: each rank owns half the rows of every
       parameter.  Validates the existing row-only path against a real model.

  2. TP=2 column sharding (new 2-D path)
       Simulates a Megatron-LM trainer with row-parallel layers (o_proj,
       down_proj) split along dim-1 into contiguous [R, C/2] shards and
       column-parallel layers (q/k/v/gate/up) split along dim-0.
       All 290 parameters must match byte-for-byte after transfer.

  3. Multi-step PrimeRL sync loop
       Simulates 3 gradient steps of an RL training loop.  The transfer plan
       is invalidated and rebuilt each step; weights verified byte-exact after
       each step.

Model : Qwen/Qwen2.5-0.5B
Params: 290
Size  : 988.1 MB bfloat16

Requirements:
  - 2 CUDA GPUs (skipped otherwise)
  - transformers installed (Qwen2.5-0.5B weights cached in HF_HOME)
  - libcudart.so.12

Both models are loaded once per test session (module scope) and reused across
all tests to avoid OOM from repeated loads.
"""

import ctypes
import math
import os
import sys
import time

import pytest
import torch

# ---------------------------------------------------------------------------
# Skip guard
# ---------------------------------------------------------------------------

_TWO_GPUS = torch.cuda.is_available() and torch.cuda.device_count() >= 2
pytestmark = pytest.mark.skipif(not _TWO_GPUS, reason="requires 2 CUDA GPUs")

# ---------------------------------------------------------------------------
# CUDA memcpy (proxy for NIXL RDMA pull)
# ---------------------------------------------------------------------------

_LIBCUDART_PATHS = [
    "/usr/local/python-3.12.6/lib/python3.12/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12",
    "/usr/lib/x86_64-linux-gnu/libcudart.so.12",
    "libcudart.so.12",
]


def _load_libcudart():
    for path in _LIBCUDART_PATHS:
        try:
            return ctypes.CDLL(path)
        except OSError:
            continue
    pytest.skip("libcudart.so.12 not found")


def _cuda_memcpy(lib, src: int, dst: int, n: int) -> None:
    rc = lib.cudaMemcpy(ctypes.c_void_p(dst), ctypes.c_void_p(src), ctypes.c_size_t(n), ctypes.c_int(4))
    if rc != 0:
        raise RuntimeError(f"cudaMemcpy failed rc={rc}")


# ---------------------------------------------------------------------------
# weight_transfer imports
# ---------------------------------------------------------------------------

sys.path.insert(0, os.path.expanduser("~/workspace/modelexpress/modelexpress_client/python"))

from modelexpress.weight_transfer.engine.lazy import BakeRecorder, LazyWeight
from modelexpress.weight_transfer.planner.local import LocalPlanner
from modelexpress.weight_transfer.planner.resolver import resolve_chain_region, region_elem_runs
from modelexpress.weight_transfer.planner.router import route_regions
from modelexpress.weight_transfer.protocol.types import (
    RdmaDescriptor,
    ResolvedRegion,
    TrainerShard,
    TrainerTable,
    TrainerTensor,
)

# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------

MODEL_ID  = "Qwen/Qwen2.5-0.5B"
N_PARAMS  = 290
SIZE_MB   = 988.1
HF_HOME   = os.path.expanduser("~/workspace/hf_cache")

_ROW_PARALLEL = {"o_proj", "down_proj"}
_COL_PARALLEL = {"q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "gate_up_proj"}


def _is_row_parallel(name: str) -> bool:
    return any(name.endswith(f".{s}.weight") for s in _ROW_PARALLEL)


def _is_col_parallel(name: str) -> bool:
    return any(name.endswith(f".{s}.weight") for s in _COL_PARALLEL)


# ---------------------------------------------------------------------------
# Module-scoped fixtures  (loaded once for the entire test session)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def libcudart():
    return _load_libcudart()


@pytest.fixture(scope="module")
def models():
    """Load trainer (GPU 0) and inference (GPU 1) models once for the whole module.

    Both models are loaded sequentially in a single fixture so there is exactly
    one CUDA context initialisation per GPU and no risk of concurrent loads.
    """
    os.environ.setdefault("HF_HOME", HF_HOME)
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    from transformers import AutoModelForCausalLM

    trainer = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="cuda:0")
    t_params = {n: p.data for n, p in trainer.named_parameters()}

    inference = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="cuda:1")
    i_params = {n: p.data for n, p in inference.named_parameters()}

    yield t_params, i_params

    del trainer, inference
    torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def trainer_params(models):
    t_params, _ = models
    return t_params


@pytest.fixture(scope="module")
def inf_params(models):
    _, i_params = models
    return i_params


def _noise(params: dict[str, torch.Tensor]) -> None:
    """Fill all inference params with noise (simulate un-synced state)."""
    for p in params.values():
        p.normal_()


# ---------------------------------------------------------------------------
# TrainerTable builders
# ---------------------------------------------------------------------------


def _fsdp_table(trainer_params: dict[str, torch.Tensor], step: int = 1) -> TrainerTable:
    """DP=2 row-only sharding: each rank owns half the dim-0 rows."""
    tensors = []
    for name, param in trainer_params.items():
        shape = list(param.shape)
        elem_size = param.element_size()
        if param.dim() == 1:
            shards = [TrainerShard(
                agent_index=0, row_start=0, row_end=1,
                device_addr=param.data_ptr(),
                row_bytes=param.numel() * elem_size, device_id=0,
            )]
            shape_use = [1, param.numel()]
        else:
            row_bytes = math.prod(shape[1:]) * elem_size
            half = max(shape[0] // 2, 1)
            shards = [
                TrainerShard(0, 0, half, param.data_ptr(), row_bytes, 0),
                TrainerShard(1, half, shape[0], param.data_ptr() + half * row_bytes, row_bytes, 0),
            ]
            shape_use = shape
        tensors.append(TrainerTensor(name=name, dtype=str(param.dtype), shape=shape_use, shards=shards))
    return TrainerTable(agents=[b"rank0", b"rank1"], tensors=tensors, step=step)


def _tp2_table(trainer_params: dict[str, torch.Tensor], step: int = 1) -> tuple[TrainerTable, list]:
    """TP=2 table: row-parallel layers column-sharded, col-parallel layers row-sharded."""
    tensors = []
    shard_buffers: list[torch.Tensor] = []

    for name, param in trainer_params.items():
        shape = list(param.shape)
        elem_size = param.element_size()

        if param.dim() == 2 and _is_row_parallel(name) and shape[1] >= 2:
            C, half = shape[1], shape[1] // 2
            s0, s1 = param[:, :half].contiguous(), param[:, half:].contiguous()
            shard_buffers.extend([s0, s1])
            shards = [
                TrainerShard(0, 0, shape[0], s0.data_ptr(), half * elem_size, 0, col_start=0, col_end=half),
                TrainerShard(1, 0, shape[0], s1.data_ptr(), (C - half) * elem_size, 0, col_start=half, col_end=C),
            ]
            tensors.append(TrainerTensor(name=name, dtype=str(param.dtype), shape=shape, shards=shards))

        elif param.dim() == 2 and _is_col_parallel(name) and shape[0] >= 2:
            R, half = shape[0], shape[0] // 2
            row_bytes = math.prod(shape[1:]) * elem_size
            shards = [
                TrainerShard(0, 0, half, param.data_ptr(), row_bytes, 0),
                TrainerShard(1, half, R, param.data_ptr() + half * row_bytes, row_bytes, 0),
            ]
            tensors.append(TrainerTensor(name=name, dtype=str(param.dtype), shape=shape, shards=shards))

        else:
            if param.dim() == 1:
                shape_use = [1, param.numel()]
                row_bytes = param.numel() * elem_size
                row_count = 1
            else:
                shape_use = shape
                row_bytes = math.prod(shape[1:]) * elem_size
                row_count = shape[0]
            shards = [TrainerShard(0, 0, row_count, param.data_ptr(), row_bytes, 0)]
            tensors.append(TrainerTensor(name=name, dtype=str(param.dtype), shape=shape_use, shards=shards))

    return TrainerTable(agents=[b"tp0", b"tp1"], tensors=tensors, step=step), shard_buffers


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------


def _build_regions(table: TrainerTable, inf_params: dict[str, torch.Tensor]) -> list[ResolvedRegion]:
    recorder = BakeRecorder()
    with recorder:
        for name, param in inf_params.items():
            tt = table.tensor_by_name(name)
            if tt is None:
                continue
            lw = LazyWeight(name, param.shape, param.dtype)
            param.copy_(lw)

    regions = []
    for copy in recorder.copies:
        tt = table.tensor_by_name(copy.src_name)
        if tt is None:
            continue
        try:
            offset, rshape, stride = resolve_chain_region(
                copy.op_chain, torch.Size(copy.dst_shape), copy.dst_dtype
            )
        except Exception:
            continue
        src_runs = [v for pair in region_elem_runs(offset, rshape, stride) for v in pair]
        dst_param = inf_params.get(copy.src_name)
        if dst_param is None:
            continue
        dst_off = (copy.dst_addr - dst_param.data_ptr()) // dst_param.element_size()
        dst_runs = [v for pair in region_elem_runs(dst_off, torch.Size(copy.dst_shape),
                                                    tuple(dst_param.stride())) for v in pair]
        regions.append(ResolvedRegion(
            tensor_name=copy.src_name,
            src_elem_runs=src_runs,
            dst_addr=dst_param.data_ptr(),
            dst_elem_runs=dst_runs,
            element_size=copy.dst_dtype.itemsize,
            dst_device_id=1,
        ))
    return regions


def _execute(lib, descs: list[RdmaDescriptor]) -> tuple[int, float]:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for d in descs:
        _cuda_memcpy(lib, d.src_addr, d.dst_addr, d.nbytes)
    torch.cuda.synchronize()
    return sum(d.nbytes for d in descs), time.perf_counter() - t0


def _mismatches(trainer_params, inf_params) -> list[str]:
    return [
        n for n in trainer_params
        if not torch.equal(trainer_params[n].cpu(), inf_params[n].cpu())
    ]


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


class TestRealModelInfo:
    """Confirm model identity before running any transfer tests."""

    def test_model_id(self, trainer_params):
        assert len(trainer_params) == N_PARAMS, (
            f"Expected {N_PARAMS} params, got {len(trainer_params)} — wrong model?"
        )

    def test_model_size(self, trainer_params):
        mb = sum(p.numel() * p.element_size() for p in trainer_params.values()) / 1e6
        assert abs(mb - SIZE_MB) < 2.0, f"Expected ~{SIZE_MB} MB, got {mb:.1f} MB"

    def test_model_on_cuda0(self, trainer_params):
        devices = {p.device.index for p in trainer_params.values()}
        assert devices == {0}, f"Trainer params not all on GPU 0: {devices}"

    def test_inference_on_cuda1(self, inf_params):
        devices = {p.device.index for p in inf_params.values()}
        assert devices == {1}, f"Inference params not all on GPU 1: {devices}"


class TestRealModelFSDP:
    """FSDP/DP=2 row-only sharding — validates the existing row-only router path."""

    def test_fsdp_table_tensor_count(self, trainer_params):
        table = _fsdp_table(trainer_params)
        assert len(table.tensors) == N_PARAMS

    def test_fsdp_full_transfer_byte_exact(self, trainer_params, inf_params, libcudart):
        _noise(inf_params)
        table = _fsdp_table(trainer_params)
        regions = _build_regions(table, inf_params)
        descs = LocalPlanner().build(regions, table, "fsdp")

        total_bytes, elapsed = _execute(libcudart, descs)

        bad = _mismatches(trainer_params, inf_params)
        assert not bad, f"FSDP: {len(bad)}/{N_PARAMS} params mismatched: {bad[:5]}"

    def test_fsdp_total_bytes_transferred(self, trainer_params, inf_params, libcudart):
        _noise(inf_params)
        table = _fsdp_table(trainer_params)
        regions = _build_regions(table, inf_params)
        descs = LocalPlanner().build(regions, table, "fsdp-bytes")
        total_bytes, _ = _execute(libcudart, descs)
        expected = int(SIZE_MB * 1e6)
        assert abs(total_bytes - expected) < 2e6, (
            f"Expected ~{SIZE_MB:.0f} MB transferred, got {total_bytes/1e6:.1f} MB"
        )

    def test_fsdp_throughput_above_floor(self, trainer_params, inf_params, libcudart):
        """PCIe PHB floor: >5 GB/s (actual measured: ~10 GB/s)."""
        _noise(inf_params)
        table = _fsdp_table(trainer_params)
        regions = _build_regions(table, inf_params)
        descs = LocalPlanner().build(regions, table, "fsdp-tput")
        total_bytes, elapsed = _execute(libcudart, descs)
        gbps = total_bytes / elapsed / 1e9
        assert gbps > 5.0, f"Transfer throughput too low: {gbps:.2f} GB/s"


class TestRealModelTP2:
    """TP=2 column sharding — exercises the new 2-D router path on a real model."""

    def test_tp2_row_parallel_count(self, trainer_params):
        rp = [n for n in trainer_params if _is_row_parallel(n) and trainer_params[n].dim() == 2]
        assert len(rp) == 48, f"Expected 48 row-parallel params (o_proj+down_proj × 24 layers), got {len(rp)}"

    def test_tp2_full_transfer_byte_exact(self, trainer_params, inf_params, libcudart):
        """All 290 params — including all 48 col-sharded row-parallel layers — must match."""
        _noise(inf_params)
        table, shard_buffers = _tp2_table(trainer_params)
        regions = _build_regions(table, inf_params)
        descs = LocalPlanner().build(regions, table, "tp2")

        total_bytes, elapsed = _execute(libcudart, descs)

        bad = _mismatches(trainer_params, inf_params)
        assert not bad, (
            f"TP=2: {len(bad)}/{N_PARAMS} params mismatched\n"
            + "\n".join(f"  {n}" for n in bad[:10])
        )

    def test_tp2_col_sharded_params_correct(self, trainer_params, inf_params, libcudart):
        """Spot-check: every o_proj and down_proj param matches exactly."""
        _noise(inf_params)
        table, shard_buffers = _tp2_table(trainer_params)
        regions = _build_regions(table, inf_params)
        descs = route_regions(regions, table)
        _execute(libcudart, descs)

        rp_names = [n for n in trainer_params if _is_row_parallel(n) and trainer_params[n].dim() == 2]
        all_bad = _mismatches(trainer_params, inf_params)
        rp_bad = [n for n in rp_names if n in all_bad]
        assert not rp_bad, f"{len(rp_bad)}/{len(rp_names)} row-parallel params mismatched"

    def test_tp2_both_agents_used(self, trainer_params, inf_params):
        table, shard_buffers = _tp2_table(trainer_params)
        regions = _build_regions(table, inf_params)
        descs = route_regions(regions, table)
        agents = {d.agent_index for d in descs}
        assert agents == {0, 1}, f"Expected descriptors from both TP agents, got {agents}"

    def test_tp2_descriptor_count(self, trainer_params, inf_params):
        table, shard_buffers = _tp2_table(trainer_params)
        regions = _build_regions(table, inf_params)
        descs = route_regions(regions, table)
        # Each of the 48 col-sharded params produces ≥2 descriptors (one per agent per row run)
        assert len(descs) >= N_PARAMS, f"Too few descriptors: {len(descs)}"

    def test_tp2_throughput_reported(self, trainer_params, inf_params, libcudart, capsys):
        _noise(inf_params)
        table, shard_buffers = _tp2_table(trainer_params)
        regions = _build_regions(table, inf_params)
        descs = route_regions(regions, table)
        total_bytes, elapsed = _execute(libcudart, descs)
        gbps = total_bytes / elapsed / 1e9
        with capsys.disabled():
            print(
                f"\n  model={MODEL_ID}  params={N_PARAMS}  size={SIZE_MB:.1f} MB"
                f"  descriptors={len(descs)}  time={elapsed*1000:.1f} ms  throughput={gbps:.2f} GB/s"
            )
        assert gbps > 5.0


class TestRealModelPrimeRLLoop:
    """3-step PrimeRL RL loop: perturb trainer weights → sync → verify byte-exact."""

    def test_three_step_loop_byte_exact(self, trainer_params, inf_params, libcudart):
        planner = LocalPlanner()
        for step in range(1, 4):
            # Simulate gradient step: small perturbation to trainer weights
            for p in trainer_params.values():
                p.add_(torch.randn_like(p) * 1e-3)

            _noise(inf_params)

            table, shard_buffers = _tp2_table(trainer_params, step=step)
            regions = _build_regions(table, inf_params)

            key = f"rl-step{step}"
            if step > 1:
                planner.invalidate(f"rl-step{step - 1}")
            descs = planner.build(regions, table, key)
            _execute(libcudart, descs)

            bad = _mismatches(trainer_params, inf_params)
            assert not bad, f"Step {step}: {len(bad)} params mismatched: {bad[:5]}"

    def test_plan_cached_within_step(self, trainer_params, inf_params):
        planner = LocalPlanner()
        table, shard_buffers = _tp2_table(trainer_params, step=99)
        regions = _build_regions(table, inf_params)
        d1 = planner.build(regions, table, "cached-step")
        d2 = planner.build(regions, table, "cached-step")
        assert d1 is d2, "Plan must be the same object (cached) within one RL step"

    def test_plan_rebuilt_after_invalidate(self, trainer_params, inf_params):
        planner = LocalPlanner()
        table, shard_buffers = _tp2_table(trainer_params, step=1)
        regions = _build_regions(table, inf_params)
        d1 = planner.build(regions, table, "rebuild-step")
        planner.invalidate("rebuild-step")
        d2 = planner.build(regions, table, "rebuild-step")
        assert d1 is not d2
        assert d1 == d2  # same math, different object
