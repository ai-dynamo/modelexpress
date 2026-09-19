# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Time the NCCL M2N collective TRANSPORT, through the real control plane.

Not an end-to-end refit, and the distinction is the whole point of this line:
the engine boundary here is ``BenchEngine``, a stub over pre-allocated tensors
whose shapes this file chooses. The MX control plane and ``nccl.m2n.reshard``
are real; no model, no framework and no checkpoint is involved, so nothing here
measures what refitting a model costs. ``examples/rl/m2n_collective_refit``
does that, against a live vLLM engine and an FSDP2-sharded checkpoint.

Deliberately not a pytest test, and the filename says so: it needs a live
ModelExpress server, one process per rank and several GPUs, and its output is
numbers rather than a pass. ``test_collective_reshard.py`` covers correctness
on four devices; this measures what the transport costs.

The split it exists to report is bootstrap versus transfer. Bootstrap is what
MX adds over a bare TCPStore -- registration, admission, the brokered unique
ids, READY, communicator init -- and it is paid once per group. Transfer is
paid on every refit. Reporting one number for both would hide which of them a
deployment should care about, so every phase is timed separately and the
per-round loop runs several times so the steady-state round is visible next to
the first one.

Correctness is not optional here: a transfer that moves nothing is very fast.
Round 0 is verified byte-exact on every generator rank against a value the
receiver computes for itself, and the run reports FAILED if it does not match.

One process per rank, ``RANK`` in the environment, results as JSON per rank.
``bench_launch.sh`` starts the cohort and aggregates.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from typing import Any

import grpc
import torch

from modelexpress_rl.collective import (
    LocalParamSpec,
    MeshSpec,
    ParamPlan,
    Placement,
    RefitClientGenerator,
    RefitClientTrainer,
    ReshardPlan,
)
from modelexpress_rl.collective import client as client_module
from modelexpress_rl.collective.backend import loaded_nccl_version
from modelexpress_rl.collective.rendezvous import CollectiveRendezvous

MIN_NCCL = (2, 30, 7)


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


RANK = _env_int("RANK", 0)
ENDPOINT = os.environ.get("MX_ENDPOINT", "127.0.0.1:8001")
N_TRAINERS = _env_int("BENCH_TRAINERS", 2)
N_GENERATORS = _env_int("BENCH_GENERATORS", 2)
PARTITIONS = _env_int("BENCH_PARTITIONS", 1)
ROUNDS = _env_int("BENCH_ROUNDS", 5)
N_PARAMS = _env_int("BENCH_PARAMS", 32)
ROWS = _env_int("BENCH_ROWS", 8192)
COLS = _env_int("BENCH_COLS", 8192)
DTYPE_NAME = os.environ.get("BENCH_DTYPE", "bfloat16")
RUN_ID = os.environ.get("BENCH_RUN_ID", "bench")
OUT_DIR = os.environ.get("BENCH_OUT", "/tmp/mxbench/out")
#: First global rank this host runs. A cross-node cohort is one process group
#: split over several pods, so local device index is the global rank minus the
#: host's base -- never the global rank, which would index off the end.
RANK_BASE = _env_int("BENCH_RANK_BASE", 0)
VERIFY = os.environ.get("BENCH_VERIFY", "1") == "1"

TRAINER_RANKS = tuple(range(N_TRAINERS))
GENERATOR_RANKS = tuple(range(N_TRAINERS, N_TRAINERS + N_GENERATORS))
WORLD = N_TRAINERS + N_GENERATORS

TORCH_DTYPE = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}[DTYPE_NAME]
ITEMSIZE = torch.empty((), dtype=TORCH_DTYPE).element_size()

#: Values are integers below this, which every dtype here represents exactly,
#: so a byte-exact comparison stays meaningful in bf16.
MODULUS = 251


def log(*args: Any) -> None:
    print(f"[rank {RANK}]", *args, flush=True)


def param_name(index: int) -> str:
    return f"layers.{index}.weight"


def build_plan() -> ReshardPlan:
    return ReshardPlan(
        bulk=[
            ParamPlan(
                name=param_name(index),
                global_shape=(ROWS, COLS),
                dtype=DTYPE_NAME,
                # Every parameter rides partition 0 when there is one source
                # partition; with more, they are dealt round robin so each
                # lane carries a share.
                partition_id=index % PARTITIONS,
                src_mesh=MeshSpec(
                    shape=(N_TRAINERS // PARTITIONS,),
                    rank_offset=(index % PARTITIONS) * (N_TRAINERS // PARTITIONS),
                ),
                src_placements=(Placement.shard(0),),
                dst_mesh=MeshSpec(shape=(N_GENERATORS,), rank_offset=N_TRAINERS),
                dst_placements=(Placement.shard(1),),
            )
            for index in range(N_PARAMS)
        ],
        misc=[],
        source_partition_count=PARTITIONS,
    )


def global_block(
    index: int,
    row_lo: int,
    row_hi: int,
    col_lo: int,
    col_hi: int,
    device: torch.device,
) -> torch.Tensor:
    """The declared global tensor for ``index``, restricted to one block.

    Built from index arithmetic rather than an RNG so both sides compute the
    same bytes without exchanging anything, which is what makes the
    verification independent of the transfer it checks.
    """
    rows = torch.arange(row_lo, row_hi, device=device, dtype=torch.int64).unsqueeze(1)
    cols = torch.arange(col_lo, col_hi, device=device, dtype=torch.int64).unsqueeze(0)
    values = (rows * COLS + cols + index * 7919) % MODULUS
    out = values.to(TORCH_DTYPE)
    del rows, cols, values
    return out


class BenchEngine:
    """Both engine boundaries over pre-allocated local storage."""

    def __init__(self, plan: ReshardPlan, specs: dict[str, LocalParamSpec]) -> None:
        self._plan = plan
        self._specs = specs
        self.installed: list[int] = []

    def capture(self) -> ReshardPlan:
        return self._plan

    def parameter_names(self) -> list[str]:
        return self._plan.parameter_names()

    def local_params(self) -> dict[str, LocalParamSpec]:
        return self._specs

    def start_new_round(self, version: str) -> None:
        pass

    def install(self, layer_group_id: int) -> None:
        self.installed.append(layer_group_id)

    def finish(self) -> None:
        pass

    def cleanup(self) -> None:
        pass


class PhaseTimer:
    """Wall time per control-plane phase, accumulated across calls."""

    def __init__(self) -> None:
        self.totals: dict[str, float] = {}
        self.counts: dict[str, int] = {}

    def wrap(self, owner: Any, attr: str, label: str) -> None:
        original = getattr(owner, attr)

        def timed(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - start
                self.totals[label] = self.totals.get(label, 0.0) + elapsed
                self.counts[label] = self.counts.get(label, 0) + 1

        setattr(owner, attr, timed)


def main() -> int:
    process_start = time.time()
    local_rank = RANK - RANK_BASE
    if not 0 <= local_rank < torch.cuda.device_count():
        log(
            f"rank {RANK} maps to local device {local_rank} on a host with "
            f"{torch.cuda.device_count()}; check BENCH_RANK_BASE"
        )
        return 2
    version = loaded_nccl_version()
    if version is None or version < MIN_NCCL:
        log(
            f"libnccl mapped into this process is {version}, reshard needs "
            f"{MIN_NCCL}; LD_PRELOAD the one nccl-extensions installed"
        )
        return 2

    device_index = local_rank
    torch.cuda.set_device(device_index)
    device = torch.device(f"cuda:{device_index}")
    streams = [torch.cuda.Stream(device=device) for _ in range(max(1, PARTITIONS))]

    is_trainer = RANK in TRAINER_RANKS
    plan = build_plan()

    trainers_per_lane = N_TRAINERS // PARTITIONS
    if is_trainer:
        source_partition = RANK // trainers_per_lane
        index_in_role = RANK
        rank_in_lane = RANK % trainers_per_lane
        rows_each = ROWS // trainers_per_lane
        row_lo = rank_in_lane * rows_each
        row_hi = row_lo + rows_each
    else:
        source_partition = None
        index_in_role = RANK - N_TRAINERS
        cols_each = COLS // N_GENERATORS
        col_lo = index_in_role * cols_each
        col_hi = col_lo + cols_each

    alloc_start = time.perf_counter()
    specs: dict[str, LocalParamSpec] = {}
    for index in range(N_PARAMS):
        if is_trainer:
            if index % PARTITIONS != source_partition:
                continue
            local = global_block(index, row_lo, row_hi, 0, COLS, device)
        else:
            local = torch.full(
                (ROWS, col_hi - col_lo), -1, dtype=TORCH_DTYPE, device=device
            )
        specs[param_name(index)] = LocalParamSpec(base=local)
    torch.cuda.synchronize()
    alloc_s = time.perf_counter() - alloc_start

    engine = BenchEngine(plan, specs)
    timer = PhaseTimer()

    channel_start = time.perf_counter()
    channel = grpc.insecure_channel(ENDPOINT)
    grpc.channel_ready_future(channel).result(timeout=120)
    channel_s = time.perf_counter() - channel_start

    rendezvous = CollectiveRendezvous(channel)
    timer.wrap(rendezvous, "join", "join")
    timer.wrap(rendezvous, "publish_bootstrap", "publish_bootstrap")
    timer.wrap(rendezvous, "await_ready", "await_ready")
    timer.wrap(client_module, "_bootstrap_barrier", "bootstrap_barrier")

    common = dict(
        rendezvous=rendezvous,
        model_name=f"bench-{RUN_ID}",
        trainer_slots=[f"t{i}" for i in range(N_TRAINERS)],
        generator_slots=[f"g{i}" for i in range(N_GENERATORS)],
        source_partition_count=PARTITIONS,
        device=device,
        streams=streams,
    )
    if is_trainer:
        client = RefitClientTrainer(
            slot_id=f"t{index_in_role}",
            worker_id=f"t{index_in_role}-{RUN_ID}",
            index_in_role=index_in_role,
            **common,
        )
    else:
        client = RefitClientGenerator(
            slot_id=f"g{index_in_role}",
            worker_id=f"g{index_in_role}-{RUN_ID}",
            index_in_role=index_in_role,
            **common,
        )
    timer.wrap(client._cache, "create", "comm_init")

    init_start = time.perf_counter()
    if is_trainer:
        client.initialize(engine, source_partition=source_partition)
    else:
        client.initialize(engine)
    initialize_s = time.perf_counter() - init_start

    bootstrap_start = time.perf_counter()
    join_wall = time.time()
    membership = client.compute_plan()
    bootstrap_s = time.perf_counter() - bootstrap_start
    log(
        f"joined group={membership.group_id} epoch={membership.epoch} "
        f"lanes={[lane.lane_id for lane in membership.lanes]} in {bootstrap_s:.3f}s"
    )

    bulk_bytes = sum(
        ROWS * COLS * ITEMSIZE for _ in range(N_PARAMS)
    )
    rounds: list[dict[str, float]] = []
    verified: bool | None = None

    for round_index in range(ROUNDS):
        label = f"v{round_index}"
        round_start = time.perf_counter()
        client.start_weight_update(label)
        start_s = time.perf_counter() - round_start

        publish_start = time.perf_counter()
        if is_trainer:
            client.publish_weights(label)
        else:
            client.update_weights(label)
        publish_s = time.perf_counter() - publish_start

        finish_start = time.perf_counter()
        client.finish_weight_update(label)
        finish_s = time.perf_counter() - finish_start

        torch.cuda.synchronize()
        total_s = time.perf_counter() - round_start
        rounds.append(
            {
                "round": round_index,
                "start_s": start_s,
                "publish_s": publish_s,
                "finish_s": finish_s,
                "total_s": total_s,
            }
        )
        log(
            f"round {round_index}: total {total_s * 1e3:.1f}ms "
            f"(enqueue {publish_s * 1e3:.1f}ms, drain {finish_s * 1e3:.1f}ms)"
        )

        if round_index == 0 and VERIFY and not is_trainer:
            ok = True
            for index in range(N_PARAMS):
                want = global_block(index, 0, ROWS, col_lo, col_hi, device)
                got = specs[param_name(index)].base
                if not torch.equal(got, want):
                    ok = False
                    log(f"VERIFY FAIL on {param_name(index)}")
                    break
                del want
            verified = ok
            log("VERIFY", "PASS" if ok else "FAIL")

    result = {
        "rank": RANK,
        "host_rank_base": RANK_BASE,
        "role": "trainer" if is_trainer else "generator",
        "run_id": RUN_ID,
        "geometry": {
            "trainers": N_TRAINERS,
            "generators": N_GENERATORS,
            "partitions": PARTITIONS,
            "params": N_PARAMS,
            "rows": ROWS,
            "cols": COLS,
            "dtype": DTYPE_NAME,
        },
        "global_bulk_bytes": bulk_bytes,
        "nccl": ".".join(str(part) for part in version),
        "group_id": membership.group_id,
        "epoch": membership.epoch,
        "process_start_wall": process_start,
        "join_start_wall": join_wall,
        "alloc_s": alloc_s,
        "channel_s": channel_s,
        "initialize_s": initialize_s,
        "bootstrap_s": bootstrap_s,
        "phases": {
            label: {"total_s": total, "calls": timer.counts[label]}
            for label, total in timer.totals.items()
        },
        "rounds": rounds,
        "verified": verified,
    }

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, f"rank{RANK}.json"), "w") as handle:
        json.dump(result, handle, indent=2)

    try:
        client.cleanup()
    except Exception as error:  # noqa: BLE001 - teardown noise must not mask the run
        log("cleanup:", type(error).__name__, error)
    return 0 if verified is not False else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
