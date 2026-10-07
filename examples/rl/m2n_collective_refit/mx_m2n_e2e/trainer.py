# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trainer side: a real checkpoint, sharded by a real framework, published.

One process per trainer rank, holding a real Hugging Face checkpoint under a
real sharding framework. Two are wired, because the Publisher is where a
framework's storage layout is encoded and one backend cannot show that the
boundary is general:

``fsdp2``
    ``torch.distributed.fsdp.fully_shard`` -- DTensor, dim-0 per parameter. The
    layout NeMo RL's DTensor policy and verl's FSDP worker hold policy weights
    in. The Publisher hands the wire op the local shard itself, so the trainer
    side does no staging copy at all.

``deepspeed``
    ZeRO-3, which partitions each parameter's FLATTENED storage across ranks.
    That is not a dim-0 shard and cannot be declared as one, so this backend
    stages: ``start_new_round`` gathers one parameter at a time and copies this
    rank's dim-0 slice into a proxy buffer, which is the case ``LocalParamSpec``
    exists for. One parameter is gathered at a time rather than the model, and
    the staging happens outside the collective rather than in a ``pre`` hook,
    because a gather on the framework's own communicator while reshard lanes
    are in flight is how two NCCL communicators deadlock.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import grpc
import torch
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor, Shard

from modelexpress_rl.collective import LocalParamSpec, RefitClientTrainer
from modelexpress_rl.collective.rendezvous import CollectiveRendezvous

from .plan_from_hf import build_plan


class FsdpPublisher:
    """``Publisher`` over an FSDP2-sharded Hugging Face model."""

    def __init__(self, model, plan, groupings, mesh_size: int) -> None:
        self._plan = plan
        self._groupings = groupings
        self._specs: dict[str, LocalParamSpec] = {}

        named = dict(model.named_parameters())
        missing = [entry.name for entry in plan.bulk if entry.name not in named]
        if missing:
            raise KeyError(
                f"{len(missing)} planned parameter(s) are absent from the trainer "
                f"model: {', '.join(missing[:5])}"
            )
        for entry in plan.bulk:
            param = named[entry.name]
            local = self._local_shard(entry, param, mesh_size)
            self._specs[entry.name] = LocalParamSpec(base=local)

    @staticmethod
    def _local_shard(entry, param, mesh_size: int):
        """This rank's slice, checked against what the plan promised.

        A mismatch here is the failure the collective cannot report: every rank
        would issue its agreed op over storage of the wrong extent, and the
        bytes would land wrong rather than erroring.
        """
        if not isinstance(param, DTensor):
            raise TypeError(
                f"{entry.name} is not sharded; expected an FSDP2 DTensor, got "
                f"{type(param).__name__}"
            )
        placements = param.placements
        if len(placements) != 1 or not isinstance(placements[0], Shard) or placements[0].dim != 0:
            raise ValueError(
                f"{entry.name}: the plan declares Shard(0) but the trainer holds "
                f"{placements}"
            )
        local = param.to_local()
        expected = entry.global_shape[0] // mesh_size
        if local.shape[0] != expected:
            raise ValueError(
                f"{entry.name}: local shard has {local.shape[0]} rows, the plan "
                f"expects {expected} (an FSDP pad would land wrong bytes silently)"
            )
        return local.detach()

    def capture(self):
        return self._plan

    def parameter_names(self) -> list[str]:
        return self._plan.parameter_names()

    def local_params(self) -> dict[str, LocalParamSpec]:
        return self._specs

    def start_new_round(self, version: str) -> None:
        pass

    def cleanup(self) -> None:
        self._specs.clear()


class DeepSpeedPublisher:
    """``Publisher`` over a ZeRO-3 model, staging each dim-0 slice per round."""

    def __init__(self, engine, plan, groupings, mesh_size: int, rank: int, device) -> None:
        self._engine = engine
        self._plan = plan
        self._groupings = groupings
        self._mesh = mesh_size
        self._rank = rank
        self._named = dict(engine.module.named_parameters())
        self._specs: dict[str, LocalParamSpec] = {}

        dtypes = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        missing = [e.name for e in plan.bulk if e.name not in self._named]
        if missing:
            raise KeyError(
                f"{len(missing)} planned parameter(s) are absent from the trainer "
                f"model: {', '.join(missing[:5])}"
            )
        for entry in plan.bulk:
            rows = entry.global_shape[0] // mesh_size
            shape = (rows, *entry.global_shape[1:])
            self._specs[entry.name] = LocalParamSpec(
                base=torch.empty(shape, dtype=dtypes[entry.dtype], device=device)
            )

    def capture(self):
        return self._plan

    def parameter_names(self) -> list[str]:
        return self._plan.parameter_names()

    def local_params(self) -> dict[str, LocalParamSpec]:
        return self._specs

    def start_new_round(self, version: str) -> None:
        import deepspeed

        for entry in self._plan.bulk:
            param = self._named[entry.name]
            rows = entry.global_shape[0] // self._mesh
            lo = self._rank * rows
            with deepspeed.zero.GatheredParameters([param], modifier_rank=None):
                self._specs[entry.name].base.copy_(param.data[lo : lo + rows])
        torch.cuda.synchronize()

    def cleanup(self) -> None:
        self._specs.clear()


def build_fsdp2(model_dir, mesh, device, trainers):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16).to(device)
    for module in model.model.layers:
        fully_shard(module, mesh=mesh)
    fully_shard(model, mesh=mesh)
    return model


def build_deepspeed(model_dir, device, trainers):
    """A ZeRO-3 engine holding the checkpoint, partitioned as DeepSpeed does it.

    ``HfDeepSpeedConfig`` has to exist, and stay alive, before ``from_pretrained``
    runs: it is what puts transformers into ZeRO-3 loading. Entering
    ``deepspeed.zero.Init`` by hand instead partitions the parameters underneath a
    loader that then compares the partitioned extents against the checkpoint's and
    refuses them as mismatched shapes.
    """
    import deepspeed
    from transformers import AutoModelForCausalLM
    from transformers.integrations import HfDeepSpeedConfig

    config = {
        "train_micro_batch_size_per_gpu": 1,
        "bf16": {"enabled": True},
        "zero_optimization": {"stage": 3, "stage3_param_persistence_threshold": 0},
    }
    holder = HfDeepSpeedConfig(config)  # noqa: F841 - must outlive from_pretrained
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16)
    engine, *_ = deepspeed.initialize(model=model, config=config)
    engine._mx_hf_ds_config = holder
    return engine


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--trainers", type=int, required=True)
    parser.add_argument("--generators", type=int, required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--backend", default="fsdp2", choices=["fsdp2", "deepspeed"])
    parser.add_argument(
        "--dst-layout", default="replicate", choices=["replicate", "sharded"]
    )
    parser.add_argument("--out", default="/work/out")
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    local_device = int(os.environ["LOCAL_DEVICE"])
    torch.cuda.set_device(local_device)
    device = torch.device(f"cuda:{local_device}")

    dist.init_process_group(backend="nccl", world_size=args.trainers, rank=rank)
    mesh = dist.device_mesh.init_device_mesh("cuda", (args.trainers,))

    load_start = time.perf_counter()
    if args.backend == "fsdp2":
        model = build_fsdp2(args.model_dir, mesh, device, args.trainers)
    else:
        model = build_deepspeed(args.model_dir, device, args.trainers)
    load_s = time.perf_counter() - load_start

    plan, groupings = build_plan(
        args.model_dir,
        trainers=args.trainers,
        generators=args.generators,
        dst_layout=args.dst_layout,
    )
    if args.backend == "fsdp2":
        publisher = FsdpPublisher(model, plan, groupings, args.trainers)
    else:
        publisher = DeepSpeedPublisher(
            model, plan, groupings, args.trainers, rank, device
        )

    channel = grpc.insecure_channel(args.endpoint)
    grpc.channel_ready_future(channel).result(timeout=120)
    client = RefitClientTrainer(
        rendezvous=CollectiveRendezvous(channel),
        model_name=args.model_name,
        trainer_slots=[f"t{i}" for i in range(args.trainers)],
        generator_slots=[f"g{i}" for i in range(args.generators)],
        source_partition_count=1,
        slot_id=f"t{rank}",
        worker_id=f"t{rank}-{args.run_id}",
        index_in_role=rank,
        device=device,
        streams=[torch.cuda.Stream(device=device)],
    )
    client.setup_layer_groups(groupings)
    client.initialize(publisher, source_partition=0)

    bootstrap_start = time.perf_counter()
    membership = client.compute_plan()
    bootstrap_s = time.perf_counter() - bootstrap_start
    print(
        f"[trainer {rank}] joined group={membership.group_id} epoch={membership.epoch} "
        f"in {bootstrap_s:.2f}s, {len(plan.bulk)} params in {len(groupings)} layer groups",
        flush=True,
    )

    rounds = []
    for index in range(args.rounds):
        version = f"v{index}"
        started = time.perf_counter()
        client.start_weight_update(version)
        publish_start = time.perf_counter()
        for group_id in range(len(groupings)):
            client.publish_weights(version, group_id)
        publish_s = time.perf_counter() - publish_start
        finish_start = time.perf_counter()
        client.finish_weight_update(version)
        finish_s = time.perf_counter() - finish_start
        torch.cuda.synchronize()
        total_s = time.perf_counter() - started
        rounds.append(
            {
                "round": index,
                "publish_s": publish_s,
                "finish_s": finish_s,
                "total_s": total_s,
            }
        )
        print(
            f"[trainer {rank}] round {index}: {total_s * 1e3:.1f}ms "
            f"(enqueue {publish_s * 1e3:.1f}ms, drain {finish_s * 1e3:.1f}ms)",
            flush=True,
        )

    bulk_bytes = sum(
        _numel(entry.global_shape) * _itemsize(entry.dtype) for entry in plan.bulk
    )
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"trainer{rank}.json"), "w") as handle:
        json.dump(
            {
                "rank": rank,
                "role": "trainer",
                "backend": args.backend,
                "dst_layout": args.dst_layout,
                "model_dir": args.model_dir,
                "model_load_s": load_s,
                "bootstrap_s": bootstrap_s,
                "bulk_params": len(plan.bulk),
                "layer_groups": len(groupings),
                "global_bulk_bytes": bulk_bytes,
                "rounds": rounds,
            },
            handle,
            indent=2,
        )

    try:
        client.cleanup()
    except Exception as error:  # noqa: BLE001 - teardown noise must not mask the run
        print(f"[trainer {rank}] cleanup: {type(error).__name__} {error}", flush=True)
    dist.destroy_process_group()
    return 0


def _numel(shape) -> int:
    total = 1
    for extent in shape:
        total *= extent
    return total


def _itemsize(dtype: str) -> int:
    return {"bfloat16": 2, "float16": 2, "float32": 4}[dtype]


if __name__ == "__main__":
    raise SystemExit(main())
