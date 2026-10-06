# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trainer side, JAX: a sharded ``jax.Array`` tree published over M2N.

The counterpart of ``trainer.py``'s ``FsdpPublisher``, and deliberately shaped
the same way, because the two together are what show the engine boundary is
general rather than a torch interface with one other caller.

What differs from the torch side is only where the local bytes are found.
FSDP2 hands a ``DTensor`` whose ``to_local()`` is this rank's slice; JAX hands
a globally-sharded array whose ``addressable_shards[0].data`` is. Both are a
dense row-major buffer of the same extent, and on GPU the two produce
byte-identical storage for every dtype measured, which is what lets one plan
describe both sides of a mixed transfer.

The weights are real checkpoint bytes, sharded across processes by JAX's own
``NamedSharding`` and then passed through a jitted kernel each round, so the
buffers the wire op reads are XLA-produced under an XLA-chosen layout rather
than host arrays copied to a device. What is deliberately absent is a JAX
*model implementation*: no Flax port of this architecture exists, and inventing
one would test a model rather than the transfer. The engine on the other side
is a live vLLM, not a stub.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import time
from typing import Any

import grpc
import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np

from modelexpress_rl.collective import LocalParamSpec, RefitClientTrainer
from modelexpress_rl.collective import jax_interop
from modelexpress_rl.collective.rendezvous import CollectiveRendezvous

from .plan_from_hf import build_plan, safetensors_header


class JaxPublisher:
    """``Publisher`` over a sharded JAX parameter tree.

    ``params`` maps each canonical parameter name onto its **global**
    ``jax.Array``. The shard is taken here rather than by the caller so the
    slice it holds can be checked against what the plan promised.
    """

    def __init__(
        self,
        params: dict[str, Any],
        plan,
        groupings,
        rank: int,
    ) -> None:
        self._plan = plan
        self._groupings = groupings
        self._specs: dict[str, LocalParamSpec] = {}
        self._arrays: list[Any] = []

        self._rank = rank

        missing = [entry.name for entry in plan.bulk if entry.name not in params]
        if missing:
            raise KeyError(
                f"{len(missing)} planned parameter(s) are absent from the trainer "
                f"tree: {', '.join(missing[:5])}"
            )
        self.update(params)

    def update(self, params: dict[str, Any]) -> None:
        """Point the existing specs at this round's buffers.

        The client resolves ``local_params()`` once, at ``compute_plan``, and
        the sender keeps that dict, so a round's new storage has to arrive by
        mutating it in place. This is the JAX shape of what the ZeRO-3
        publisher does when it stages into a proxy buffer each round.
        """
        self._arrays.clear()
        for entry in self._plan.bulk:
            shard = jax_interop.local_shard(
                params[entry.name],
                name=entry.name,
                expect_index=jax_interop.expected_index(
                    entry.global_shape, entry.src_mesh, entry.src_placements, self._rank
                ),
            )
            self._arrays.append(shard)
            self._specs[entry.name] = LocalParamSpec(
                base=jax_interop.JaxDeviceBuffer(shard)
            )

    def capture(self):
        return self._plan

    def parameter_names(self) -> list[str]:
        return self._plan.parameter_names()

    def local_params(self) -> dict[str, LocalParamSpec]:
        return self._specs

    def start_new_round(self, version: str) -> None:
        """Wait for every shard's producing computation before the wire op.

        JAX enqueues on its own stream and ``reshard`` takes a bare device
        pointer with no stream handshake, so nothing else orders the transfer
        against the step that produced these weights. The race is measured
        rather than assumed: a consumer reading the buffer on another stream
        mid-computation sees intermediate values. One synchronize per round
        rather than per parameter, which the protocol allows because no
        parameter may change after the round opens.
        """
        jax.block_until_ready(self._arrays)

    def cleanup(self) -> None:
        self._specs.clear()
        self._arrays.clear()


def _shard_files(model_dir: str) -> dict[str, str]:
    """Map each checkpoint tensor onto the safetensors file that holds it."""
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as handle:
            return json.load(handle)["weight_map"]
    header = safetensors_header(os.path.join(model_dir, "model.safetensors"))
    return {name: "model.safetensors" for name in header if name != "__metadata__"}


_NUMPY_DTYPE = {
    "bfloat16": ml_dtypes.bfloat16,
    "float16": np.float16,
    "float32": np.float32,
}


def load_local_piece(model_dir: str, entry, *, rank: int) -> np.ndarray:
    """This rank's slice of one parameter, read straight out of the checkpoint.

    The slice is the one the plan says the collective reads from this rank, on
    every sharded dim. A safetensors tensor is stored dense and row-major, so a
    memory map of its byte range is the whole tensor viewed in place, and
    slicing it reads only this rank's piece off disk, which is what keeps a
    trainer process off the whole model.
    """
    shard = _shard_files(model_dir)[entry.name]
    path = os.path.join(model_dir, shard)
    with open(path, "rb") as handle:
        (length,) = struct.unpack("<Q", handle.read(8))
        header = json.loads(handle.read(length))
    begin, end = header[entry.name]["data_offsets"]
    dtype = np.dtype(_NUMPY_DTYPE[entry.dtype])
    if end - begin != _numel(entry.global_shape) * dtype.itemsize:
        raise ValueError(
            f"{entry.name}: stored bytes do not match the plan's shape and dtype"
        )
    whole = np.memmap(
        path,
        dtype=dtype,
        mode="r",
        offset=8 + length + begin,
        shape=tuple(entry.global_shape),
    )
    index = jax_interop.expected_index(
        entry.global_shape, entry.src_mesh, entry.src_placements, rank
    )
    return np.ascontiguousarray(whole[index])


def build_params(model_dir: str, plan, sharding_for, *, rank: int):
    """Every planned parameter as a globally-sharded ``jax.Array``."""
    params = {}
    for entry in plan.bulk:
        local = load_local_piece(model_dir, entry, rank=rank)
        params[entry.name] = jax.make_array_from_process_local_data(
            sharding_for(entry),
            local,
            tuple(entry.global_shape),
        )
    return params


@jax.jit
def _step(tree, scale):
    """One jitted pass over the whole tree, standing in for an optimizer step.

    ``scale`` is a traced operand rather than a literal so XLA cannot fold the
    multiply away and hand back the input buffer: the point is that each round's
    weights are produced by a kernel, on JAX's stream, the way a real trainer's
    are. It is numerically the identity at scale 1.0, which keeps the refit
    checkable against the checkpoint.

    The input is deliberately not donated. Donation would let XLA write the
    result back into the buffer the previous round's specs still point at,
    which is correct only by accident of ordering; a fresh allocation makes
    the per-round ``update`` the thing that keeps them current.
    """
    return jax.tree.map(lambda x: (x.astype(jnp.float32) * scale).astype(x.dtype), tree)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--trainers", type=int, required=True)
    parser.add_argument("--generators", type=int, required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument(
        "--dst-layout", default="replicate", choices=["replicate", "sharded"]
    )
    parser.add_argument(
        "--src-layout", default="dim0", choices=["dim0", "dim1", "2d"]
    )
    parser.add_argument("--out", default="/work/out")
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    jax.distributed.initialize(
        coordinator_address=f"{os.environ['MASTER_ADDR']}:{os.environ['JAX_PORT']}",
        num_processes=args.trainers,
        process_id=rank,
    )
    devices = jax.devices()
    if len(devices) != args.trainers:
        raise RuntimeError(
            f"expected {args.trainers} global JAX devices, got {len(devices)}; "
            "each trainer process must see exactly one"
        )
    sharding_for = jax_interop.plan_sharding(devices)

    plan, groupings = build_plan(
        args.model_dir,
        trainers=args.trainers,
        generators=args.generators,
        dst_layout=args.dst_layout,
        src_layout=args.src_layout,
    )

    load_start = time.perf_counter()
    params = build_params(args.model_dir, plan, sharding_for, rank=rank)
    scale = jnp.asarray(1.0, dtype=jnp.float32)
    params = _step(params, scale)
    jax.block_until_ready(params)
    load_s = time.perf_counter() - load_start

    publisher = JaxPublisher(params, plan, groupings, rank)

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
        device=0,
        barrier_alloc=jax_interop.barrier_buffer,
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
        # A fresh kernel per round, so every round's buffers are XLA output
        # rather than the ones the previous round already drained.
        params = _step(params, scale)
        publisher.update(params)
        client.start_weight_update(version)
        publish_start = time.perf_counter()
        for group_id in range(len(groupings)):
            client.publish_weights(version, group_id)
        publish_s = time.perf_counter() - publish_start
        finish_start = time.perf_counter()
        client.finish_weight_update(version)
        finish_s = time.perf_counter() - finish_start
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
                "backend": "jax",
                "dst_layout": args.dst_layout,
                "src_layout": args.src_layout,
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
