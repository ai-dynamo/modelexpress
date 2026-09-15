# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Derive the canonical refit plan from a Hugging Face checkpoint on disk.

Both sides call this and must agree byte for byte, so it reads only the
checkpoint's own metadata -- never a live model object, whose parameter set
differs between a trainer and an inference engine. The safetensors header
carries name, shape and dtype for every tensor, which is exactly the plan's
bulk record, and reading it needs no torch and no GPU.
"""

from __future__ import annotations

import json
import os
import struct

from modelexpress_rl.collective import MeshSpec, ParamPlan, Placement, ReshardPlan

#: Checkpoint dtypes the plan records; the wire op carries the stored dtype.
_DTYPE_NAMES = {
    "BF16": "bfloat16",
    "F16": "float16",
    "F32": "float32",
}


def safetensors_header(path: str) -> dict:
    """Parse one safetensors file's JSON header without reading the tensors."""
    with open(path, "rb") as handle:
        (length,) = struct.unpack("<Q", handle.read(8))
        return json.loads(handle.read(length))


def checkpoint_tensors(model_dir: str) -> dict[str, tuple[tuple[int, ...], str]]:
    """Map every checkpoint tensor to its global shape and plan dtype name."""
    shards = []
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as handle:
            index = json.load(handle)
        shards = sorted(set(index["weight_map"].values()))
    else:
        shards = ["model.safetensors"]

    out: dict[str, tuple[tuple[int, ...], str]] = {}
    for shard in shards:
        header = safetensors_header(os.path.join(model_dir, shard))
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            dtype = _DTYPE_NAMES.get(meta["dtype"])
            if dtype is None:
                raise ValueError(f"{name}: unsupported checkpoint dtype {meta['dtype']}")
            out[name] = (tuple(meta["shape"]), dtype)
    return out


def tied_names(model_dir: str) -> set[str]:
    """Checkpoint tensors the loaded model will not hold as distinct parameters.

    A tied output head is stored in the checkpoint and then aliased onto the
    embedding by every framework that loads it, so neither a trainer nor an
    inference engine owns storage for it. Planning it would declare a transfer
    with no source and no destination, and the failure surfaces only once a
    worker looks for local storage.
    """
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        return set()
    with open(config_path) as handle:
        config = json.load(handle)
    text = config.get("text_config", config)
    if text.get("tie_word_embeddings") or config.get("tie_word_embeddings"):
        return {"lm_head.weight", "language_model.lm_head.weight"}
    return set()


def layer_index(name: str) -> int | None:
    """The transformer layer a parameter belongs to, if it belongs to one."""
    parts = name.split(".")
    for position, part in enumerate(parts[:-1]):
        if part == "layers" and parts[position + 1].isdigit():
            return int(parts[position + 1])
    return None


#: How a tensor-parallel inference engine splits each parameter. The suffix rule
#: is a fact about this model family's architecture, and it lives HERE, in the
#: example's plan builder, rather than in the shared core -- the core never
#: infers geometry from a parameter name, which is what keeps it portable.
_COLUMN_PARALLEL = (
    "q_proj.weight",
    "k_proj.weight",
    "v_proj.weight",
    "gate_proj.weight",
    "up_proj.weight",
    "embed_tokens.weight",
    "lm_head.weight",
)
_ROW_PARALLEL = ("o_proj.weight", "down_proj.weight")


def engine_placement(name: str, shape: tuple[int, ...], generators: int):
    """Where one canonical parameter lands in a tensor-parallel engine.

    Returns None when it cannot be delivered pre-split -- a dimension that does
    not divide the engine's world, or a parameter this rule does not classify.
    The caller falls back to replicating those, which is always correct and
    costs bandwidth rather than correctness.
    """
    if name.endswith(_COLUMN_PARALLEL):
        return Placement.shard(0) if shape[0] % generators == 0 else None
    if name.endswith(_ROW_PARALLEL):
        if len(shape) < 2 or shape[1] % generators != 0:
            return None
        return Placement.shard(1)
    if len(shape) == 1:
        return Placement.replicate()
    return None


def build_plan(
    model_dir: str,
    *,
    trainers: int,
    generators: int,
    dst_layout: str = "replicate",
) -> tuple[ReshardPlan, list[list[str]]]:
    """Build the plan and its layer groups for one trainer/generator geometry.

    The trainer holds each parameter sharded on dim 0, which is what a
    dim-0-sharding data-parallel framework produces.

    ``dst_layout`` decides what the generator receives. ``replicate`` hands each
    rank the whole tensor and lets the engine's own loader split it, which is
    correct for any architecture and costs the engine's world size in wire
    bytes. ``sharded`` delivers each rank exactly the slice it will keep, which
    carries the bytes once but needs the receiver to know its engine's fused
    layout, so the Loader stages through ``pre``/``post`` hooks.

    Layer groups follow the transformer's own layers, so a receiver installs
    one layer at a time instead of holding the whole model in scratch.
    """
    tensors = checkpoint_tensors(model_dir)
    for name in tied_names(model_dir):
        tensors.pop(name, None)

    src_mesh = MeshSpec(shape=(trainers,))
    dst_mesh = MeshSpec(shape=(generators,), rank_offset=trainers)

    bulk: list[ParamPlan] = []
    undivided: list[str] = []
    for name in sorted(tensors):
        shape, dtype = tensors[name]
        if not shape:
            undivided.append(name)
            continue
        if shape[0] % trainers != 0:
            # The plan rejects a placement that does not divide evenly, and
            # guessing a different shard dim here would move wrong bytes
            # silently. Name it instead.
            undivided.append(name)
            continue
        dst = Placement.replicate()
        if dst_layout == "sharded":
            dst = engine_placement(name, shape, generators) or Placement.replicate()
        bulk.append(
            ParamPlan(
                name=name,
                global_shape=shape,
                dtype=dtype,
                partition_id=0,
                src_mesh=src_mesh,
                src_placements=(Placement.shard(0),),
                dst_mesh=dst_mesh,
                dst_placements=(dst,),
            )
        )
    if undivided:
        raise ValueError(
            f"{len(undivided)} checkpoint tensor(s) cannot be sharded on dim 0 across "
            f"{trainers} trainers: {', '.join(undivided[:5])}"
        )

    groups: dict[int, list[str]] = {}
    for entry in bulk:
        index = layer_index(entry.name)
        # Everything outside the layer stack rides one trailing group, so the
        # embedding and the final norm are installed together at the end.
        key = -1 if index is None else index
        groups.setdefault(key, []).append(entry.name)

    ordered = [groups[key] for key in sorted(groups)]
    plan = ReshardPlan(bulk=bulk, misc=[], source_partition_count=1)
    return plan, ordered
