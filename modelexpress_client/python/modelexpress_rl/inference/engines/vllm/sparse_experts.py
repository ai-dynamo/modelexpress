# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Write staged routed-expert deltas straight into live FP8 MoE kernel tensors.

Under per-expert MoE backends, such as FlashInfer TRT-LLM block FP8, each
expert's live weights are a byte permutation of its checkpoint tensors, and its
live scales a permutation plus a clamp. Both maps are derived once per expert
shape by sending position codes through vLLM's own per-rank expert loader and
kernel-format conversion. Staged weight bytes are then XORed into the live bytes
on the device, and changed scales, rebuilt from the checkpoint, are written as
absolute values.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import torch

# The FusedMoE shard each checkpoint projection loads into, and its dimension.
_SHARD = {"gate": "w1", "up": "w3", "down": "w2"}
_SHARD_DIM = {"w1": 0, "w3": 0, "w2": 1}
WEIGHT = "weight"
SCALE = "weight_scale_inv"
# Backends whose kernel-format conversion treats every expert alone.
_PER_EXPERT_BACKENDS = frozenset({"FLASHINFER_TRTLLM"})
# Codes start at 1, so a live element left at 0 has no checkpoint source.
_CODE_BITS = 24
_INT_VIEW = {1: torch.uint8, 2: torch.int16, 4: torch.int32}
# Staged changes placed per pass.
_WINDOW = 1 << 23


class SparseExpertError(RuntimeError):
    """A module's live expert layout is not the permutation sparse writes assume."""


def live_tensors(module: torch.nn.Module) -> list[torch.Tensor]:
    """The w13 and w2 weights, then their scales, as the kernel reads them."""
    scale = module.quant_method.weight_scale_name
    return [
        module.w13_weight,
        module.w2_weight,
        getattr(module, f"w13_{scale}"),
        getattr(module, f"w2_{scale}"),
    ]


def _target(proj: str, kind: str) -> int:
    """Index in ``live_tensors`` of the tensor a checkpoint projection and kind load into."""
    return (1 if _SHARD[proj] == "w2" else 0) + (2 if kind == SCALE else 0)


def scale_copies(module: torch.nn.Module) -> list[list[torch.Tensor]] | None:
    """Every live copy of the w13 and w2 scales, or None if the kernel's differ.

    After a layerwise reload, the MoE quant config can keep the scales processing
    produced while the layer holds its original storage again: eager calls read
    the config's copy and captured graphs the layer's.
    """
    _w13, _w2, w13_scale, w2_scale = live_tensors(module)
    config = getattr(module.quant_method, "moe_quant_config", None)
    copies = []
    for own, kernel in (
        (w13_scale, getattr(config, "w1_scale", None)),
        (w2_scale, getattr(config, "w2_scale", None)),
    ):
        if not isinstance(kernel, torch.Tensor):
            return None
        if kernel.data_ptr() == own.data_ptr():
            copies.append([own])
        elif (kernel.shape, kernel.dtype, kernel.device) == (
            own.shape,
            own.dtype,
            own.device,
        ):
            copies.append([own, kernel])
        else:
            return None
    return copies


def ineligible(module: torch.nn.Module) -> str | None:
    """Why sparse writes cannot install this module's experts, or None if they can."""
    method = getattr(module, "quant_method", None)
    backend = getattr(getattr(method, "fp8_backend", None), "name", None)
    if backend not in _PER_EXPERT_BACKENDS:
        return f"MoE backend {backend}"
    if not hasattr(module, "_load_model_weight_or_group_weight_scale"):
        return "vLLM has no per-expert loader"
    if not getattr(method, "block_quant", False):
        return "experts are not block quantized"
    config = module.moe_config
    if not getattr(config, "is_act_and_mul", False):
        return "experts have no gated activation"
    if getattr(config, "tp_shard_with_padding", False):
        return "tensor-parallel shards are padded"
    if getattr(config.moe_parallel_config, "ep_size", 1) != 1:
        return "expert parallelism"
    if any(not tensor.is_contiguous() for tensor in live_tensors(module)):
        return "live expert tensors are not contiguous"
    if scale_copies(module) is None:
        return "the kernel's scales differ from the layer's"
    return None


@dataclass(frozen=True)
class ExpertLayout:
    """Where each checkpoint element of one expert lands in its live slot.

    ``maps[(proj, kind)][e]`` is the live element, within the expert's slot of
    the tensor that kind loads into, of checkpoint element ``e``, or -1 when
    another rank owns it. ``sources[(proj, SCALE)]`` lists the scale elements
    this rank owns, in order, and ``floor`` is the value processing clamps
    scales up to, 0 if none.
    """

    maps: dict[tuple[str, str], torch.Tensor]
    sources: dict[tuple[str, str], torch.Tensor]
    slots: tuple[int, ...]
    floor: float


def derive_layout(
    module: torch.nn.Module,
    shapes: Mapping[tuple[str, str], tuple[int, ...]],
    device: torch.device,
) -> ExpertLayout:
    """Map one expert's checkpoint elements to its live elements on this rank.

    ``shapes`` holds the checkpoint shape of one expert's tensor of every
    projection and kind.
    """
    try:
        from vllm.model_executor.layers.fused_moe.oracle.fp8 import (
            convert_to_fp8_moe_kernel_format,
        )
    except ImportError as error:
        raise SparseExpertError(
            f"vLLM has no FP8 MoE kernel-format conversion: {error}"
        ) from error

    live = live_tensors(module)
    tp_size = module.moe_config.moe_parallel_config.tp_size
    tp_rank = module.moe_config.tp_rank

    def run(proj: str, kind: str, value: torch.Tensor) -> list[torch.Tensor]:
        temps = _expert_temps(shapes, live, tp_size, device)
        shard = _SHARD[proj]
        module._load_model_weight_or_group_weight_scale(
            shard_dim=_SHARD_DIM[shard],
            expert_data=temps[_target(proj, kind)][0],
            shard_id=shard,
            loaded_weight=value,
            tp_rank=tp_rank,
        )
        converted = convert_to_fp8_moe_kernel_format(
            fp8_backend=module.quant_method.fp8_backend,
            layer=module,
            w13=temps[0],
            w2=temps[1],
            w13_scale=temps[2],
            w2_scale=temps[3],
            w13_input_scale=None,
            w2_input_scale=None,
        )
        for out, tensor in zip(converted, live, strict=True):
            if out.shape[1:] != tensor.shape[1:] or out.dtype != tensor.dtype:
                raise SparseExpertError(
                    f"converted expert {tuple(out.shape)} {out.dtype} does not fit "
                    f"live {tuple(tensor.shape)} {tensor.dtype}"
                )
        return list(converted)

    maps, sources = {}, {}
    for (proj, kind), shape in shapes.items():
        numel = math.prod(shape)
        if numel >= 1 << _CODE_BITS:
            raise SparseExpertError(
                f"{proj} {kind} has too many elements for position codes"
            )
        target = _target(proj, kind)
        codes = torch.arange(1, numel + 1, dtype=torch.int64, device=device).reshape(
            shape
        )
        if kind == SCALE:
            if live[target].dtype != torch.float32:
                raise SparseExpertError(f"live {proj} scales are {live[target].dtype}")
            recovered = (
                run(proj, kind, codes.to(torch.float32))[target]
                .reshape(-1)
                .to(torch.int64)
            )
        else:
            dtype = live[target].dtype
            size = dtype.itemsize
            bits = 8 * size
            recovered = torch.zeros(
                live[target][0].numel(), dtype=torch.int64, device=device
            )
            for shift in range(0, _CODE_BITS, bits):
                digit = (
                    ((codes >> shift) & ((1 << bits) - 1))
                    .to(_INT_VIEW[size])
                    .view(dtype)
                )
                plane = (
                    run(proj, kind, digit)[target][0].reshape(-1).view(_INT_VIEW[size])
                )
                recovered |= (plane.to(torch.int64) & ((1 << bits) - 1)) << shift
        owned = torch.nonzero(recovered, as_tuple=True)[0]
        source = recovered[owned] - 1
        # Each rank owns one tensor-parallel share of every projection, once.
        if source.numel() != numel // tp_size or (
            source.numel()
            and (
                int(source.max()) >= numel
                or torch.unique(source).numel() != source.numel()
            )
        ):
            raise SparseExpertError(f"{proj} {kind} processing is not a permutation")
        mapping = torch.full((numel,), -1, dtype=torch.int64, device=device)
        mapping[source] = owned
        maps[(proj, kind)] = mapping
        if kind == SCALE:
            sources[(proj, kind)] = torch.sort(source).values
    # Zero scales come back as whatever processing clamps them to.
    zeros = run("gate", SCALE, torch.zeros(shapes[("gate", SCALE)], device=device))
    floors = torch.unique(torch.cat([zeros[2].reshape(-1), zeros[3].reshape(-1)]))
    if floors.numel() != 1:
        raise SparseExpertError("processing changes zero scales unevenly")
    return ExpertLayout(
        maps, sources, tuple(tensor[0].numel() for tensor in live), float(floors[0])
    )


def _expert_temps(
    shapes: Mapping[tuple[str, str], tuple[int, ...]],
    live: list[torch.Tensor],
    tp_size: int,
    device: torch.device,
) -> list[torch.Tensor]:
    """Zeroed load-time tensors for one expert on this rank: w13, w2 and their scales."""
    gate, down = shapes[("gate", WEIGHT)], shapes[("down", WEIGHT)]
    gate_scale, down_scale = shapes[("gate", SCALE)], shapes[("down", SCALE)]

    def zeros(rows: int, cols: int, dtype: torch.dtype) -> torch.Tensor:
        size = dtype.itemsize
        return torch.zeros(
            (1, rows, cols * size), dtype=torch.uint8, device=device
        ).view(dtype)

    return [
        zeros(2 * (gate[0] // tp_size), gate[1], live[0].dtype),
        zeros(down[0], down[1] // tp_size, live[1].dtype),
        zeros(2 * (gate_scale[0] // tp_size), gate_scale[1], torch.float32),
        zeros(down_scale[0], down_scale[1] // tp_size, torch.float32),
    ]


class SparseExperts:
    """Sparse expert writes on one rank, with a layout per expert shape."""

    def __init__(self, device: torch.device) -> None:
        self._device = device
        self._layouts: dict[tuple, ExpertLayout] = {}

    def layout(
        self, module: torch.nn.Module, shapes: Mapping[tuple[str, str], tuple[int, ...]]
    ) -> ExpertLayout:
        key = (
            tuple(sorted(shapes.items())),
            tuple(
                (tuple(tensor.shape), tensor.dtype) for tensor in live_tensors(module)
            ),
        )
        layout = self._layouts.get(key)
        if layout is None:
            layout = derive_layout(module, shapes, self._device)
            self._layouts[key] = layout
        return layout

    @torch.no_grad()
    def write_weights(
        self,
        module: torch.nn.Module,
        layout: ExpertLayout,
        rules: torch.Tensor,
        counts: torch.Tensor,
        begin: int,
        runs: Iterable,
    ) -> int:
        """XOR staged weight bytes into the live experts; return the bytes written.

        ``rules`` has one column per staged row from ``begin``: the projection
        (0 gate, 1 up, 2 down, -1 to skip) and the local expert. ``counts`` holds
        the rows' staged byte counts.
        """
        live = live_tensors(module)
        written = 0
        for run in runs:
            low, high = run.begin - begin, run.end - begin
            total = run.values.numel()
            if total == 0:
                continue
            rows = torch.repeat_interleave(
                torch.arange(high - low, device=self._device),
                counts[low:high],
                output_size=total,
            )
            for start in range(0, total, _WINDOW):
                stop = min(start + _WINDOW, total)
                proj, expert = rules[:, low:high][:, rows[start:stop]]
                offsets = run.offsets[start:stop].to(torch.int64)
                if run.offsets.dtype == torch.int32:
                    offsets &= 0xFFFFFFFF
                values = run.values[start:stop]
                for code, name in enumerate(("gate", "up", "down")):
                    target = _target(name, WEIGHT)
                    size = live[target].element_size()
                    pick = proj == code
                    position = offsets[pick]
                    mapped = layout.maps[(name, WEIGHT)][
                        torch.div(position, size, rounding_mode="floor")
                    ]
                    keep = mapped >= 0
                    index = (
                        expert[pick][keep] * layout.slots[target] + mapped[keep]
                    ) * size
                    index += position[keep] % size
                    flat = live[target].view(-1).view(torch.uint8)
                    flat[index] ^= values[pick][keep]
                    written += index.numel()
        return written

    @torch.no_grad()
    def write_scales(
        self,
        module: torch.nn.Module,
        layout: ExpertLayout,
        scales: Iterable[tuple[str, int, torch.Tensor]],
    ) -> int:
        """Write target scales as absolute values; return the tensors written.

        ``scales`` yields each changed scale's projection, local expert and full
        checkpoint value.
        """
        copies = scale_copies(module)
        if copies is None:
            raise SparseExpertError("the kernel's scales differ from the layer's")
        # A projection's changed scales go to the device as one stacked copy.
        grouped: dict[str, tuple[list[int], list[torch.Tensor]]] = {}
        for proj, expert, value in scales:
            experts, values = grouped.setdefault(proj, ([], []))
            experts.append(expert)
            values.append(value.reshape(-1))
        writes: dict[int, tuple[list[torch.Tensor], list[torch.Tensor]]] = {
            2: ([], []),
            3: ([], []),
        }
        count = 0
        for proj, (experts, values) in grouped.items():
            key = (proj, SCALE)
            source = layout.sources[key]
            target = _target(proj, SCALE)
            local = torch.tensor(experts, dtype=torch.int64).to(self._device)
            stacked = torch.stack(values).to(device=self._device, dtype=torch.float32)
            writes[target][0].append(
                (
                    local[:, None] * layout.slots[target] + layout.maps[key][source]
                ).reshape(-1)
            )
            writes[target][1].append(stacked[:, source].reshape(-1))
            count += len(experts)
        for target, (indices, values) in writes.items():
            if not indices:
                continue
            index, value = torch.cat(indices), torch.cat(values)
            if layout.floor > 0:
                value = value.clamp(min=layout.floor)
            for copy in copies[target - 2]:
                copy.view(-1)[index] = value.to(copy.dtype)
        return count


__all__ = [
    "SCALE",
    "WEIGHT",
    "ExpertLayout",
    "SparseExpertError",
    "SparseExperts",
    "derive_layout",
    "ineligible",
    "live_tensors",
]
