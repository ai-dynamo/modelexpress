# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tiny optimizer-step source payload helpers for runtime refit smokes.

These helpers intentionally model only the source-payload side of a trainer:
each source rank owns one shard, runs a real ``torch.optim.SGD`` step over a
small synthetic objective for that shard, and publishes the post-step tensor for
NIXL reads. This is not a production RL/FSDP trainer integration, but it removes
the older static replacement formula from the runtime bridge proofs.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import prod
from typing import Any, Sequence

import torch

from .resharding import SliceOwnership, TensorRange

DEFAULT_TRAINER_STEP_COUNT = 1
DEFAULT_TRAINER_LR = 0.125


@dataclass(frozen=True)
class TrainerStepSourcePublication:
    """Post-step source tensor plus the metadata a trainer rank publishes."""

    ownership: SliceOwnership
    tensor: torch.Tensor
    provenance: dict[str, Any]

    def to_artifact_metadata(self) -> dict[str, Any]:
        return {
            "ownership": self.ownership.to_dict(),
            "source_payload_provenance": dict(self.provenance),
            "tensor_shape": [int(dim) for dim in self.tensor.shape],
            "tensor_dtype": str(self.tensor.dtype).removeprefix("torch."),
            "tensor_device": str(self.tensor.device),
            "tensor_bytes": int(self.tensor.numel() * self.tensor.element_size()),
        }


@dataclass(frozen=True)
class TrainerLoopStepPublication:
    """All source-rank publications for one versioned trainer-loop step."""

    model_name: str
    base_model_version: str
    model_version: str
    step_index: int
    source_publications: tuple[TrainerStepSourcePublication, ...]
    provenance: dict[str, Any]

    def to_artifact_metadata(self) -> dict[str, Any]:
        source_publications = [
            publication.to_artifact_metadata()
            for publication in self.source_publications
        ]
        return {
            "model_name": self.model_name,
            "base_model_version": self.base_model_version,
            "model_version": self.model_version,
            "step_index": int(self.step_index),
            "source_publication_count": len(source_publications),
            "source_ids": [
                publication.ownership.source_id
                for publication in self.source_publications
            ],
            "total_tensor_bytes": sum(
                publication.tensor.numel() * publication.tensor.element_size()
                for publication in self.source_publications
            ),
            "trainer_loop_provenance": dict(self.provenance),
            "source_publications": source_publications,
        }


def publish_trainer_step_source(
    owner: SliceOwnership,
    *,
    dtype: torch.dtype,
    device: torch.device,
    step_count: int = DEFAULT_TRAINER_STEP_COUNT,
    learning_rate: float = DEFAULT_TRAINER_LR,
) -> TrainerStepSourcePublication:
    """Publish a source-owned post-step shard and its ownership metadata."""

    published_owner = annotate_trainer_step_ownership(
        owner,
        step_count=step_count,
        learning_rate=learning_rate,
    )
    tensor = materialize_trainer_step_source_tensor(
        published_owner,
        dtype=dtype,
        device=device,
        step_count=step_count,
        learning_rate=learning_rate,
    )
    return TrainerStepSourcePublication(
        ownership=published_owner,
        tensor=tensor,
        provenance=trainer_step_source_provenance(
            step_count=step_count,
            learning_rate=learning_rate,
        ),
    )


def trainer_loop_model_version(base_model_version: str, step_index: int) -> str:
    """Return the model version name for one trainer-loop publication step."""

    if not base_model_version:
        raise ValueError("base_model_version is required")
    if int(step_index) <= 0:
        raise ValueError("step_index must be positive")
    return f"{base_model_version}-trainer-loop-step-{int(step_index):06d}"


def publish_trainer_loop_step(
    ownerships: Sequence[SliceOwnership],
    *,
    dtype: torch.dtype,
    device: torch.device,
    step_index: int,
    learning_rate: float = DEFAULT_TRAINER_LR,
    model_version: str | None = None,
) -> TrainerLoopStepPublication:
    """Publish all source-owned shards for one versioned trainer-loop step.

    This is still a deterministic trainer-loop smoke over a tiny synthetic
    objective. The important boundary change is that one trainer loop step
    publishes a coherent model version across all source ranks, with
    step-specific leases/descriptors that downstream MX planning can verify.
    """

    owners = tuple(ownerships)
    if not owners:
        raise ValueError("at least one source ownership is required")
    if int(step_index) <= 0:
        raise ValueError("step_index must be positive")

    model_names = {owner.model_name for owner in owners}
    if len(model_names) != 1:
        raise ValueError(f"trainer loop ownerships span model names: {model_names}")
    base_versions = {owner.model_version for owner in owners}
    if len(base_versions) != 1:
        raise ValueError(
            f"trainer loop ownerships span base model versions: {base_versions}"
        )

    model_name = owners[0].model_name
    base_model_version = owners[0].model_version
    step_model_version = model_version or trainer_loop_model_version(
        base_model_version, step_index
    )
    publications = tuple(
        _annotate_trainer_loop_publication(
            publish_trainer_step_source(
                replace(
                    owner,
                    model_version=step_model_version,
                    source_lease="",
                    nixl_descriptor_id="",
                ),
                dtype=dtype,
                device=device,
                step_count=step_index,
                learning_rate=learning_rate,
            ),
            base_model_version=base_model_version,
            step_index=step_index,
            learning_rate=learning_rate,
        )
        for owner in owners
    )
    return TrainerLoopStepPublication(
        model_name=model_name,
        base_model_version=base_model_version,
        model_version=step_model_version,
        step_index=int(step_index),
        source_publications=publications,
        provenance=trainer_loop_source_provenance(
            base_model_version=base_model_version,
            model_version=step_model_version,
            step_index=step_index,
            learning_rate=learning_rate,
        ),
    )


def annotate_trainer_step_ownership(
    owner: SliceOwnership,
    *,
    step_count: int = DEFAULT_TRAINER_STEP_COUNT,
    learning_rate: float = DEFAULT_TRAINER_LR,
) -> SliceOwnership:
    """Attach optimizer-step publisher metadata to a source ownership record."""

    source_id = owner.source_id or owner.worker_id
    source_lease = owner.source_lease or (
        f"{source_id}-{owner.model_version}-optimizer-step-{int(step_count)}"
    )
    nixl_descriptor_id = owner.nixl_descriptor_id or source_lease
    layout_tags = dict(owner.layout_tags)
    layout_tags.update(
        {
            "trainer_update_source": "torch.optim.SGD-step-publisher",
            "source_payload_generator": "torch.optim.SGD",
            "optimizer": "SGD",
            "optimizer_step_count": int(step_count),
            "learning_rate": str(float(learning_rate)),
            "optimizer_step_publisher": True,
            "synthetic_training_objective": True,
            "static_replacement_formula": False,
        }
    )
    return replace(
        owner,
        source_lease=source_lease,
        nixl_descriptor_id=nixl_descriptor_id,
        layout_tags=layout_tags,
    )


def trainer_loop_source_provenance(
    *,
    base_model_version: str,
    model_version: str,
    step_index: int,
    learning_rate: float = DEFAULT_TRAINER_LR,
) -> dict[str, Any]:
    """Return artifact metadata for a versioned trainer-loop publication step."""

    provenance = trainer_step_source_provenance(
        step_count=step_index,
        learning_rate=learning_rate,
    )
    provenance.update(
        {
            "trainer_loop_publisher_used": True,
            "trainer_loop_step_index": int(step_index),
            "trainer_loop_base_model_version": base_model_version,
            "trainer_loop_model_version": model_version,
            "real_distributed_trainer_loop_used": False,
            "synthetic_trainer_loop_smoke_used": True,
        }
    )
    return provenance


def trainer_step_source_provenance(
    *,
    step_count: int = DEFAULT_TRAINER_STEP_COUNT,
    learning_rate: float = DEFAULT_TRAINER_LR,
) -> dict[str, Any]:
    """Return artifact metadata for the synthetic optimizer-step publisher."""

    return {
        "source_payload_generator": "torch.optim.SGD",
        "optimizer": "SGD",
        "optimizer_step_count": int(step_count),
        "learning_rate": float(learning_rate),
        "trainer_owned_parameter_tensor_used": True,
        "optimizer_step_publisher_used": True,
        "synthetic_training_objective_used": True,
        "static_replacement_formula_used": False,
        "real_rl_training_loop_used": False,
    }


def trainer_step_replacement_tensor(
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    device: torch.device | str,
    step_count: int = DEFAULT_TRAINER_STEP_COUNT,
    learning_rate: float = DEFAULT_TRAINER_LR,
) -> torch.Tensor:
    """Materialize the full expected post-step tensor for receiver validation."""

    shape = _normalize_shape(shape)
    tensor_range = tuple((0, dim) for dim in shape)
    return trainer_step_tensor_for_range(
        shape,
        tensor_range,
        dtype=dtype,
        device=device,
        step_count=step_count,
        learning_rate=learning_rate,
    )


def materialize_trainer_step_source_tensor(
    owner: SliceOwnership,
    *,
    dtype: torch.dtype,
    device: torch.device,
    step_count: int = DEFAULT_TRAINER_STEP_COUNT,
    learning_rate: float = DEFAULT_TRAINER_LR,
) -> torch.Tensor:
    """Materialize only the source-owned post-step shard for one owner."""

    return trainer_step_tensor_for_range(
        owner.global_shape,
        owner.source_range,
        dtype=dtype,
        device=device,
        step_count=step_count,
        learning_rate=learning_rate,
    )


def trainer_step_tensor_for_range(
    global_shape: tuple[int, ...],
    tensor_range: TensorRange,
    *,
    dtype: torch.dtype,
    device: torch.device | str,
    step_count: int = DEFAULT_TRAINER_STEP_COUNT,
    learning_rate: float = DEFAULT_TRAINER_LR,
) -> torch.Tensor:
    """Run a deterministic optimizer step for one global tensor range."""

    global_shape = _normalize_shape(global_shape)
    tensor_range = _normalize_range(tensor_range, global_shape)
    extents = tuple(end - start for start, end in tensor_range)
    if prod(extents) == 0:
        return torch.empty(extents, dtype=dtype, device=device)

    linear = _linear_indices_for_range(global_shape, tensor_range, device=device)
    initial = _initial_values(linear)
    objective_target = _objective_values(linear)
    param = torch.nn.Parameter(initial.clone())
    if step_count > 0:
        optimizer = torch.optim.SGD([param], lr=float(learning_rate))
        for _ in range(int(step_count)):
            optimizer.zero_grad(set_to_none=True)
            loss = 0.5 * torch.sum((param - objective_target) ** 2)
            loss.backward()
            optimizer.step()
    return param.detach().to(dtype=dtype).contiguous()


def _annotate_trainer_loop_publication(
    publication: TrainerStepSourcePublication,
    *,
    base_model_version: str,
    step_index: int,
    learning_rate: float,
) -> TrainerStepSourcePublication:
    owner = publication.ownership
    layout_tags = dict(owner.layout_tags)
    layout_tags.update(
        {
            "trainer_loop_publisher": True,
            "trainer_loop_step_index": int(step_index),
            "trainer_loop_base_model_version": base_model_version,
            "trainer_loop_model_version": owner.model_version,
            "trainer_loop_update_source": "torch.optim.SGD-trainer-loop-smoke",
            "synthetic_trainer_loop_smoke": True,
        }
    )
    provenance = trainer_loop_source_provenance(
        base_model_version=base_model_version,
        model_version=owner.model_version,
        step_index=step_index,
        learning_rate=learning_rate,
    )
    return replace(
        publication,
        ownership=replace(owner, layout_tags=layout_tags),
        provenance=provenance,
    )


def _normalize_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
    normalized = tuple(int(dim) for dim in shape)
    if not normalized or any(dim <= 0 for dim in normalized):
        raise ValueError(f"global shape must be positive, got {shape!r}")
    return normalized


def _normalize_range(
    tensor_range: TensorRange,
    global_shape: tuple[int, ...],
) -> tuple[tuple[int, int], ...]:
    normalized = tuple((int(start), int(end)) for start, end in tensor_range)
    if len(normalized) != len(global_shape):
        raise ValueError(
            f"range rank {len(normalized)} does not match shape rank {len(global_shape)}"
        )
    for axis, ((start, end), dim) in enumerate(zip(normalized, global_shape)):
        if start < 0 or end < start or end > dim:
            raise ValueError(
                f"invalid range on axis {axis}: {(start, end)!r} for dim {dim}"
            )
    return normalized


def _linear_indices_for_range(
    global_shape: tuple[int, ...],
    tensor_range: tuple[tuple[int, int], ...],
    *,
    device: torch.device | str,
) -> torch.Tensor:
    axes = [
        torch.arange(start, end, device=device, dtype=torch.float32)
        for start, end in tensor_range
    ]
    grids = torch.meshgrid(*axes, indexing="ij")
    strides = []
    for axis in range(len(global_shape)):
        strides.append(
            prod(global_shape[axis + 1 :]) if axis + 1 < len(global_shape) else 1
        )
    linear = torch.zeros_like(grids[0], dtype=torch.float32)
    for grid, stride in zip(grids, strides):
        linear = linear + grid * float(stride)
    return linear


def _initial_values(linear_indices: torch.Tensor) -> torch.Tensor:
    return ((linear_indices % 1021.0) - 510.0) / 32.0


def _objective_values(linear_indices: torch.Tensor) -> torch.Tensor:
    return ((linear_indices % 257.0) - 128.0) / 8.0
