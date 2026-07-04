# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ModelExpress-neutral inputs for GMS snapshot restore.

Environment variables:
    MX_GDS_CHUNK_SIZE_BYTES: Positive, GDS-aligned restore chunk size in bytes
    MX_GDS_MAX_INFLIGHT_BATCHES: Positive maximum concurrent restore batch count
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from .. import envs
from ..accelerators import AcceleratorBackend, CudaAcceleratorBackend
from ..gds_constants import _GDS_ALIGNMENT
from ..gds_loader import MxDeviceReadTarget, MxFileReadSource


def _optional_int(value: str | None) -> int | None:
    value = (value or "").strip()
    return int(value) if value else None


@dataclass(frozen=True)
class GmsRestoreContext:
    """Inputs shared by the ModelExpress GMS restore strategies."""

    sources: Sequence[MxFileReadSource]
    targets: Mapping[str, MxDeviceReadTarget]
    # The caller guarantees consistency with sources/targets and groups pairs by
    # file path with each sequence sorted by file_offset.
    grouped_sources: Mapping[
        str, Sequence[tuple[MxFileReadSource, MxDeviceReadTarget]]
    ]
    device: int
    max_workers: int
    backend_config: Mapping[str, object]
    gds_chunk_size: int | None
    gds_max_inflight: int
    accelerator_backend: AcceleratorBackend = field(
        default_factory=CudaAcceleratorBackend
    )

    @classmethod
    def from_env(
        cls,
        *,
        sources: Sequence[MxFileReadSource],
        targets: Mapping[str, MxDeviceReadTarget],
        grouped_sources: Mapping[
            str, Sequence[tuple[MxFileReadSource, MxDeviceReadTarget]]
        ],
        device: int,
        max_workers: int,
        backend_config: Mapping[str, object],
        gds_chunk_size: int | str | None = None,
        gds_max_inflight: int | str | None = None,
        accelerator_backend: AcceleratorBackend | None = None,
    ) -> GmsRestoreContext:
        """Build a context with optional GDS configuration from the environment.

        Non-None GDS arguments take precedence over environment values
        (MX_GDS_CHUNK_SIZE_BYTES, MX_GDS_MAX_INFLIGHT_BATCHES); empty or unset
        environment values fall back to no chunking and ``max_workers``.
        """
        if gds_chunk_size is None:
            gds_chunk_size = _optional_int(envs.MX_GDS_CHUNK_SIZE_BYTES)
        else:
            gds_chunk_size = int(gds_chunk_size)
        if gds_chunk_size is not None and (
            gds_chunk_size <= 0 or gds_chunk_size % _GDS_ALIGNMENT != 0
        ):
            raise ValueError(
                "gds_chunk_size must be a positive multiple of "
                f"{_GDS_ALIGNMENT}, got {gds_chunk_size}"
            )

        if gds_max_inflight is None:
            gds_max_inflight = _optional_int(envs.MX_GDS_MAX_INFLIGHT_BATCHES)
        gds_max_inflight = (
            max(1, max_workers) if gds_max_inflight is None else int(gds_max_inflight)
        )
        if gds_max_inflight < 1:
            raise ValueError(
                f"gds_max_inflight must be at least 1, got {gds_max_inflight}"
            )
        if accelerator_backend is None:
            accelerator_backend = CudaAcceleratorBackend()
        return cls(
            sources=sources,
            targets=targets,
            grouped_sources=grouped_sources,
            device=device,
            max_workers=max_workers,
            backend_config=backend_config,
            gds_chunk_size=gds_chunk_size,
            gds_max_inflight=gds_max_inflight,
            accelerator_backend=accelerator_backend,
        )
