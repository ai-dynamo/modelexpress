# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reshard receiver seam for native Megatron staging and translation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from modelexpress.refit.reshard.receiver import ReshardReceiver

from .layout import (
    MegatronTargetLayout,
    MegatronTargetSpec,
    lower_megatron_target,
)


def _dtype_label(dtype: Any) -> str:
    return str(dtype).removeprefix("torch.")


class MegatronReshardReceiver(ReshardReceiver):
    """Pull destination-owned native tensors, then invoke framework translation.

    The generic reshard receiver owns discovery, exact segment planning, NIXL reads,
    and persistent buffers. This subclass replaces vLLM loader capture with
    native Megatron target geometry; its install callback translates those
    target-local native buffers into the engine's parameter representation.
    """

    def __init__(
        self,
        *,
        target_specs: list[MegatronTargetSpec],
        target_layout: MegatronTargetLayout,
        install_native: Callable[[dict[str, Any]], None],
        **base_kwargs: Any,
    ) -> None:
        if not target_specs:
            raise ValueError("target_specs must not be empty")
        self._target_specs = list(target_specs)
        self._target_layout = target_layout
        self._install_native = install_native
        super().__init__(**base_kwargs)

    def _capture(self, manifest: list) -> tuple:
        published = {
            name: (_dtype_label(dtype), tuple(int(dim) for dim in shape))
            for name, dtype, shape in manifest
        }
        missing = [
            spec.source_name
            for spec in self._target_specs
            if spec.source_name not in published
        ]
        if missing:
            raise RuntimeError(
                f"Megatron target plan references {len(missing)} unpublished "
                f"tensor(s): {missing[:10]}"
            )
        mismatched = []
        for spec in self._target_specs:
            dtype, shape = published[spec.source_name]
            if shape != tuple(spec.global_shape) or dtype != _dtype_label(spec.dtype):
                mismatched.append(
                    (
                        spec.source_name,
                        (shape, dtype),
                        (tuple(spec.global_shape), _dtype_label(spec.dtype)),
                    )
                )
        if mismatched:
            raise RuntimeError(
                "Megatron source manifest disagrees with target geometry: "
                f"{mismatched[:5]}"
            )
        return lower_megatron_target(self._target_specs, self._target_layout)

    def _install(self, recv_buffers: dict) -> None:
        self._install_native(recv_buffers)


__all__ = ["MegatronReshardReceiver"]
