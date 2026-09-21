# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""JAX CUDA buffer descriptions for NIXL source registration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class JaxCudaBuffer:
    """A retained JAX allocation described for raw NIXL registration."""

    owner: Any
    addr: int
    size: int
    device_id: int
    dtype: str

    def registration_tuple(self) -> tuple[int, int, int, str]:
        """Return NIXL's raw registration descriptor."""
        return self.addr, self.size, self.device_id, ""


def _load_jax_array_type() -> type:
    try:
        import jax
    except ImportError as exc:
        raise RuntimeError(
            "JAX CUDA source registration requires JAX to be installed "
            "by the runtime environment"
        ) from exc
    return jax.Array


def is_jax_array(value: Any) -> bool:
    """Return whether a JAX-like accelerator value is a real ``jax.Array``."""
    if not all(
        hasattr(value, attribute)
        for attribute in (
            "addressable_shards",
            "block_until_ready",
            "devices",
            "is_fully_addressable",
            "unsafe_buffer_pointer",
        )
    ):
        return False
    return isinstance(value, _load_jax_array_type())


def _torch_dtype_name(dtype: object, name: str) -> str:
    dtype_name = str(dtype)
    torch_dtype = getattr(torch, dtype_name, None)
    if not isinstance(torch_dtype, torch.dtype):
        raise TypeError(
            f"JAX array '{name}' uses unsupported dtype {dtype_name!r}; "
            "a byte-identical Torch dtype is required for ModelExpress metadata"
        )
    return str(torch_dtype)


def _validate_layout(value: Any, name: str) -> None:
    array_format = getattr(value, "format", None)
    layout = getattr(array_format, "layout", None)
    shape = tuple(getattr(value, "shape", ()))
    if layout is None:
        raise ValueError(
            f"JAX array '{name}' does not expose a verifiable dense layout"
        )
    if tuple(getattr(layout, "major_to_minor", ())) != tuple(range(len(shape))):
        raise ValueError(f"JAX array '{name}' must use dense row-major layout")
    if getattr(layout, "tiling", None):
        raise ValueError(f"JAX array '{name}' must not use a tiled layout")
    if int(getattr(layout, "sub_byte_element_size_in_bits", 0)) != 0:
        raise ValueError(f"JAX array '{name}' must not use sub-byte packing")

    on_device_size = getattr(value, "on_device_size_in_bytes", None)
    if not callable(on_device_size):
        raise ValueError(
            f"JAX array '{name}' does not expose on_device_size_in_bytes()"
        )
    if int(on_device_size()) != int(value.nbytes):
        raise ValueError(f"JAX array '{name}' has padded or non-dense device storage")


def describe_jax_cuda_array(
    name: str,
    value: Any,
    *,
    device_id: int,
) -> JaxCudaBuffer:
    """Validate and describe one read-only, single-device JAX CUDA source."""
    jax_array_type = _load_jax_array_type()
    if not isinstance(value, jax_array_type):
        raise TypeError(
            f"JAX source '{name}' must be a jax.Array, got {type(value).__name__}"
        )

    is_deleted = getattr(value, "is_deleted", None)
    if not callable(is_deleted) or is_deleted():
        raise RuntimeError(f"JAX array '{name}' has been deleted")

    block_until_ready = getattr(value, "block_until_ready", None)
    if not callable(block_until_ready):
        raise TypeError(f"JAX array '{name}' does not expose block_until_ready()")
    try:
        block_until_ready()
    except Exception as exc:
        raise RuntimeError(f"JAX array '{name}' failed to become ready") from exc

    if not bool(getattr(value, "is_fully_addressable", False)):
        raise ValueError(f"JAX array '{name}' must be fully addressable")
    devices = getattr(value, "devices", None)
    if not callable(devices):
        raise TypeError(f"JAX array '{name}' does not expose devices()")
    device_set = set(devices())
    shards = getattr(value, "addressable_shards", None)
    if len(device_set) != 1 or shards is None or len(shards) != 1:
        raise ValueError(f"JAX array '{name}' must have exactly one local CUDA shard")

    device = next(iter(device_set))
    platform = getattr(device, "platform", None)
    platform_version = str(
        getattr(getattr(device, "client", None), "platform_version", "")
    ).lower()
    if platform not in {"cuda", "gpu"} or (
        platform == "gpu" and "cuda" not in platform_version
    ):
        raise ValueError(
            f"JAX array '{name}' must reside on NVIDIA CUDA, got {platform!r}"
        )
    local_device_id = getattr(device, "local_hardware_id", None)
    if local_device_id is None:
        raise ValueError(
            f"JAX array '{name}' does not expose a local CUDA device ordinal"
        )
    if int(local_device_id) != device_id:
        raise ValueError(
            f"JAX array '{name}' is on device {local_device_id}, "
            f"but this NIXL manager owns device {device_id}"
        )

    _validate_layout(value, name)

    size = int(value.nbytes)
    if size < 0:
        raise ValueError(f"JAX array '{name}' has a negative byte size")
    unsafe_buffer_pointer = getattr(value, "unsafe_buffer_pointer", None)
    if not callable(unsafe_buffer_pointer):
        raise TypeError(f"JAX array '{name}' does not expose unsafe_buffer_pointer()")
    addr = int(unsafe_buffer_pointer())
    if size > 0 and addr == 0:
        raise ValueError(f"JAX array '{name}' has a null device pointer")

    return JaxCudaBuffer(
        owner=value,
        addr=addr,
        size=size,
        device_id=device_id,
        dtype=_torch_dtype_name(value.dtype, name),
    )
