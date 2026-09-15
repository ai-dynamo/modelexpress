# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""JAX storage adapter for the NCCL M2N collective refit path.

A ``jax.Array`` cannot be handed to ``nccl.m2n.reshard`` directly, and the
reason is narrow enough to be worth stating: ``reshard`` resolves a buffer by
trying ``data_ptr()`` first and ``__cuda_array_interface__`` second, and
``jax.Array`` has no ``data_ptr()`` and *raises* from the interface property
for bfloat16 and float8 buffers, which are the dtypes a refit actually carries.
The raise is swallowed by the ``getattr(..., None)`` the library probes with, so
a bfloat16 array arrives looking like an object with no interface at all.

``unsafe_buffer_pointer()`` carries no dtype restriction, so that is the route
taken here. It returns the same address DLPack hands a consumer, which is what
establishes it is the live buffer rather than a staging copy.

Everything here imports jax lazily, so this module stays importable in a
process that has never installed it.
"""

from __future__ import annotations

from typing import Any


def local_shard(
    array: Any, *, name: str = "array", expect_rows: int | None = None
) -> Any:
    """This process's single-device piece of a possibly-sharded array.

    ``addressable_shards[i].data`` is the JAX counterpart of DTensor's
    ``to_local()``: a ``jax.Array`` living on exactly one device, holding the
    slice named by ``shard.index``. A globally-sharded array cannot be
    addressed as one buffer, which is why the shard is the unit the wire op
    receives.

    One addressable shard is required rather than assumed. A process driving
    several local devices has several, and picking the first would silently
    transfer one device's weights under every rank's name.

    ``expect_rows`` turns on the check that matters for a refit: that this
    rank's piece is the dim-0 slice the plan promised, and the extent the plan
    promised. Every rank would otherwise issue its agreed op over storage of
    the wrong size, and the bytes would land wrong rather than erroring. The
    shard's own ``index`` is read rather than the array's ``PartitionSpec``,
    because the index describes what the transfer will actually cover.
    """
    shards = array.addressable_shards
    if len(shards) != 1:
        raise ValueError(
            f"{name}: expected exactly one addressable shard on this process, "
            f"got {len(shards)}. A refit rank owns one device; drive one client "
            "per local device rather than choosing a shard here."
        )
    shard = shards[0]
    if expect_rows is None:
        return shard.data

    index = shard.index
    if any(axis != slice(None) for axis in index[1:]):
        raise ValueError(
            f"{name}: the plan declares a dim-0 shard but this rank holds "
            f"{index}, which is split on another axis"
        )
    local = shard.data
    if local.shape[0] != expect_rows:
        raise ValueError(
            f"{name}: local shard has {local.shape[0]} rows, the plan expects "
            f"{expect_rows} (an uneven split would land wrong bytes silently)"
        )
    return local


class JaxDeviceBuffer:
    """A single-device ``jax.Array`` in the shape ``reshard`` resolves.

    Exposes the ``data_ptr()`` / ``shape`` / ``dtype`` trio the M2N buffer
    resolver reads, and nothing else. ``dtype`` is passed through untouched:
    the resolver normalizes by ``str()``, and ``str(jax_array.dtype)`` is
    already ``"bfloat16"``, ``"float8_e4m3fn"`` and so on, which are the names
    its table carries.

    The source array is held for the object's whole life. The pointer is raw,
    so letting the array be collected would free the device allocation under a
    transfer already in flight.

    ``is_contiguous`` is deliberately not defined. The resolver checks it only
    when it is present, and defining it would be asserting a property of XLA's
    device layout rather than reading one. It holds for every dtype and shape
    measured -- float32, float16, bfloat16, int8 and float8_e4m3fn over six
    shapes, byte-identical to the canonical row-major image -- and a dense
    row-major layout is what XLA produces for these arrays on GPU.
    """

    __slots__ = ("_array", "_ptr", "shape", "dtype")

    def __init__(self, array: Any) -> None:
        devices = array.sharding.device_set
        if len(devices) != 1:
            raise ValueError(
                f"a reshard buffer must live on one device, got {len(devices)}. "
                "Pass local_shard(array) rather than the global array."
            )
        self._array = array
        self._ptr = int(array.unsafe_buffer_pointer())
        self.shape = tuple(int(dim) for dim in array.shape)
        self.dtype = array.dtype

    def data_ptr(self) -> int:
        return self._ptr

    def __repr__(self) -> str:
        return (
            f"JaxDeviceBuffer(shape={self.shape}, dtype={self.dtype}, "
            f"ptr=0x{self._ptr:x})"
        )


def as_reshard_buffer(array: Any) -> JaxDeviceBuffer:
    """Wrap this process's shard of ``array`` for the wire op."""
    return JaxDeviceBuffer(local_shard(array))


def barrier_buffer(device: Any) -> Any:
    """One device byte for the bootstrap barrier, allocated through JAX.

    The broadcast writes into this buffer and nobody reads the result, so its
    contents are irrelevant; what matters is that all ranks present a byte on
    the right device. It is returned rather than stored so the caller's
    reference is what keeps the allocation alive for the length of the
    collective.
    """
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    zero = jnp.zeros(1, dtype=jnp.uint8)
    if device is not None:
        zero = jax.device_put(zero, device)
    return jax.block_until_ready(zero)


def ready(*arrays: Any) -> None:
    """Block until every array's producing computation has completed.

    JAX enqueues on its own stream and ``reshard`` takes a bare device pointer
    with no stream handshake, so nothing orders the transfer against the
    computation that produced the weights. This is not a precaution: the race
    was measured. ``unsafe_buffer_pointer()`` returns in microseconds without
    synchronizing, and a consumer reading those bytes on another stream while
    the producing computation is still in flight observes intermediate values
    -- in every trial, and even when the read itself took hundreds of
    milliseconds. Skipping this transfers partially-computed weights with
    nothing anywhere reporting an error.

    One host synchronize per round is the cost, and it is negligible beside
    the transfer; per-parameter synchronizing is not, which is why this takes
    a batch.
    """
    import jax  # noqa: PLC0415

    jax.block_until_ready(list(arrays))
