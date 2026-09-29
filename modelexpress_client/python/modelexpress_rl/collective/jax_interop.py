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

from collections.abc import Sequence
from typing import Any


def expected_index(
    global_shape: Sequence[int],
    mesh: Any,
    placements: Sequence[Any],
    rank: int,
) -> tuple[slice, ...]:
    """The slice of a parameter that ``rank`` holds under a plan's source side.

    ``mesh`` and ``placements`` are a ``ParamPlan``'s ``src_mesh`` and
    ``src_placements``. Ranks map onto the mesh in row-major order, which is
    the order ``MeshSpec.nested()`` hands the reshard op, so this is the slice
    the collective will read from this rank. Every tensor dim is covered: a
    dim named by a ``Shard`` placement is split across that mesh axis, and any
    other dim is whole.
    """
    shape = tuple(int(extent) for extent in mesh.shape)
    offset = rank - int(mesh.rank_offset)
    size = 1
    for extent in shape:
        size *= extent
    if not 0 <= offset < size:
        raise ValueError(
            f"rank {rank} is not in the source mesh {shape}@{mesh.rank_offset}"
        )
    coord = []
    for extent in reversed(shape):
        coord.append(offset % extent)
        offset //= extent
    coord.reverse()

    index = [slice(0, int(extent)) for extent in global_shape]
    for axis, placement in enumerate(placements):
        dim = getattr(placement, "dim", None)
        if dim is None:
            continue
        piece = int(global_shape[dim]) // shape[axis]
        index[dim] = slice(coord[axis] * piece, (coord[axis] + 1) * piece)
    return tuple(index)


def _bounds(index: Sequence[slice], global_shape: Sequence[int]) -> tuple:
    bounds = []
    for dim, extent in enumerate(global_shape):
        axis = index[dim] if dim < len(index) else slice(None)
        start, stop, _ = axis.indices(int(extent))
        bounds.append((start, stop))
    return tuple(bounds)


def local_shard(
    array: Any,
    *,
    name: str = "array",
    expect_index: Sequence[slice] | None = None,
) -> Any:
    """This process's single-device piece of a possibly-sharded array.

    ``addressable_shards[i].data`` is the JAX counterpart of DTensor's
    ``to_local()``: a ``jax.Array`` living on exactly one device, holding the
    slice named by ``shard.index``. A globally-sharded array cannot be
    addressed as one buffer, which is why the shard is the unit the wire op
    receives. Whichever dim the array is split on, each device's piece is its
    own allocation, so the pointer covers exactly that piece. Which splits a
    refit can carry is the plan's rule rather than this function's: one
    sharded dim per source tensor.

    One addressable shard is required rather than assumed. A process driving
    several local devices has several, and picking the first would silently
    transfer one device's weights under every rank's name.

    ``expect_index`` (see ``expected_index``) turns on the check that matters
    for a refit: that this rank holds the slice the plan says the collective
    will read from it, on every dim, extent and position both. A mesh whose
    device order differs from the plan's rank order, or a ``PartitionSpec``
    that disagrees with the plan's placements, would otherwise have every rank
    issue its agreed op over the wrong piece, and the bytes would land wrong
    rather than erroring. The shard's own ``index`` is read rather than the
    array's ``PartitionSpec``, because the index describes what the transfer
    will actually cover.
    """
    shards = array.addressable_shards
    if len(shards) != 1:
        raise ValueError(
            f"{name}: expected exactly one addressable shard on this process, "
            f"got {len(shards)}. A refit rank owns one device; drive one client "
            "per local device rather than choosing a shard here."
        )
    shard = shards[0]
    if expect_index is None:
        return shard.data

    global_shape = tuple(int(extent) for extent in array.shape)
    held = _bounds(shard.index, global_shape)
    wanted = _bounds(expect_index, global_shape)
    if held != wanted:
        raise ValueError(
            f"{name}: this rank holds {held} of {global_shape}, the plan expects "
            f"{wanted} (start, stop per dim). The array's sharding or its mesh's "
            "device order disagrees with the plan's source side"
        )
    return shard.data


def plan_sharding(devices: Sequence[Any]) -> Any:
    """Build ``NamedSharding`` objects that agree with a plan's source side.

    Returns a function taking a ``ParamPlan`` entry. The JAX mesh is built
    with ``devices`` in process order, reshaped row-major to the entry's
    ``src_mesh``, which is the rank order ``MeshSpec.nested()`` hands the
    collective; each ``Shard(d)`` placement names the mesh axis that splits
    tensor dim ``d``. ``jax.make_mesh`` is deliberately not used: it is free to
    reorder devices for the interconnect, and a mesh whose order differs from
    the plan's puts every rank's bytes under another rank's name.
    """
    import numpy as np  # noqa: PLC0415
    import jax  # noqa: PLC0415
    from jax.sharding import PartitionSpec  # noqa: PLC0415

    ordered = np.array(
        sorted(devices, key=lambda device: (device.process_index, device.id)),
        dtype=object,
    )
    meshes: dict[tuple[int, ...], Any] = {}

    def sharding_for(entry: Any) -> Any:
        shape = tuple(int(extent) for extent in entry.src_mesh.shape)
        mesh = meshes.get(shape)
        if mesh is None:
            names = tuple(f"a{axis}" for axis in range(len(shape)))
            mesh = jax.sharding.Mesh(ordered.reshape(shape), names)
            meshes[shape] = mesh
        spec: list[Any] = [None] * len(entry.global_shape)
        for axis, placement in enumerate(entry.src_placements):
            dim = getattr(placement, "dim", None)
            if dim is not None:
                spec[dim] = mesh.axis_names[axis]
        return jax.sharding.NamedSharding(mesh, PartitionSpec(*spec))

    return sharding_for


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


class _ByteView:
    """A one-byte CUDA Array Interface view over a JAX buffer.

    Handing the ``jax.Array`` itself to nccl4py does not work: ``cuda.core``
    prefers DLPack when an object offers both, and calls ``__dlpack__`` with
    the ``-1`` stream sentinel nccl4py uses to mean "do not synchronize". That
    value is legal in the DLPack protocol for ROCm and not for CUDA, so JAX
    passes it to the driver and the barrier dies with
    ``CUDA_ERROR_INVALID_HANDLE``. Exposing only the array interface takes the
    other branch, which carries no stream handshake at all.

    ``uint8`` has an array-interface typestr, so nothing here runs into the
    bfloat16 restriction that shapes the rest of this module.
    """

    __slots__ = ("_array", "__cuda_array_interface__")

    def __init__(self, array: Any) -> None:
        self._array = array
        self.__cuda_array_interface__ = {
            "data": (int(array.unsafe_buffer_pointer()), False),
            "shape": (int(array.size),),
            "typestr": "|u1",
            "version": 3,
            "strides": None,
        }


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
        # The client carries whatever the communicator wants, and nccl4py wants
        # an ordinal. ``device_put`` refuses an int, so the ordinal is resolved
        # against this process's own devices rather than passed through.
        target = jax.local_devices()[device] if isinstance(device, int) else device
        zero = jax.device_put(zero, target)
    return _ByteView(jax.block_until_ready(zero))


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
