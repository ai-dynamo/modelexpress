# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The JAX adapter against a real JAX, on four host devices.

``test_collective_jax.py`` runs without jax and checks the shared path against
stand-ins; this file checks the assumptions those stand-ins encode, against the
library itself, so an upstream change to how JAX reports shard indices, orders
a mesh, or lays out a shard's buffer fails here. It needs no GPU: the host
platform is split into four devices, which is enough to exercise one-axis and
two-axis meshes. It is skipped where jax is not installed.
"""

from __future__ import annotations

import ctypes
import os
from types import SimpleNamespace

os.environ["XLA_FLAGS"] = (
    os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=4"
).strip()

import numpy as np  # noqa: E402
import pytest  # noqa: E402

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from modelexpress_rl.collective import MeshSpec, ParamPlan, Placement  # noqa: E402
from modelexpress_rl.collective import jax_interop  # noqa: E402

pytestmark = pytest.mark.skipif(
    len(jax.devices("cpu")) < 4,
    reason="jax was imported before the host device count could be set",
)

LAYOUTS = {
    "dim0": ((4,), (Placement.shard(0),)),
    "dim1": ((4,), (Placement.shard(1),)),
    "2d": ((2, 2), (Placement.shard(0), Placement.shard(1))),
    "2d-transposed": ((2, 2), (Placement.shard(1), Placement.shard(0))),
    "2d-half-replicated": ((2, 2), (Placement.replicate(), Placement.shard(1))),
}


def _entry(shape, mesh_shape, placements, dtype="float32"):
    """A plan entry, or a bare stand-in where the plan itself would refuse it.

    Two sharded dims on one source is a split the collective rejects, so
    ``ParamPlan`` refuses it; the JAX-side mapping is still worth pinning for
    it, because it is the case where a wrong axis order is easiest to write.
    """
    mesh = MeshSpec(shape=mesh_shape)
    if sum(p.dim is not None for p in placements) > 1:
        return SimpleNamespace(
            global_shape=shape, src_mesh=mesh, src_placements=placements
        )
    return ParamPlan(
        name="w",
        global_shape=shape,
        dtype=dtype,
        partition_id=0,
        src_mesh=mesh,
        src_placements=placements,
        dst_mesh=MeshSpec(shape=(1,), rank_offset=mesh.size),
        dst_placements=(Placement.replicate(),),
    )


def _ranked_shards(array, devices):
    """Each addressable shard paired with the plan rank its device holds."""
    order = sorted(devices, key=lambda device: (device.process_index, device.id))
    rank_of = {device: rank for rank, device in enumerate(order)}
    return sorted(
        ((rank_of[shard.device], shard) for shard in array.addressable_shards),
        key=lambda pair: pair[0],
    )


def _bytes_at(pointer, count):
    return ctypes.string_at(pointer, count)


@pytest.mark.parametrize("layout", sorted(LAYOUTS))
def test_the_plan_slice_is_the_slice_jax_places_on_each_rank(layout):
    """``plan_sharding`` and ``expected_index`` agree with JAX's own placement."""
    mesh_shape, placements = LAYOUTS[layout]
    devices = jax.devices("cpu")[:4]
    entry = _entry((8, 12), mesh_shape, placements)
    host = np.arange(96, dtype=np.float32).reshape(8, 12)
    array = jax.device_put(host, jax_interop.plan_sharding(devices)(entry))

    ranked = _ranked_shards(array, devices)
    assert [rank for rank, _ in ranked] == [0, 1, 2, 3]
    for rank, shard in ranked:
        expect = jax_interop.expected_index(
            entry.global_shape, entry.src_mesh, entry.src_placements, rank
        )
        assert jax_interop._bounds(shard.index, entry.global_shape) == (
            jax_interop._bounds(expect, entry.global_shape)
        ), f"rank {rank}"
        np.testing.assert_array_equal(np.asarray(shard.data), host[expect])


@pytest.mark.parametrize("layout", ["dim1", "2d"])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_the_buffer_pointer_covers_exactly_the_rank_piece(layout, dtype):
    """A non-dim-0 piece is its own dense row-major allocation, not a strided view.

    This is the property the wire op depends on once a shard is split on a
    trailing dim: the bytes at ``data_ptr()`` must be the piece in row-major
    order, with nothing of the neighbouring ranks' columns interleaved.
    """
    mesh_shape, placements = LAYOUTS[layout]
    devices = jax.devices("cpu")[:4]
    entry = _entry((8, 12), mesh_shape, placements, dtype=dtype)
    host = np.arange(96, dtype=np.float32).reshape(8, 12).astype(jnp.dtype(dtype))
    array = jax.device_put(host, jax_interop.plan_sharding(devices)(entry))

    for rank, shard in _ranked_shards(array, devices):
        expect = jax_interop.expected_index(
            entry.global_shape, entry.src_mesh, entry.src_placements, rank
        )
        piece = np.ascontiguousarray(host[expect])
        buffer = jax_interop.JaxDeviceBuffer(shard.data)
        assert buffer.shape == piece.shape
        assert _bytes_at(buffer.data_ptr(), piece.nbytes) == piece.tobytes(), (
            f"rank {rank}"
        )


def test_local_shard_refuses_a_process_holding_several_devices():
    devices = jax.devices("cpu")[:4]
    entry = _entry((8, 12), *LAYOUTS["2d"])
    array = jax.device_put(
        np.zeros((8, 12), np.float32), jax_interop.plan_sharding(devices)(entry)
    )
    with pytest.raises(ValueError, match="exactly one addressable shard"):
        jax_interop.local_shard(array)


def test_a_single_device_piece_passes_the_plan_check():
    """The check the publisher runs, on a real one-shard array."""
    device = jax.devices("cpu")[0]
    host = np.arange(24, dtype=np.float32).reshape(2, 12)
    array = jax.device_put(host, device)
    expect = (slice(0, 2), slice(0, 12))
    assert jax_interop.local_shard(array, expect_index=expect) is (
        array.addressable_shards[0].data
    )


def test_the_default_layout_of_a_real_array_is_accepted():
    """What XLA hands back by default must pass the row-major check."""
    for dtype in ("float32", "bfloat16", "float8_e4m3fn", "int8"):
        array = jax.device_put(jnp.zeros((2, 3, 4), dtype), jax.devices("cpu")[0])
        assert array.format.layout is not None, "the layout check would not run"
        jax_interop.JaxDeviceBuffer(array)


def test_a_real_transposed_layout_is_refused():
    """A non-default device layout is caught before a pointer is handed out."""
    from jax.experimental.layout import Format, Layout  # noqa: PLC0415

    device = jax.devices("cpu")[0]
    host = jax.device_put(jnp.arange(24.0).reshape(4, 6), device)
    array = jax.device_put(host, Format(Layout((1, 0)), host.sharding))
    assert array.format.layout.major_to_minor == (1, 0)
    with pytest.raises(ValueError, match="dense row-major"):
        jax_interop.JaxDeviceBuffer(array)
