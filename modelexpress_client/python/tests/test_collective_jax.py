# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The torch-free half of the collective refit path.

None of this needs jax installed. What is checked here is that the shared path
stops requiring torch, and that the JAX storage adapter refuses the shapes it
cannot address rather than transferring the wrong bytes under the right name.
The device pointer itself is not reachable without a GPU, so the reading that
JAX and torch produce byte-identical storage lives in the GPU bench rather
than here.
"""

from __future__ import annotations

import sys
import types as pytypes

import pytest

from modelexpress_rl.collective import jax_interop
from modelexpress_rl.collective import client as collective_client
from modelexpress_rl.collective.types import Placement


def _fake_nccl_m2n(monkeypatch):
    """A stand-in for the library's own placement classes.

    ``_placement_to_int`` duck-types on ``__class__.__name__``, so the names
    are the contract and these are faithful stand-ins for it.
    """

    class Replicate:
        def __eq__(self, other):
            return type(other).__name__ == "Replicate"

    class Shard:
        def __init__(self, dim):
            self.dim = dim

    module = pytypes.ModuleType("nccl.m2n")
    module.Replicate = Replicate
    module.Shard = Shard
    package = pytypes.ModuleType("nccl")
    package.m2n = module
    monkeypatch.setitem(sys.modules, "nccl", package)
    monkeypatch.setitem(sys.modules, "nccl.m2n", module)
    return module


class TestPlacementConversion:
    def test_torch_placements_are_used_when_torch_is_installed(self):
        """The call stays identical to NeMo RL's wherever torch exists."""
        torch_tensor = pytest.importorskip("torch.distributed.tensor")

        assert isinstance(Placement.replicate().to_wire(), torch_tensor.Replicate)
        shard = Placement.shard(1).to_wire()
        assert isinstance(shard, torch_tensor.Shard)
        assert shard.dim == 1

    def test_the_library_s_own_placements_are_used_without_torch(self, monkeypatch):
        """A JAX trainer has no torch, and the op does not need one."""
        module = _fake_nccl_m2n(monkeypatch)
        monkeypatch.setitem(sys.modules, "torch.distributed.tensor", None)

        assert isinstance(Placement.replicate().to_wire(), module.Replicate)
        shard = Placement.shard(2).to_wire()
        assert isinstance(shard, module.Shard)
        assert shard.dim == 2

    def test_a_replicate_and_a_shard_stay_distinguishable_without_torch(
        self, monkeypatch
    ):
        """The fallback must not collapse the two kinds onto one object."""
        _fake_nccl_m2n(monkeypatch)
        monkeypatch.setitem(sys.modules, "torch.distributed.tensor", None)

        replicate = Placement.replicate().to_wire()
        shard = Placement.shard(0).to_wire()
        assert type(replicate).__name__ == "Replicate"
        assert type(shard).__name__ == "Shard"


class _Lane:
    def __init__(self):
        self.stream = None
        self.broadcast_calls = []
        self.sync_calls = []
        self.handle = self

    def broadcast(self, *, sendbuf, recvbuf, root, stream):
        self.broadcast_calls.append((sendbuf, recvbuf, root, stream))

    def synchronize(self, timeout_s=None):
        self.sync_calls.append(timeout_s)


class _Err:
    cudaSuccess = "success"
    cudaErrorNotReady = "not-ready"


class _FakeRuntime:
    """cuda.bindings.runtime as far as the torch-free lane path touches it.

    Every call returns a tuple with the status first, which is the binding's
    own convention and the thing ``_cudart_check`` unpacks.
    """

    cudaError_t = _Err
    cudaEventDisableTiming = 2

    def __init__(self, *, ready_after=0, current=3):
        self.calls = []
        self._ready_after = ready_after
        self._current = current

    def cudaGetDevice(self):
        self.calls.append(("get_device",))
        return (_Err.cudaSuccess, self._current)

    def cudaSetDevice(self, ordinal):
        self.calls.append(("set_device", ordinal))
        self._current = ordinal
        return (_Err.cudaSuccess,)

    def cudaEventCreateWithFlags(self, flags):
        self.calls.append(("event_create", flags))
        return (_Err.cudaSuccess, "event")

    def cudaEventRecord(self, event, stream):
        self.calls.append(("event_record", event, stream))
        return (_Err.cudaSuccess,)

    def cudaEventQuery(self, event):
        self.calls.append(("event_query", event))
        if self._ready_after > 0:
            self._ready_after -= 1
            return (_Err.cudaErrorNotReady,)
        return (_Err.cudaSuccess,)

    def cudaEventDestroy(self, event):
        self.calls.append(("event_destroy", event))
        return (_Err.cudaSuccess,)

    def cudaStreamSynchronize(self, stream):
        self.calls.append(("stream_sync", stream))
        return (_Err.cudaSuccess,)


class _BroadcastComm:
    def __init__(self):
        self.broadcast_calls = []

    def broadcast(self, *, sendbuf, recvbuf, root, stream):
        self.broadcast_calls.append((sendbuf, recvbuf, root, stream))


@pytest.fixture
def no_torch(monkeypatch):
    """Block torch the way a JAX-only worker lacks it, and fake the CUDA runtime."""
    from modelexpress_rl.collective import comm

    runtime = _FakeRuntime()
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setattr(comm, "_cudart", lambda: runtime)
    return runtime


def _real_lane(device=0, stream=None):
    from modelexpress_rl.collective.comm import LaneCommunicator

    return LaneCommunicator(
        _BroadcastComm(), rank=0, world_size=2, stream=stream, device=device
    )


class TestTorchFreeLane:
    """The real ``LaneCommunicator`` with torch absent, not a recording double."""

    def test_the_barrier_completes_without_torch(self, no_torch):
        """Broadcast, then a bounded wait through the CUDA runtime."""
        lane = _real_lane(device=0)
        seen = []

        def alloc(device):
            seen.append(device)
            return "jax::barrier"

        collective_client._bootstrap_barrier(lane, 0, timeout_s=5.0, alloc=alloc)

        assert seen == [0]
        assert lane.handle.broadcast_calls == [("jax::barrier", "jax::barrier", 0, None)]
        assert ("event_record", "event", 0) in no_torch.calls
        assert no_torch.calls[-1] == ("set_device", 3), "the previous device is restored"
        assert ("event_destroy", "event") in no_torch.calls

    def test_the_bounded_wait_polls_until_the_event_lands(self, no_torch):
        no_torch._ready_after = 3
        _real_lane(device=None).synchronize(timeout_s=5.0)
        queries = [c for c in no_torch.calls if c[0] == "event_query"]
        assert len(queries) == 4

    def test_the_bounded_wait_still_times_out(self, no_torch):
        no_torch._ready_after = 10**9
        with pytest.raises(TimeoutError, match="did not finish"):
            _real_lane(device=None).synchronize(timeout_s=0.05)
        assert ("event_destroy", "event") in no_torch.calls

    def test_an_unbounded_wait_synchronizes_the_lane_stream(self, no_torch):
        class _Stream:
            cuda_stream = 0xABC

        _real_lane(device=1, stream=_Stream()).synchronize()
        assert ("stream_sync", 0xABC) in no_torch.calls
        assert ("set_device", 1) in no_torch.calls

    def test_the_device_is_restored_when_the_block_raises(self, no_torch):
        from modelexpress_rl.collective.comm import cuda_device

        with pytest.raises(ValueError):
            with cuda_device("cuda:2"):
                raise ValueError("boom")
        assert no_torch.calls == [
            ("get_device",),
            ("set_device", 2),
            ("set_device", 3),
        ]


class TestBootstrapBarrierAllocator:
    def test_the_timeout_still_reaches_synchronize_through_the_allocator_path(self):
        lane = _Lane()
        collective_client._bootstrap_barrier(
            lane, None, timeout_s=1.5, alloc=lambda device: "buf"
        )
        assert lane.sync_calls == [1.5]

    def test_no_allocator_still_takes_the_torch_path(self, monkeypatch):
        """The default must not change for anyone already on this path."""
        allocated = []

        class _FakeCuda:
            @staticmethod
            def device(dev):
                from contextlib import nullcontext

                return nullcontext()

        fake_torch = pytypes.ModuleType("torch")
        fake_torch.uint8 = "uint8"
        fake_torch.cuda = _FakeCuda

        def zeros(count, *, dtype, device):
            allocated.append((count, dtype, device))
            return "torch::barrier"

        fake_torch.zeros = zeros
        monkeypatch.setitem(sys.modules, "torch", fake_torch)

        lane = _Lane()
        collective_client._bootstrap_barrier(lane, "cuda:1")

        assert allocated == [(1, "uint8", "cuda:1")]
        assert lane.broadcast_calls == [("torch::barrier", "torch::barrier", 0, None)]


class _Shard:
    def __init__(self, data, index=None):
        self.data = data
        self.index = index


class _Sharding:
    def __init__(self, devices):
        self.device_set = set(devices)


class _Layout:
    def __init__(self, major_to_minor, tiling):
        self.major_to_minor = major_to_minor
        self.tiling = tiling

    def __repr__(self):
        return f"Layout(major_to_minor={self.major_to_minor}, tiling={self.tiling})"


class _Array:
    """Enough of a ``jax.Array`` to exercise the adapter's refusals."""

    def __init__(
        self,
        *,
        shards,
        devices=("d0",),
        shape=(4, 8),
        dtype="bfloat16",
        ptr=0x1000,
        indices=None,
        layout=None,
    ):
        self.addressable_shards = [
            _Shard(s, None if indices is None else indices[i])
            for i, s in enumerate(shards)
        ]
        self.sharding = _Sharding(devices)
        self.shape = shape
        self.dtype = dtype
        self.size = 1
        for extent in shape:
            self.size *= extent
        self._ptr = ptr
        if layout is not None:
            self.format = pytypes.SimpleNamespace(layout=layout, sharding=self.sharding)

    def unsafe_buffer_pointer(self):
        return self._ptr


class TestLocalShard:
    def test_the_single_addressable_shard_is_returned(self):
        array = _Array(shards=["only"])
        assert jax_interop.local_shard(array) is array.addressable_shards[0].data

    def test_several_addressable_shards_are_refused(self):
        """Picking the first would ship one device's weights under every rank."""
        array = _Array(shards=["a", "b"])
        with pytest.raises(ValueError, match="exactly one addressable shard"):
            jax_interop.local_shard(array)

    def test_no_addressable_shard_is_refused(self):
        with pytest.raises(ValueError, match="exactly one addressable shard"):
            jax_interop.local_shard(_Array(shards=[]))


class TestJaxDeviceBuffer:
    def test_it_exposes_the_trio_the_resolver_reads(self):
        array = _Array(shards=["x"], shape=(11008, 4096), dtype="bfloat16", ptr=0xDEAD000)
        buffer = jax_interop.JaxDeviceBuffer(array)

        assert buffer.data_ptr() == 0xDEAD000
        assert buffer.shape == (11008, 4096)
        assert buffer.dtype == "bfloat16"

    def test_it_holds_the_array_so_the_pointer_cannot_dangle(self):
        array = _Array(shards=["x"])
        buffer = jax_interop.JaxDeviceBuffer(array)
        assert any(getattr(buffer, slot, None) is array for slot in buffer.__slots__)

    def test_a_multi_device_array_is_refused(self):
        """A globally-sharded array has no one address to transfer from."""
        array = _Array(shards=["x"], devices=("d0", "d1"))
        with pytest.raises(ValueError, match="must live on one device"):
            jax_interop.JaxDeviceBuffer(array)

    def test_it_does_not_define_is_contiguous(self):
        """The resolver checks contiguity only when the attribute is present.

        The layout is read and refused at construction instead; a constant
        attribute would be an assertion on backends that cannot report one.
        """
        assert not hasattr(jax_interop.JaxDeviceBuffer(_Array(shards=["x"])), "is_contiguous")

    @pytest.mark.parametrize("shape", [(), (7,), (4, 8), (2, 3, 4)])
    def test_the_default_row_major_layout_is_accepted(self, shape):
        """JAX spells row-major as major_to_minor (0, ..., ndim-1), no tiling."""
        layout = _Layout(tuple(range(len(shape))), ())
        array = _Array(shards=["x"], shape=shape, layout=layout, ptr=0x3000)
        assert jax_interop.JaxDeviceBuffer(array).data_ptr() == 0x3000

    def test_a_none_tiling_is_accepted(self):
        array = _Array(shards=["x"], layout=_Layout((0, 1), None))
        jax_interop.JaxDeviceBuffer(array)

    @pytest.mark.parametrize(
        "major_to_minor, shape",
        [((1, 0), (4, 8)), ((0, 2, 1), (2, 3, 4)), ((2, 1, 0), (2, 3, 4))],
    )
    def test_a_non_row_major_layout_is_refused(self, major_to_minor, shape):
        """The wire op would read a transposed buffer as scrambled row-major bytes."""
        array = _Array(shards=["x"], shape=shape, layout=_Layout(major_to_minor, ()))
        with pytest.raises(ValueError, match="dense row-major"):
            jax_interop.JaxDeviceBuffer(array)

    def test_a_tiled_layout_is_refused(self):
        array = _Array(shards=["x"], layout=_Layout((0, 1), ((8, 128),)))
        with pytest.raises(ValueError, match="dense row-major"):
            jax_interop.JaxDeviceBuffer(array)

    def test_no_format_attribute_falls_through(self):
        """jax without Array.format keeps the old behaviour."""
        array = _Array(shards=["x"])
        assert not hasattr(array, "format")
        assert jax_interop.JaxDeviceBuffer(array).shape == (4, 8)

    def test_a_backend_that_cannot_report_a_layout_falls_through(self):
        """Array.format returns Format(None, sharding) on UNIMPLEMENTED."""
        array = _Array(shards=["x"])
        array.format = pytypes.SimpleNamespace(layout=None, sharding=array.sharding)
        jax_interop.JaxDeviceBuffer(array)

    def test_a_format_without_layout_falls_through(self):
        """jax 0.6.2 has Array.format, but its field is device_local_layout."""
        array = _Array(shards=["x"])
        array.format = pytypes.SimpleNamespace(
            device_local_layout=_Layout((1, 0), ()), sharding=array.sharding
        )
        jax_interop.JaxDeviceBuffer(array)


class _Piece:
    """A shard payload that only has to report its shape."""

    def __init__(self, shape):
        self.shape = shape


def _mesh(shape, rank_offset=0):
    from modelexpress_rl.collective import MeshSpec

    return MeshSpec(shape=shape, rank_offset=rank_offset)


class TestExpectedIndex:
    """The slice the collective reads from each rank, on every sharded dim."""

    def test_dim0_over_a_flat_mesh(self):
        from modelexpress_rl.collective import Placement

        got = jax_interop.expected_index((8, 6), _mesh((4,)), (Placement.shard(0),), 2)
        assert got == (slice(4, 6), slice(0, 6))

    def test_dim1_over_a_flat_mesh(self):
        from modelexpress_rl.collective import Placement

        got = jax_interop.expected_index((8, 6), _mesh((2,)), (Placement.shard(1),), 1)
        assert got == (slice(0, 8), slice(3, 6))

    def test_two_dims_over_a_2d_mesh_is_row_major(self):
        """Rank 1 on a (2, 2) mesh is coordinate (0, 1), matching nested()."""
        from modelexpress_rl.collective import Placement

        mesh = _mesh((2, 2))
        placements = (Placement.shard(0), Placement.shard(1))
        assert mesh.nested() == [[0, 1], [2, 3]]
        got = [jax_interop.expected_index((8, 6), mesh, placements, r) for r in range(4)]
        assert got == [
            (slice(0, 4), slice(0, 3)),
            (slice(0, 4), slice(3, 6)),
            (slice(4, 8), slice(0, 3)),
            (slice(4, 8), slice(3, 6)),
        ]

    def test_a_replicated_axis_leaves_the_tensor_whole_on_it(self):
        from modelexpress_rl.collective import Placement

        placements = (Placement.replicate(), Placement.shard(1))
        got = jax_interop.expected_index((8, 6), _mesh((2, 2)), placements, 3)
        assert got == (slice(0, 8), slice(3, 6))

    def test_a_rank_offset_is_subtracted(self):
        from modelexpress_rl.collective import Placement

        mesh = _mesh((2,), rank_offset=4)
        assert jax_interop.expected_index((8,), mesh, (Placement.shard(0),), 5) == (
            slice(4, 8),
        )

    def test_a_rank_outside_the_mesh_is_refused(self):
        from modelexpress_rl.collective import Placement

        with pytest.raises(ValueError, match="not in the source mesh"):
            jax_interop.expected_index((8,), _mesh((2,)), (Placement.shard(0),), 2)


class TestLocalShardPlanChecks:
    """``expect_index`` is what stops a wrong-slice op landing silently."""

    @staticmethod
    def _array(piece_shape, index, global_shape=(1024, 8)):
        return _Array(shards=[_Piece(piece_shape)], indices=[index], shape=global_shape)

    def test_a_matching_dim0_shard_passes(self):
        array = self._array((512, 8), (slice(0, 512), slice(None)))
        expect = (slice(0, 512), slice(0, 8))
        assert jax_interop.local_shard(array, name="w", expect_index=expect).shape == (512, 8)

    def test_a_matching_two_axis_shard_passes(self):
        array = self._array((512, 4), (slice(512, 1024), slice(4, 8)))
        expect = (slice(512, 1024), slice(4, 8))
        assert jax_interop.local_shard(array, name="w", expect_index=expect).shape == (512, 4)

    def test_a_short_shard_is_refused(self):
        array = self._array((511, 8), (slice(0, 511), slice(None)))
        with pytest.raises(ValueError, match="the plan expects"):
            jax_interop.local_shard(
                array, name="w", expect_index=(slice(0, 512), slice(0, 8))
            )

    def test_the_right_extent_at_the_wrong_position_is_refused(self):
        """A mesh ordered differently from the plan's ranks: same size, wrong rows."""
        array = self._array((512, 8), (slice(512, 1024), slice(None)))
        with pytest.raises(ValueError, match="device order disagrees"):
            jax_interop.local_shard(
                array, name="w", expect_index=(slice(0, 512), slice(0, 8))
            )

    def test_a_shard_split_on_a_dim_the_plan_keeps_whole_is_refused(self):
        array = self._array((1024, 4), (slice(None), slice(0, 4)))
        with pytest.raises(ValueError, match="the plan expects"):
            jax_interop.local_shard(
                array, name="w", expect_index=(slice(0, 512), slice(0, 8))
            )

    def test_the_parameter_name_reaches_the_message(self):
        array = self._array((1, 8), (slice(0, 1), slice(None)))
        with pytest.raises(ValueError, match="model.layers.0.weight"):
            jax_interop.local_shard(
                array,
                name="model.layers.0.weight",
                expect_index=(slice(0, 512), slice(0, 8)),
            )

    def test_without_expect_index_no_layout_check_runs(self):
        """The plain lookup must stay usable where there is no plan to check."""
        array = self._array((7, 8), (slice(None), slice(0, 4)))
        assert jax_interop.local_shard(array).shape == (7, 8)


class TestBarrierByteView:
    """The barrier hands nccl4py an array interface and no ``__dlpack__``."""

    def test_it_exposes_a_one_byte_interface_at_the_buffer_address(self):
        view = jax_interop._ByteView(_Array(shards=["x"], ptr=0xBEEF000))
        cai = view.__cuda_array_interface__
        assert cai["data"] == (0xBEEF000, False)
        assert cai["typestr"] == "|u1"
        assert cai["strides"] is None
        assert cai["version"] == 3

    def test_it_offers_no_dlpack(self):
        """cuda.core prefers DLPack when both are present, and JAX's rejects
        the -1 stream sentinel nccl4py passes, which killed the barrier."""
        view = jax_interop._ByteView(_Array(shards=["x"]))
        assert not hasattr(view, "__dlpack__")

    def test_it_holds_the_array(self):
        array = _Array(shards=["x"])
        view = jax_interop._ByteView(array)
        assert view._array is array
