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


class TestBootstrapBarrierAllocator:
    def test_a_supplied_allocator_provides_the_byte_and_torch_is_not_imported(
        self, monkeypatch
    ):
        """The whole point: a worker with no torch still barriers."""
        lane = _Lane()
        seen = []

        def alloc(device):
            seen.append(device)
            return "jax::barrier"

        monkeypatch.setitem(sys.modules, "torch", None)
        collective_client._bootstrap_barrier(lane, "dev0", alloc=alloc)

        assert seen == ["dev0"]
        assert lane.broadcast_calls == [("jax::barrier", "jax::barrier", 0, None)]
        assert len(lane.sync_calls) == 1

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
    ):
        self.addressable_shards = [
            _Shard(s, None if indices is None else indices[i])
            for i, s in enumerate(shards)
        ]
        self.sharding = _Sharding(devices)
        self.shape = shape
        self.dtype = dtype
        self._ptr = ptr

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

        Defining it would assert a property of XLA's device layout rather than
        read one, so its absence is deliberate and worth pinning.
        """
        assert not hasattr(jax_interop.JaxDeviceBuffer(_Array(shards=["x"])), "is_contiguous")

    def test_as_reshard_buffer_goes_through_the_shard(self):
        inner = _Array(shards=["inner"], ptr=0x2000)
        outer = _Array(shards=[inner], devices=("d0", "d1"))
        buffer = jax_interop.as_reshard_buffer(outer)
        assert buffer.data_ptr() == 0x2000


class _Rows:
    """A shard payload that only has to report a row count."""

    def __init__(self, rows):
        self.shape = (rows, 8)


class TestLocalShardPlanChecks:
    """``expect_rows`` is what stops a wrong-extent op landing silently."""

    @staticmethod
    def _array(rows, index):
        return _Array(shards=[_Rows(rows)], indices=[index])

    def test_a_matching_dim0_shard_passes(self):
        array = self._array(512, (slice(0, 512), slice(None)))
        assert jax_interop.local_shard(array, name="w", expect_rows=512).shape == (512, 8)

    def test_a_short_shard_is_refused(self):
        array = self._array(511, (slice(0, 511), slice(None)))
        with pytest.raises(ValueError, match="expects 512"):
            jax_interop.local_shard(array, name="w", expect_rows=512)

    def test_a_shard_split_on_another_axis_is_refused(self):
        array = self._array(512, (slice(None), slice(0, 4)))
        with pytest.raises(ValueError, match="split on another axis"):
            jax_interop.local_shard(array, name="w", expect_rows=512)

    def test_the_parameter_name_reaches_the_message(self):
        array = self._array(1, (slice(0, 1), slice(None)))
        with pytest.raises(ValueError, match="model.layers.0.weight"):
            jax_interop.local_shard(
                array, name="model.layers.0.weight", expect_rows=512
            )

    def test_without_expect_rows_no_layout_check_runs(self):
        """The plain lookup must stay usable where there is no plan to check."""
        array = self._array(7, (slice(None), slice(0, 4)))
        assert jax_interop.local_shard(array).shape == (7, 8)
