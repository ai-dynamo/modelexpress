# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""JAX CUDA source-registration coverage for the low-level NIXL manager."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from modelexpress import _accelerator_buffer
from modelexpress._accelerator_buffer import describe_jax_cuda_array, is_jax_array
from modelexpress.nixl_transfer import NixlTransferManager
from modelexpress.types import TensorDescriptor


@dataclass(frozen=True)
class _FakeJaxDevice:
    platform: str = "gpu"
    local_hardware_id: int = 0
    client: object = field(
        default_factory=lambda: SimpleNamespace(platform_version="CUDA 13.0"),
        compare=False,
        hash=False,
    )


@dataclass(frozen=True)
class _FakeLayout:
    major_to_minor: tuple[int, ...] = (0,)
    tiling: tuple[tuple[int, ...], ...] | None = ()
    sub_byte_element_size_in_bits: int = 0


class _FakeJaxArray:
    def __init__(
        self,
        *,
        ptr: int = 0xCAFE,
        nbytes: int = 16,
        dtype: str = "float32",
        device_id: int = 0,
        platform: str = "gpu",
        platform_version: str = "CUDA 13.0",
        fully_addressable: bool = True,
        device_count: int = 1,
        shard_count: int = 1,
        deleted: bool = False,
        shape: tuple[int, ...] = (4,),
        layout: _FakeLayout | None = None,
        on_device_size: int | None = None,
        ready_error: Exception | None = None,
    ) -> None:
        self._ptr = ptr
        self.nbytes = nbytes
        self.dtype = dtype
        self.shape = shape
        self.format = SimpleNamespace(layout=layout or _FakeLayout())
        self.is_fully_addressable = fully_addressable
        self.addressable_shards = [object() for _ in range(shard_count)]
        self._deleted = deleted
        self._ready_error = ready_error
        self._on_device_size = nbytes if on_device_size is None else on_device_size
        self.ready_calls = 0
        self.pointer_observed_after_ready = False
        self._devices = {
            _FakeJaxDevice(
                platform=platform,
                local_hardware_id=index if device_count > 1 else device_id,
                client=SimpleNamespace(platform_version=platform_version),
            )
            for index in range(device_count)
        }

    def is_deleted(self) -> bool:
        return self._deleted

    def block_until_ready(self):
        self.ready_calls += 1
        if self._ready_error is not None:
            raise self._ready_error
        return self

    def devices(self):
        return self._devices

    def on_device_size_in_bytes(self) -> int:
        return self._on_device_size

    def unsafe_buffer_pointer(self) -> int:
        self.pointer_observed_after_ready = self.ready_calls > 0
        return self._ptr


@pytest.fixture(autouse=True)
def fake_jax_type(monkeypatch):
    monkeypatch.setattr(
        _accelerator_buffer,
        "_load_jax_array_type",
        lambda: _FakeJaxArray,
    )


def _manager() -> NixlTransferManager:
    backend = MagicMock()
    backend.name = "cuda"
    backend.nixl_mem_type = "VRAM"
    manager = NixlTransferManager(
        agent_name="test",
        device_id=0,
        accelerator_backend=backend,
    )
    manager._agent = MagicMock()
    manager._agent.get_agent_metadata.return_value = b"metadata"
    return manager


def test_describe_jax_cuda_source_blocks_before_pointer_and_normalizes_dtype():
    array = _FakeJaxArray()

    buffer = describe_jax_cuda_array("weight", array, device_id=0)

    assert buffer.owner is array
    assert buffer.addr == 0xCAFE
    assert buffer.size == 16
    assert buffer.device_id == 0
    assert buffer.dtype == "torch.float32"
    assert array.ready_calls == 1
    assert array.pointer_observed_after_ready is True


def test_missing_jax_fails_before_nixl_side_effects(monkeypatch):
    manager = _manager()

    def missing_jax():
        raise RuntimeError("JAX CUDA source registration requires JAX")

    monkeypatch.setattr(_accelerator_buffer, "_load_jax_array_type", missing_jax)

    with pytest.raises(RuntimeError, match="requires JAX"):
        manager.register_jax_arrays({"weight": _FakeJaxArray()})

    manager._agent.register_memory.assert_not_called()


def test_torch_value_does_not_load_jax(monkeypatch):
    def unexpected_jax_import():
        raise AssertionError("ordinary Torch values must not load JAX")

    monkeypatch.setattr(
        _accelerator_buffer,
        "_load_jax_array_type",
        unexpected_jax_import,
    )

    assert is_jax_array(torch.empty(0)) is False


def test_register_jax_arrays_uses_raw_vram_descriptors_and_retains_owners():
    first = _FakeJaxArray(ptr=0x1000, nbytes=16)
    empty = _FakeJaxArray(ptr=0, nbytes=0, shape=(0,))
    caller = {"first": first, "empty": empty}
    manager = _manager()

    assert manager.register_jax_arrays(caller) == b"metadata"
    caller.clear()

    manager._agent.register_memory.assert_called_once_with(
        [(0x1000, 16, 0, "")],
        mem_type="VRAM",
        backends=["UCX"],
    )
    assert manager._jax_arrays == {"first": first, "empty": empty}
    assert manager.tensor_descriptors == [
        TensorDescriptor("first", 0x1000, 16, 0, "torch.float32"),
        TensorDescriptor("empty", 0, 0, 0, "torch.float32"),
    ]


def test_register_jax_arrays_validates_entire_catalog_before_side_effects():
    manager = _manager()

    with pytest.raises(ValueError, match="device 1"):
        manager.register_jax_arrays(
            {
                "valid": _FakeJaxArray(ptr=0x1000),
                "wrong_device": _FakeJaxArray(ptr=0x2000, device_id=1),
            }
        )

    manager._agent.register_memory.assert_not_called()
    assert manager._jax_arrays == {}
    assert manager.tensor_descriptors == []


def test_metadata_failure_retains_registration_for_shutdown_cleanup():
    array = _FakeJaxArray()
    manager = _manager()
    registered = object()
    manager._agent.register_memory.return_value = registered
    manager._agent.get_agent_metadata.side_effect = RuntimeError("metadata failed")

    with pytest.raises(RuntimeError, match="metadata failed"):
        manager.register_jax_arrays({"weight": array})

    assert manager._registered_memory == [registered]
    assert manager._jax_arrays == {"weight": array}
    assert manager.nixl_metadata == b""


def test_repeated_registration_retains_every_registered_catalog():
    first = _FakeJaxArray(ptr=0x1000)
    second = _FakeJaxArray(ptr=0x2000)
    manager = _manager()

    manager.register_jax_arrays({"first": first})
    manager.register_jax_arrays({"second": second})

    assert manager._jax_arrays == {"second": second}
    assert manager._jax_array_registrations == [
        {"first": first},
        {"second": second},
    ]


def test_jax_registration_supersedes_implicit_torch_receive_catalog():
    torch_tensor = torch.empty(4, dtype=torch.float32)
    manager = _manager()
    manager.register_tensors({"weight": torch_tensor})
    manager.register_jax_arrays({"weight": _FakeJaxArray()})

    with pytest.raises(TypeError, match="active registration.*read-only JAX"):
        manager.receive_from_source(b"source-metadata", [])

    assert manager._tensors == {}
    assert manager._torch_tensor_registrations == [{"weight": torch_tensor}]
    manager._accelerator_backend.set_device.assert_not_called()
    manager._agent.add_remote_agent.assert_not_called()


def test_receive_rejects_jax_destination_before_transfer_side_effects():
    manager = _manager()

    with pytest.raises(TypeError, match="jax.Array.*immutable"):
        manager.receive_from_source(
            b"source-metadata",
            [],
            destination_tensors={"weight": _FakeJaxArray()},
        )

    manager._accelerator_backend.set_device.assert_not_called()
    manager._agent.add_remote_agent.assert_not_called()
    manager._agent.prep_xfer_dlist.assert_not_called()


def test_shutdown_deregisters_jax_memory_before_releasing_owners():
    array = _FakeJaxArray()
    manager = _manager()
    registered = object()
    manager._agent.register_memory.return_value = registered
    manager.register_jax_arrays({"weight": array})
    observed_owners = []

    def deregister_memory(handle):
        assert handle is registered
        observed_owners.append(manager._jax_arrays.copy())

    manager._agent.deregister_memory.side_effect = deregister_memory

    manager.shutdown()

    assert observed_owners == [{"weight": array}]
    assert manager._jax_arrays == {}
    assert manager._jax_array_registrations == []
    assert manager._active_registration_kind is None


@pytest.mark.parametrize(
    ("array", "match"),
    [
        (_FakeJaxArray(platform="cpu", platform_version=""), "NVIDIA CUDA"),
        (
            _FakeJaxArray(platform="gpu", platform_version="ROCm 7"),
            "NVIDIA CUDA",
        ),
        (_FakeJaxArray(fully_addressable=False), "fully addressable"),
        (_FakeJaxArray(device_count=2, shard_count=2), "exactly one local"),
        (_FakeJaxArray(shard_count=2), "exactly one local"),
        (_FakeJaxArray(device_id=1), "device 1"),
        (_FakeJaxArray(ptr=0, nbytes=16), "null"),
        (
            _FakeJaxArray(layout=_FakeLayout(major_to_minor=())),
            "row-major",
        ),
        (
            _FakeJaxArray(layout=_FakeLayout(tiling=((2,),))),
            "tiled",
        ),
        (
            _FakeJaxArray(
                layout=_FakeLayout(sub_byte_element_size_in_bits=4),
            ),
            "sub-byte",
        ),
        (_FakeJaxArray(on_device_size=32), "padded"),
        (_FakeJaxArray(dtype="custom_dtype"), "unsupported dtype"),
        (_FakeJaxArray(deleted=True), "deleted"),
    ],
)
def test_invalid_jax_sources_fail_before_nixl_registration(array, match):
    manager = _manager()

    with pytest.raises((TypeError, ValueError, RuntimeError), match=match):
        manager.register_jax_arrays({"weight": array})

    manager._agent.register_memory.assert_not_called()


def test_wrong_object_and_readiness_failure_include_tensor_name():
    manager = _manager()

    with pytest.raises(TypeError, match="wrong.*object"):
        manager.register_jax_arrays({"wrong": object()})
    with pytest.raises(RuntimeError, match="weight.*ready"):
        manager.register_jax_arrays(
            {"weight": _FakeJaxArray(ready_error=RuntimeError("device failure"))}
        )

    manager._agent.register_memory.assert_not_called()
