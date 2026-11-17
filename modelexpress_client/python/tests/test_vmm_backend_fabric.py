# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CUDA-driver boundary tests for VMM FABRIC allocation selection."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from modelexpress.vmm import backend as backend_module
from modelexpress.vmm.backend import CudaBackendError, CudaVmmBackend


@pytest.fixture
def driver(monkeypatch):
    cu = SimpleNamespace(
        CUresult=SimpleNamespace(
            CUDA_SUCCESS=0,
            CUDA_ERROR_INVALID_VALUE=1,
            CUDA_ERROR_OUT_OF_MEMORY=2,
            CUDA_ERROR_NOT_INITIALIZED=3,
            CUDA_ERROR_NOT_SUPPORTED=801,
            CUDA_ERROR_NOT_PERMITTED=800,
        ),
        CUdevice_attribute=SimpleNamespace(
            CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED=128,
        ),
        CUmemAllocationHandleType=SimpleNamespace(CU_MEM_HANDLE_TYPE_FABRIC=8),
        CUmemAllocationType=SimpleNamespace(CU_MEM_ALLOCATION_TYPE_PINNED=1),
        CUmemLocationType=SimpleNamespace(CU_MEM_LOCATION_TYPE_DEVICE=1),
        CUmemAccess_flags=SimpleNamespace(CU_MEM_ACCESS_FLAGS_PROT_READWRITE=3),
        CUmemAllocationGranularity_flags=SimpleNamespace(
            CU_MEM_ALLOC_GRANULARITY_MINIMUM=0,
        ),
        CUmemAllocationProp=lambda: SimpleNamespace(
            location=SimpleNamespace(),
            allocFlags=SimpleNamespace(),
            requestedHandleTypes=0,
        ),
        CUmemAccessDesc=lambda: SimpleNamespace(location=SimpleNamespace()),
        cuCtxGetCurrent=lambda: (0, 1),
        cuMemGetAllocationGranularity=lambda prop, flags: (0, 2097152),
        cuDeviceGetAttribute=MagicMock(return_value=(0, 1)),
        cuMemCreate=MagicMock(return_value=(0, 42)),
        cuMemMap=MagicMock(return_value=(0,)),
        cuMemSetAccess=MagicMock(return_value=(0,)),
        cuGetErrorString=lambda code: (0, b"driver error"),
    )
    monkeypatch.setattr(backend_module, "_import_cuda", lambda: cu)
    return cu


def test_unsupported_device_skips_fabric_allocations(driver, caplog, monkeypatch):
    monkeypatch.setattr(backend_module.logger, "handlers", [caplog.handler])
    driver.cuDeviceGetAttribute.return_value = (0, 0)

    def create(size, prop, flags):
        if prop.requestedHandleTypes == 8:
            return (1, 0)  # Unsupported FABRIC returns INVALID_VALUE.
        assert prop.location.id == 2
        assert prop.allocFlags.gpuDirectRDMACapable == 1
        return (0, 42)

    driver.cuMemCreate.side_effect = create
    backend = CudaVmmBackend(device=2)
    assert backend.allocate(0x1000000, 2097152) == 42
    assert backend.allocate(0x1200000, 2097152) == 42
    driver.cuDeviceGetAttribute.assert_called_once_with(128, 2)
    assert [
        call.args[1].requestedHandleTypes
        for call in driver.cuMemCreate.call_args_list
    ] == [0, 0]
    assert "MNNVL cuda_ipc peer-to-peer will NOT be available" in caplog.text


def test_supported_device_uses_fabric(driver):
    backend = CudaVmmBackend()
    assert backend.allocate(0x1000000, 2097152) == 42
    assert driver.cuMemCreate.call_args.args[1].requestedHandleTypes == 8


@pytest.mark.parametrize("error", [800, 801])
def test_permission_or_support_refusal_still_falls_back(driver, error):
    driver.cuMemCreate.side_effect = [(error, 0), (0, 42), (0, 43)]
    backend = CudaVmmBackend()
    assert backend.allocate(0x1000000, 2097152) == 42
    assert backend.allocate(0x1200000, 2097152) == 43
    assert [
        call.args[1].requestedHandleTypes
        for call in driver.cuMemCreate.call_args_list
    ] == [8, 0, 0]


@pytest.mark.parametrize("supported", [0, 1])
@pytest.mark.parametrize("error", [1, 2])
def test_allocation_errors_are_not_hidden(driver, supported, error):
    driver.cuDeviceGetAttribute.return_value = (0, supported)
    driver.cuMemCreate.return_value = (error, 0)
    backend = CudaVmmBackend()
    with pytest.raises(CudaBackendError) as exc:
        backend.allocate(0x1000000, 2097152)
    assert exc.value.op == "cuMemCreate"
    assert exc.value.result_code == error
    assert driver.cuMemCreate.call_count == 1
    driver.cuMemMap.assert_not_called()


def test_capability_query_errors_are_not_hidden(driver):
    driver.cuDeviceGetAttribute.return_value = (1, 0)
    with pytest.raises(CudaBackendError) as exc:
        CudaVmmBackend()
    assert exc.value.op == "cuDeviceGetAttribute"
    assert exc.value.result_code == 1
    driver.cuMemCreate.assert_not_called()


def test_older_bindings_keep_allocation_probe(driver):
    del driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED
    backend = CudaVmmBackend()
    assert backend.allocate(0x1000000, 2097152) == 42
    assert driver.cuMemCreate.call_args.args[1].requestedHandleTypes == 8
