# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for accelerator backend abstractions and capability gates."""

from __future__ import annotations

from contextlib import nullcontext
from unittest.mock import MagicMock, patch

import pytest
import torch

from modelexpress.accelerators import (
    CudaAcceleratorBackend,
    accelerator_backend_for,
)
from modelexpress.adapter import EngineAdapter
from modelexpress.load_strategy.context import LoadResult
from modelexpress.nixl_transfer import NixlTransferManager
from modelexpress.types import TensorDescriptor


class TestCudaAcceleratorBackend:
    def test_cuda_backend_uses_nixl_vram_segment(self):
        backend = CudaAcceleratorBackend()

        assert backend.name == "cuda"
        assert backend.torch_device_type == "cuda"
        assert backend.nixl_mem_type == "VRAM"
        assert backend.supports_rdma_p2p() is True
        assert backend.supports_pool_reg() is True
        assert backend.supports_vmm() is True
        assert backend.supports_gds() is True

    def test_cuda_backend_delegates_torch_calls(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            torch.cuda,
            "set_device",
            lambda device_id: calls.append(("set", device_id)),
        )
        monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)
        monkeypatch.setattr(
            torch.cuda,
            "synchronize",
            lambda device_id=None: calls.append(("sync", device_id)),
        )
        monkeypatch.setattr(
            torch.cuda,
            "empty_cache",
            lambda: calls.append(("empty", None)),
        )

        backend = CudaAcceleratorBackend()
        backend.set_device(2)
        assert backend.current_device() == 3
        backend.synchronize(2)
        backend.synchronize()
        backend.empty_cache()

        assert calls == [
            ("set", 2),
            ("sync", 2),
            ("sync", None),
            ("empty", None),
        ]

    def test_cuda_backend_is_accel_tensor_uses_tensor_cuda_flag(self):
        backend = CudaAcceleratorBackend()

        assert backend.is_accel_tensor(torch.zeros(1)) is False

        class FakeCudaTensor:
            is_cuda = True

        assert backend.is_accel_tensor(FakeCudaTensor()) is True

    def test_accelerator_backend_for_cuda(self):
        assert isinstance(
            accelerator_backend_for(torch.device("cuda:0")),
            CudaAcceleratorBackend,
        )

    def test_accelerator_backend_for_rejects_unsupported_device(self):
        with pytest.raises(ValueError, match="Unsupported accelerator backend"):
            accelerator_backend_for(torch.device("cpu"))


class TestAcceleratorCapabilityGates:
    def _make_manager(self, backend) -> NixlTransferManager:
        mgr = NixlTransferManager(
            agent_name="test",
            device_id=0,
            accelerator_backend=backend,
        )
        mgr._agent = MagicMock()
        mgr._agent.get_agent_metadata.return_value = b"metadata"
        return mgr

    def test_pool_reg_unsupported_falls_back_to_tensor_registration(
        self,
        monkeypatch,
        mock_accelerator_backend_cls,
    ):
        backend = mock_accelerator_backend_cls(pool_reg=False)
        mgr = self._make_manager(backend)
        tensor = torch.zeros(4, dtype=torch.float32)
        monkeypatch.setenv("MX_POOL_REG", "1")

        with patch.object(
            NixlTransferManager,
            "_find_cuda_allocations",
            side_effect=AssertionError("pool discovery should not run"),
        ):
            assert mgr.register_tensors({"w": tensor}) == b"metadata"

        mgr._agent.register_memory.assert_called_once_with(
            [tensor],
            backends=["UCX"],
        )

    def test_vmm_arena_unsupported_falls_back_to_tensor_registration(
        self,
        mock_accelerator_backend_cls,
    ):
        class FakeArena:
            def registered_range(self):
                return 0x1000, 0x2000

        backend = mock_accelerator_backend_cls(vmm=False)
        mgr = self._make_manager(backend)
        tensor = torch.zeros(1)

        assert mgr.register_arena(FakeArena(), {"w": tensor}) == b"metadata"

        mgr._agent.register_memory.assert_called_once_with(
            [tensor],
            backends=["UCX"],
        )

    def test_receive_uses_backend_device_ops_and_mem_type(
        self,
        mock_accelerator_backend_cls,
    ):
        backend = mock_accelerator_backend_cls(nixl_mem_type="VRAM")
        mgr = self._make_manager(backend)
        local = torch.zeros(4, dtype=torch.float32)
        mgr._tensors = {"w": local}
        mgr._agent.prep_xfer_dlist.side_effect = ["src", "dst"]
        mgr._agent.make_prepped_xfer.return_value = "handle"
        mgr._agent.check_xfer_state.return_value = "DONE"

        bytes_transferred, tensor_count, _ = mgr.receive_from_source(
            source_metadata=b"",
            source_tensors=[
                TensorDescriptor(
                    name="w",
                    addr=0x1000,
                    size=local.numel() * local.element_size(),
                    device_id=0,
                    dtype=str(local.dtype),
                )
            ],
            remote_agent_name="source",
        )

        assert bytes_transferred == local.numel() * local.element_size()
        assert tensor_count == 1
        assert backend.set_device_calls == [0]
        assert backend.synchronize_calls == [0]

    def test_gds_strategy_unavailable_when_backend_does_not_support_gds(
        self,
        mock_accelerator_backend_cls,
    ):
        from modelexpress.load_strategy.context import LoadContext
        from modelexpress.load_strategy.gds_strategy import GdsStrategy

        class Adapter(EngineAdapter):
            def apply_weight_iter(self, result: LoadResult, weights_iter):
                return result

        ctx = LoadContext(
            model_config=MagicMock(),
            load_config=MagicMock(),
            target_device=torch.device("cpu"),
            global_rank=0,
            worker_rank=0,
            device_id=0,
            identity=MagicMock(),
            mx_client=MagicMock(),
            worker_id="test-worker",
            adapter=Adapter(),
            accelerator_backend=mock_accelerator_backend_cls(gds=False),
        )

        with patch(
            "modelexpress.gds_transfer.is_gds_available",
            side_effect=AssertionError("system GDS probe should not run"),
        ):
            assert GdsStrategy().is_available(ctx) is False

    def test_vmm_runtime_noops_when_backend_does_not_support_arena(
        self,
        monkeypatch,
        mock_accelerator_backend_cls,
    ):
        from modelexpress.vmm import runtime as vmm_runtime

        monkeypatch.setenv("MX_VMM_ARENA", "1")

        class Ctx:
            global_rank = 0
            device_id = 0
            target_device = nullcontext()

        ctx = Ctx()
        ctx.accelerator_backend = mock_accelerator_backend_cls(vmm=False)

        entered = False
        with vmm_runtime.maybe_enter_vmm_arena(ctx):
            entered = True

        assert entered is True
