# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""How a reshard receiver wires itself to its accelerator.

The receiver's device-specific work - the allocation scope for NIXL-registered
buffers, the per-stage synchronizes, and the backend the transfer manager
registers memory through - all resolve from the constructor's ``device`` rather
than assuming CUDA. These tests pin that wiring for both implemented families.

The transfer-manager assertion is the one that catches a silent regression:
``NixlTransferManager`` defaults to a CUDA backend when none is passed, so
forgetting to hand it the receiver's backend registers XPU memory through CUDA
device calls while every other part of the receiver looks correct.

Nothing here asserts a publisher/target compatibility policy, because the
receiver has none to assert: the rendezvous identity and the shard table carry no
publisher accelerator, so both same-family and cross-family pairings are
unchecked by accelerator family. See the ``TODO(publisher-accelerator)`` in
``receiver.py``.

Run: pytest tests/test_reshard_refit_accelerator_wiring.py
"""

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from modelexpress.accelerators import CudaAcceleratorBackend, XpuAcceleratorBackend
from modelexpress.refit.reshard.alloc_scope import registered_buffer_alloc_scope
from modelexpress.refit.reshard.receiver import ReshardReceiver

_KWARGS = dict(
    model_name="m",
    mx_server="127.0.0.1:8011",
    agent_name="agent0",
    local_rank=0,
    global_rank=0,
    num_trainer_sources=1,
    listen_port=9000,
)


@pytest.fixture
def build_receiver():
    """Construct a receiver with the NIXL agent and metadata client stubbed out."""

    def _build(device):
        with patch(
            "modelexpress.refit.reshard.receiver.NixlTransferManager"
        ) as manager_cls, patch("modelexpress.refit.reshard.receiver.MxClient"):
            manager_cls.return_value = MagicMock()
            receiver = ReshardReceiver(device=device, **_KWARGS)
        return receiver, manager_cls

    return _build


@pytest.fixture
def xpu_available(monkeypatch):
    monkeypatch.setattr(
        torch, "xpu", SimpleNamespace(is_available=lambda: True), raising=False
    )


class TestBackendResolution:
    def test_cuda_device_resolves_the_cuda_backend(self, build_receiver):
        receiver, _ = build_receiver(torch.device("cuda", 0))

        assert isinstance(receiver._backend, CudaAcceleratorBackend)
        assert receiver._backend.requires_classic_alloc_pool() is True

    def test_xpu_device_resolves_the_xpu_backend(self, build_receiver, xpu_available):
        receiver, _ = build_receiver(torch.device("xpu", 0))

        assert isinstance(receiver._backend, XpuAcceleratorBackend)
        assert receiver._backend.requires_classic_alloc_pool() is False

    @pytest.mark.parametrize("device_type", ["cuda", "xpu"])
    def test_backend_is_passed_to_the_transfer_manager(
        self,
        build_receiver,
        xpu_available,
        device_type,
    ):
        """Otherwise the manager falls back to its own CUDA default and registers
        this backend's memory through CUDA device calls."""
        receiver, manager_cls = build_receiver(torch.device(device_type, 0))

        passed = manager_cls.call_args.kwargs["accelerator_backend"]
        assert passed is receiver._backend
        assert passed.name == device_type


class TestAllocationScope:
    def test_cuda_receiver_scopes_registered_buffers_into_the_classic_pool(
        self,
        build_receiver,
    ):
        receiver, _ = build_receiver(torch.device("cuda", 0))
        entered = []

        @contextmanager
        def fake_use_mem_pool(pool, device=None):
            entered.append(pool)
            yield

        with patch(
            "modelexpress.refit.reshard.cuda_pool._get_pool", return_value="pool"
        ), patch.object(torch.cuda, "use_mem_pool", fake_use_mem_pool):
            with registered_buffer_alloc_scope(receiver._backend):
                pass

        assert entered == ["pool"]

    def test_xpu_receiver_allocates_registered_buffers_normally(
        self,
        build_receiver,
        xpu_available,
    ):
        receiver, _ = build_receiver(torch.device("xpu", 0))

        with patch(
            "modelexpress.refit.reshard.cuda_pool._get_pool",
            side_effect=AssertionError("classic pool should not be built"),
        ):
            scope = registered_buffer_alloc_scope(receiver._backend)
            assert isinstance(scope, type(nullcontext()))
            with scope:
                pass

    def test_prepare_scopes_every_registered_buffer_class(self, monkeypatch):
        from tests.test_reshard_refit_fused_wire import _build, _RecordingTransport

        receiver, keepalive = _build(_RecordingTransport())
        plan = receiver._plan
        receiver._manager = MagicMock()
        receiver._mx_client = MagicMock()
        receiver._model_name = "m"
        receiver._num_trainer_sources = 3
        receiver._capture = MagicMock(
            return_value=(
                SimpleNamespace(
                    copies=[
                        SimpleNamespace(param_name="exact"),
                        SimpleNamespace(param_name="strided"),
                        SimpleNamespace(param_name="router"),
                    ],
                    unsupported=[],
                ),
                {
                    "exact": ((8,), torch.float32),
                    "strided": ((4, 2), torch.float32),
                    "router": ((4,), torch.float32),
                },
            )
        )
        receiver._log_coverage = MagicMock()
        sessions = {"s0": "a0", "s1": "a1", "s2": "a2"}
        sources = {
            name: SimpleNamespace(dtype=torch.float32, global_shape=(1,))
            for name in ("exact", "strided", "router")
        }
        entered = []

        @contextmanager
        def recording_scope(_backend):
            entered.append(True)
            yield

        monkeypatch.setattr(
            "modelexpress.refit.reshard.receiver.gather_sources",
            lambda *_args, **_kwargs: (
                sources,
                sessions,
                {session: 0 for session in sessions},
                {agent: "host:1" for agent in sessions.values()},
            ),
        )
        monkeypatch.setattr(
            "modelexpress.refit.reshard.receiver.plan_transfer",
            lambda *_args, **_kwargs: plan,
        )
        monkeypatch.setattr(
            "modelexpress.refit.reshard.receiver.handshake_with_peers",
            lambda *_args, **_kwargs: None,
        )
        monkeypatch.setattr(
            "modelexpress.refit.reshard.receiver.NixlReshardTransport",
            lambda *_args, **_kwargs: MagicMock(),
        )
        monkeypatch.setattr(
            "modelexpress.refit.reshard.receiver.registered_buffer_alloc_scope",
            recording_scope,
        )

        receiver._prepare(timeout=1.0)

        assert len(entered) == 3
        assert receiver._manager.register_tensors.call_count == 3
        assert all(tensor.data_ptr() for tensor in keepalive)


class TestStageSynchronization:
    def test_every_stage_sync_goes_through_the_backend(self, monkeypatch):
        """The receiver holds no ``torch.cuda`` call of its own, so a refit on a
        non-CUDA target synchronizes through that target's backend. Asserted by
        running a real refit against a stub backend: were any stage still calling
        ``torch.cuda`` directly, the count here would be short."""
        from tests.test_reshard_refit_fused_wire import _RecordingTransport, _build

        monkeypatch.setenv("MX_RESHARD_FUSED_WIRE", "1")
        harness, _keepalive = _build(_RecordingTransport())

        harness.update_weights(step=1)

        # re-slice, dtype cast, install - the three stages this plan exercises.
        assert harness._backend.synchronize_calls == [None, None, None]

    def test_receiver_module_holds_no_direct_torch_cuda_call(self):
        """A regression fence: the point of the backend boundary is that this file
        names no accelerator directly."""
        from pathlib import Path

        import modelexpress.refit.reshard.receiver as receiver_module

        source = Path(receiver_module.__file__).read_text()
        code = "\n".join(
            line for line in source.splitlines() if not line.strip().startswith("#")
        )
        assert "torch.cuda" not in code
        assert "torch.xpu" not in code
