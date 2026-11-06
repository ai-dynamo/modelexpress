# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the NIXL OBJ transfer manager."""

from unittest.mock import patch

import pytest
import torch

from modelexpress import s3_rdma_transfer
from modelexpress.s3_rdma_transfer import (
    S3RdmaBackendUnavailable,
    S3RdmaEndpoint,
    S3RdmaTransferError,
    S3RdmaTransferManager,
    is_s3_rdma_available,
)
from tests.test_s3_rdma_loader import ENDPOINT, install_fake_nixl


@pytest.fixture
def fake_nixl(monkeypatch):
    return install_fake_nixl(monkeypatch)


class TestIsS3RdmaAvailable:
    @patch("modelexpress.s3_rdma_transfer.NIXL_AVAILABLE", False)
    def test_nixl_missing(self):
        assert is_s3_rdma_available() is False

    @patch("modelexpress.s3_rdma_transfer.NIXL_AVAILABLE", True)
    @patch("modelexpress.s3_rdma_transfer._cuobj_client_loadable", return_value=False)
    def test_cuobj_missing(self, _probe):
        assert is_s3_rdma_available() is False

    @patch("modelexpress.s3_rdma_transfer.NIXL_AVAILABLE", True)
    @patch("modelexpress.s3_rdma_transfer._cuobj_client_loadable", return_value=True)
    def test_available(self, _probe):
        assert is_s3_rdma_available() is True


class TestS3RdmaEndpoint:
    def test_backend_params(self):
        assert ENDPOINT.backend_params(8) == {
            "accelerated": "true",
            "bucket": "models",
            "endpoint_override": "http://aistor:9100",
            "scheme": "http",
            "region": "us-east-1",
            "use_virtual_addressing": "false",
            "num_threads": "8",
        }

    def test_optional_params_and_scheme(self):
        endpoint = S3RdmaEndpoint(
            bucket="b",
            endpoint_url="https://aistor:9443",
            region="r",
            ca_bundle="/etc/ca.pem",
        )
        params = endpoint.backend_params(2)
        assert params["scheme"] == "https"
        assert params["ca_bundle"] == "/etc/ca.pem"


class _SizedSlot:
    """Stand-in for a tensor too large to allocate in a unit test."""

    def __init__(self, nbytes):
        self._nbytes = nbytes

    def numel(self):
        return self._nbytes

    def element_size(self):
        return 1

    def data_ptr(self):
        return 0x1000

    def is_contiguous(self):
        return True

    @property
    def device(self):
        return torch.device("cpu")


@pytest.fixture
def manager(fake_nixl, mock_accelerator_backend_cls):
    state = fake_nixl({"k0": b"x" * 100, "k1": b"y" * 50})
    mgr = S3RdmaTransferManager(
        "mx-s3-rdma-test",
        ENDPOINT,
        device_id=3,
        num_threads=4,
        accelerator_backend=mock_accelerator_backend_cls(s3_rdma=True),
    )
    return mgr, state


class TestS3RdmaTransferManager:
    def test_initialize_sets_device(self, manager):
        mgr, _ = manager
        mgr.initialize()
        assert mgr._accelerator_backend.set_device_calls == [3]

    @pytest.mark.parametrize(
        "agent_kwargs",
        [
            {"plugins": ("UCX",)},
            {"create_error": RuntimeError("fabric down")},
            {"mem_types": ("DRAM_SEG", "OBJ_SEG")},
        ],
    )
    def test_initialize_unavailable(self, fake_nixl, mock_accelerator_backend_cls, agent_kwargs):
        fake_nixl({}, **agent_kwargs)
        mgr = S3RdmaTransferManager(
            "a", ENDPOINT, device_id=0, num_threads=1, accelerator_backend=mock_accelerator_backend_cls()
        )
        with pytest.raises(S3RdmaBackendUnavailable):
            mgr.initialize()

    def test_initialize_without_nixl(self, mock_accelerator_backend_cls, monkeypatch):
        monkeypatch.setattr(s3_rdma_transfer, "NIXL_AVAILABLE", False)
        mgr = S3RdmaTransferManager(
            "a", ENDPOINT, device_id=0, num_threads=1, accelerator_backend=mock_accelerator_backend_cls()
        )
        with pytest.raises(S3RdmaBackendUnavailable):
            mgr.initialize()

    def test_requires_initialize(self, manager):
        mgr, _ = manager
        with pytest.raises(S3RdmaTransferError, match="not initialized"):
            mgr.register_objects([("k0", 100)])

    def test_register_objects_assigns_unique_ids(self, manager):
        mgr, state = manager
        mgr.initialize()
        assert mgr.register_objects([("k0", 100), ("k1", 50)]) == [0, 1]
        assert state["agents"][0].obj_keys == {0: "k0", 1: "k1"}
        with pytest.raises(S3RdmaTransferError, match="already registered"):
            mgr.register_objects([("k0", 100)])

    def test_register_slots_accepts_slots_past_4_gib(self, manager):
        mgr, _ = manager
        mgr.initialize()
        mgr.register_slots([_SizedSlot(8 * 1024**3)])

    def test_register_slots_rejects_wrong_device(self, manager):
        mgr, _ = manager
        mgr.initialize()
        with pytest.raises(S3RdmaTransferError, match="expected cpu"):
            mgr.register_slots([torch.zeros(16, dtype=torch.uint8, device="meta")])

    def test_register_slots_rejects_non_contiguous(self, manager):
        mgr, _ = manager
        mgr.initialize()
        with pytest.raises(S3RdmaTransferError, match="not contiguous"):
            mgr.register_slots([torch.zeros(32, dtype=torch.uint8)[::2]])

    def test_register_host_slots_as_dram(self, manager):
        mgr, state = manager
        mgr.initialize()
        mgr.register_objects([("k0", 100)])
        with pytest.raises(S3RdmaTransferError, match="is on meta, expected cpu"):
            mgr.register_slots([torch.zeros(16, dtype=torch.uint8, device="meta")], host=True)
        mgr.register_slots([torch.zeros(16, dtype=torch.uint8)], host=True)
        handle = mgr.post(0, 0, 0, 16)
        while not mgr.poll(handle):
            pass
        mgr.release(handle)
        agent = state["agents"][0]
        assert ("register", "DRAM", 1) in agent.events
        assert agent.local_types == [("DRAM", 0)]

    def test_register_slots_reports_nixl_failure(self, fake_nixl, mock_accelerator_backend_cls):
        fake_nixl({"k0": b"x" * 100}, vram_register_error=RuntimeError("cuMemObjGetDescriptor failed"))
        mgr = S3RdmaTransferManager(
            "a", ENDPOINT, device_id=0, num_threads=1, accelerator_backend=mock_accelerator_backend_cls()
        )
        mgr.initialize()
        with pytest.raises(S3RdmaTransferError, match="could not register VRAM staging slots"):
            mgr.register_slots([torch.zeros(16, dtype=torch.uint8)])
        mgr.register_slots([torch.zeros(16, dtype=torch.uint8)], host=True)

    def test_post_rejects_busy_or_unknown_slot(self, manager):
        mgr, _ = manager
        mgr.initialize()
        mgr.register_objects([("k0", 100)])
        mgr.register_slots([torch.zeros(16, dtype=torch.uint8)])
        with pytest.raises(S3RdmaTransferError, match="not registered"):
            mgr.post(1, 0, 0, 16)
        handle = mgr.post(0, 0, 0, 16)
        with pytest.raises(S3RdmaTransferError, match="already has a read in flight"):
            mgr.post(0, 0, 16, 16)
        while not mgr.poll(handle):
            pass
        mgr.release(handle)
        mgr.post(0, 0, 16, 16)
        assert mgr.drain(1.0)

    def test_post_poll_release(self, manager):
        mgr, _ = manager
        mgr.initialize()
        mgr.register_objects([("k0", 100), ("k1", 50)])
        slot = torch.zeros(64, dtype=torch.uint8)
        mgr.register_slots([slot])
        with pytest.raises(S3RdmaTransferError, match="invalid read length"):
            mgr.post(0, 0, 0, 65)
        handle = mgr.post(0, 1, 10, 20)
        assert mgr.inflight == 1
        while not mgr.poll(handle):
            pass
        mgr.release(handle)
        assert mgr.inflight == 0
        assert bytes(slot[:20].tolist()) == b"y" * 20

    def test_poll_maps_errors(self, fake_nixl, mock_accelerator_backend_cls):
        fake_nixl({"k0": b"x" * 100}, fail={0}, max_polls=1)
        mgr = S3RdmaTransferManager(
            "a", ENDPOINT, device_id=0, num_threads=1, accelerator_backend=mock_accelerator_backend_cls()
        )
        mgr.initialize()
        mgr.register_objects([("k0", 100)])
        mgr.register_slots([torch.zeros(16, dtype=torch.uint8)])
        handle = mgr.post(0, 0, 0, 16)
        with pytest.raises(S3RdmaTransferError, match="OBJ read failed"):
            mgr.poll(handle)
        assert mgr.inflight == 0

    def test_post_maps_err_state(self, manager, monkeypatch):
        mgr, state = manager
        mgr.initialize()
        mgr.register_objects([("k0", 100)])
        mgr.register_slots([torch.zeros(16, dtype=torch.uint8)])
        monkeypatch.setattr(state["agents"][0], "transfer", lambda handle: "ERR")
        with pytest.raises(S3RdmaTransferError, match="could not be posted"):
            mgr.post(0, 0, 0, 16)
        assert mgr.inflight == 0

    def test_shutdown_deregisters_slots_then_objects(self, manager):
        mgr, state = manager
        mgr.initialize()
        mgr.register_objects([("k0", 100)])
        mgr.register_slots([torch.zeros(16, dtype=torch.uint8)])
        mgr.shutdown()
        events = state["agents"][0].events
        assert events[-2:] == [("deregister", "VRAM"), ("deregister", "OBJ")]
        mgr.shutdown()

    def test_shutdown_quarantines_undrained_reads(self, fake_nixl, mock_accelerator_backend_cls):
        state = fake_nixl({"k0": b"x" * 100}, stuck={0})
        mgr = S3RdmaTransferManager(
            "a", ENDPOINT, device_id=0, num_threads=1, accelerator_backend=mock_accelerator_backend_cls()
        )
        mgr.initialize()
        mgr.register_objects([("k0", 100)])
        slot = torch.zeros(16, dtype=torch.uint8)
        mgr.register_slots([slot])
        mgr.post(0, 0, 0, 16)
        assert mgr.drain(0.01) is False
        mgr.shutdown(drain_timeout_s=0.01)
        assert len(s3_rdma_transfer._QUARANTINE) == 1
        agent, slots = s3_rdma_transfer._QUARANTINE[0]
        assert agent is state["agents"][0]
        assert slots[0] is slot
        assert not any(event[0] == "deregister" for event in agent.events)
