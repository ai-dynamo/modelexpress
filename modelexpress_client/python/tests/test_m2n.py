# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for NIXL M2N integration: M2nDescriptor, M2nPlanner, M2nExecutor."""

import threading
import time
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from modelexpress.weight_transfer.protocol.types import (
    M2nDescriptor,
    RdmaDescriptor,
    ResolvedRegion,
    TrainerShard,
    TrainerTable,
    TrainerTensor,
)
from modelexpress.weight_transfer.protocol.serialization import (
    encode_m2n_descriptors,
    decode_m2n_descriptors,
)
from modelexpress.weight_transfer.planner.m2n_planner import M2nPlanner, _decode_m2n_proto_descriptors
from modelexpress.weight_transfer.planner.local import LocalPlanner


# ---------------------------------------------------------------------------
# M2nDescriptor
# ---------------------------------------------------------------------------


class TestM2nDescriptor:
    def test_fields(self):
        d = M2nDescriptor(
            src_agent_index=2,
            dst_agent_index=1,
            src_addr=0x1000,
            dst_addr=0x2000,
            nbytes=512,
        )
        assert d.src_agent_index == 2
        assert d.dst_agent_index == 1
        assert d.nbytes == 512

    def test_to_rdma_descriptor(self):
        d = M2nDescriptor(src_agent_index=3, dst_agent_index=0, src_addr=0xA000, dst_addr=0xB000, nbytes=256)
        rdma = d.to_rdma_descriptor()
        assert isinstance(rdma, RdmaDescriptor)
        assert rdma.agent_index == 3          # src_agent_index becomes agent_index
        assert rdma.src_addr == 0xA000
        assert rdma.dst_addr == 0xB000
        assert rdma.nbytes == 256

    def test_to_rdma_descriptor_drops_dst_agent_index(self):
        d = M2nDescriptor(src_agent_index=0, dst_agent_index=99, src_addr=0, dst_addr=0, nbytes=1)
        rdma = d.to_rdma_descriptor()
        assert not hasattr(rdma, "dst_agent_index")


# ---------------------------------------------------------------------------
# M2nDescriptor serialization
# ---------------------------------------------------------------------------


class TestM2nDescriptorSerialization:
    def test_empty_roundtrip(self):
        assert decode_m2n_descriptors(encode_m2n_descriptors([])) == []

    def test_single_roundtrip(self):
        descs = [M2nDescriptor(src_agent_index=0, dst_agent_index=1, src_addr=100, dst_addr=200, nbytes=64)]
        assert decode_m2n_descriptors(encode_m2n_descriptors(descs)) == descs

    def test_multi_roundtrip(self):
        descs = [
            M2nDescriptor(0, 0, 0x1000, 0x2000, 128),
            M2nDescriptor(1, 0, 0x3000, 0x4000, 256),
            M2nDescriptor(0, 1, 0x5000, 0x6000, 512),
            M2nDescriptor(1, 1, 0x7000, 0x8000, 1024),
        ]
        assert decode_m2n_descriptors(encode_m2n_descriptors(descs)) == descs

    def test_encoded_is_bytes(self):
        descs = [M2nDescriptor(0, 0, 1, 2, 3)]
        assert isinstance(encode_m2n_descriptors(descs), bytes)


# ---------------------------------------------------------------------------
# _decode_m2n_proto_descriptors helper
# ---------------------------------------------------------------------------


class TestDecodeM2nProtoDescriptors:
    def _make_proto(self, src, dst, sa, da, nb):
        p = MagicMock()
        p.src_agent_index = src
        p.dst_agent_index = dst
        p.src_addr = sa
        p.dst_addr = da
        p.nbytes = nb
        return p

    def test_empty(self):
        assert _decode_m2n_proto_descriptors([]) == []

    def test_single(self):
        proto = self._make_proto(1, 2, 0x100, 0x200, 64)
        result = _decode_m2n_proto_descriptors([proto])
        assert result == [M2nDescriptor(1, 2, 0x100, 0x200, 64)]

    def test_multiple(self):
        protos = [
            self._make_proto(0, 0, 10, 20, 30),
            self._make_proto(1, 1, 40, 50, 60),
        ]
        result = _decode_m2n_proto_descriptors(protos)
        assert len(result) == 2
        assert result[0] == M2nDescriptor(0, 0, 10, 20, 30)
        assert result[1] == M2nDescriptor(1, 1, 40, 50, 60)


# ---------------------------------------------------------------------------
# M2nPlanner — mock client
# ---------------------------------------------------------------------------


def _make_table():
    shards = [
        TrainerShard(agent_index=0, row_start=0, row_end=4, device_addr=0x0000, row_bytes=8, device_id=0),
        TrainerShard(agent_index=1, row_start=4, row_end=8, device_addr=0x1000, row_bytes=8, device_id=1),
    ]
    tensors = [TrainerTensor(name="w", dtype="torch.bfloat16", shape=[8, 4], shards=shards)]
    return TrainerTable(agents=[b"a0", b"a1"], tensors=tensors, step=1)


def _make_regions():
    return [ResolvedRegion(
        tensor_name="w",
        src_elem_runs=[0, 16],
        dst_addr=0xABCD0000,
        dst_elem_runs=[0, 16],
        element_size=2,
        dst_device_id=0,
    )]


def _make_proto_desc(src_agent, dst_agent, src_addr, dst_addr, nbytes):
    p = MagicMock()
    p.src_agent_index = src_agent
    p.dst_agent_index = dst_agent
    p.src_addr = src_addr
    p.dst_addr = dst_addr
    p.nbytes = nbytes
    return p


class TestM2nPlannerBuild:
    def _make_client(self, plan_id="test-plan-id", ready_after=0):
        """Mock MX client. Returns plan_id immediately; plan is always ready."""
        client = MagicMock()

        reg_resp = MagicMock()
        reg_resp.m2n_plan_id = plan_id
        client.register_m2n_worker.return_value = reg_resp

        get_resp = MagicMock()
        get_resp.ready = True
        get_resp.descriptors = [
            _make_proto_desc(0, 0, 0x0000, 0xABCD0000, 32),
            _make_proto_desc(1, 0, 0x1000, 0xABCD0020, 32),
        ]
        client.get_m2n_plan.return_value = get_resp

        return client

    def test_build_returns_rdma_descriptors(self):
        client = self._make_client()
        planner = M2nPlanner(
            mx_client=client,
            model_key="model-key",
            worker_rank=0,
            total_workers=2,
            nixl_metadata=b"nixl-meta",
        )
        descs = planner.build(_make_regions(), _make_table(), "plan-key-0")
        assert len(descs) == 2
        assert all(isinstance(d, RdmaDescriptor) for d in descs)

    def test_build_m2n_returns_m2n_descriptors(self):
        client = self._make_client()
        planner = M2nPlanner(
            mx_client=client,
            model_key="model-key",
            worker_rank=0,
            total_workers=2,
            nixl_metadata=b"nixl-meta",
        )
        descs = planner.build_m2n(_make_regions(), _make_table(), "plan-key-0")
        assert len(descs) == 2
        assert all(isinstance(d, M2nDescriptor) for d in descs)
        assert descs[0].src_agent_index == 0
        assert descs[1].src_agent_index == 1

    def test_build_caches_result(self):
        client = self._make_client()
        planner = M2nPlanner(
            mx_client=client,
            model_key="model-key",
            worker_rank=0,
            total_workers=2,
            nixl_metadata=b"nixl-meta",
        )
        planner.build(_make_regions(), _make_table(), "plan-key-cache")
        planner.build(_make_regions(), _make_table(), "plan-key-cache")
        # register_m2n_worker should only be called once (cached after first build)
        assert client.register_m2n_worker.call_count == 1

    def test_build_passes_correct_args_to_client(self):
        client = self._make_client()
        planner = M2nPlanner(
            mx_client=client,
            model_key="my-model",
            worker_rank=1,
            total_workers=4,
            nixl_metadata=b"meta-bytes",
        )
        planner.build_m2n(_make_regions(), _make_table(), "pk")
        call_kwargs = client.register_m2n_worker.call_args.kwargs
        assert call_kwargs["model_key"] == "my-model"
        assert call_kwargs["worker_rank"] == 1
        assert call_kwargs["total_workers"] == 4
        assert call_kwargs["nixl_metadata"] == b"meta-bytes"

    def test_invalidate_clears_cache_and_calls_server(self):
        client = self._make_client()
        planner = M2nPlanner(
            mx_client=client,
            model_key="model-key",
            worker_rank=0,
            total_workers=2,
            nixl_metadata=b"meta",
        )
        planner.build_m2n(_make_regions(), _make_table(), "pk")
        planner.invalidate("pk")
        # After invalidate, a second build should call the server again
        planner.build_m2n(_make_regions(), _make_table(), "pk")
        assert client.register_m2n_worker.call_count == 2
        client.invalidate_m2n_plan.assert_called_once_with(model_key="model-key")


class TestM2nPlannerBarrierPolling:
    """Test that M2nPlanner polls when plan_id is initially empty (barrier not met)."""

    def test_polls_until_plan_id_available(self):
        client = MagicMock()

        # First call returns empty plan_id (barrier not met)
        empty_resp = MagicMock()
        empty_resp.m2n_plan_id = ""

        ready_resp = MagicMock()
        ready_resp.m2n_plan_id = "final-plan-id"

        client.register_m2n_worker.side_effect = [empty_resp, ready_resp]

        get_resp = MagicMock()
        get_resp.ready = True
        get_resp.descriptors = [_make_proto_desc(0, 0, 0, 0, 64)]
        client.get_m2n_plan.return_value = get_resp

        planner = M2nPlanner(
            mx_client=client,
            model_key="m",
            worker_rank=0,
            total_workers=2,
            nixl_metadata=b"",
            poll_interval=0.0,
        )
        descs = planner.build_m2n(_make_regions(), _make_table(), "pk")
        assert len(descs) == 1
        assert client.register_m2n_worker.call_count == 2

    def test_timeout_raises(self):
        client = MagicMock()
        empty_resp = MagicMock()
        empty_resp.m2n_plan_id = ""
        client.register_m2n_worker.return_value = empty_resp

        planner = M2nPlanner(
            mx_client=client,
            model_key="m",
            worker_rank=0,
            total_workers=2,
            nixl_metadata=b"",
            timeout=0.05,
            poll_interval=0.01,
        )
        with pytest.raises(TimeoutError, match="M2nPlanner"):
            planner.build_m2n(_make_regions(), _make_table(), "pk")


class TestM2nPlannerFallback:
    def test_falls_back_to_local_on_server_error(self):
        client = MagicMock()
        client.register_m2n_worker.side_effect = ConnectionError("server down")

        planner = M2nPlanner(
            mx_client=client,
            model_key="m",
            worker_rank=0,
            total_workers=2,
            nixl_metadata=b"",
        )
        # Should not raise — falls back to LocalPlanner
        descs = planner.build(_make_regions(), _make_table(), "pk")
        # LocalPlanner returns descriptors (router is pure-math, no NIXL needed)
        assert isinstance(descs, list)
        assert all(isinstance(d, RdmaDescriptor) for d in descs)
        total = sum(d.nbytes for d in descs)
        assert total == 16 * 2  # 16 elements × 2 bytes


# ---------------------------------------------------------------------------
# M2nPlanner is a drop-in for AbstractPlanner
# ---------------------------------------------------------------------------


class TestM2nPlannerInterface:
    def test_is_abstract_planner_subclass(self):
        from modelexpress.weight_transfer.planner.base import AbstractPlanner
        assert issubclass(M2nPlanner, AbstractPlanner)

    def test_has_build_method(self):
        assert callable(M2nPlanner.build)

    def test_has_invalidate_method(self):
        assert callable(M2nPlanner.invalidate)

    def test_build_m2n_distinct_from_build(self):
        assert M2nPlanner.build is not M2nPlanner.build_m2n
