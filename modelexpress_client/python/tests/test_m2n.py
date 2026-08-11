# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for NIXL M2N integration: M2nDescriptor, M2nExecutor."""

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
