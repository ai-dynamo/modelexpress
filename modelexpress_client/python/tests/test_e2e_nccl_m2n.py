# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for NCCL M2N transport pipeline (no NIXL, no GPU)."""

import math
from unittest.mock import MagicMock

import pytest

from modelexpress.weight_transfer.planner.local import LocalPlanner
from modelexpress.weight_transfer.planner.m2n_planner import M2nPlanner
from modelexpress.weight_transfer.planner.router import route_regions
from modelexpress.weight_transfer.protocol.serialization import (
    decode_m2n_descriptors,
    encode_m2n_descriptors,
)
from modelexpress.weight_transfer.protocol.types import (
    M2nDescriptor,
    RdmaDescriptor,
    ResolvedRegion,
    TrainerShard,
    TrainerTable,
    TrainerTensor,
)


def _row_shard(agent, r0, r1, shape, base=0x10000):
    row_bytes = math.prod(shape[1:]) * 2
    return TrainerShard(
        agent_index=agent, row_start=r0, row_end=r1,
        device_addr=base + r0 * row_bytes, row_bytes=row_bytes, device_id=agent,
    )


def _col_shard(agent, r0, r1, c0, c1, base=0x20000):
    row_bytes = (c1 - c0) * 2
    return TrainerShard(
        agent_index=agent, row_start=r0, row_end=r1, col_start=c0, col_end=c1,
        device_addr=base + agent * (r1 - r0) * row_bytes, row_bytes=row_bytes, device_id=agent,
    )


def _table(tensors, agents=None, step=1):
    if agents is None:
        n = max(s.agent_index for tt in tensors for s in tt.shards) + 1
        agents = [f"agent{i}".encode() for i in range(n)]
    return TrainerTable(agents=agents, tensors=tensors, step=step)


def _region(name, n_elems, dst_addr=0xDEAD0000, elem_size=2):
    return ResolvedRegion(
        tensor_name=name, src_elem_runs=[0, n_elems],
        dst_addr=dst_addr, dst_elem_runs=[0, n_elems],
        element_size=elem_size, dst_device_id=0,
    )


def _mock_m2n_client(descriptors=None, plan_id="plan-1"):
    descriptors = descriptors or []
    client = MagicMock()
    reg = MagicMock()
    reg.m2n_plan_id = plan_id
    client.register_m2n_worker.return_value = reg
    get = MagicMock()
    get.ready = True
    get.descriptors = [_proto(d) for d in descriptors]
    client.get_m2n_plan.return_value = get
    return client


def _proto(d: M2nDescriptor):
    p = MagicMock()
    p.src_agent_index = d.src_agent_index
    p.dst_agent_index = d.dst_agent_index
    p.src_addr = d.src_addr
    p.dst_addr = d.dst_addr
    p.nbytes = d.nbytes
    return p


class TestM2nDescriptorAccounting:
    def test_single_tensor_single_shard_full_coverage(self):
        shape = [8, 16]
        n = math.prod(shape)
        tt = TrainerTensor("w", "torch.bfloat16", list(shape), [_row_shard(0, 0, 8, shape)])
        assert sum(d.nbytes for d in route_regions([_region("w", n)], _table([tt]))) == n * 2

    def test_two_row_shards_full_coverage(self):
        shape = [16, 32]
        n = math.prod(shape)
        tt = TrainerTensor("w", "torch.bfloat16", list(shape), [
            _row_shard(0, 0, 8,  shape),
            _row_shard(1, 8, 16, shape),
        ])
        assert sum(d.nbytes for d in route_regions([_region("w", n)], _table([tt]))) == n * 2

    def test_two_col_shards_full_coverage(self):
        shape = [8, 16]
        n = math.prod(shape)
        tt = TrainerTensor("w", "torch.bfloat16", list(shape), [
            _col_shard(0, 0, 8, 0,  8),
            _col_shard(1, 0, 8, 8, 16),
        ])
        assert sum(d.nbytes for d in route_regions([_region("w", n)], _table([tt]))) == n * 2

    def test_multi_tensor_full_coverage(self):
        names = ["embed", "proj", "norm"]
        shapes = [[128, 64], [64, 32], [64]]
        tensors = []
        total_bytes = 0
        for name, shape in zip(names, shapes):
            n = math.prod(shape)
            total_bytes += n * 2
            if len(shape) == 2:
                tt = TrainerTensor(name, "torch.bfloat16", shape, [
                    _row_shard(0, 0, shape[0] // 2, shape),
                    _row_shard(1, shape[0] // 2, shape[0], shape),
                ])
            else:
                tt = TrainerTensor(name, "torch.bfloat16", [1, shape[0]], [
                    _row_shard(0, 0, 1, [1, shape[0]]),
                ])
            tensors.append(tt)
        regions = [_region(name, math.prod(s), dst_addr=0xA000 + i * 0x10000)
                   for i, (name, s) in enumerate(zip(names, shapes))]
        assert sum(d.nbytes for d in route_regions(regions, _table(tensors))) == total_bytes

    def test_no_byte_double_counted(self):
        """Each destination byte range must appear exactly once."""
        shape = [4, 8]
        n = math.prod(shape)
        tt = TrainerTensor("w", "torch.bfloat16", list(shape), [
            _row_shard(0, 0, 2, shape),
            _row_shard(1, 2, 4, shape),
        ])
        descs = route_regions([_region("w", n, dst_addr=0xF000)], _table([tt]))
        ranges = sorted((d.dst_addr, d.dst_addr + d.nbytes) for d in descs)
        for (_, end_a), (start_b, _) in zip(ranges, ranges[1:]):
            assert end_a <= start_b, "Destination byte ranges overlap"


class TestM2nMultiWorkerDescriptors:
    def _model_table(self):
        shape = [8, 8]
        tt = TrainerTensor("mlp", "torch.bfloat16", list(shape), [
            _row_shard(0, 0, 4, shape),
            _row_shard(1, 4, 8, shape),
        ])
        return _table([tt]), math.prod(shape)

    def test_two_workers_independent_plans_same_bytes(self):
        """Each worker gets the full model; independent dst_addr → no collision."""
        table, n = self._model_table()
        expected = n * 2
        for worker in range(2):
            regions = [_region("mlp", n, dst_addr=0x1000 + worker * 0x100000)]
            assert sum(d.nbytes for d in route_regions(regions, table)) == expected

    def test_four_workers_all_get_same_descriptor_structure(self):
        """Routing is deterministic: same table → same descriptor structure."""
        table, n = self._model_table()
        reference = route_regions([_region("mlp", n, dst_addr=0x1000)], table)
        for i in range(1, 4):
            descs = route_regions([_region("mlp", n, dst_addr=0x1000 + i * 0x200000)], table)
            assert len(descs) == len(reference)
            for ref, got in zip(reference, descs):
                assert ref.agent_index == got.agent_index
                assert ref.nbytes == got.nbytes


class TestM2nTP2Routing:
    def _tp2_table(self):
        shape_o = [4, 8]
        tt_o = TrainerTensor("o_proj.weight", "torch.bfloat16", shape_o, [
            _col_shard(0, 0, 4, 0, 4),
            _col_shard(1, 0, 4, 4, 8),
        ])
        shape_q = [8, 4]
        tt_q = TrainerTensor("q_proj.weight", "torch.bfloat16", shape_q, [
            _row_shard(0, 0, 4, shape_q, base=0x30000),
            _row_shard(1, 4, 8, shape_q, base=0x40000),
        ])
        return _table([tt_o, tt_q], agents=[b"tp0", b"tp1"])

    def test_tp2_o_proj_uses_both_agents(self):
        table = self._tp2_table()
        descs = route_regions([_region("o_proj.weight", 4 * 8)], table)
        assert {d.agent_index for d in descs} == {0, 1}

    def test_tp2_q_proj_uses_both_agents(self):
        table = self._tp2_table()
        descs = route_regions([_region("q_proj.weight", 8 * 4, dst_addr=0xB000)], table)
        assert {d.agent_index for d in descs} == {0, 1}

    def test_tp2_full_model_byte_coverage(self):
        table = self._tp2_table()
        regions = [_region("o_proj.weight", 32, dst_addr=0xA000), _region("q_proj.weight", 32, dst_addr=0xB000)]
        assert sum(d.nbytes for d in route_regions(regions, table)) == (32 + 32) * 2

    def test_tp2_col_shard_agent_byte_balance(self):
        """Each TP agent provides exactly half the o_proj bytes."""
        table = self._tp2_table()
        descs = route_regions([_region("o_proj.weight", 32)], table)
        by_agent = {0: 0, 1: 0}
        for d in descs:
            by_agent[d.agent_index] += d.nbytes
        assert by_agent[0] == by_agent[1] == 32


class TestM2nWireProtocol:
    def _descriptors(self, n=10):
        return [
            M2nDescriptor(
                src_agent_index=i % 2, dst_agent_index=i % 3,
                src_addr=0x1000 * i, dst_addr=0x2000 * i, nbytes=64 * (i + 1),
            )
            for i in range(n)
        ]

    def test_empty_roundtrip(self):
        assert decode_m2n_descriptors(encode_m2n_descriptors([])) == []

    def test_single_roundtrip(self):
        descs = self._descriptors(1)
        assert decode_m2n_descriptors(encode_m2n_descriptors(descs)) == descs

    def test_large_roundtrip(self):
        descs = self._descriptors(500)
        assert decode_m2n_descriptors(encode_m2n_descriptors(descs)) == descs

    def test_encoded_is_bytes_type(self):
        assert isinstance(encode_m2n_descriptors(self._descriptors(3)), bytes)

    def test_nbytes_preserved(self):
        descs = [M2nDescriptor(0, 0, 0x1000, 0x2000, 123456)]
        assert decode_m2n_descriptors(encode_m2n_descriptors(descs))[0].nbytes == 123456

    def test_large_addresses_roundtrip(self):
        """64-bit addresses must survive encoding (RDMA addresses can exceed 32 bits)."""
        descs = [M2nDescriptor(0, 0, 0xDEAD_BEEF_CAFE_0000, 0xABCD_1234_5678_9000, 4096)]
        out = decode_m2n_descriptors(encode_m2n_descriptors(descs))
        assert out[0].src_addr == 0xDEAD_BEEF_CAFE_0000
        assert out[0].dst_addr == 0xABCD_1234_5678_9000

    def test_to_rdma_descriptor_conversion(self):
        d = M2nDescriptor(src_agent_index=2, dst_agent_index=7, src_addr=0xA, dst_addr=0xB, nbytes=512)
        rdma = d.to_rdma_descriptor()
        assert rdma.agent_index == 2
        assert rdma.src_addr == 0xA
        assert rdma.dst_addr == 0xB
        assert rdma.nbytes == 512


class TestM2nPlannerE2EPipeline:
    def _setup(self, shape=(8, 16), n_tp_shards=1):
        n = math.prod(shape)
        if n_tp_shards == 2:
            cols = shape[1]
            shards = [
                _col_shard(0, 0, shape[0], 0, cols // 2),
                _col_shard(1, 0, shape[0], cols // 2, cols),
            ]
        else:
            shards = [_row_shard(0, 0, shape[0], list(shape))]
        tt = TrainerTensor("layer", "torch.bfloat16", list(shape), shards)
        table = _table([tt])
        region = _region("layer", n)
        return table, [region], n * 2

    def test_local_planner_full_pipeline(self):
        table, regions, expected_bytes = self._setup()
        descs = LocalPlanner().build(regions, table, "pk")
        assert sum(d.nbytes for d in descs) == expected_bytes

    def test_m2n_planner_build_with_server(self):
        table, regions, _ = self._setup()
        client = _mock_m2n_client([M2nDescriptor(0, 0, 0x1000, 0x2000, 64)])
        planner = M2nPlanner(mx_client=client, model_key="m", worker_rank=0, total_workers=1, nixl_metadata=b"")
        descs = planner.build(regions, table, "pk")
        assert all(isinstance(d, RdmaDescriptor) for d in descs)
        assert len(descs) == 1

    def test_m2n_planner_build_m2n_returns_m2n_type(self):
        table, regions, _ = self._setup()
        client = _mock_m2n_client([M2nDescriptor(0, 0, 0xA000, 0xB000, 128)])
        planner = M2nPlanner(mx_client=client, model_key="m", worker_rank=0, total_workers=1, nixl_metadata=b"")
        descs = planner.build_m2n(regions, table, "pk")
        assert all(isinstance(d, M2nDescriptor) for d in descs)

    def test_m2n_planner_tp2_fallback_covers_all_bytes(self):
        """When the server is down, LocalPlanner fallback correctly routes TP=2 shards."""
        table, regions, expected_bytes = self._setup(shape=(4, 8), n_tp_shards=2)
        client = MagicMock()
        client.register_m2n_worker.side_effect = RuntimeError("no server")
        planner = M2nPlanner(mx_client=client, model_key="m", worker_rank=0, total_workers=2, nixl_metadata=b"")
        assert sum(d.nbytes for d in planner.build(regions, table, "pk")) == expected_bytes

    def test_m2n_planner_multi_worker_barrier_releases_all(self):
        """Server holds barrier until all workers register; all N workers get a plan."""
        N = 4
        plan_id = "global-plan"
        table, regions, _ = self._setup()
        planners = []
        clients = []
        for rank in range(N):
            client = _mock_m2n_client(
                [M2nDescriptor(0, rank, 0x1000, 0x2000 + rank * 0x1000, 64)],
                plan_id=plan_id,
            )
            clients.append(client)
            planners.append(M2nPlanner(
                mx_client=client, model_key="policy",
                worker_rank=rank, total_workers=N,
                nixl_metadata=f"meta{rank}".encode(),
            ))
        for rank, planner in enumerate(planners):
            descs = planner.build_m2n(regions, table, f"pk-{rank}")
            assert len(descs) == 1
            assert descs[0].dst_agent_index == rank
