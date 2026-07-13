# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for RDMA strategy source selection helpers."""

from types import SimpleNamespace

from modelexpress.load_strategy.rdma_strategy import RdmaStrategy


def test_rdma_strategy_skips_local_fixed_worker_endpoint(monkeypatch):
    monkeypatch.setenv("POD_IP", "10.1.2.3")
    monkeypatch.setenv("MX_WORKER_GRPC_PORT", "6555")
    monkeypatch.setenv("MX_METADATA_PORT", "7555")
    ctx = SimpleNamespace(device_id=2)
    worker = SimpleNamespace(
        worker_grpc_endpoint="10.1.2.3:6557",
        metadata_endpoint="10.1.2.3:7557",
    )

    assert RdmaStrategy._is_local_worker_metadata(ctx, worker)


def test_rdma_strategy_keeps_remote_worker_endpoint(monkeypatch):
    monkeypatch.setenv("POD_IP", "10.1.2.3")
    monkeypatch.setenv("MX_WORKER_GRPC_PORT", "6555")
    monkeypatch.setenv("MX_METADATA_PORT", "7555")
    ctx = SimpleNamespace(device_id=2)
    worker = SimpleNamespace(
        worker_grpc_endpoint="10.9.8.7:6557",
        metadata_endpoint="10.9.8.7:7557",
    )

    assert not RdmaStrategy._is_local_worker_metadata(ctx, worker)
