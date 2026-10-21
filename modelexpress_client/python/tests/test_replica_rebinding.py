# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU simulated-transport coverage for repeated replica value rebinding.

This exercises the existing bounded-transfer fixture and its byte-copy transport;
it is component evidence, not real NIXL or GPU coverage.
"""

from contextlib import nullcontext
import hashlib

import pytest
import torch
import modelexpress_rl.inference.nixl_staged_transfer as transfer_module

from tests.test_refit_bounded_descriptors import harness
from modelexpress_rl.refit.reshard.rendezvous import (
    PublishedShard, PublishedTensor, wrap_rendezvous_blob,
)
from modelexpress_rl.inference.methods import LoadTimeTensorNixlUpdateMethod

from modelexpress_rl.inference.adapter import TrainerSourceShard
from modelexpress_rl.inference.plan import ResolvedTrainerSource, TrainerSourceSnapshot
from modelexpress_rl.version import MeshIdentity


def _initialize_replicas(harness):
    if not hasattr(harness, "replica_sources"):
        harness.replica_sources = {
            name: {replica_id: tensor.clone() for replica_id in ("A", "B")}
            for name, tensor in harness.sources.items()
        }


def _snapshot(harness, replica, cycle):
    agent = f"agent-{replica}"
    tensors = harness.replica_sources
    blob = wrap_rendezvous_blob(
        agent.encode(), agent, f"{replica}:19000",
        [PublishedTensor(
            name=name, dtype="torch.float32", elsize=4, full_shape=(4, 4),
            shards=[PublishedShard(agent_name=agent, device_id=0, addr=tensor[replica].data_ptr(),
                                   shard_offset=(0, 0), shape=(4, 4))],
        ) for name, tensor in tensors.items()],
    )
    shard = TrainerSourceShard(
        "slot", replica, hashlib.sha256(blob).hexdigest(), f"{replica}:19000", blob,
    )
    return TrainerSourceSnapshot(MeshIdentity("mesh", 1), (shard,))


def _prepare(method, source, cycle):
    if hasattr(method, "prepare_streaming"):
        return method.prepare_streaming(
            version=cycle,
            source=source,
            max_staging_bytes=1024,
            staging_device="cpu",
        )
    return method.prepare(version=cycle, source=source)


def _load_replica(harness, method, replica: str, cycle: int):
    _initialize_replicas(harness)
    # Separate, persistent source allocations make A/B rebinding observable.
    base = 10.0 if replica == "A" else 100.0
    for index, (name, tensor) in enumerate(
        (name, sources[replica]) for name, sources in harness.replica_sources.items()
    ):
        tensor.copy_(torch.arange(16, dtype=torch.float32).reshape(4, 4) + base + cycle + index)
    raw_source = _snapshot(harness, replica, cycle)
    source = ResolvedTrainerSource(raw_source)
    prepared = _prepare(method, source, cycle)
    try:
        return {name: value.clone() for batch in prepared.batches() for name, value in batch.items()}
    finally:
        method.release(prepared)


def _assert_replica_values(installed, replica, cycle):
    offset = (10.0 if replica == "A" else 100.0) + cycle
    for name, index, dtype, transpose in (
        ("a.weight", 0, torch.float32, False),
        ("b.weight", 1, torch.float32, True),
        ("c.weight", 2, torch.bfloat16, False),
        ("d.weight", 0, torch.float32, False),
    ):
        source = torch.arange(16, dtype=torch.float32).reshape(4, 4) + offset + index
        if transpose:
            source = source.T
        expected = torch.zeros(20, dtype=dtype)
        expected[2:18].copy_(source.to(dtype).reshape(-1))
        assert torch.equal(installed[name], expected), f"stale or incorrect value for {name}"


def _setup_method(harness, monkeypatch):
    monkeypatch.setattr(transfer_module, "classic_cuda_alloc", nullcontext)
    method = LoadTimeTensorNixlUpdateMethod(transfer=harness.transfer)
    if not hasattr(method, "prepare_streaming"):
        monkeypatch.setattr(harness.transfer, "_device", torch.device("cpu"))
    return method


@pytest.mark.parametrize("sequence", ["AA", "AB", "ABA", "ABAB", "AABBAB"])
def test_replica_value_rebinding_sequence(harness, monkeypatch, sequence):
    """Every cycle must install that replica's current values, including returns."""
    method = _setup_method(harness, monkeypatch)
    for cycle, replica in enumerate(sequence, start=1):
        installed = _load_replica(
            harness, method, replica, cycle
        )
        _assert_replica_values(installed, replica, cycle)


def test_replica_failure_recovers_with_current_values(harness, monkeypatch):
    method = _setup_method(harness, monkeypatch)
    _load_replica(harness, method, "A", 1)
    _initialize_replicas(harness)
    for index, (name, tensor) in enumerate(
        (name, sources["B"]) for name, sources in harness.replica_sources.items()
    ):
        tensor.fill_(102.0 + index)
    raw_source = _snapshot(harness, "B", 2)
    source = ResolvedTrainerSource(raw_source) if ResolvedTrainerSource is not None else raw_source
    prepared = _prepare(method, source, 2)
    # The fixture transport is created during prepare and acts as the simulated wire.
    prepared_transport = harness.transports[-1]
    prepared_transport.fail_post = True
    with pytest.raises(RuntimeError):
        next(prepared.batches())
    prepared_transport.fail_post = False
    method.release(prepared)
    installed = _load_replica(harness, method, "B", 3)
    _assert_replica_values(installed, "B", 3)
