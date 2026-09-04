# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""The throughput floor on the staged FSDP receiver.

There are two receivers over one transport. ``test_reshard_refit_floor`` covers
the Megatron one; this file covers the staged one, which is what the FSDP
trainers refit over and what the collapse that motivated the floor was actually
measured on. The floor first shipped covering only the Megatron receiver, so an
operator could set ``MX_RESHARD_MIN_GBPS`` on an FSDP job, breach it by 20x, and
get silence - the guard was configurable but unreachable. These tests exist so
that cannot come back.
"""

from __future__ import annotations

import json
import logging
from contextlib import nullcontext

import pytest
import torch

import modelexpress_rl.inference.nixl_staged_transfer as transfer_module
from modelexpress import p2p_pb2
from modelexpress.refit.reshard import throughput
from modelexpress.refit.reshard.slice_plan import Shard
from modelexpress.refit.reshard.transfer_plan import SourceInfo, TransferPlan
from modelexpress.refit.reshard.types import CaptureResult, RecordedCopy
from modelexpress.refit.reshard.verify import tensor_digest
from modelexpress_rl.inference.nixl_staged_transfer import (
    _NixlStagedTransfer,
    _PreparedNixlTransfer,
)

# The measured incident and the measured healthy run on the same node: 16.64 GB
# moved at ~1.5 GB/s per reader when four converged on one rail, against ~26 GB/s
# when they were spread. A floor between them separates the two.
SLOW_BYTES = 16_640_000_000
SLOW_WIRE_S = 10.741
FAST_WIRE_S = 0.632
FLOOR_GBPS = 50.0
DEVICE_ID = 3


class _Clock:
    """A perf_counter that hands out a scripted sequence.

    The rate under test is bytes over a measured span, so a real clock would make
    the assertion depend on how fast the stub transport happens to return.
    """

    def __init__(self, *stamps: float) -> None:
        self._stamps = iter(stamps)

    def perf_counter(self) -> float:
        return next(self._stamps)


class _Transport:
    def __init__(self) -> None:
        self.reads: list = []

    def read(self, descriptors) -> None:
        self.reads.append(descriptors)


class _Descriptor:
    def __init__(self, nbytes: int) -> None:
        self.nbytes = nbytes


def _floor(monkeypatch, value) -> None:
    if value is None:
        monkeypatch.delenv("MX_RESHARD_MIN_GBPS", raising=False)
    else:
        monkeypatch.setenv("MX_RESHARD_MIN_GBPS", str(value))


def _prepared(tensor: torch.Tensor, nbytes: int) -> _PreparedNixlTransfer:
    copy = RecordedCopy(
        src_name="weight",
        op_chain=(),
        param_name="weight",
        dest_offset=0,
        dest_shape=tuple(tensor.shape),
        dest_stride=tuple(tensor.stride()),
        dest_dtype=tensor.dtype,
    )
    source = SourceInfo(
        global_shape=tuple(tensor.shape),
        dtype=tensor.dtype,
        elsize=tensor.element_size(),
        shards=[
            Shard(
                shard_offset=(0,),
                shape=tuple(tensor.shape),
                session="trainer",
                addr=0,
                elsize=tensor.element_size(),
                digest=tensor_digest(tensor),
            )
        ],
    )
    return _PreparedNixlTransfer(
        plan=TransferPlan(),
        capture=CaptureResult(copies=[copy]),
        sources={"weight": source},
        descriptors=(_Descriptor(nbytes),),
        transport=_Transport(),
    )


def _stage(monkeypatch, *, nbytes: int, wire_s: float):
    """Run stage() to completion on CPU with a scripted wire duration."""
    monkeypatch.setattr(
        transfer_module,
        "time",
        _Clock(0.0, wire_s, wire_s, wire_s),
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    tensor = torch.arange(64, dtype=torch.int32)
    prepared = _prepared(tensor, nbytes)

    transfer = object.__new__(_NixlStagedTransfer)
    transfer._closed = False
    transfer._device = torch.device("cpu")
    transfer._device_id = DEVICE_ID
    transfer._recv_buffers = {"weight": tensor}
    transfer._convert_buffers = {}
    transfer._full_buffers = {}
    transfer._active = prepared
    return transfer.stage(prepared)


def _peer_stage(monkeypatch, *, nbytes: int, wire_s: float):
    """Run stage_peer() with the manager reporting a scripted transfer."""

    class _Manager:
        def register_tensors(self, tensors) -> None:
            pass

        def add_remote_agent(self, metadata) -> str:
            return "peer-agent"

        def receive_from_source(self, **kwargs):
            return nbytes, 1, wire_s

        def remove_remote_agent(self, agent_name) -> None:
            pass

    monkeypatch.setattr(transfer_module, "classic_cuda_alloc", nullcontext)
    transfer = object.__new__(_NixlStagedTransfer)
    transfer._closed = False
    transfer._device = torch.device("cpu")
    transfer._device_id = DEVICE_ID
    transfer._timeout = 30.0
    transfer._manager = _Manager()
    transfer._recv_buffers = {}
    transfer._registered_recv_params = set()
    transfer._published_peer_rank = None
    transfer._active = None
    source = p2p_pb2.WorkerMetadata(
        nixl_metadata=b"peer-metadata",
        tensors=[
            p2p_pb2.TensorDescriptor(
                name="weight",
                addr=1234,
                size=16,
                device_id=0,
                dtype="torch.float32",
            )
        ],
    )
    return transfer.stage_peer(
        source=source,
        parameter_layout={"weight": ((4,), torch.float32)},
    )


def _record(caplog) -> dict:
    line = next(m for m in caplog.messages if "MX_REFIT_SLOW_THROUGHPUT" in m)
    return json.loads(line.split("MX_REFIT_SLOW_THROUGHPUT ", 1)[1])


def _records(caplog) -> list:
    return [m for m in caplog.messages if "MX_REFIT_SLOW_THROUGHPUT" in m]


def test_the_fsdp_pull_reports_a_slow_refit(monkeypatch, caplog):
    """The test that would have caught the gap.

    This is the call path an FSDP generator takes every refit. Before the floor
    reached it, this exact scenario -- 12.4 Gbps against a 50 Gbps floor -- logged
    nothing at all.
    """
    _floor(monkeypatch, FLOOR_GBPS)
    with caplog.at_level(logging.WARNING):
        staged = _stage(monkeypatch, nbytes=SLOW_BYTES, wire_s=SLOW_WIRE_S)

    payload = _record(caplog)
    assert payload["schema"] == throughput.SLOW_THROUGHPUT_SCHEMA
    assert payload["wire_bytes"] == SLOW_BYTES
    assert payload["implied_gbps"] == pytest.approx(12.4, abs=0.2)
    assert payload["floor_gbps"] == FLOOR_GBPS
    # A slow refit is still a correct refit, so the weights must still come back.
    assert staged.metrics["bytes_received"] == SLOW_BYTES


def test_the_report_names_the_device_that_saw_it(monkeypatch, caplog):
    """The collapse hit some readers on a node and not others, because it was
    decided per GPU by which rail that GPU's process picked. A record that cannot
    be attributed to a device cannot describe the incident it exists to report."""
    _floor(monkeypatch, FLOOR_GBPS)
    with caplog.at_level(logging.WARNING):
        _stage(monkeypatch, nbytes=SLOW_BYTES, wire_s=SLOW_WIRE_S)

    payload = _record(caplog)
    assert payload["device_id"] == DEVICE_ID
    assert payload["phase"] == "stage"


def test_the_healthy_fsdp_pull_is_silent(monkeypatch, caplog):
    """The other half of the guard: 16.64 GB in 0.632 s is 210 Gbps, and a floor
    that fires on the spread case is a floor someone turns off."""
    _floor(monkeypatch, FLOOR_GBPS)
    with caplog.at_level(logging.WARNING):
        _stage(monkeypatch, nbytes=SLOW_BYTES, wire_s=FAST_WIRE_S)

    assert not _records(caplog)


def test_the_floor_is_off_unless_configured(monkeypatch, caplog):
    """Only the operator knows what their fabric should deliver, so absent config
    the staged path stays out of the way exactly as the Megatron one does."""
    _floor(monkeypatch, None)
    with caplog.at_level(logging.WARNING):
        _stage(monkeypatch, nbytes=SLOW_BYTES, wire_s=SLOW_WIRE_S)

    assert not _records(caplog)


def test_the_peer_pull_is_covered_too(monkeypatch, caplog):
    """A generator can refit from a peer instead of the trainer. Those bytes cross
    the same rails, so covering only the trainer pull would leave a whole refit
    path silent on the failure the floor is for."""
    _floor(monkeypatch, FLOOR_GBPS)
    with caplog.at_level(logging.WARNING):
        _peer_stage(monkeypatch, nbytes=SLOW_BYTES, wire_s=SLOW_WIRE_S)

    payload = _record(caplog)
    assert payload["phase"] == "stage_peer"
    assert payload["implied_gbps"] == pytest.approx(12.4, abs=0.2)


def test_a_healthy_peer_pull_is_silent(monkeypatch, caplog):
    _floor(monkeypatch, FLOOR_GBPS)
    with caplog.at_level(logging.WARNING):
        _peer_stage(monkeypatch, nbytes=SLOW_BYTES, wire_s=FAST_WIRE_S)

    assert not _records(caplog)


def test_both_receivers_report_the_same_marker_and_schema(monkeypatch, caplog):
    """The anti-drift test.

    A CI gate greps one marker string. If the two receivers each carried their own
    copy of this record, the same regression could fail one path and pass the
    other, and the gate would read green for the wrong reason. Both must come from
    the shared implementation.
    """
    _floor(monkeypatch, FLOOR_GBPS)
    from modelexpress.refit.reshard import receiver

    class _Rig:
        _global_rank = 3

    with caplog.at_level(logging.WARNING):
        _stage(monkeypatch, nbytes=SLOW_BYTES, wire_s=SLOW_WIRE_S)
        staged_payload = _record(caplog)
        caplog.clear()
        receiver.ReshardReceiver._check_throughput_floor(
            _Rig(), 1, SLOW_BYTES, {"wire_fused_s": SLOW_WIRE_S}
        )
        megatron_payload = _record(caplog)

    assert staged_payload["schema"] == megatron_payload["schema"]
    # The quantitative fields a gate would threshold on must be named identically
    # on both paths, or a gate written against one silently ignores the other.
    shared = {
        "schema",
        "wire_bytes",
        "wire_s",
        "implied_gbps",
        "floor_gbps",
        "shortfall_x",
    }
    assert shared <= set(staged_payload)
    assert shared <= set(megatron_payload)
    for key in shared:
        assert staged_payload[key] == megatron_payload[key]
