# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Batched vs per-view re-slice in ReshardReceiver.update_weights - CPU, no NIXL.

A full-pulled source is staged whole and then re-sliced locally into the receive
buffers, one copy per view the loader recorded. On a real model that is thousands
of views, and thousands of individual ``copy_()`` launches cost enough Python and
launch overhead to rival the RDMA they follow. The copies are now collected and
issued as a single ``torch._foreach_copy_``.

The batching is only safe because the destinations are disjoint, so the tests
that matter are the ones that would catch a wrong batch:

  * every receive buffer is byte-identical between the batched and per-view
    paths, over a plan with several sources and several views each;
  * the values are the ones the op chains actually describe, so two paths that
    are wrong in the same way still fail;
  * overlapping views, where batching would be order-dependent, are not silently
    reordered.

Run: pytest tests/test_reshard_refit_batch_install.py
"""

import torch

from modelexpress.refit.reshard.receiver import ReshardReceiver
from modelexpress.refit.reshard.slice_plan import PullSegment
from modelexpress.refit.reshard.transfer_plan import FullPullSource, TransferPlan
from modelexpress.refit.reshard.transport import InMemoryReferenceTransport

EL = 4  # float32 element size


class _Copy:
    """The subset of ``RecordedCopy`` the re-slice reads."""

    def __init__(self, *, param_name, op_chain, dest_shape, dest_stride, dest_offset):
        self.param_name = param_name
        self.op_chain = op_chain
        self.dest_shape = dest_shape
        self.dest_stride = dest_stride
        self.dest_offset = dest_offset


class _Harness(ReshardReceiver):
    """A receiver with the plan and buffers set directly.

    ``ReshardReceiver.__init__`` builds a NIXL agent and a metadata client, so
    this bypasses it. Only the state ``update_weights`` touches is populated.
    """

    def __init__(self, transport) -> None:
        self._device = torch.device("cpu")
        self._timeout = 1.0
        self._transport = transport
        self._global_rank = 0
        self._cached_descriptors = None

    def _install(self, recv_buffers) -> None:
        pass


def _build(transport):
    """Two full-pulled sources, four views in total.

    ``qkv`` is a 4x6 source re-sliced into three column blocks, the shape a
    fused QKV projection produces under tensor parallelism. ``rows`` is a 4x4
    source re-sliced into two row blocks. Four views is enough that the batched
    path passes a real list to ``_foreach_copy_`` and a mis-paired
    destination/source shows up as wrong bytes. Returns the harness plus the
    source tensors, which the caller must keep alive: the plan holds their raw
    addresses.
    """
    harness = _Harness(transport)

    qkv_src = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    rows_src = torch.arange(100, 116, dtype=torch.float32).reshape(4, 4)

    recv_q = torch.zeros(4, 2, dtype=torch.float32)
    recv_k = torch.zeros(4, 2, dtype=torch.float32)
    recv_v = torch.zeros(4, 2, dtype=torch.float32)
    recv_rows = torch.zeros(4, 4, dtype=torch.float32)
    qkv_staging = torch.zeros(4, 6, dtype=torch.float32)
    rows_staging = torch.zeros(4, 4, dtype=torch.float32)

    harness._recv_buffers = {
        "q": recv_q,
        "k": recv_k,
        "v": recv_v,
        "rows": recv_rows,
    }
    harness._param_ptr = {}
    harness._full_staging = {"qkv": qkv_staging, "rows": rows_staging}
    harness._full_staging_ptr = {
        "qkv": qkv_staging.data_ptr(),
        "rows": rows_staging.data_ptr(),
    }
    harness._staging = {}
    harness._staging_ptr = {}

    plan = TransferPlan(
        segments=[],
        full_pulls=[
            FullPullSource(
                src_name="qkv",
                global_shape=(4, 6),
                dtype=torch.float32,
                elsize=EL,
                segments=[
                    PullSegment(
                        session="s0",
                        src_addr=qkv_src.data_ptr(),
                        dst_byte=0,
                        nbytes=24 * EL,
                        param_name="qkv",
                    )
                ],
                copies=[
                    _Copy(
                        param_name=name,
                        op_chain=(("narrow", (1, start, 2), ()),),
                        dest_shape=(4, 2),
                        dest_stride=(2, 1),
                        dest_offset=0,
                    )
                    for name, start in (("q", 0), ("k", 2), ("v", 4))
                ],
            ),
            FullPullSource(
                src_name="rows",
                global_shape=(4, 4),
                dtype=torch.float32,
                elsize=EL,
                segments=[
                    PullSegment(
                        session="s1",
                        src_addr=rows_src.data_ptr(),
                        dst_byte=0,
                        nbytes=16 * EL,
                        param_name="rows",
                    )
                ],
                # Two row blocks into one buffer, so the destination offset is
                # exercised as well as the shape.
                copies=[
                    _Copy(
                        param_name="rows",
                        op_chain=(("narrow", (0, start, 2), ()),),
                        dest_shape=(2, 4),
                        dest_stride=(4, 1),
                        dest_offset=start * 4,
                    )
                    for start in (0, 2)
                ],
            ),
        ],
        converts=[],
        exact_bytes=0,
        exact_descriptor_count=0,
    )
    harness._plan = plan
    keepalive = (qkv_src, rows_src)
    return harness, keepalive


def _run(monkeypatch, *, batched: bool):
    monkeypatch.setenv("MX_RESHARD_BATCH_INSTALL", "1" if batched else "0")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    transport = InMemoryReferenceTransport()
    harness, keepalive = _build(transport)
    metrics = harness.update_weights(step=1)
    return harness, metrics, keepalive


def test_batched_and_per_view_produce_identical_buffers(monkeypatch):
    """The correctness gate on batching: same bytes, same destinations."""
    batched, batched_metrics, _k = _run(monkeypatch, batched=True)
    per_view, per_view_metrics, _k = _run(monkeypatch, batched=False)

    assert batched._recv_buffers.keys() == per_view._recv_buffers.keys()
    for name, buffer in batched._recv_buffers.items():
        assert torch.equal(buffer, per_view._recv_buffers[name]), name

    # And the accounting agrees, so a dashboard cannot tell the paths apart.
    assert batched_metrics["bytes_received"] == per_view_metrics["bytes_received"]
    assert batched_metrics["segments"] == per_view_metrics["segments"]


def test_reconstructs_the_expected_values(monkeypatch):
    """Buffer parity between two wrong paths would still pass, so check values."""
    qkv = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    rows = torch.arange(100, 116, dtype=torch.float32).reshape(4, 4)

    for batched in (True, False):
        harness, _m, _k = _run(monkeypatch, batched=batched)
        for name, start in (("q", 0), ("k", 2), ("v", 4)):
            assert torch.equal(
                harness._recv_buffers[name], qkv[:, start : start + 2]
            ), f"{name} batched={batched}"
        assert torch.equal(harness._recv_buffers["rows"], rows), f"batched={batched}"


def test_disjoint_destinations_use_the_foreach_batch(monkeypatch):
    """The normal plan is pairwise disjoint and stays on the batched path."""
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    monkeypatch.setenv("MX_RESHARD_BATCH_INSTALL", "1")
    calls = []
    real_foreach = torch._foreach_copy_

    def record(destinations, sources):
        calls.append((destinations, sources))
        return real_foreach(destinations, sources)

    monkeypatch.setattr(torch, "_foreach_copy_", record)
    harness, keepalive = _build(InMemoryReferenceTransport())
    harness.update_weights(step=1)

    assert len(calls) == 1
    assert len(calls[0][0]) == 5
    assert all(t.data_ptr() for t in keepalive)


def test_overlapping_destinations_fall_back_to_plan_order(monkeypatch):
    """Overlapping views use sequential copies because foreach order is undefined."""
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    monkeypatch.setenv("MX_RESHARD_BATCH_INSTALL", "1")
    monkeypatch.setattr(
        torch,
        "_foreach_copy_",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("overlapping destinations must not be batched")
        ),
    )
    harness, keepalive = _build(InMemoryReferenceTransport())
    for copy in harness._plan.full_pulls[1].copies:
        copy.dest_offset = 0

    harness.update_weights(step=1)

    expected = torch.arange(108, 116, dtype=torch.float32).reshape(2, 4)
    assert torch.equal(harness._recv_buffers["rows"][:2], expected)
    assert all(t.data_ptr() for t in keepalive)


def test_stage_record_reports_the_install_arm(monkeypatch, caplog):
    """A captured record must say which arm produced it, or it is unattributable."""
    import json
    import logging

    monkeypatch.setenv("MX_REFIT_STAGE_RECORD", "1")
    for batched in (True, False):
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            _run(monkeypatch, batched=batched)
        records = [
            json.loads(message.split("MX_REFIT_STAGE ", 1)[1])
            for message in caplog.messages
            if "MX_REFIT_STAGE " in message
        ]
        assert records, f"no stage record emitted (batched={batched})"
        assert records[-1]["batch_install"] is batched
        # The re-slice is attributed either way, so an A/B can be compared.
        assert "reslice_s" in records[-1]
        # Views, not sources: five views over two sources. Both arms must agree,
        # or the launch count batching removes cannot be read off the records.
        assert records[-1]["reslice_copies"] == 5
        assert records[-1]["full_pull_sources"] == 2


def test_empty_full_pulls_is_a_no_op(monkeypatch):
    """No full pulls means no batched copy, and no crash on an empty list."""
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)

    for batched in (True, False):
        monkeypatch.setenv("MX_RESHARD_BATCH_INSTALL", "1" if batched else "0")
        transport = InMemoryReferenceTransport()
        harness, keepalive = _build(transport)
        harness._plan.full_pulls = []

        metrics = harness.update_weights(step=1)

        assert metrics["full_pull_sources"] == 0
        assert all(t.data_ptr() for t in keepalive)  # keep alive
