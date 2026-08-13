# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Fused vs phased wire in ReshardReceiver.update_weights - CPU only, no NIXL.

A refit reads three groups of bytes: exact segments straight into the receive
buffers, whole sources into full-pull staging, and dtype-mismatched sources into
convert staging. Those used to be three separate drains. They target disjoint
destinations and every reader of those destinations runs after all three, so
they are now issued as one batch.

The tests that matter here are the two that would catch a wrong fusion:

  * the receive buffers are byte-identical whether the reads are fused or
    phased, checked against a plan that exercises all three groups at once;
  * the re-slice and the dtype cast run after the wire, not interleaved with it,
    which is the ordering assumption fusing depends on.

The in-memory reference transport performs real byte moves over CPU addresses,
so a fusion that mixes up a destination address fails rather than passes.

Run: pytest tests/test_reshard_refit_fused_wire.py
"""

import torch

from modelexpress.refit.reshard.receiver import ReshardReceiver
from modelexpress.refit.reshard.slice_plan import PullSegment
from modelexpress.refit.reshard.transfer_plan import (
    ConvertSource,
    FullPullSource,
    TransferPlan,
)
from modelexpress.refit.reshard.transport import InMemoryReferenceTransport

EL = 4  # float32 element size


class _RecordingTransport(InMemoryReferenceTransport):
    """Real byte moves, plus the batch boundaries so ordering can be asserted."""

    def __init__(self) -> None:
        super().__init__()
        self.batch_sizes: list[int] = []
        self.installed_after: list[int] = []

    def read(self, descriptors) -> None:
        self.batch_sizes.append(len(descriptors))
        super().read(descriptors)


class _Harness(ReshardReceiver):
    """A receiver with the plan and buffers set directly.

    ``ReshardReceiver.__init__`` builds a NIXL agent and a metadata client, so
    this bypasses it. Only the state ``update_weights`` touches is populated.
    """

    def __init__(self, transport) -> None:  # noqa: D107 - see class docstring
        self._device = torch.device("cpu")
        self._timeout = 1.0
        self._transport = transport
        self._global_rank = 0
        self._cached_descriptors = None
        self._install_order: list[str] = []

    def _install(self, recv_buffers) -> None:
        self._install_order.append("install")
        self._transport.installed_after.append(len(self._transport.batch_sizes))


def _build(transport):
    """One plan covering all three read groups.

    ``exact`` is read straight into its receive buffer. ``strided`` is a whole
    source staged contiguously and then re-sliced locally. ``router`` is served
    bf16 for an fp32 destination, so it stages at the served dtype and is cast
    afterwards. Returns the harness plus the source tensors, which the caller
    must keep alive: the plan holds their raw addresses.
    """
    harness = _Harness(transport)

    exact_src = torch.arange(8, dtype=torch.float32)
    strided_src = torch.arange(100, 116, dtype=torch.float32).reshape(4, 4)
    router_src = torch.arange(200, 204, dtype=torch.float32).to(torch.bfloat16)

    recv_exact = torch.zeros(8, dtype=torch.float32)
    recv_strided = torch.zeros(4, 2, dtype=torch.float32)
    recv_router = torch.zeros(4, dtype=torch.float32)
    full_staging = torch.zeros(4, 4, dtype=torch.float32)
    convert_staging = torch.zeros(4, dtype=torch.bfloat16)

    harness._recv_buffers = {
        "exact": recv_exact,
        "strided": recv_strided,
        "router": recv_router,
    }
    harness._param_ptr = {"exact": recv_exact.data_ptr()}
    harness._full_staging = {"strided": full_staging}
    harness._full_staging_ptr = {"strided": full_staging.data_ptr()}
    harness._staging = {"router": convert_staging}
    harness._staging_ptr = {"router": convert_staging.data_ptr()}

    plan = TransferPlan(
        segments=[
            PullSegment(
                session="s0",
                src_addr=exact_src.data_ptr(),
                dst_byte=0,
                nbytes=8 * EL,
                param_name="exact",
            )
        ],
        full_pulls=[
            FullPullSource(
                src_name="strided",
                global_shape=(4, 4),
                dtype=torch.float32,
                elsize=EL,
                segments=[
                    PullSegment(
                        session="s1",
                        src_addr=strided_src.data_ptr(),
                        dst_byte=0,
                        nbytes=16 * EL,
                        param_name="strided",
                    )
                ],
                # Take the first two columns, the shape a tensor-parallel row
                # split produces and the reason this source is descriptor-heavy.
                copies=[
                    _Copy(
                        param_name="strided",
                        op_chain=(("narrow", (1, 0, 2), ()),),
                        dest_shape=(4, 2),
                        dest_stride=(2, 1),
                        dest_offset=0,
                    )
                ],
            )
        ],
        converts=[
            ConvertSource(
                param_name="router",
                dest_shape=(4,),
                src_dtype=torch.bfloat16,
                segments=[
                    PullSegment(
                        session="s2",
                        src_addr=router_src.data_ptr(),
                        dst_byte=0,
                        nbytes=4 * 2,
                        param_name="router",
                    )
                ],
            )
        ],
        exact_bytes=8 * EL,
        exact_descriptor_count=1,
    )
    harness._plan = plan
    keepalive = (exact_src, strided_src, router_src)
    return harness, keepalive


class _Copy:
    """The subset of ``RecordedCopy`` the re-slice reads."""

    def __init__(self, *, param_name, op_chain, dest_shape, dest_stride, dest_offset):
        self.param_name = param_name
        self.op_chain = op_chain
        self.dest_shape = dest_shape
        self.dest_stride = dest_stride
        self.dest_offset = dest_offset


def _run(monkeypatch, *, fused: bool):
    monkeypatch.setenv("MX_RESHARD_FUSED_WIRE", "1" if fused else "0")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    transport = _RecordingTransport()
    harness, keepalive = _build(transport)
    metrics = harness.update_weights(step=1)
    return harness, transport, metrics, keepalive


def test_fused_issues_one_batch_and_phased_issues_three(monkeypatch):
    _h, fused_transport, _m, _k = _run(monkeypatch, fused=True)
    assert fused_transport.batch_sizes == [3]

    _h, phased_transport, _m, _k = _run(monkeypatch, fused=False)
    assert phased_transport.batch_sizes == [1, 1, 1]


def test_fused_and_phased_produce_identical_buffers(monkeypatch):
    """The correctness gate on fusing: same bytes, same destinations."""
    fused, _t, fused_metrics, _k = _run(monkeypatch, fused=True)
    phased, _t, phased_metrics, _k = _run(monkeypatch, fused=False)

    assert fused._recv_buffers.keys() == phased._recv_buffers.keys()
    for name, buffer in fused._recv_buffers.items():
        assert torch.equal(buffer, phased._recv_buffers[name]), name

    # And the accounting agrees, so a dashboard cannot tell the paths apart.
    assert fused_metrics["bytes_received"] == phased_metrics["bytes_received"]
    assert fused_metrics["segments"] == phased_metrics["segments"]


def test_reconstructs_the_expected_values(monkeypatch):
    """Buffer parity between two wrong paths would still pass, so check values."""
    harness, _t, _m, _k = _run(monkeypatch, fused=True)

    assert torch.equal(
        harness._recv_buffers["exact"], torch.arange(8, dtype=torch.float32)
    )
    # First two columns of the 4x4 source, re-sliced locally after the full pull.
    expected = torch.arange(100, 116, dtype=torch.float32).reshape(4, 4)[:, :2]
    assert torch.equal(harness._recv_buffers["strided"], expected)
    # Served bf16, cast up to the fp32 destination after the wire.
    assert torch.equal(
        harness._recv_buffers["router"],
        torch.arange(200, 204, dtype=torch.float32).to(torch.bfloat16).float(),
    )


def test_install_runs_after_every_read(monkeypatch):
    fused, fused_transport, _m, _k = _run(monkeypatch, fused=True)
    assert fused._install_order == ["install"]
    assert fused_transport.batch_sizes == [3]
    assert fused_transport.installed_after == [1]

    phased, phased_transport, _m, _k = _run(monkeypatch, fused=False)
    assert phased._install_order == ["install"]
    assert phased_transport.batch_sizes == [1, 1, 1]
    assert phased_transport.installed_after == [3]


def test_accounting_covers_all_three_groups(monkeypatch):
    _h, _t, metrics, _k = _run(monkeypatch, fused=True)

    assert metrics["segments"] == 3
    assert metrics["bytes_received"] == 8 * EL + 16 * EL + 4 * 2
    assert metrics["full_pull_sources"] == 1
    assert metrics["converts"] == 1
    assert metrics["fallback"] == 0


def test_empty_full_pull_and_convert_groups_are_skipped(monkeypatch):
    """A plan with only exact segments must still issue exactly one batch."""
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)

    for fused in (True, False):
        monkeypatch.setenv("MX_RESHARD_FUSED_WIRE", "1" if fused else "0")
        transport = _RecordingTransport()
        harness, keepalive = _build(transport)
        harness._plan.full_pulls = []
        harness._plan.converts = []

        harness.update_weights(step=1)

        assert transport.batch_sizes == [1]
        assert all(t.data_ptr() for t in keepalive)  # keep alive
