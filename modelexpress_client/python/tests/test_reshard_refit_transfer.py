# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Transfer plan + pull end-to-end CORRECTNESS test.

The strongest check in the pipeline: reconstruct inference params purely from
sharded sources via capture -> plan -> pull, and assert they equal the ground
truth the engine's own loader produces from the full tensors. Uses the
in-memory reference transport (real ctypes byte moves over CPU addresses), so a
plan that reconstructs correctly here reconstructs correctly on the wire.

Exercises the same layouts as the geometry test: column block (contiguous),
row-parallel column slice (strided, multi-run), fused qkv (per-shard offsets),
and full copy - plus unsupported-op detection.

Run: pytest tests/test_reshard_refit_transfer.py
"""

import torch

from modelexpress.refit.reshard.geometry import capture_geometry
from modelexpress.refit.reshard.slice_plan import Shard
from modelexpress.refit.reshard.transfer_plan import (
    SourceInfo,
    execute_transfer,
    plan_transfer,
)
from modelexpress.refit.reshard.transport import InMemoryReferenceTransport

# Reuse the ToyModel + manifest from the geometry test (same package, same dir).
from tests.test_reshard_refit_geometry import ToyModel, _manifest

EL = 4  # float32


class _RecordingTransport(InMemoryReferenceTransport):
    """Reference transport that also records which sessions it was asked to read
    from, so a test can assert that an unsupported source is never fetched."""

    def __init__(self) -> None:
        super().__init__()
        self.sessions: set = set()

    def read(self, descriptors: list) -> None:
        self.sessions.update(d.session for d in descriptors if d.nbytes)
        super().read(descriptors)


def _full_sources():
    """Distinct-valued full source tensors, one per manifest entry."""
    shapes = {name: shape for name, _dtype, shape in _manifest()}
    srcs = {}
    base = 0.0
    for name, shape in shapes.items():
        n = 1
        for s in shape:
            n *= s
        srcs[name] = (
            (base + torch.arange(n, dtype=torch.float32)).reshape(shape).contiguous()
        )
        base += n  # keep value ranges disjoint across tensors
    return srcs


def test_reshard_reconstructs_ground_truth():
    srcs = _full_sources()

    # Ground truth: the engine's own loader run on the FULL tensors.
    truth_model = ToyModel()
    truth_model.load_weights(list(srcs.items()))
    truth = {name: p.detach().clone() for name, p in truth_model.named_parameters()}

    # Reconstruct target: zero it, then fill only via the reshard pull.
    recon_model = ToyModel()
    for p in recon_model.parameters():
        torch.nn.init.zeros_(p)
    recon_params = dict(recon_model.named_parameters())

    # Capture geometry on a disposable meta twin (no storage touched).
    with torch.device("meta"):
        meta_model = ToyModel()
    capture = capture_geometry(meta_model, _manifest())

    # Publish each full source as a single contiguous shard covering the whole
    # tensor (shard buffer IS the source tensor; addr = its data_ptr()).
    sources = {}
    for name, tensor in srcs.items():
        shard = Shard(
            shard_offset=(0,) * tensor.dim(),
            shape=tuple(tensor.shape),
            session=name,
            addr=tensor.data_ptr(),
            elsize=EL,
        )
        sources[name] = SourceInfo(
            global_shape=tuple(tensor.shape),
            dtype=torch.float32,
            elsize=EL,
            shards=[shard],
        )

    plan = plan_transfer(capture, sources)
    assert plan.fallback == []
    assert plan.bytes_planned() > 0

    stats = execute_transfer(
        plan,
        resolve_param_ptr=lambda name: recon_params[name].data_ptr(),
        transport=InMemoryReferenceTransport(),
    )
    assert stats["bytes"] == plan.bytes_planned()

    # Every param reconstructed bit-for-bit from shards alone.
    for name in truth:
        assert torch.equal(recon_params[name], truth[name]), f"mismatch for {name}"

    # Keep source tensors alive until after the memmoves above.
    assert all(t.data_ptr() for t in srcs.values())


def test_strided_source_reconstructs_exactly():
    """Focus the row-parallel case: a strided column-slice must land correctly
    across its multiple runs (which dim-0-only shard schemes can't serve).

    Captured over a row-only manifest so the plan has an empty ``fallback``.
    Capturing the whole manifest while publishing only ``row`` would leave the
    other five sources unsupported, and a plan carrying unsupported sources is
    one the receiver refuses outright - so executing it here would assert
    reconstruction against a plan that never reaches a real receiver.
    """
    srcs = _full_sources()
    row_manifest = [entry for entry in _manifest() if entry[0] == "row"]

    truth_model = ToyModel()
    truth_model.load_weights(list(srcs.items()))
    truth_row = dict(truth_model.named_parameters())["row"].detach().clone()

    recon_model = ToyModel()
    for p in recon_model.parameters():
        torch.nn.init.zeros_(p)
    recon_row = dict(recon_model.named_parameters())["row"]

    with torch.device("meta"):
        meta_model = ToyModel()
    capture = capture_geometry(meta_model, row_manifest)

    row_src = srcs["row"]
    shard = Shard(
        shard_offset=(0, 0),
        shape=tuple(row_src.shape),
        session="row",
        addr=row_src.data_ptr(),
        elsize=EL,
    )
    sources = {"row": SourceInfo(tuple(row_src.shape), torch.float32, EL, [shard])}

    plan = plan_transfer(capture, sources)
    assert plan.fallback == []
    execute_transfer(
        plan,
        resolve_param_ptr=lambda name: dict(recon_model.named_parameters())[
            name
        ].data_ptr(),
        transport=InMemoryReferenceTransport(),
    )
    assert torch.equal(recon_row, truth_row)
    assert row_src.data_ptr()  # keep alive


def test_unsupported_source_is_marked_for_fail_closed():
    srcs = _full_sources()
    srcs["bad"] = torch.arange(16, dtype=torch.float32).reshape(4, 4).contiguous()

    with torch.device("meta"):
        meta_model = ToyModel(with_bad=True)
    capture = capture_geometry(meta_model, _manifest(with_bad=True))

    sources = {}
    for name, tensor in srcs.items():
        shard = Shard(
            (0,) * tensor.dim(), tuple(tensor.shape), name, tensor.data_ptr(), EL
        )
        sources[name] = SourceInfo(tuple(tensor.shape), torch.float32, EL, [shard])

    plan = plan_transfer(capture, sources)
    assert "bad" in plan.fallback
    # The good sources are still planned, not dropped.
    planned_params = {seg.param_name for seg in plan.segments}
    assert {"col", "row", "qkv", "norm"} <= planned_params
    assert all(t.data_ptr() for t in srcs.values())

    # Fail-closed means the unsupported source is never fetched. `plan_transfer`
    # records it and plans nothing for it; the refusal itself is the receiver's
    # policy, not this layer's - see the `if plan.fallback:` guard in
    # `ReshardReceiver._prepare`, which raises UnsupportedReshard before any
    # transport call rather than serving stale weights for `bad`.
    assert "bad" not in planned_params

    recon_model = ToyModel(with_bad=True)
    for p in recon_model.parameters():
        torch.nn.init.zeros_(p)
    recon_ptrs = {n: p.data_ptr() for n, p in recon_model.named_parameters()}

    reads = _RecordingTransport()
    execute_transfer(
        plan,
        resolve_param_ptr=lambda name: recon_ptrs[name],
        transport=reads,
    )
    assert "bad" not in reads.sessions
    assert reads.sessions  # the supported sources were still fetched


def test_execute_transfer_surfaces_fallback_for_the_receiver_to_reject():
    """`execute_transfer` reports `fallback` rather than acting on it.

    The split is deliberate: this layer stays mechanism and the receiver holds
    the policy, so the contract that has to hold here is that the list survives
    into the returned stats. A receiver that cannot see it cannot fail closed.
    """
    srcs = _full_sources()
    srcs["bad"] = torch.arange(16, dtype=torch.float32).reshape(4, 4).contiguous()

    with torch.device("meta"):
        meta_model = ToyModel(with_bad=True)
    capture = capture_geometry(meta_model, _manifest(with_bad=True))

    recon_model = ToyModel(with_bad=True)
    for p in recon_model.parameters():
        torch.nn.init.zeros_(p)
    recon_ptrs = {n: p.data_ptr() for n, p in recon_model.named_parameters()}

    sources = {}
    for name, tensor in srcs.items():
        shard = Shard(
            (0,) * tensor.dim(), tuple(tensor.shape), name, tensor.data_ptr(), EL
        )
        sources[name] = SourceInfo(tuple(tensor.shape), torch.float32, EL, [shard])

    plan = plan_transfer(capture, sources)
    stats = execute_transfer(
        plan,
        resolve_param_ptr=lambda name: recon_ptrs[name],
        transport=InMemoryReferenceTransport(),
    )
    assert stats["fallback"] == plan.fallback
    assert "bad" in stats["fallback"]
    assert all(t.data_ptr() for t in srcs.values())


if __name__ == "__main__":
    test_reshard_reconstructs_ground_truth()
    test_strided_source_reconstructs_exactly()
    test_unsupported_source_is_marked_for_fail_closed()
    test_execute_transfer_surfaces_fallback_for_the_receiver_to_reject()
    print("OK: reshard reconstructs ground truth + strided + fail-closed")
