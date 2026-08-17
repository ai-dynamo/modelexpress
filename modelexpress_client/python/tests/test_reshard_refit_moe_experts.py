# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Fused grouped-expert (MoE) capture, mirroring vLLM's real expert loader.

A trainer publishes MoE experts in per-expert HF form
(``...experts.<e>.gate_proj.weight``) while vLLM's destination is a fused
grouped-expert buffer (``...experts.w13_weight`` of shape
``[local_experts, 2 * inter_per_tp, hidden]``). vLLM unifies its fused and
per-expert loader paths with::

    experts_shard = loaded_weight.unsqueeze(0)   # per-expert source
    loaded_experts = experts_shard.unbind()
    for expert_id, loaded_expert in enumerate(loaded_experts, start=start):
        param.weight_loader(param, loaded_expert, ..., expert_id=expert_id)

``unbind`` is a pure multi-return view, but until it was allowlisted every
expert source in the model was classified unsupported and the refit failed
closed at ~5% coverage (48 layers x 128 experts x 3 projections = 18432 sources
on Qwen3-30B-A3B). Note the ``unsqueeze(0)`` / ``unbind()`` pair cancels, so the
resolved view is rank-preserving and the plan stays an axis-aligned box.

Runs on CPU/meta in any torch env: no GPU, no vLLM.
"""

import torch

from modelexpress.refit.reshard.geometry import capture_geometry
from modelexpress.refit.reshard.slice_plan import Shard, plan_pull

HIDDEN = 4
INTER = 8  # per-expert intermediate size, full (unsharded)
TP_SIZE = 2
TP_RANK = 0
INTER_PER_TP = INTER // TP_SIZE
LOCAL_EXPERTS = 2


class FusedMoEModel(torch.nn.Module):
    """Destination holds experts fused: ``w13`` stacks gate (w1) then up (w3)."""

    def __init__(self):
        super().__init__()
        self.w13_weight = torch.nn.Parameter(
            torch.empty(LOCAL_EXPERTS, 2 * INTER_PER_TP, HIDDEN)
        )
        self.w13_weight.weight_loader = self._expert_loader

    def _expert_loader(self, *, param, loaded_weight, shard_id, expert_id):
        """Mirrors vLLM's ``_load_w13``: pick this rank's slice of the source, then
        the member half (w1 low, w3 high) of the stacked destination slot."""
        expert_data = param.data[expert_id]
        member = 0 if shard_id == "w1" else 1
        dest = expert_data.narrow(0, member * INTER_PER_TP, INTER_PER_TP)
        src = loaded_weight.narrow(0, TP_RANK * INTER_PER_TP, INTER_PER_TP)
        dest.copy_(src)

    def load_weights(self, weights):
        for name, loaded in weights:
            expert_id = int(name.split(".experts.")[1].split(".")[0])
            shard_id = "w1" if "gate_proj" in name else "w3"
            # The vLLM path under test: add a dummy expert dim, then unbind it.
            experts_shard = loaded.unsqueeze(0)
            for local_id, loaded_expert in enumerate(
                experts_shard.unbind(), start=expert_id
            ):
                # Invoked entirely by keyword, exactly as vLLM's RoutedExperts
                # loader does. A capture stamp that names its first parameter
                # positionally raises TypeError here.
                self.w13_weight.weight_loader(
                    param=self.w13_weight,
                    loaded_weight=loaded_expert,
                    shard_id=shard_id,
                    expert_id=local_id,
                )


def _name(expert: int, proj: str) -> str:
    return f"model.layers.0.mlp.experts.{expert}.{proj}.weight"


def _manifest():
    return [
        (_name(e, proj), torch.float32, [INTER, HIDDEN])
        for e in range(LOCAL_EXPERTS)
        for proj in ("gate_proj", "up_proj")
    ]


def test_per_expert_sources_capture_into_fused_destination():
    """Every expert source is captured; none lands in ``unsupported``."""
    with torch.device("meta"):
        model = FusedMoEModel()
    result = capture_geometry(model, _manifest())

    assert result.unsupported == []
    assert result.unsupported_reasons == {}
    assert result.unattributed == 0
    # One copy per (expert, projection): both halves of every expert's w13 slot.
    assert len(result.copies) == 2 * LOCAL_EXPERTS

    by_src = {c.src_name: c for c in result.copies}
    assert set(by_src) == {n for n, _, _ in _manifest()}
    for copy in result.copies:
        assert copy.param_name == "w13_weight"
        assert copy.dest_shape == (INTER_PER_TP, HIDDEN)
        assert ("unbind", (), ()) in copy.op_chain


def test_unbind_is_recorded_and_does_not_change_rank():
    """``unsqueeze(0)`` then ``unbind()`` cancel, so the recorded chain still
    resolves to a rank-preserving box. This is why an allowlist entry suffices
    and no rank-collapse support is needed in the slice arithmetic."""
    with torch.device("meta"):
        model = FusedMoEModel()
    result = capture_geometry(model, _manifest())

    copy = next(c for c in result.copies if c.src_name == _name(0, "gate_proj"))
    ops = [name for name, _args, _kw in copy.op_chain]
    assert ops[:3] == ["unsqueeze", "unbind", "__getitem__"]
    assert "narrow" in ops


def test_expert_copies_plan_to_the_right_source_bytes():
    """The captured geometry must resolve to this TP rank's half of each expert,
    proving the chain is a real box and not merely accepted by the allowlist."""
    with torch.device("meta"):
        model = FusedMoEModel()
    result = capture_geometry(model, _manifest())
    copy = next(c for c in result.copies if c.src_name == _name(1, "up_proj"))

    # One publisher offering the whole per-expert source, contiguous row-major.
    shard = Shard(
        shard_offset=(0, 0),
        shape=(INTER, HIDDEN),
        session="pub0",
        addr=0,
        elsize=4,
    )
    segments = plan_pull(
        copy,
        global_shape=(INTER, HIDDEN),
        src_dtype=torch.float32,
        elsize=4,
        shards=[shard],
    )

    assert segments, "expert copy produced no pull segments"
    pulled = sum(s.nbytes for s in segments)
    # Exactly this rank's slice: INTER_PER_TP rows of HIDDEN float32 elements.
    assert pulled == INTER_PER_TP * HIDDEN * 4
    # Rank 0 reads from the start of the source.
    assert min(s.src_addr for s in segments) == 0
    # It lands in expert 1's w3 (upper) half of the fused destination.
    expected_dst = (
        1 * (2 * INTER_PER_TP) * HIDDEN + INTER_PER_TP * HIDDEN
    ) * 4
    assert min(s.dst_byte for s in segments) == expected_dst
