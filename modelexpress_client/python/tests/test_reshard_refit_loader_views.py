# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Loader idioms beyond plain ``narrow(...).copy_``, captured without an engine.

* mamba-style loaders assign ``param.data[a:b, ...] = src[c:d].view_as(...)``;
* a trainer that publishes the gate and up projections separately is mapped to
  the checkpoint's fused tensor with ``torch.cat``, which the loader re-slices;
* any other op that takes a list of sources must be classified unsupported.

Every recorded copy is checked against real data, both through ``plan_pull`` and
by replaying its op-chain on the source the way ordered source replay does.
"""

import pytest
import torch
from modelexpress_rl.refit.reshard.geometry import (
    LazyWeight,
    build_lazy_weights,
    capture_geometry,
    capture_weights,
    convert_source_weights,
)
from modelexpress_rl.refit.reshard.slice_plan import Shard, plan_pull
from modelexpress_rl.refit.reshard.types import UnsupportedReshard, run_epilogue

F32 = torch.float32


def _source(shape, base=0.0):
    return torch.arange(torch.Size(shape).numel(), dtype=F32).reshape(shape) + base


def _reconstruct(copies, sources, params):
    """Apply every copy twice: via the byte plan and via op-chain replay."""
    planned = {name: torch.zeros_like(value) for name, value in params.items()}
    replayed = {name: torch.zeros_like(value) for name, value in params.items()}
    for copy in copies:
        source = sources[copy.src_name]
        shard = Shard(
            shard_offset=(0,) * source.dim(), shape=tuple(source.shape),
            session="s", addr=0, elsize=4,
        )
        flat_source = source.reshape(-1)
        flat_dest = planned[copy.param_name].view(-1)
        for seg in plan_pull(
            copy, global_shape=tuple(source.shape), src_dtype=F32, elsize=4,
            shards=[shard], buffer_indices=(0,),
        ):
            s0, d0, n = seg.src_byte // 4, seg.dst_byte // 4, seg.nbytes // 4
            flat_dest[d0 : d0 + n] = flat_source[s0 : s0 + n]
        value = source
        for op, args, kwargs in copy.op_chain:
            value = getattr(value, op)(*args, **dict(kwargs))
        replayed[copy.param_name].as_strided(
            copy.dest_shape, copy.dest_stride, copy.dest_offset
        ).copy_(value)
    return planned, replayed


def _mamba_sharded_loader(shard_spec):
    """The shape of a mamba-v2 sharded loader at tensor-parallel size 1."""

    def loader(param, loaded_weight):
        boundary = loaded_boundary = 0
        for full_dim in shard_spec:
            target_slice = param.data[boundary : boundary + full_dim, ...]
            param.data[boundary : boundary + full_dim, ...] = loaded_weight[
                loaded_boundary : loaded_boundary + full_dim
            ].view_as(target_slice)
            boundary += full_dim
            loaded_boundary += full_dim

    return loader


class ConvModel(torch.nn.Module):
    def __init__(self, spec=(4, 4, 6), kernel=3):
        super().__init__()
        self.conv = torch.nn.Parameter(torch.empty(sum(spec), 1, kernel))
        self.conv.weight_loader = _mamba_sharded_loader(spec)

    def load_weights(self, weights):
        for _, loaded in weights:
            self.conv.weight_loader(self.conv, loaded)


def test_view_as_and_slice_assignment_are_captured():
    with torch.device("meta"):
        model = ConvModel()
    result = capture_geometry(model, [("conv", F32, (14, 1, 3))])

    assert not result.unsupported and not result.unattributed
    assert [c.op_chain for c in result.copies] == [
        (("__getitem__", (slice(lo, hi, None),), ()), ("view", ((n, 1, 3),), ()))
        for lo, hi, n in ((0, 4, 4), (4, 8, 4), (8, 14, 6))
    ]
    assert [c.dest_offset for c in result.copies] == [0, 12, 24]
    source = _source((14, 1, 3))
    for dest in _reconstruct(result.copies, {"conv": source}, {"conv": source})[0:2]:
        assert torch.equal(dest["conv"], source)


def test_broadcasting_slice_assignment_is_unsupported():
    with torch.device("meta"):
        model = ConvModel()

    def loader(param, loaded):
        param.data[0:4] = loaded[0]

    model.conv.weight_loader = loader
    result = capture_geometry(model, [("conv", F32, (14, 1, 3))])
    assert result.unsupported == ["conv"] and not result.copies
    assert "broadcasting assignment" in result.unsupported_reasons["conv"]


def test_assignment_through_advanced_indexing_is_unsupported():
    with torch.device("meta"):
        model = ConvModel()

    def loader(param, loaded):
        param.data[[0, 1]] = loaded[0:2]

    model.conv.weight_loader = loader
    result = capture_geometry(model, [("conv", F32, (14, 1, 3))])
    assert result.unsupported == ["conv"] and not result.copies


def test_unsupported_op_over_a_list_of_sources_does_not_recurse():
    with torch.device("meta"):
        model = ConvModel()
    model.conv.weight_loader = lambda param, loaded: param.data.copy_(
        torch.hstack([loaded, loaded])[:, :, 0:3]
    )
    result = capture_geometry(model, [("conv", F32, (14, 1, 3))])
    assert result.unsupported == ["conv"] and not result.copies
    assert "unsupported op aten." in result.unsupported_reasons["conv"]


_E, _I, _H = 3, 2, 5


class FusedExpertModel(torch.nn.Module):
    """A fused ``[E, 2I, H]`` checkpoint tensor loaded into ``w13[e, half]``."""

    def __init__(self, loader):
        super().__init__()
        self.w13 = torch.nn.Parameter(torch.empty(_E, 2 * _I, _H))
        self.w13.weight_loader = loader

    def load_weights(self, weights):
        for _, loaded in weights:
            self.w13.weight_loader(self.w13, loaded)


def _per_half_loader(transposed):
    def loader(param, fused):
        if transposed:  # checkpoint stores [E, H, 2I]
            fused = fused.transpose(1, 2)
        for half, experts in enumerate(fused.chunk(2, dim=1)):
            for expert, weight in enumerate(experts.unbind()):
                param.data[expert].narrow(0, half * _I, _I).copy_(weight)

    return loader


def _whole_loader(param, fused):
    param.data.copy_(fused)


def _concat_convert(dim):
    def convert(sd):
        return {"gate_up": torch.cat([sd["gate"], sd["up"]], dim=dim)}

    return convert


@pytest.mark.parametrize(
    ("loader", "transposed", "copies"),
    [
        (_per_half_loader(False), False, 2 * _E),
        (_per_half_loader(True), True, 2 * _E),
        (_whole_loader, False, 2),
    ],
)
def test_concatenated_sources_resolve_to_one_source_per_copy(loader, transposed, copies):
    shape = (_E, _H, _I) if transposed else (_E, _I, _H)
    sources = {"gate": _source(shape), "up": _source(shape, 1000.0)}
    with torch.device("meta"):
        model = FusedExpertModel(loader)
    weights = convert_source_weights(
        _concat_convert(2 if transposed else 1),
        [(name, F32, tuple(value.shape)) for name, value in sources.items()],
    )
    result = capture_weights(model, weights)

    assert not result.unsupported and not result.unattributed
    assert len(result.copies) == copies
    assert {c.src_name for c in result.copies} == {"gate", "up"}
    fused = torch.cat([sources["gate"], sources["up"]], dim=2 if transposed else 1)
    expected = fused.transpose(1, 2) if transposed else fused
    planned, replayed = _reconstruct(
        result.copies, sources, {"w13": torch.empty(_E, 2 * _I, _H)}
    )
    assert torch.equal(planned["w13"], expected)
    assert torch.equal(replayed["w13"], expected)


def test_copy_through_a_flattened_concatenation_splits_per_source():
    def loader(param, fused):
        param.data.view(_E, -1).copy_(fused.reshape(_E, -1))

    sources = {"gate": _source((_E, _I, _H)), "up": _source((_E, _I, _H), 1000.0)}
    with torch.device("meta"):
        model = FusedExpertModel(loader)
    weights = convert_source_weights(
        _concat_convert(1), [(name, F32, (_E, _I, _H)) for name in sources]
    )
    result = capture_weights(model, weights)
    assert not result.unsupported and len(result.copies) == 2
    planned, replayed = _reconstruct(
        result.copies, sources, {"w13": torch.empty(_E, 2 * _I, _H)}
    )
    expected = torch.cat([sources["gate"], sources["up"]], dim=1)
    assert torch.equal(planned["w13"], expected) and torch.equal(replayed["w13"], expected)


def test_copy_of_a_reordered_copy_of_concatenated_sources_is_unsupported():
    def loader(param, fused):
        param.data.view(_E, -1).copy_(fused.transpose(1, 2).reshape(_E, -1))

    with torch.device("meta"):
        model = FusedExpertModel(loader)
    weights = convert_source_weights(
        _concat_convert(1), [("gate", F32, (_E, _I, _H)), ("up", F32, (_E, _I, _H))]
    )
    result = capture_weights(model, weights)
    assert result.unsupported and not result.copies
    assert "concatenated sources" in next(iter(result.unsupported_reasons.values()))


@pytest.mark.parametrize(
    "ops",
    [
        lambda t: t.chunk(2, dim=1)[1],
        lambda t: t.narrow(1, 0, 2).unbind()[2],
        lambda t: t.narrow(1, 3, 2).select(0, 1),
        lambda t: t.transpose(1, 2)[:, :, 0:2],
        lambda t: t.permute(2, 0, 1)[:, 1:3, 2:4],
        lambda t: t.select(1, 4),
        lambda t: t[1, 2:4].t(),
        lambda t: t.unsqueeze(0).chunk(2, dim=2)[1].squeeze(0),
    ],
)
def test_views_inside_one_concatenated_source_replay_exactly(ops):
    gate, up = _source((_E, _I, _H)), _source((_E, 3, _H), 1000.0)
    lazies = build_lazy_weights([("gate", F32, (_E, _I, _H)), ("up", F32, (_E, 3, _H))])
    view = ops(torch.cat([lazies["gate"], lazies["up"]], dim=1))

    assert isinstance(view, LazyWeight) and view._concat is None
    value = {"gate": gate, "up": up}[view._name]
    for op, args, kwargs in view._ops:
        value = getattr(value, op)(*args, **dict(kwargs))
    assert torch.equal(value, ops(torch.cat([gate, up], dim=1)))


def test_concatenation_rejects_a_synthesized_part():
    lazies = build_lazy_weights([("gate", F32, (_E, _I, _H))])
    with pytest.raises(UnsupportedReshard, match="another tensor"):
        torch.cat([lazies["gate"], torch.zeros(_E, _I, _H)], dim=1)


def _composed(param, loaded):
    """vLLM's ``composed_weight_loader`` as Mamba2 uses it for ``A``."""
    param.data.copy_(loaded)
    param.data.copy_(-torch.exp(param.float()))


def test_post_copy_transform_is_unsupported_without_source_replay():
    """Replaying only the copy would install ``A_log`` itself, so a capture that
    cannot carry the transform must reject the source rather than record it."""
    with torch.device("meta"):
        model = ConvModel()
    model.conv.weight_loader = _composed
    result = capture_geometry(model, [("conv", F32, (14, 1, 3))])
    assert result.unsupported == ["conv"]
    assert "rewrites destination 'conv'" in result.unsupported_reasons["conv"]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_post_copy_transform_is_recorded_as_an_epilogue_for_source_replay(dtype):
    with torch.device("meta"):
        model = ConvModel()
        model.conv = torch.nn.Parameter(torch.empty(14, 1, 3, dtype=dtype))
    model.conv.weight_loader = _composed
    weights = build_lazy_weights([("conv", dtype, (14, 1, 3))])
    result = capture_weights(model, weights, source_replay=True)

    assert not result.unsupported and not result.unattributed
    (copy,) = result.copies
    assert copy.op_chain == () and copy.epilogue
    source = torch.randn(14, 1, 3).to(dtype)
    destination = torch.zeros(14, 1, 3, dtype=dtype)
    destination.copy_(source)
    destination.copy_(run_epilogue(destination, copy.epilogue))
    expected = torch.zeros(14, 1, 3, dtype=dtype)
    _composed(expected, source)
    assert torch.equal(destination, expected)


def test_in_place_post_copy_scale_is_recorded_as_an_epilogue():
    """A loader that scales its destination in place after the copy."""

    def scaled(param, loaded):
        param.data.copy_(loaded.view(param.shape)).mul_(0.25)

    with torch.device("meta"):
        model = ConvModel()
    model.conv.weight_loader = scaled
    weights = build_lazy_weights([("conv", F32, (14, 3))])
    result = capture_weights(model, weights, source_replay=True)

    assert not result.unsupported
    (copy,) = result.copies
    assert copy.op_chain == (("view", ((14, 1, 3),), ()),) and copy.epilogue
    source = torch.randn(14, 3)
    destination = torch.zeros(14, 1, 3)
    destination.copy_(source.view(14, 1, 3))
    destination.copy_(run_epilogue(destination, copy.epilogue))
    assert torch.equal(destination, source.view(14, 1, 3) * 0.25)


@pytest.mark.parametrize(
    "rewrite",
    [
        # Only part of the destination is recomputed.
        lambda param: param.data[0:4].copy_(-torch.exp(param[0:4].float())),
        # The new value depends on a tensor other than the destination.
        lambda param: param.data.copy_(param * torch.ones(14, 1, 3, device="meta")),
        # An in-place rewrite by a tensor other than the destination.
        lambda param: param.data.mul_(torch.full((14, 1, 3), 2.0)),
        # A second copy lands between the read and the rewrite.
        None,
    ],
)
def test_post_copy_rewrites_that_are_not_a_function_of_the_destination_are_unsupported(rewrite):
    def loader(param, loaded):
        param.data.copy_(loaded)
        if rewrite is None:
            value = -param.float()
            param.data.copy_(loaded)
            param.data.copy_(value)
        else:
            rewrite(param)

    with torch.device("meta"):
        model = ConvModel()
    model.conv.weight_loader = loader
    weights = build_lazy_weights([("conv", F32, (14, 1, 3))])
    result = capture_weights(model, weights, source_replay=True)
    assert result.unsupported == ["conv"]


def test_loader_may_prepare_its_destination_before_the_copy():
    def zero_then_copy(param, loaded):
        param.data.zero_()
        param.data.copy_(loaded)

    with torch.device("meta"):
        model = ConvModel()
    model.conv.weight_loader = zero_then_copy
    result = capture_geometry(model, [("conv", F32, (14, 1, 3))])
    assert not result.unsupported and len(result.copies) == 1


def test_loader_may_zero_padding_past_the_copied_rows():
    """vLLM's vocabulary-parallel embedding copies the checkpoint rows, then
    zeroes the padding rows after them (an empty slice without padding)."""

    def padded(param, loaded):
        param[: loaded.shape[0]].data.copy_(loaded)
        param[loaded.shape[0] :].data.fill_(0)

    with torch.device("meta"):
        model = ConvModel()
    model.conv.weight_loader = padded
    for rows in (10, 14):
        result = capture_geometry(model, [("conv", F32, (rows, 1, 3))])
        assert not result.unsupported and len(result.copies) == 1


def _replayed(copies, sources, params):
    """Install the way ordered source replay does: op-chain on the full source."""
    out = {name: torch.zeros_like(value) for name, value in params.items()}
    for copy in copies:
        value = sources[copy.src_name]
        for op, args, kwargs in copy.op_chain:
            value = getattr(value, op)(*args, **dict(kwargs))
        out[copy.param_name].as_strided(
            copy.dest_shape, copy.dest_stride, copy.dest_offset
        ).copy_(value)
    return out


class InterleavedExpertModel(torch.nn.Module):
    """An ``[E, H, 2I]`` checkpoint whose columns alternate gate and up, loaded
    into ``w13 [E, 2I, H]`` by slicing, permuting and a ``contiguous`` copy."""

    def __init__(self):
        super().__init__()
        self.w13 = torch.nn.Parameter(torch.empty(_E, 2 * _I, _H))
        self.w13.weight_loader = lambda param, loaded: param.copy_(loaded)
        self.b13 = torch.nn.Parameter(torch.empty(_E, 2 * _I))
        self.b13.weight_loader = lambda param, loaded: param.copy_(loaded)

    def load_weights(self, weights):
        for name, loaded in weights:
            if name == "gate_up_proj":
                weight = loaded[:, :, 0 : 2 * _I].permute(0, 2, 1).contiguous()
                self.w13.weight_loader(self.w13, weight)
            else:
                self.b13.weight_loader(self.b13, loaded[:, 0 : 2 * _I])


def test_interleaved_sources_land_in_alternating_destination_rows():
    def convert(sd):
        gate, up = sd["gate"].transpose(-2, -1), sd["up"].transpose(-2, -1)
        return {
            "gate_up_proj": torch.stack((gate, up), dim=-1).flatten(-2),
            "gate_up_proj_bias": torch.stack((sd["gate_bias"], sd["up_bias"]), dim=-1).flatten(-2),
        }

    sources = {
        "gate": _source((_E, _I, _H)), "up": _source((_E, _I, _H), 1000.0),
        "gate_bias": _source((_E, _I), 2000.0), "up_bias": _source((_E, _I), 3000.0),
    }
    with torch.device("meta"):
        model = InterleavedExpertModel()
    weights = convert_source_weights(
        convert, [(name, F32, tuple(value.shape)) for name, value in sources.items()]
    )
    result = capture_weights(model, weights, source_replay=True)

    assert not result.unsupported and not result.unattributed
    assert sorted(c.src_name for c in result.copies) == sorted(sources)
    out = _replayed(
        result.copies, sources,
        {"w13": torch.empty(_E, 2 * _I, _H), "b13": torch.empty(_E, 2 * _I)},
    )
    fused = torch.empty(_E, _H, 2 * _I)
    fused[..., ::2] = sources["gate"].transpose(-2, -1)
    fused[..., 1::2] = sources["up"].transpose(-2, -1)
    bias = torch.empty(_E, 2 * _I)
    bias[..., ::2], bias[..., 1::2] = sources["gate_bias"], sources["up_bias"]
    assert torch.equal(out["w13"], fused.permute(0, 2, 1))
    assert torch.equal(out["b13"], bias)


def test_detached_source_is_the_same_source():
    """A loader may detach a weight before tagging it with routing attributes."""

    def loader(param, loaded):
        loaded.shard_id = "k"
        detached = loaded.detach()
        detached.shard_id = 1
        param.data.copy_(detached)
        assert loaded.shard_id == "k"

    with torch.device("meta"):
        model = ConvModel()
    model.conv.weight_loader = loader
    result = capture_geometry(model, [("conv", F32, (14, 1, 3))])
    assert not result.unsupported
    (copy,) = result.copies
    assert copy.src_name == "conv" and copy.op_chain == ()
