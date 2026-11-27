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

from types import SimpleNamespace

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


def test_concatenation_skips_an_empty_rank_one_source():
    """``torch.cat`` skips a ``(0,)`` operand whatever the dim, as PyTorch does."""
    source = _source((14, 1, 3))
    with torch.device("meta"):
        model = ConvModel()
    weights = convert_source_weights(
        lambda sd: {"conv": torch.cat([sd["empty"], sd["conv"]], dim=2)},
        [("empty", F32, (0,)), ("conv", F32, (14, 1, 3))],
    )
    result = capture_weights(model, weights)

    assert not result.unsupported and not result.unattributed
    assert {copy.src_name for copy in result.copies} == {"conv"}
    for dest in _reconstruct(result.copies, {"conv": source}, {"conv": source}):
        assert torch.equal(dest["conv"], source)


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


@pytest.mark.parametrize(
    "reduce", [lambda param: param.sum(), lambda param: param.amax(0)]
)
def test_post_copy_rewrite_broadcasts_like_copy(reduce):
    """``copy_`` broadcasts a reduction over the destination; replay must too."""

    def loader(param, loaded):
        param.data.copy_(loaded)
        param.data.copy_(reduce(param))

    with torch.device("meta"):
        model = ConvModel()
    model.conv.weight_loader = loader
    result = capture_weights(
        model, build_lazy_weights([("conv", F32, (14, 1, 3))]), source_replay=True
    )

    assert not result.unsupported
    (copy,) = result.copies
    assert copy.epilogue
    source = torch.randn(14, 1, 3)
    destination = torch.zeros(14, 1, 3)
    destination.copy_(source)
    destination.copy_(run_epilogue(destination, copy.epilogue))
    expected = torch.zeros(14, 1, 3)
    loader(expected, source)
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
        # An in-place view op changes only the metadata of ``param.data``; its
        # out-of-place form is a view of another shape.
        lambda param: param.data.squeeze_(),
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


@pytest.mark.parametrize("source_replay", [False, True])
@pytest.mark.parametrize(
    ("zeroed", "allowed"), [(slice(1, 3), True), (slice(0, 2), False)]
)
def test_zeroing_columns_beside_a_copied_column(source_replay, zeroed, allowed):
    """A strided copy and a strided zero only collide if they share bytes."""

    def loader(param, loaded):
        param.data[..., 0].copy_(loaded)
        param.data[..., zeroed].zero_()

    with torch.device("meta"):
        model = ConvModel()
    model.conv.weight_loader = loader
    result = capture_weights(
        model, build_lazy_weights([("conv", F32, (14, 1))]), source_replay=source_replay
    )
    if allowed:
        assert not result.unsupported and len(result.copies) == 1
        source = _source((14, 1))
        expected = torch.zeros(14, 1, 3)
        expected[..., 0] = source
        for dest in _reconstruct(result.copies, {"conv": source}, {"conv": expected}):
            assert torch.equal(dest["conv"], expected)
    else:
        assert result.unsupported == ["conv"]
        assert "rewrites destination 'conv'" in result.unsupported_reasons["conv"]


def _replay_into(copies, sources, out):
    """Install into ``out`` the way ordered source replay does, epilogues included."""
    for copy in copies:
        value = sources[copy.src_name]
        for op, args, kwargs in copy.op_chain:
            value = getattr(value, op)(*args, **dict(kwargs))
        destination = out[copy.param_name]
        destination.as_strided(
            copy.dest_shape, copy.dest_stride,
            destination.storage_offset() + copy.dest_offset,
        ).copy_(value)
        if copy.epilogue:
            destination.copy_(run_epilogue(destination, copy.epilogue))
    return out


class SharedStorageModel(torch.nn.Module):
    """Two registered views ``a = base[0:2]`` and ``b = base[at:at + 2]``."""

    def __init__(self, loader, at):
        super().__init__()
        base = torch.empty(at + 2)
        self.a = torch.nn.Parameter(base[0:2])
        self.b = torch.nn.Parameter(base[at : at + 2])
        self.loader = loader

    def load_weights(self, weights):
        for _, loaded in weights:
            self.loader(self, loaded)


@pytest.mark.parametrize("at", [1, 2])
def test_post_copy_rewrite_after_a_later_copy_into_shared_bytes(at):
    """Replay runs ``a``'s rewrite right after ``a``'s copy, before ``b``'s, so
    a copy into ``b`` that the rewrite then overwrites must reject the rewrite."""

    def loader(model, loaded):
        model.a.data.copy_(loaded[0])
        negated = -model.a
        model.b.data.copy_(loaded[1])
        model.a.data.copy_(negated)

    with torch.device("meta"):
        model = SharedStorageModel(loader, at)
    result = capture_weights(
        model, build_lazy_weights([("w", F32, (2, 2))]), source_replay=True
    )
    if at == 1:
        assert result.unsupported == ["w"]
        assert "replay would run the rewrite first" in result.unsupported_reasons["w"]
        return
    assert not result.unsupported
    source = _source((2, 2), 1.0)
    base = torch.zeros(at + 2)
    _replay_into(result.copies, {"w": source}, {"a": base[0:2], "b": base[at:]})
    expected = torch.zeros(at + 2)
    loader(SimpleNamespace(a=expected[0:2], b=expected[at:]), source)
    assert torch.equal(base, expected)


@pytest.mark.parametrize(("fill", "allowed"), [(5.0, False), (0.0, True)])
def test_post_copy_rewrite_that_reads_an_unrecorded_write(fill, allowed):
    """Replay starts the destination from zeros, so the rewrite may read bytes
    no copy wrote only if the loader left them zero."""

    def loader(param, loaded):
        param.data[1:].fill_(fill)
        param.data[:1].copy_(loaded)
        param.data.copy_(param + param.sum())

    with torch.device("meta"):
        model = ConvModel()
    model.conv.weight_loader = loader
    result = capture_weights(
        model, build_lazy_weights([("conv", F32, (1, 1, 3))]), source_replay=True
    )
    if not allowed:
        assert result.unsupported == ["conv"]
        assert "unrecorded non-zero write" in result.unsupported_reasons["conv"]
        return
    assert not result.unsupported
    source = _source((1, 1, 3), 1.0)
    out = _replay_into(result.copies, {"conv": source}, {"conv": torch.zeros(14, 1, 3)})
    expected = torch.zeros(14, 1, 3)
    loader(expected, source)
    assert torch.equal(out["conv"], expected)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.mul_(2),
        lambda value: value.view(-1)[0:3].add_(1),
        lambda value: value.data.neg_(),
        lambda value: value.copy_(torch.ones(14, 1, 3, device="meta")),
    ],
    ids=["in_place", "through_a_view", "through_data", "copy_into"],
)
@pytest.mark.parametrize(
    "compute", [lambda param: -param, torch.empty_like], ids=["neg", "empty_like"]
)
def test_post_copy_rewrite_from_a_value_written_after_its_trace_is_unsupported(
    compute, mutate
):
    """A trace replays the ops that computed a value, which no longer describe
    it once a write lands in its storage."""

    def loader(param, loaded):
        param.data.copy_(loaded)
        value = compute(param)
        mutate(value)
        param.data.copy_(value)

    with torch.device("meta"):
        model = ConvModel()
    model.conv.weight_loader = loader
    result = capture_weights(
        model, build_lazy_weights([("conv", F32, (14, 1, 3))]), source_replay=True
    )
    assert result.unsupported == ["conv"]
    assert "rewrites destination 'conv'" in result.unsupported_reasons["conv"]


def test_post_copy_rewrite_from_a_value_beside_a_written_scratch_tensor():
    """A write to another tensor leaves the traced value replayable."""

    def loader(param, loaded):
        param.data.copy_(loaded)
        value = -param
        scratch = torch.zeros_like(value)
        scratch.add_(1)
        param.data.copy_(value * 2)

    with torch.device("meta"):
        model = ConvModel()
    model.conv.weight_loader = loader
    result = capture_weights(
        model, build_lazy_weights([("conv", F32, (14, 1, 3))]), source_replay=True
    )
    assert not result.unsupported
    source = torch.randn(14, 1, 3)
    out = _replay_into(result.copies, {"conv": source}, {"conv": torch.zeros(14, 1, 3)})
    assert torch.equal(out["conv"], -source * 2)


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
