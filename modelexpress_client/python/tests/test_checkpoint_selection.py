# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import safetensors
import torch
from modelexpress_rl.inference.checkpoint_selection import (
    CheckpointChanges,
    DependencyGroup,
    read_selected_tensors,
    select_checkpoint_sources,
)
from safetensors.torch import save_file


def select(names, groups, checkpoint_names=None, **kwargs):
    return select_checkpoint_sources(
        CheckpointChanges("base", "target", frozenset(names)),
        serving_version=kwargs.get("serving_version", "base"),
        target_version=kwargs.get("target_version", "target"),
        checkpoint_names=frozenset(checkpoint_names or names),
        groups=tuple(groups),
    )


def test_dependency_closure_includes_fused_siblings_and_refresh():
    groups = (
        DependencyGroup("consumer", frozenset({"up", "scale"}), frozenset({"host"})),
        DependencyGroup("fused", frozenset({"gate", "up"}), frozenset({"repack"})),
        DependencyGroup("unrelated", frozenset({"expert"})),
    )
    result = select({"gate"}, groups, {"gate", "up", "scale", "expert"})
    assert result.fallback_reason is None
    assert result.sources == {"gate", "up", "scale"}
    assert result.groups == {"fused", "consumer"}
    assert result.refresh == {"repack", "host"}


def test_shared_projection_selects_all_consumers():
    groups = (
        DependencyGroup("projection", frozenset({"kv"})),
        DependencyGroup("mla", frozenset({"kv"}), frozenset({"W_UV", "W_UK_T"})),
        DependencyGroup("tied", frozenset({"kv", "alias"})),
    )
    result = select({"kv"}, groups, {"kv", "alias"})
    assert result.groups == {"projection", "mla", "tied"}
    assert result.sources == {"kv", "alias"}
    assert result.refresh == {"W_UV", "W_UK_T"}


@pytest.mark.parametrize(
    "names,groups,available,kwargs,reason",
    [
        ({"w"}, (), {"w"}, {}, "unsupported"),
        ({"missing"}, (), {"w"}, {}, "unknown"),
        ({"w"}, (), {"w"}, {"serving_version": "peer-version"}, "version"),
        ({"w"}, (), {"w"}, {"target_version": "other"}, "version"),
        (
            {"w"},
            (DependencyGroup("g", frozenset({"w", "absent"})),),
            {"w"},
            {},
            "catalog",
        ),
        ({"w"}, (DependencyGroup("g", frozenset({"w"})),) * 2, {"w"}, {}, "catalog"),
    ],
)
def test_fallback_precedes_reads(tmp_path, names, groups, available, kwargs, reason):
    result = select(names, groups, available, **kwargs)
    assert reason in result.fallback_reason
    assert not result.sources
    with pytest.raises(ValueError, match="full reload required"):
        read_selected_tensors(result, checkpoint=tmp_path / "absent", weight_map={})


def test_empty_delta_is_distinct_from_unknown_metadata(tmp_path):
    empty = select(set(), ())
    assert empty.fallback_reason is None
    assert read_selected_tensors(empty, checkpoint=tmp_path, weight_map={}) == {}
    unknown = select_checkpoint_sources(
        None,
        serving_version="base",
        target_version="target",
        checkpoint_names=frozenset(),
        groups=(),
    )
    assert unknown.fallback_reason is not None


def test_reads_only_selected_sources_and_returns_owned_tensors(tmp_path, monkeypatch):
    save_file(
        {"gate": torch.ones(2), "up": torch.full((2,), 3.0)}, tmp_path / "a.safetensors"
    )
    (tmp_path / "untouched.safetensors").write_bytes(b"unreadable payload")
    original = (tmp_path / "a.safetensors").read_bytes()
    opened = []
    safe_open = safetensors.safe_open

    def tracked(path, **kwargs):
        opened.append(path.name)
        return safe_open(path, **kwargs)

    monkeypatch.setattr(safetensors, "safe_open", tracked)
    plan = select(
        {"gate"}, [DependencyGroup("fused", frozenset({"gate", "up"}))], {"gate", "up"}
    )
    staged = read_selected_tensors(
        plan,
        checkpoint=tmp_path,
        weight_map={
            "gate": "a.safetensors",
            "up": "a.safetensors",
            "other": "untouched.safetensors",
        },
    )
    assert opened == ["a.safetensors"]
    assert set(staged) == {"gate", "up"}
    assert torch.equal(staged["up"], torch.full((2,), 3.0))
    staged["gate"].zero_()
    assert (tmp_path / "a.safetensors").read_bytes() == original
    assert (tmp_path / "untouched.safetensors").read_bytes() == b"unreadable payload"


@pytest.mark.parametrize(
    "filename", ["../escape.safetensors", "/escape.safetensors", "bad.bin"]
)
def test_rejects_invalid_shard_mapping(tmp_path, filename):
    plan = select({"w"}, [DependencyGroup("g", frozenset({"w"}))])
    with pytest.raises(ValueError, match="invalid shard"):
        read_selected_tensors(plan, checkpoint=tmp_path, weight_map={"w": filename})


def test_read_failure_does_not_modify_checkpoint_or_return_partial_result(tmp_path):
    save_file({"a": torch.ones(2)}, tmp_path / "a.safetensors")
    save_file({"wrong": torch.ones(2)}, tmp_path / "b.safetensors")
    before = {p: p.read_bytes() for p in tmp_path.iterdir()}
    plan = select({"a", "b"}, [DependencyGroup("g", frozenset({"a", "b"}))])
    with pytest.raises(ValueError, match="selected tensors missing"):
        read_selected_tensors(
            plan,
            checkpoint=tmp_path,
            weight_map={"a": "a.safetensors", "b": "b.safetensors"},
        )
    assert all(p.read_bytes() == contents for p, contents in before.items())
