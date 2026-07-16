# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for TensorRT-LLM live-transfer catalog validation."""

import pytest
import torch
from torch import nn

from modelexpress.trtllm_live_transfer import (
    _canonical_named_parameters,
    _require_exact_catalog_match,
)


class _AliasLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.next_attn = None
        self.self_attn = None


class _AliasedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_AliasLayer(), _AliasLayer()])
        self.layers[1].self_attn = nn.Linear(2, 2, bias=False)
        self.layers[0].next_attn = self.layers[1].self_attn


def test_runtime_aliases_are_excluded_from_canonical_catalog():
    model = _AliasedModel()

    default_names = dict(model.named_parameters())
    canonical_names = dict(_canonical_named_parameters(model))

    assert "layers.0.next_attn.weight" in default_names
    assert "layers.1.self_attn.weight" not in default_names
    assert "layers.0.next_attn.weight" not in canonical_names
    assert canonical_names["layers.1.self_attn.weight"] is model.layers[1].self_attn.weight


def test_exact_catalog_match_is_accepted():
    tensor = torch.zeros(1)
    _require_exact_catalog_match({"model.weight": object()}, {"model.weight": tensor})


@pytest.mark.parametrize(
    ("source", "target", "message"),
    [
        ({"source.only": object()}, {}, "1 source tensors are absent"),
        ({}, {"target.only": torch.zeros(1)}, "1 target tensors are absent"),
    ],
)
def test_incomplete_catalogs_fail_closed(source, target, message):
    with pytest.raises(RuntimeError, match=message):
        _require_exact_catalog_match(source, target)
