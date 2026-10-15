# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import pytest
from modelexpress.refit.reshard.slice_plan import Shard
from modelexpress.refit.reshard.transfer_plan import SourceInfo
from modelexpress_rl.inference import _source_snapshot as snapshot

from tests.test_refit_warm_caches import _bounded_cache_inputs



def test_source_rows_are_isolated_from_mutable_parser_rows() -> None:
    original = _bounded_cache_inputs()["resolved"]
    frozen_sources = snapshot._freeze_sources(original.sources)

    assert frozen_sources is not None
    original.sources["weight"].shards[0].addr += 16
    original.sources["weight"].shards.reverse()
    assert frozen_sources["weight"].shards[0].addr == 100
    with pytest.raises(TypeError):
        frozen_sources["other"] = frozen_sources["weight"]
    with pytest.raises((AttributeError, TypeError)):
        frozen_sources["weight"].shards[0].addr = 999


@pytest.mark.parametrize(
    "field_name",
    [
        "shape",
        "offset",
        "source_shape",
        "address_bool",
        "dtype",
        "digest",
        "session",
        "source_extra",
        "shard_extra",
        "source_subclass",
        "shard_subclass",
        "source_dict_subclass",
    ],
)
def test_nonordinary_sources_keep_original_mutable_path(field_name):
    resolved = _bounded_cache_inputs()["resolved"]
    source = resolved.sources["weight"]
    shard = source.shards[0]
    if field_name == "shape":
        shard.shape = [2]
    elif field_name == "offset":
        shard.shard_offset = (True,)
    elif field_name == "source_shape":
        source.global_shape = [4]
    elif field_name == "address_bool":
        shard.addr = True
    elif field_name == "dtype":
        source.dtype = object()
    elif field_name == "digest":
        shard.digest = []
    elif field_name == "session":
        shard.session = object()
    elif field_name == "source_extra":
        source.extra = []
    elif field_name == "shard_extra":
        shard.extra = []
    elif field_name == "source_subclass":
        source.__class__ = type("CustomSource", (SourceInfo,), {})
    elif field_name == "shard_subclass":
        shard.__class__ = type("CustomShard", (Shard,), {})
    else:
        resolved = replace(
            resolved, sources=type("CustomDict", (dict,), {})(resolved.sources)
        )
    assert snapshot._freeze_sources(resolved.sources) is None
