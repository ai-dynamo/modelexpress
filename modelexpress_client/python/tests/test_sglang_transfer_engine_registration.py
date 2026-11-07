# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TransferEngine registration coverage and cleanup without RDMA hardware."""

from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest
import torch
from modelexpress.engines.sglang.loader import (
    MxModelLoader,
    _coalesce_cuda_memory_regions,
)


def _segment(address, *blocks, expandable=False):
    return {
        "address": address,
        "is_expandable": expandable,
        "blocks": [
            {"address": addr, "size": size, "state": state}
            for addr, size, state in blocks
        ],
    }


def _tensor(address, size, *, cuda=True):
    return SimpleNamespace(
        data_ptr=lambda: address,
        numel=lambda: size,
        element_size=lambda: 1,
        is_cuda=cuda,
    )


@pytest.fixture
def snapshot(monkeypatch):
    snapshot = Mock(return_value=[])
    monkeypatch.setattr(torch.cuda, "memory_snapshot", snapshot)
    return snapshot


def test_registration_merges_blocks_and_preserves_tensor_metadata(snapshot):
    snapshot.return_value = [
        _segment(
            1024,
            (1024, 256, "active_allocated"),
            (1280, 256, "active_allocated"),
            (1536, 256, "active_allocated"),
        )
    ]
    tensors = {
        "weight": _tensor(1056, 32),
        "weight_alias": _tensor(1056, 64),
        "offset_view": _tensor(1152, 64),
        "scale_buffer": _tensor(1280, 16),
        "projection.__storage": _tensor(1536, 256),
        "empty": _tensor(0, 0),
    }
    engine = Mock(register_memory=Mock(return_value=0))
    loader = MxModelLoader(SimpleNamespace())

    weight_info, registered_ptrs = loader._register_transfer_engine_tensors(
        tensors, engine
    )

    engine.register_memory.assert_called_once_with(1024, 768)
    assert registered_ptrs == {1024}
    assert weight_info == {
        "weight": (1056, 32, 1),
        "weight_alias": (1056, 64, 1),
        "offset_view": (1152, 64, 1),
        "scale_buffer": (1280, 16, 1),
        "projection.__storage": (1536, 256, 1),
        "empty": (0, 0, 1),
    }
    engine.batch_register_memory.assert_not_called()


@pytest.mark.parametrize(
    "gap_state", ["inactive", "active_awaiting_free", "active_allocated"]
)
def test_registration_does_not_merge_across_unselected_blocks_or_segments(
    snapshot, gap_state
):
    snapshot.return_value = [
        _segment(1024, (1024, 256, "active_allocated")),
        _segment(
            1280,
            (1280, 256, "active_allocated"),
            (1536, 256, gap_state),
            (1792, 256, "active_allocated"),
        ),
    ]

    assert _coalesce_cuda_memory_regions({1056: 64, 1280: 64, 1824: 64}) == [
        (1024, 256),
        (1280, 256),
        (1792, 256),
    ]


def test_registration_requires_full_block_coverage_and_keeps_unknown_ranges(snapshot):
    snapshot.return_value = [
        _segment(1024, (1024, 256, "active_allocated")),
        _segment(2048, (2048, 256, "active_allocated")),
        _segment(3072, (3072, 512, "active_allocated"), expandable=True),
    ]

    assert _coalesce_cuda_memory_regions(
        {1024: 300, 2080: 64, 3072: 128, 3200: 128, 4096: 32}
    ) == [
        (1024, 300),
        (2048, 256),
        (3072, 128),
        (3200, 128),
        (4096, 32),
    ]


@pytest.mark.parametrize("regions", [{1000: 32, 1056: 64}, {1056: 64, 1200: 100}])
def test_block_expansion_cannot_overlap_an_unrecognized_range(snapshot, regions):
    snapshot.return_value = [_segment(1024, (1024, 256, "active_allocated"))]

    assert _coalesce_cuda_memory_regions(regions) == list(regions.items())


@pytest.mark.parametrize(
    "error",
    [None, RuntimeError("snapshot unavailable"), AttributeError("no snapshot API")],
)
def test_unavailable_snapshot_keeps_individual_regions(snapshot, error):
    snapshot.side_effect = error

    assert _coalesce_cuda_memory_regions({1024: 16, 1040: 16}) == [
        (1024, 16),
        (1040, 16),
    ]


def test_non_cuda_tensors_skip_snapshot_and_deduplicate_full_ranges(snapshot):
    engine = Mock(register_memory=Mock(return_value=0))
    loader = MxModelLoader(SimpleNamespace())

    _, registered_ptrs = loader._register_transfer_engine_tensors(
        {
            "short": _tensor(1024, 16, cuda=False),
            "long": _tensor(1024, 32, cuda=False),
            "host": _tensor(2048, 16, cuda=False),
        },
        engine,
    )

    snapshot.assert_not_called()
    assert engine.register_memory.call_args_list == [
        call(1024, 32),
        call(2048, 16),
    ]
    assert registered_ptrs == {1024, 2048}


def test_empty_tensors_need_no_registration(snapshot):
    engine = Mock()
    loader = MxModelLoader(SimpleNamespace())

    weight_info, registered_ptrs = loader._register_transfer_engine_tensors(
        {
            "empty": _tensor(0, 0),
        },
        engine,
    )

    assert weight_info == {"empty": (0, 0, 1)}
    assert registered_ptrs == set()
    snapshot.assert_not_called()
    engine.register_memory.assert_not_called()


@pytest.mark.parametrize("failure", [-1, RuntimeError("registration failed")])
def test_registration_failure_rolls_back_merged_region_bases(snapshot, failure):
    snapshot.return_value = [
        _segment(
            1024,
            (1024, 256, "active_allocated"),
            (1280, 256, "active_allocated"),
        )
    ]
    engine = Mock(
        register_memory=Mock(side_effect=[0, failure]),
        unregister_memory=Mock(return_value=0),
    )
    loader = MxModelLoader(SimpleNamespace())

    with pytest.raises(RuntimeError):
        loader._register_transfer_engine_tensors(
            {
                "weight": _tensor(1056, 64),
                "buffer": _tensor(1312, 64),
                "external": _tensor(2048, 64),
            },
            engine,
        )

    assert engine.register_memory.call_args_list == [call(1024, 512), call(2048, 64)]
    engine.unregister_memory.assert_called_once_with(1024)
