# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TransferEngine registration coverage and cleanup without RDMA hardware."""

from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest
import torch
from modelexpress import p2p_pb2
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


@pytest.mark.parametrize(
    "invalid_segment",
    [
        pytest.param({"is_expandable": False}, id="missing-blocks"),
        pytest.param({"is_expandable": False, "blocks": None}, id="invalid-blocks"),
        pytest.param(
            {
                "is_expandable": False,
                "address": 2048,
                "blocks": [{"state": "active_allocated", "address": 2048}],
            },
            id="missing-block-size",
        ),
    ],
)
def test_malformed_snapshot_keeps_individual_registration(snapshot, invalid_segment):
    snapshot.return_value = [
        _segment(1024, (1024, 256, "active_allocated")),
        invalid_segment,
    ]
    engine = Mock(register_memory=Mock(return_value=0))
    loader = MxModelLoader(SimpleNamespace())

    weight_info, registered_ptrs = loader._register_transfer_engine_tensors(
        {"weight": _tensor(1056, 32), "buffer": _tensor(1152, 64)},
        engine,
    )

    assert engine.register_memory.call_args_list == [call(1056, 32), call(1152, 64)]
    assert registered_ptrs == {1056, 1152}
    assert weight_info == {"weight": (1056, 32, 1), "buffer": (1152, 64, 1)}


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


@pytest.fixture
def tensor_views(snapshot):
    snapshot.return_value = [
        _segment(
            1024,
            (1024, 256, "active_allocated"),
            (1280, 256, "active_allocated"),
        ),
        _segment(2048, (2048, 256, "active_allocated")),
    ]
    return {
        "alias": _tensor(1056, 32),
        "weight": _tensor(1056, 64),
        "buffer": _tensor(1312, 64),
        "scale": _tensor(2064, 16),
    }


@pytest.mark.parametrize("failure", [-1, RuntimeError("registration failed")])
def test_registration_retries_tensor_ranges_after_rollback(
    tensor_views, failure, caplog
):
    engine = Mock()
    engine.register_memory.side_effect = [0, failure, 0, 0, 0]
    engine.unregister_memory.return_value = 0
    loader = MxModelLoader(SimpleNamespace())

    with caplog.at_level("INFO", logger="modelexpress.engines.sglang.loader"):
        weight_info, registered_ptrs = loader._register_transfer_engine_tensors(
            tensor_views, engine
        )

    assert engine.method_calls == [
        call.register_memory(1024, 512),
        call.register_memory(2048, 256),
        call.unregister_memory(1024),
        call.register_memory(1056, 64),
        call.register_memory(1312, 64),
        call.register_memory(2064, 16),
    ]
    assert registered_ptrs == {1056, 1312, 2064}
    assert weight_info["alias"] == (1056, 32, 1)
    assert "registration failed (coalesced attempt," in caplog.text
    assert (
        "Registered 3 TransferEngine regions for 4 tensors (144 bytes)" in caplog.text
    )


def test_failed_tensor_retry_unregisters_only_successful_bases(tensor_views, caplog):
    engine = Mock()
    engine.register_memory.side_effect = [-1, 0, -2]
    engine.unregister_memory.return_value = 0

    with pytest.raises(RuntimeError, match="error=-2"):
        MxModelLoader(SimpleNamespace())._register_transfer_engine_tensors(
            tensor_views, engine
        )

    assert engine.method_calls == [
        call.register_memory(1024, 512),
        call.register_memory(1056, 64),
        call.register_memory(1312, 64),
        call.unregister_memory(1056),
    ]
    assert caplog.messages == [
        "TransferEngine registration failed (coalesced attempt, register_memory returned error=-1)",
        "TransferEngine registration failed (original attempt, register_memory returned error=-2)",
    ]


@pytest.mark.parametrize("cleanup_failure", [-1, RuntimeError("unregister failed")])
def test_incomplete_rollback_skips_tensor_retry(tensor_views, cleanup_failure, caplog):
    engine = Mock()
    engine.register_memory.side_effect = [0, 0, -2]
    engine.unregister_memory.side_effect = [cleanup_failure, 0]
    tensors = {**tensor_views, "external": _tensor(4096, 64)}

    with pytest.raises(RuntimeError, match="error=-2"):
        MxModelLoader(SimpleNamespace())._register_transfer_engine_tensors(
            tensors, engine
        )

    assert engine.register_memory.call_args_list == [
        call(1024, 512),
        call(2048, 256),
        call(4096, 64),
    ]
    assert engine.unregister_memory.call_count == 2
    engine.unregister_memory.assert_has_calls([call(1024), call(2048)], any_order=True)
    assert "registration failed (coalesced attempt," in caplog.text


@pytest.mark.parametrize("cuda", [False, True], ids=["cpu", "cuda-order-only"])
def test_unchanged_ranges_get_one_original_attempt(snapshot, caplog, cuda):
    snapshot.return_value = [
        _segment(1024, (1024, 64, "active_allocated")),
        _segment(2048, (2048, 64, "active_allocated")),
    ]
    tensors = {
        "high": _tensor(2048, 64, cuda=cuda),
        "low": _tensor(1024, 64, cuda=cuda),
    }
    engine = Mock()
    engine.register_memory.side_effect = [0, -1]
    engine.unregister_memory.return_value = 0

    with pytest.raises(RuntimeError, match="error=-1"):
        MxModelLoader(SimpleNamespace())._register_transfer_engine_tensors(
            tensors, engine
        )

    assert engine.method_calls == [
        call.register_memory(2048, 64),
        call.register_memory(1024, 64),
        call.unregister_memory(2048),
    ]
    assert "registration failed (original attempt," in caplog.text


def test_adjacent_full_blocks_retry_as_original_ranges(snapshot):
    snapshot.return_value = [
        _segment(1024, (1024, 256, "active_allocated"), (1280, 256, "active_allocated"))
    ]
    engine = Mock()
    engine.register_memory.side_effect = [-1, 0, 0]

    MxModelLoader(SimpleNamespace())._register_transfer_engine_tensors(
        {"weight": _tensor(1024, 256), "buffer": _tensor(1280, 256)}, engine
    )

    assert engine.method_calls == [
        call.register_memory(1024, 512),
        call.register_memory(1024, 256),
        call.register_memory(1280, 256),
    ]


@pytest.mark.parametrize("has_source", [False, True], ids=["seed", "receiver"])
def test_single_block_retry_allows_transfer_engine_load(
    snapshot, monkeypatch, has_source
):
    snapshot.return_value = [_segment(1024, (1024, 256, "active_allocated"))]
    engine = Mock()
    engine.register_memory.side_effect = [-1, 0]
    engine.submit_scatter_read.return_value = SimpleNamespace(
        drained=True, status=SimpleNamespace(name="COMPLETED")
    )
    loader = MxModelLoader(
        SimpleNamespace(
            remote_instance_weight_loader_transfer_engine=engine,
            remote_instance_weight_loader_transfer_engine_session_id="local",
        )
    )
    model = torch.nn.Module()
    result = SimpleNamespace(model=model)
    adapter = Mock()
    adapter.load_via_native.return_value = result
    adapter.before_rdma_receive.return_value = result
    adapter.after_rdma_receive.return_value = result
    adapter.discover_tensors.return_value = {"weight": _tensor(1024, 64)}
    ctx = SimpleNamespace(
        global_rank=0, identity=SimpleNamespace(model_name="model"), adapter=adapter
    )
    source = p2p_pb2.WorkerMetadata(
        transfer_engine_session_id="peer",
        tensors=[p2p_pb2.TensorDescriptor(name="weight", addr=4096, size=64)],
    )
    monkeypatch.setattr(
        "modelexpress.engines.sglang.loader.build_sglang_load_context", lambda *_: ctx
    )
    monkeypatch.setattr(
        loader, "_find_transfer_engine_source", lambda _: source if has_source else None
    )
    publish = Mock(return_value=True)
    monkeypatch.setattr(loader, "_publish_transfer_engine_source", publish)

    loaded = loader._load_model_via_transfer_engine(
        model=model, model_config=SimpleNamespace(), device_config=SimpleNamespace()
    )

    assert loaded is model
    assert engine.register_memory.call_args_list == [call(1024, 256), call(1024, 64)]
    publish.assert_called_once_with(
        ctx=ctx, session_id="local", weight_info={"weight": (1024, 64, 1)}
    )
    if has_source:
        adapter.load_via_native.assert_not_called()
        engine.submit_scatter_read.assert_called_once_with(
            "peer", [1024], [64], [4096], [64], [[0]], [[0]], [[64]]
        )
    else:
        adapter.load_via_native.assert_called_once()
        engine.submit_scatter_read.assert_not_called()
