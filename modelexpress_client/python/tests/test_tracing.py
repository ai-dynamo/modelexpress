# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for opt-in OpenTelemetry tracing."""

from unittest.mock import MagicMock, patch

import pytest
import torch
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from modelexpress import p2p_pb2
from modelexpress.adapter import StrategyFailed
from modelexpress.load_strategy import LoadContext, LoadResult, LoadStrategyChain


@pytest.fixture
def tracer_and_exporter():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider.get_tracer("modelexpress.test"), exporter


def _ctx():
    return LoadContext(
        model_config=MagicMock(),
        load_config=MagicMock(),
        target_device=torch.device("cpu"),
        global_rank=3,
        worker_rank=3,
        device_id=0,
        identity=p2p_pb2.SourceIdentity(model_name="test/model"),
        mx_client=MagicMock(),
        worker_id="w",
    )


@pytest.mark.parametrize("load_result,expect_used", [(True, True), (False, False)])
def test_load_chain_emits_real_span(tracer_and_exporter, load_result, expect_used):
    tracer, exporter = tracer_and_exporter
    base = "modelexpress.load_strategy"
    model = MagicMock()
    strategy_result = (
        LoadResult(value=model, model=model)
        if load_result
        else StrategyFailed("not available", mutated=False)
    )

    with (
        patch(f"{base}.tracer", tracer),
        patch(f"{base}.rdma_strategy.RdmaStrategy.is_available", return_value=False),
        patch(f"{base}.model_streamer_strategy.ModelStreamerStrategy.is_available", return_value=False),
        patch(f"{base}.gds_strategy.GdsStrategy.is_available", return_value=False),
        patch(f"{base}.default_strategy.DefaultStrategy.is_available", return_value=True),
        patch(
            f"{base}.default_strategy.DefaultStrategy.load",
            return_value=strategy_result,
            side_effect=strategy_result if not load_result else None,
        ),
        patch(f"{base}.default_strategy.DefaultStrategy.rollback"),
    ):
        if load_result:
            LoadStrategyChain.run(model, _ctx())
        else:
            with pytest.raises(RuntimeError):
                LoadStrategyChain.run(model, _ctx())

    [span] = exporter.get_finished_spans()
    assert span.name == "Load model"
    assert span.attributes["model_name"] == "test/model"
    assert span.attributes["global_rank"] == 3
    assert span.attributes["eligible_strategies"] == ("default",)
    assert ("weight_loading_strategy" in span.attributes) is expect_used
    if expect_used:
        assert span.attributes["weight_loading_strategy"] == "default"
