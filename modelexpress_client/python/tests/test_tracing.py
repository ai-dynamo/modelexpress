# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for opt-in OpenTelemetry tracing."""

from unittest.mock import MagicMock, patch

import pytest
import torch
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


@pytest.fixture
def real_tracer_and_exporter():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider.get_tracer("modelexpress.test"), exporter


def _make_load_context(**overrides):
    """Return a LoadContext with mocked dependencies."""
    from modelexpress.load_strategy import LoadContext
    from modelexpress import p2p_pb2
    defaults = dict(
        vllm_config=MagicMock(),
        model_config=MagicMock(),
        load_config=MagicMock(),
        target_device=torch.device("cpu"),
        global_rank=3,
        device_id=0,
        identity=p2p_pb2.SourceIdentity(model_name="test/model"),
        mx_client=MagicMock(),
        worker_id="w",
    )
    defaults.update(overrides)
    return LoadContext(**defaults)


@pytest.mark.parametrize("load_result,expect_used", [(True, True), (False, False)])
def test_load_chain_emits_real_span(real_tracer_and_exporter, load_result, expect_used):
    from modelexpress.load_strategy import LoadStrategyChain

    tracer, exporter = real_tracer_and_exporter

    with patch("modelexpress.load_strategy.tracer", tracer), patch(
        "modelexpress.load_strategy.rdma_strategy.RdmaStrategy.is_available", return_value=False
    ), patch(
        "modelexpress.load_strategy.model_streamer_strategy.ModelStreamerStrategy.is_available",
        return_value=False,
    ), patch(
        "modelexpress.load_strategy.gds_strategy.GdsStrategy.is_available", return_value=False
    ), patch(
        "modelexpress.load_strategy.default_strategy.DefaultStrategy.is_available",
        return_value=True,
    ), patch(
        "modelexpress.load_strategy.default_strategy.DefaultStrategy.load",
        return_value=load_result,
    ), patch(
        "modelexpress.load_strategy.default_strategy.DefaultStrategy.rollback",
        return_value=False,
    ):
        if load_result:
            LoadStrategyChain.run(MagicMock(), _make_load_context())
        else:
            with pytest.raises(RuntimeError):
                LoadStrategyChain.run(MagicMock(), _make_load_context())

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "Load model"
    assert span.attributes["model_name"] == "test/model"
    assert span.attributes["global_rank"] == 3
    assert span.attributes["eligible_strategies"] == ("default",)
    assert ("strategy_used" in span.attributes) is expect_used
    if expect_used:
        assert span.attributes["strategy_used"] == "default"
