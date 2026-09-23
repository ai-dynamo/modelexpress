# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A reload target must not be put through vLLM post-load processing twice.

``prepare()`` exists to bring a freshly dummy-allocated target up to the
processed layout the source published. A reload target is already in that
layout -- it was post-processed before the checkpoint -- and vLLM's post-load
processing is not idempotent, so running it again corrupts the model.

Observed on a CRIU-restored Qwen3-Next-80B replica pulling weights from a peer:

    convert_moe_weights_to_flashinfer_trtllm_block_layout
      w13_rows, w13_cols = w13_weight[0].view(torch.uint8).shape
    ValueError: too many values to unpack (expected 2)

That conversion reshapes MoE experts from ``[E, rows, cols]`` to
``[E, cols/block_k, rows, block_k]``. On the second pass it is handed its own
4-D output, so ``w13_weight[0]`` is 3-D and the two-way unpack fails. The RDMA
attempt dies, the chain falls back to reinit_for_retry, and on a model this size
that rebuild then OOMs -- so a non-idempotent reshape surfaces as an
out-of-memory crash several layers away from its cause.

RdmaStrategy prepares the target inside load(), once a source is selected, and
skips that and finalize() on a reload. Its prepare() and abort() are no-ops, so
it never enters or leaves a layerwise reload.

Run: pytest tests/test_rdma_reload_post_process_gate.py
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from modelexpress.load_strategy import _strategy_phases
from modelexpress.load_strategy.rdma_strategy import RdmaStrategy


def _ctx(*, is_reload: bool):
    return SimpleNamespace(
        global_rank=0,
        is_reload=is_reload,
        adapter=MagicMock(),
    )


def _result():
    return SimpleNamespace(model=MagicMock())


@pytest.fixture
def strategy(monkeypatch):
    strat = RdmaStrategy()
    # Source selection and the transfer itself are not under test; only which
    # adapter hooks run around the receive.
    monkeypatch.setattr(strat, "_receive_from_peer", lambda *a, **k: None)
    monkeypatch.setattr(
        strat,
        "load",
        lambda result, ctx: strat._load_as_target(
            result, ctx, MagicMock(), "mx-source", "worker"
        ),
    )
    return strat


def test_reload_skips_both_phases(strategy):
    """The restore path: the model is already allocated and already processed."""
    ctx = _ctx(is_reload=True)

    result = _result()
    with _strategy_phases(strategy, result, ctx):
        strategy.load(result, ctx)

    ctx.adapter.prepare_rdma_target.assert_not_called()
    ctx.adapter.before_rdma_receive.assert_not_called()
    ctx.adapter.after_rdma_receive.assert_not_called()


def test_cold_load_runs_both_phases(strategy):
    """The gate must not disarm the cold path, where the work is required.

    A dummy-allocated target has none of the processed layout, so skipping here
    would hand the source's weights to a model whose tensors are the wrong shape.
    """
    ctx = _ctx(is_reload=False)
    ctx.adapter.prepare_rdma_target.side_effect = lambda r: r
    ctx.adapter.before_rdma_receive.side_effect = lambda r: r
    ctx.adapter.after_rdma_receive.side_effect = lambda r: r

    result = _result()
    with _strategy_phases(strategy, result, ctx):
        strategy.load(result, ctx)

    ctx.adapter.prepare_rdma_target.assert_called_once()
    ctx.adapter.before_rdma_receive.assert_called_once()
    ctx.adapter.after_rdma_receive.assert_called_once()


def test_reload_does_not_enter_layerwise_reload(strategy):
    """RDMA writes into the buffers layerwise reload would have released.

    Deferring them to meta would leave the transfer landing nowhere the model
    executes against, so a reload must leave this strategy's storage intact.
    """
    ctx = _ctx(is_reload=True)

    result = _result()
    with _strategy_phases(strategy, result, ctx):
        strategy.load(result, ctx)

    ctx.adapter.begin_streaming_reload.assert_not_called()
    ctx.adapter.end_streaming_reload.assert_not_called()
