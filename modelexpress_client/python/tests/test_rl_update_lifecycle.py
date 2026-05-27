# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest

from modelexpress.rl_update_lifecycle import (
    RlWeightUpdateLifecycleHooks,
    iter_weight_update_lifecycle,
)


async def _items(values, events):
    for value in values:
        events.append(f"produce:{value}")
        yield value


def test_weight_update_lifecycle_orders_pause_flush_resume():
    events = []

    async def collect():
        hooks = RlWeightUpdateLifecycleHooks(
            pause_generation=lambda: events.append("pause"),
            flush_cache=lambda: events.append("flush"),
            resume_generation=lambda: events.append("resume"),
        )
        output = []
        async for item in iter_weight_update_lifecycle(
            _items([1, 2], events),
            hooks=hooks,
        ):
            events.append(f"consume:{item}")
            output.append(item)
        return output

    assert asyncio.run(collect()) == [1, 2]
    assert events == [
        "pause",
        "produce:1",
        "consume:1",
        "produce:2",
        "consume:2",
        "flush",
        "resume",
    ]


def test_weight_update_lifecycle_rejects_non_callable_hooks():
    with pytest.raises(ValueError, match="pause_generation hook must be callable"):
        RlWeightUpdateLifecycleHooks(pause_generation="pause")


def test_weight_update_lifecycle_resumes_when_consumer_raises():
    events = []

    async def consume():
        hooks = RlWeightUpdateLifecycleHooks(
            pause_generation=lambda: events.append("pause"),
            flush_cache=lambda: events.append("flush"),
            resume_generation=lambda: events.append("resume"),
        )
        async for item in iter_weight_update_lifecycle(
            _items([1, 2], events),
            hooks=hooks,
        ):
            events.append(f"consume:{item}")
            raise RuntimeError("refit failed")

    with pytest.raises(RuntimeError, match="refit failed"):
        asyncio.run(consume())
    assert events == ["pause", "produce:1", "consume:1", "resume"]


def test_weight_update_lifecycle_preserves_primary_error_when_resume_fails(caplog):
    events = []

    async def broken_items():
        events.append("produce")
        raise ValueError("receive failed")
        yield  # pragma: no cover

    def resume():
        events.append("resume")
        raise RuntimeError("resume failed")

    async def collect():
        hooks = RlWeightUpdateLifecycleHooks(
            pause_generation=lambda: events.append("pause"),
            resume_generation=resume,
        )
        async for _item in iter_weight_update_lifecycle(
            broken_items(),
            hooks=hooks,
        ):
            pass

    with pytest.raises(ValueError, match="receive failed"):
        asyncio.run(collect())
    assert events == ["pause", "produce", "resume"]
    assert "resume hook failed" in caplog.text


def test_weight_update_lifecycle_raises_resume_error_after_success():
    events = []

    async def collect():
        hooks = RlWeightUpdateLifecycleHooks(
            resume_generation=_raise_resume_error,
        )
        async for _item in iter_weight_update_lifecycle(
            _items([1], events),
            hooks=hooks,
        ):
            pass

    with pytest.raises(RuntimeError, match="resume failed"):
        asyncio.run(collect())


def _raise_resume_error():
    raise RuntimeError("resume failed")
