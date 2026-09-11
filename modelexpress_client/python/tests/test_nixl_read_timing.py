# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Distinguish transfer completion from handle cleanup and device synchronization."""
import json
import logging
from unittest.mock import MagicMock

import pytest

from modelexpress import nixl_transfer
from modelexpress.nixl_transfer import NixlTransferManager, PostedRead
from modelexpress.refit.timing import RefitTimingRecorder, use_refit_timing


def test_read_timing_separates_completion_release_and_sync(monkeypatch, caplog):
    now = [0.2]
    monkeypatch.setattr(nixl_transfer.time, "perf_counter", lambda: now[0])
    backend = MagicMock()
    manager = NixlTransferManager("receiver", 2, accelerator_backend=backend)
    manager._agent = MagicMock()
    handles = [object(), object()]

    def complete(handle):
        now[0] += 0.1
        return "DONE"

    def release(handle):
        now[0] += 0.025

    def synchronize(device):
        assert device == 2
        now[0] += 0.7

    manager._agent.check_xfer_state.side_effect = complete
    manager._agent.release_xfer_handle.side_effect = release
    backend.synchronize.side_effect = synchronize
    batches = [
        PostedRead(handles[0], "source0", 100, 1, 0.0, 0.05, 0.1, (0,)),
        PostedRead(handles[1], "source1", 200, 2, 0.1, 0.15, 0.2, (1,)),
    ]
    recorder = RefitTimingRecorder(backend="test", version=3, version_id="v3", rank=2)
    with caplog.at_level(logging.INFO), use_refit_timing(recorder):
        total, ranges, duration = manager.await_read_batches(batches)
    records = [r.message for r in caplog.records if r.message.startswith("MX_NIXL_READ_TIMING ")]
    assert len(records) == 1
    record = json.loads(records[0].split(" ", 1)[1])
    assert (total, ranges) == (300, 3)
    assert duration == pytest.approx(1.15)
    assert record["version_id"] == "v3"
    assert record["rank"] == record["device_id"] == 2
    assert record["wait_ms"] == pytest.approx(200)
    assert record["release_ms"] == pytest.approx(50)
    assert record["device_sync_ms"] == pytest.approx(700)
    assert [b["completion_offset_ms"] for b in record["batches"]] == pytest.approx([300, 400])
    assert [b["bytes"] for b in record["batches"]] == [100, 200]
    assert [b["remote_devices"] for b in record["batches"]] == [[0], [1]]
    assert manager._agent.release_xfer_handle.call_count == 2
    backend.synchronize.assert_called_once_with(2)
