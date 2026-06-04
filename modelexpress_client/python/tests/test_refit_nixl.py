# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

from modelexpress.refit_nixl import _post_submit_probe


def test_post_submit_probe_writes_marker_without_sleep(monkeypatch, tmp_path):
    marker = tmp_path / "nixl-submitted.json"
    monkeypatch.setenv("MX_REFIT_NIXL_POST_SUBMIT_MARKER", str(marker))
    monkeypatch.setenv("MX_REFIT_NIXL_POST_SUBMIT_SLEEP_SECONDS", "0")

    _post_submit_probe(
        trace_label="trainer-rank0",
        remote_agent_name=b"remote-rank0",
        remote_descs=[(0x1000, 16, 0), (0x2000, 32, 0)],
        local_descs=[(0x3000, 16, 1), (0x4000, 32, 1)],
    )

    payload = json.loads(marker.read_text(encoding="utf-8"))
    assert payload["phase"] == "nixl.read_submitted"
    assert payload["trace_label"] == "trainer-rank0"
    assert payload["remote_agent_name"] == {
        "bytes": len(b"remote-rank0"),
        "hex_prefix": b"remote-rank0"[:16].hex(),
    }
    assert payload["remote_desc_count"] == 2
    assert payload["local_desc_count"] == 2
    assert payload["bytes"] == 48
    assert payload["sleep_seconds"] == 0.0
