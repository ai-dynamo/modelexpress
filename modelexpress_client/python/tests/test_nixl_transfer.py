# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for nixl_transfer region-layout validation.

These tests exercise only pure-Python logic and do NOT require NIXL, CUDA,
or a GPU, so they can run in CI on any worker.
"""

from modelexpress.nixl_transfer import (
    NixlTransferManager,
    RegionLayoutMismatchError,
)
from modelexpress.types import TensorDescriptor


def _region_desc(i: int, addr: int, size: int) -> TensorDescriptor:
    """Build a region-style TensorDescriptor as emitted by the source side."""
    return TensorDescriptor(
        name=f"__region_{i}__",
        addr=addr,
        size=size,
        device_id=0,
        dtype="contiguous_region",
    )


class TestValidateRegionLayoutMatch:
    """Tests for NixlTransferManager._validate_region_layout_match.

    This guards against the bug where mismatched region layouts between
    source and recv produced `(src, local)` pairs of unequal length that
    NIXL rejected with NIXL_ERR_INVALID_PARAM after having silently logged
    per-region WARNINGs.
    """

    def test_identical_layouts_match(self):
        """Equal count and equal sizes (addresses may differ) -> match."""
        source = [
            _region_desc(0, 0x10000, 1024),
            _region_desc(1, 0x20000, 2048),
            _region_desc(2, 0x30000, 512),
        ]
        # Local addresses intentionally different — they're VAs in another process.
        local = [(0xAA000, 1024), (0xBB000, 2048), (0xCC000, 512)]

        ok, msg = NixlTransferManager._validate_region_layout_match(source, local)
        assert ok is True
        assert msg == ""

    def test_count_mismatch_does_not_match(self):
        """Different region counts must be rejected."""
        source = [_region_desc(i, 0x10000 * (i + 1), 1024) for i in range(3)]
        local = [(0xA000, 1024), (0xB000, 1024)]

        ok, msg = NixlTransferManager._validate_region_layout_match(source, local)
        assert ok is False
        assert "region count mismatch" in msg
        assert "3" in msg and "2" in msg

    def test_size_mismatch_does_not_match(self):
        """Same count but one region differs in size -> rejected."""
        source = [
            _region_desc(0, 0x10000, 1024),
            _region_desc(1, 0x20000, 2048),
            _region_desc(2, 0x30000, 512),
        ]
        local = [(0xA000, 1024), (0xB000, 9999), (0xC000, 512)]  # index 1 diverges

        ok, msg = NixlTransferManager._validate_region_layout_match(source, local)
        assert ok is False
        assert "size mismatch" in msg
        assert "region 1" in msg
        assert "2048" in msg
        assert "9999" in msg

    def test_size_mismatch_summary_caps_output(self):
        """Many mismatches produce a bounded summary, not spam per region."""
        # 20 regions, all size-mismatched
        source = [_region_desc(i, 0x1000 * (i + 1), 1024) for i in range(20)]
        local = [(0x10000 + 0x1000 * i, 4096) for i in range(20)]

        ok, msg = NixlTransferManager._validate_region_layout_match(source, local)
        assert ok is False
        # Summary should name the total but not list every one of the 20.
        assert "20 region size mismatch" in msg
        # Expect a "+N more" suffix indicating truncation.
        assert "more" in msg

    def test_empty_layouts_match_trivially(self):
        """Two empty layouts are (vacuously) equal."""
        ok, msg = NixlTransferManager._validate_region_layout_match([], [])
        assert ok is True
        assert msg == ""

    def test_regression_logged_failure_from_llama31_8b(self):
        """Regression test matching the exact scenario from the 2026-04-20 log.

        Source produced 223 regions, recv produced 219 regions, with specific
        size mismatches at indices 84, 85, 88-92, 216-218. Under the old
        code this proceeded anyway and NIXL rejected the transfer. Under
        the fix it must raise RegionLayoutMismatchError.
        """
        # Build two layouts with different counts (223 vs 219)
        source = [_region_desc(i, 0x10000 * (i + 1), 33554432) for i in range(223)]
        local = [(0xA0000 + i * 0x1000, 33554432) for i in range(219)]

        ok, msg = NixlTransferManager._validate_region_layout_match(source, local)
        assert ok is False
        assert "region count mismatch" in msg
        assert "223" in msg
        assert "219" in msg


class TestRegionLayoutMismatchError:
    """The exception used to signal layout disagreement to callers."""

    def test_is_exception(self):
        err = RegionLayoutMismatchError("layouts differ")
        assert isinstance(err, Exception)
        assert "layouts differ" in str(err)
