# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for weight source modules."""

from unittest.mock import MagicMock

import pytest


class TestDiskWeightSource:
    """Tests for disk weight source."""

    def test_returns_none(self):
        from modelexpress.gms.weight_sources.disk import get_weights_iterator

        assert get_weights_iterator() is None


class TestGdsWeightSource:
    """Tests for GDS weight source."""

    def test_raises_not_implemented(self):
        from modelexpress.gms.weight_sources.gds import get_weights_iterator

        with pytest.raises(NotImplementedError, match="GDS"):
            get_weights_iterator("model/path", MagicMock())


class TestS3WeightSource:
    """Tests for S3 weight source."""

    def test_raises_not_implemented(self):
        from modelexpress.gms.weight_sources.s3 import get_weights_iterator

        with pytest.raises(NotImplementedError, match="S3"):
            get_weights_iterator("bucket", "prefix", MagicMock())
