# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared test fixtures for modelexpress tests."""

import pytest


@pytest.fixture
def default_gms_config():
    """Create a default GmsConfig for testing."""
    from modelexpress.gms.config import GmsConfig

    return GmsConfig(model="test-model")


@pytest.fixture
def default_mx_config():
    """Create a default MxConfig for testing."""
    from modelexpress.gms.config import MxConfig

    return MxConfig(
        mx_server="localhost:8001",
        model_name="test-model",
        expected_workers=1,
    )


@pytest.fixture
def multi_gpu_gms_config():
    """Create a multi-GPU GmsConfig for testing."""
    from modelexpress.gms.config import GmsConfig

    return GmsConfig(
        model="test-model",
        tp_size=4,
        ep_size=2,
        mx_server="mx-server:8001",
    )
