# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


def pytest_configure(config):
    """Set asyncio_mode to auto so async tests run without explicit markers."""
    config.option.asyncio_mode = "auto"
