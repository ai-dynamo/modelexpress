# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Disk weight source (default).

Returns None to signal the engine launcher to use the engine's
built-in disk loading mechanism.
"""


def get_weights_iterator():
    """Return None to use the engine's built-in disk loader."""
    return None
