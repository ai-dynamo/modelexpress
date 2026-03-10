# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPUDirect Storage weight source for fast NVMe reads.

Bypasses the page cache and reads directly from NVMe into GPU memory,
achieving up to ~25 GB/s per GPU vs ~6 GB/s for regular disk I/O.
"""

from __future__ import annotations


def get_weights_iterator(
    model_path: str,
    model_config: object,
):
    """Read model weight tensors via GDS (GPUDirect Storage).

    Raises:
        NotImplementedError: GDS weight source is not yet implemented.
    """
    raise NotImplementedError("GDS weight source is not yet implemented")
