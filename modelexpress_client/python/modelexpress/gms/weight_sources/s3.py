# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""S3 weight source using model streamer for high-throughput reads."""

from __future__ import annotations


def get_weights_iterator(
    bucket: str,
    prefix: str,
    model_config: object,
):
    """Stream model weight tensors from S3.

    Raises:
        NotImplementedError: S3 weight source is not yet implemented.
    """
    raise NotImplementedError("S3 weight source is not yet implemented")
