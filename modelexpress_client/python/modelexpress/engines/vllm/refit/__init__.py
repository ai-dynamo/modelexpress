# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM-specific capture and installation support for live model refit."""

from .installer import MdlLoader

__all__ = ["MdlLoader"]
