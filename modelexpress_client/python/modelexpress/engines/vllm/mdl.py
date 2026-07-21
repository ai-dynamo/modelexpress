# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility export for the vLLM refit installer.

New code should import :class:`MdlLoader` from
``modelexpress.engines.vllm.refit``.
"""

from .refit import MdlLoader

__all__ = ["MdlLoader"]
