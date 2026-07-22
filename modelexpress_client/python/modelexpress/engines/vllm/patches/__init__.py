# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime compatibility patches for vLLM."""

from .patch_humming_regex_ignore import patch_humming_regex_ignore
from .patch_object_storage_format_check import patch_object_storage_format_check

PATCHES = (
    patch_object_storage_format_check,
    patch_humming_regex_ignore,
)

__all__ = ["PATCHES"]
