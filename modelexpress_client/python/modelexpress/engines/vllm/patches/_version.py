# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared vLLM version detection for compatibility patches."""

from importlib.metadata import PackageNotFoundError, version as package_version
import re


def installed_vllm_minor() -> tuple[int, int] | None:
    try:
        installed_version = package_version("vllm")
    except PackageNotFoundError:
        return None
    version_match = re.match(r"^(\d+)\.(\d+)", installed_version)
    if version_match is None:
        return None
    major, minor = version_match.groups()
    return int(major), int(minor)
