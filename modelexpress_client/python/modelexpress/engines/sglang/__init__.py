# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang integration for ModelExpress."""

from .adapter import (
    SglangAdapter,
    build_sglang_load_context,
    build_sglang_source_identity,
)
from .loader import MxModelLoader
from .p2p_metadata_sync import (
    build_metadata_tar_bytes,
    discover_metadata_source_urls,
    extract_metadata_tar_bytes,
    fetch_metadata_from_source,
    publish_meta_source,
    start_meta_server,
    sync_or_serve_metadata,
)
from .registration import register as register_modelexpress_sglang

__all__ = [
    "MxModelLoader",
    "SglangAdapter",
    "build_metadata_tar_bytes",
    "build_sglang_load_context",
    "build_sglang_source_identity",
    "discover_metadata_source_urls",
    "extract_metadata_tar_bytes",
    "fetch_metadata_from_source",
    "publish_meta_source",
    "register_modelexpress_sglang",
    "start_meta_server",
    "sync_or_serve_metadata",
]
