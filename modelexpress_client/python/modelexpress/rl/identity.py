# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Portable identities for Miles delta weight versions."""

from __future__ import annotations

import hashlib
import os
from importlib.metadata import version as package_version

from modelexpress import p2p_pb2

_INDEX_NAME = "model.safetensors.index.json"
_READ_CHUNK_SIZE = 8 * 1024 * 1024


def _hash_file(digest, path: str) -> None:
    with open(path, "rb") as file:
        while chunk := file.read(_READ_CHUNK_SIZE):
            digest.update(chunk)


def content_digest(version_dir: str) -> str:
    """Hash delta shard bytes in filename order followed by the index bytes."""
    shard_names = sorted(
        name
        for name in os.listdir(version_dir)
        if name.endswith(".safetensors")
        and os.path.isfile(os.path.join(version_dir, name))
    )

    digest = hashlib.sha256()
    for shard_name in shard_names:
        _hash_file(digest, os.path.join(version_dir, shard_name))
    _hash_file(digest, os.path.join(version_dir, _INDEX_NAME))
    return digest.hexdigest()


def build_delta_identity(
    model_name: str,
    revision: str,
    training_step: int,
    base_version: int,
    layout_signature: str,
) -> p2p_pb2.SourceIdentity:
    """Build the weights-masquerade identity for one delta version."""
    try:
        mx_version = package_version("modelexpress")
    except Exception:
        mx_version = "0.0.0"

    return p2p_pb2.SourceIdentity(
        mx_version=mx_version,
        mx_source_type=p2p_pb2.MX_SOURCE_TYPE_WEIGHTS,
        model_name=model_name,
        revision=revision,
        extra_parameters={
            "training_step": str(training_step),
            "base_version": str(base_version),
            "layout_signature": str(layout_signature),
        },
    )
