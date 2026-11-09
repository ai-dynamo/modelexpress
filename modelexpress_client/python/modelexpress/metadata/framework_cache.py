# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Framework-owned file caches that share one artifact source type.

Kernel JIT caches each have their own ``MxSourceType``. Caches that belong to
one serving framework, such as vLLM's persisted startup plans, instead use
``MX_SOURCE_TYPE_FRAMEWORK_CACHE`` and identify themselves with a cache name,
a schema and an opaque compatibility key. Adding such a cache therefore
needs no protocol, server or transport change: an engine integration builds a
``FrameworkCacheArtifact`` and exposes it on its adapter as
``framework_cache_artifacts``.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

from .. import envs, p2p_pb2
from .artifact_transfer import ArtifactCacheRoot, ArtifactTransfer, TarredArtifactTransfer
from .source_id import compute_mx_source_id

FRAMEWORK_CACHE_NAME_KEY = "framework_cache_name"
FRAMEWORK_CACHE_SCHEMA_KEY = "framework_cache_schema"
_SUPPORTED_FRAMEWORKS = (
    p2p_pb2.BACKEND_FRAMEWORK_VLLM,
    p2p_pb2.BACKEND_FRAMEWORK_SGLANG,
    p2p_pb2.BACKEND_FRAMEWORK_TRT_LLM,
)


@dataclass(frozen=True)
class FrameworkCacheArtifact:
    """One framework-owned cache directory and the identity it is shared under.

    ``compatibility_key`` is an opaque string chosen by the engine integration.
    Two processes may exchange the cache only if their keys are equal, so it
    must cover every setting that makes the cache content invalid elsewhere.
    Local paths are placement policy and never enter the identity.

    ``include`` optionally selects which locally produced files under the root
    are published. It receives each file's POSIX path relative to the root and
    is evaluated when the bundle is prepared, after the engine has finished
    starting. Files installed for the same identity are published as well
    while their content is unchanged, so NIXL peers relay them like any other
    cache. Without ``include`` the whole root is published. Installation
    always extracts into ``root`` and keeps unrelated files.
    """

    name: str
    backend_framework: int
    backend_framework_version: str
    compatibility_key: str
    schema: str
    root: Path
    include: Callable[[str], bool] | None = None

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[a-z][a-z0-9_]*", self.name):
            raise ValueError("framework cache name must be a lowercase cache identifier")
        if self.backend_framework not in _SUPPORTED_FRAMEWORKS:
            raise ValueError("framework cache requires a known backend_framework")
        for field in ("backend_framework_version", "compatibility_key", "schema"):
            if not getattr(self, field).strip():
                raise ValueError(f"framework cache requires {field}")

    def identity(self, model_name: str) -> p2p_pb2.SourceIdentity:
        """Return the source identity used to publish and discover this cache."""
        if not model_name.strip():
            raise ValueError("framework cache requires model_name")
        return p2p_pb2.SourceIdentity(
            mx_source_type=p2p_pb2.MX_SOURCE_TYPE_FRAMEWORK_CACHE,
            model_name=model_name,
            backend_framework=self.backend_framework,
            backend_framework_version=self.backend_framework_version,
            compile_config_digest=_compile_config_digest(self.compatibility_key),
            extra_parameters={
                FRAMEWORK_CACHE_NAME_KEY: self.name,
                FRAMEWORK_CACHE_SCHEMA_KEY: self.schema,
            },
        )

    def entry(
        self, model_name: str, bundle_root: Path
    ) -> tuple[ArtifactTransfer, p2p_pb2.SourceIdentity]:
        """Return the transfer and identity consumed by the artifact lifecycle."""
        identity = self.identity(model_name)
        transfer = TarredArtifactTransfer(
            name=self.name,
            mx_source_type=identity.mx_source_type,
            roots=(ArtifactCacheRoot("primary", self.root, self.root),),
            # Kernel caches use bundle_root/<cache name>; the separate parent
            # keeps a framework cache from sharing a bundle directory with one.
            bundle_root=(
                bundle_root / "framework_caches" / self.name
                / compute_mx_source_id(identity)
            ),
        )
        if self.include is not None:
            transfer = replace(
                transfer,
                include=_SelectedOrInstalled(self.include, transfer, identity),
            )
        return transfer, identity


def framework_cache_entries(
    ctx, bundle_root: Path
) -> list[tuple[ArtifactTransfer, p2p_pb2.SourceIdentity]]:
    """Return artifact entries for the caches the engine adapter registered."""
    caches = getattr(getattr(ctx, "adapter", None), "framework_cache_artifacts", ())
    return [cache.entry(ctx.identity.model_name, bundle_root) for cache in caches]


class _SelectedOrInstalled:
    """Publish files the engine selects plus files installed for this identity.

    The artifact lifecycle records an install receipt for filtered transfers.
    Files listed there are relayed while their content is unchanged, so NIXL
    peers republish what they fetched like any other cache. The receipt is
    parsed again only when it changes.
    """

    def __init__(
        self,
        include: Callable[[str], bool],
        transfer: ArtifactTransfer,
        identity: p2p_pb2.SourceIdentity,
    ) -> None:
        self._include = include
        self._transfer = transfer
        self._identity = identity
        self._receipt: Path | None = None
        self._installed: tuple[int, dict[str, str]] | None = None

    def __call__(self, relative_path: str) -> bool:
        # Imported lazily: the lifecycle module pulls in engine-facing loading
        # code that cache descriptors do not otherwise need.
        from .artifact_lifecycle import file_digest

        expected = self._installed_files().get(relative_path)
        if expected is not None:
            root = self._transfer.roots[0].target_root
            if file_digest(root / relative_path) == expected:
                return True
        return self._include(relative_path)

    def _installed_files(self) -> dict[str, str]:
        from .artifact_lifecycle import install_receipt_path, read_install_receipt

        if self._receipt is None:
            self._receipt = install_receipt_path(self._transfer, self._identity)
        try:
            version = self._receipt.stat().st_mtime_ns
        except OSError:
            return {}
        if self._installed is None or self._installed[0] != version:
            self._installed = (version, read_install_receipt(self._receipt))
        return self._installed[1]


def _compile_config_digest(compatibility_key: str) -> str:
    # The raw key may be long and is case-sensitive, while identity strings are
    # normalized before hashing, so only its digest enters the identity. An
    # operator-supplied MX_ARTIFACT_COMPILE_CONFIG_DIGEST partitions framework
    # caches the same way it partitions kernel caches.
    material = {"compatibility_key": compatibility_key}
    operator_digest = envs.MX_ARTIFACT_COMPILE_CONFIG_DIGEST
    if operator_digest:
        material["operator_digest"] = operator_digest
    return hashlib.sha256(
        json.dumps(material, sort_keys=True).encode("utf-8")
    ).hexdigest()
