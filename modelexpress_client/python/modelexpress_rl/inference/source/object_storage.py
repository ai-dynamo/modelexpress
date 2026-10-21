# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Version-level object-storage source resolution."""

import logging
from collections.abc import Iterator

from ...control import WeightVersion
from ...object_storage import ObjectStorageType
from ..plan import (
    ObjectStorageUpdateSource,
    ResolvedSource,
    SourceResolver,
    WeightSource,
)

logger = logging.getLogger("modelexpress_rl.inference.source.object_storage")


class ObjectStorageSourceResolver(SourceResolver):
    """Resolve the object-storage record carried by a weight version."""

    @property
    def kind(self) -> WeightSource:
        return WeightSource.OBJECT_STORAGE

    def supports(self, version: WeightVersion) -> bool:
        return version.object_storage is not None

    def payload_format(self, version: WeightVersion):
        return version.payload_format

    def candidates(self, version: WeightVersion) -> Iterator[ResolvedSource]:
        source = version.object_storage
        if source is None:
            return
        if source.storage_type is not ObjectStorageType.S3:
            logger.warning(
                "object-storage refit currently requires S3; skipping %s for "
                "version %s",
                source.storage_type,
                version.version_id,
            )
            return
        yield ObjectStorageUpdateSource(
            storage=source,
            payload_format=version.payload_format,
        )


__all__ = ["ObjectStorageSourceResolver"]
