# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public object-storage values shared across ModelExpress RL clients."""

from dataclasses import dataclass
from enum import Enum


class ObjectStorageType(str, Enum):
    """Object storage provider holding one canonical weight version."""

    S3 = "S3"
    AZURE = "AZURE"
    GCS = "GCS"


@dataclass(frozen=True)
class ObjectStorageSource:
    """Typed object storage location for one canonical weight version."""

    storage_type: ObjectStorageType
    uri: str

    def __post_init__(self) -> None:
        if not isinstance(self.storage_type, ObjectStorageType):
            raise TypeError("storage_type must be an ObjectStorageType")
        if not self.uri.strip():
            raise ValueError("uri is required")


__all__ = ["ObjectStorageSource", "ObjectStorageType"]
