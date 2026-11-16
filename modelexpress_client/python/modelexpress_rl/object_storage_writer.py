# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Internal write interface for canonical checkpoint artifacts."""

from typing import Protocol


class ObjectStorageWriter(Protocol):
    """Publish immutable objects; callers may issue concurrent writes."""

    def put(self, *, uri: str, data: bytes) -> None:
        """Create an object or accept identical bytes; reject conflicting bytes."""
        ...

    def close(self) -> None:
        """Release owned resources after writes have finished."""
        ...
