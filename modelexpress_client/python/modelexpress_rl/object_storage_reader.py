# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Internal read interface for canonical checkpoint artifacts."""

from typing import Protocol


class ObjectStorageReader(Protocol):
    """Read immutable objects; callers may issue concurrent reads."""

    def get(self, uri: str) -> bytes:
        """Return the complete object, raising on a missing or failed read."""
        ...

    def size(self, uri: str) -> int:
        """Return the object size in bytes without downloading its contents."""
        ...

    def close(self) -> None:
        """Release resources owned by this reader after reads have finished."""
        ...
