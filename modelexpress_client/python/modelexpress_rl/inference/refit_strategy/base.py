# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public source-strategy contract for ModelExpress RL refit."""

from abc import ABC, abstractmethod

from ...control import WeightVersion


class RefitStrategy(ABC):
    """Stage one exact weight version from a particular class of sources.

    Strategies may populate adapter-owned staging buffers but must not install
    weights into the live model. The generator client owns installation and
    staged-buffer release, so strategies do not expose rollback or close hooks.
    """

    @abstractmethod
    def stage(self, version: WeightVersion) -> object | None:
        """Return verified staged weights, or ``None`` after a clean source miss."""


__all__ = ["RefitStrategy"]
