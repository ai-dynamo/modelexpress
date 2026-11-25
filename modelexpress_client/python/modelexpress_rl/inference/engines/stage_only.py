# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""An engine stand-in for a generator client that stages but never installs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..adapter import GeneratorEngineContext
from ..plan import EngineCapabilities, EngineInstaller, PreparedArtifact
from ..runtime import EngineRuntime


@dataclass(frozen=True)
class StageOnlyGeneratorContext(GeneratorEngineContext):
    """What staging needs to know about an engine that lives in another process."""

    # The engine's model name, which keys node-local state.
    model_name: str
    # What the engine's installer accepts.
    capabilities: EngineCapabilities


class _NoInstaller(EngineInstaller):
    def __init__(self, capabilities: EngineCapabilities) -> None:
        self._capabilities = capabilities

    @property
    def capabilities(self) -> EngineCapabilities:
        return self._capabilities

    def install(self, prepared: PreparedArtifact) -> Any:
        raise RuntimeError("a stage-only generator client cannot install weights")


def _create_stage_only_engine_runtime(
    context: StageOnlyGeneratorContext,
) -> EngineRuntime:
    return EngineRuntime(
        model_name=context.model_name,
        installer=_NoInstaller(context.capabilities),
    )


__all__ = ["StageOnlyGeneratorContext"]
