# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ModelExpress clients and protobuf bindings for RL weight refit.

Exports are resolved lazily. The two refit paths have disjoint dependencies --
the NIXL pull client needs a NIXL agent, the collective client needs nccl4py --
and eagerly importing either here would force both on every deployment. That
would also quietly break the property the collective path is built around:
``import modelexpress_rl.collective`` must not pull in NIXL.
"""

from typing import Any

_LAZY = {
    "ModelExpressTrainerClient": ".client",
    "StagedWeightVersionShard": ".client",
    "WeightVersionRef": ".client",
    "CompletionFence": ".train",
    "StagedWeightVersionShardData": ".train",
    "TrainerEngineAdapter": ".train",
    "TrainerStagingMode": ".train",
    "WeightPayloadFormat": ".train",
    "WeightVersionShardManifest": ".train",
    "WeightVersionShardManifestPublisher": ".train",
    "WeightVersionShardManifestService": ".train",
}

__all__ = sorted(_LAZY)


def __getattr__(name: str) -> Any:
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals()))
