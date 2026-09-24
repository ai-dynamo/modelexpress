# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ModelExpress clients and protobuf bindings for RL weight refit.

Exports are resolved lazily. The two refit paths have disjoint dependencies:
the NIXL pull client needs a NIXL agent, the collective client needs nccl4py.
Importing either eagerly here would force both on every deployment, and would
break the property the collective path is built around: ``import
modelexpress_rl.collective`` must not pull in NIXL.
"""

from typing import Any

_LAZY = {
    # Framework-facing clients.
    "ModelExpressControlClient": ".control",
    "ModelExpressGeneratorClient": ".inference",
    "ModelExpressTrainerClient": ".train",
    # Configuration fixed when a worker client is initialized.
    "ModelExpressGeneratorConfig": ".inference",
    "ModelExpressTrainerConfig": ".train",
    "FSDPTrainerContext": ".train",
    "MegatronTrainerContext": ".train",
    "ObjectStorageConfig": ".train",
    "ObjectStorageGeneratorConfig": ".inference",
    "SglangGeneratorContext": ".inference",
    "TrainerStagingMode": ".train",
    "TrainerEngineContext": ".train",
    "WeightPayloadFormat": ".train",
    "WeightSource": ".inference",
    "VllmGeneratorContext": ".inference",
    # Version values shared across the control, trainer, and generator clients.
    "ObjectStorageSource": ".object_storage",
    "ObjectStorageType": ".object_storage",
    "WeightVersion": ".control",
    "WeightVersionRef": ".version",
    "WeightVersionState": ".control",
}

__all__ = [  # noqa: RUF022 - grouped by public API role, not alphabetically.
    # Framework-facing clients.
    "ModelExpressControlClient",
    "ModelExpressGeneratorClient",
    "ModelExpressTrainerClient",
    # Configuration fixed when a worker client is initialized.
    "ModelExpressGeneratorConfig",
    "ModelExpressTrainerConfig",
    "FSDPTrainerContext",
    "MegatronTrainerContext",
    "ObjectStorageConfig",
    "ObjectStorageGeneratorConfig",
    "SglangGeneratorContext",
    "TrainerStagingMode",
    "TrainerEngineContext",
    "WeightPayloadFormat",
    "WeightSource",
    "VllmGeneratorContext",
    # Version values shared across the control, trainer, and generator clients.
    "ObjectStorageSource",
    "ObjectStorageType",
    "WeightVersion",
    "WeightVersionRef",
    "WeightVersionState",
]


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
