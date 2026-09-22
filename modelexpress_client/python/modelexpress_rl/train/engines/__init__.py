# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Training-engine adapters for ModelExpress RL."""
"""Trainer engine selection."""

from ..adapter import NixlMetadataProvider, TrainerEngineAdapter
from ..context import FSDPTrainerContext, MegatronTrainerContext, TrainerEngineContext


def _create_trainer_adapter(
    context: TrainerEngineContext,
    *,
    manager: NixlMetadataProvider,
    nixl_metadata_endpoint: str,
) -> TrainerEngineAdapter:
    if isinstance(context, FSDPTrainerContext):
        from .fsdp import FSDPTrainerAdapter

        adapter_type = FSDPTrainerAdapter
    elif isinstance(context, MegatronTrainerContext):
        from .megatron import MegatronTrainerAdapter

        adapter_type = MegatronTrainerAdapter
    else:
        raise TypeError(f"unsupported trainer context {type(context).__name__}")
    return adapter_type(
        manager=manager,
        nixl_metadata_endpoint=nixl_metadata_endpoint,
    )


__all__: list[str] = []
