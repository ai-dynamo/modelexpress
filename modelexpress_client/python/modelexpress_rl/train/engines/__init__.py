# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Training-engine adapters for ModelExpress RL."""

from ..adapter import NixlMetadataProvider, TrainerEngineAdapter
from ..context import FSDPTrainerContext, MegatronTrainerContext, TrainerEngineContext


def _create_trainer_adapter(
    context: TrainerEngineContext,
    *,
    manager: NixlMetadataProvider,
    nixl_metadata_endpoint: str,
) -> TrainerEngineAdapter:
    engine_kwargs: dict[str, object] = {}
    if isinstance(context, FSDPTrainerContext):
        from .fsdp import FSDPTrainerAdapter

        adapter_type = FSDPTrainerAdapter
        engine_kwargs["wire_dtype_overrides"] = context.wire_dtype_overrides
    elif isinstance(context, MegatronTrainerContext):
        from .megatron import MegatronTrainerAdapter

        adapter_type = MegatronTrainerAdapter
    else:
        raise TypeError(f"unsupported trainer context {type(context).__name__}")
    return adapter_type(
        manager=manager,
        nixl_metadata_endpoint=nixl_metadata_endpoint,
        **engine_kwargs,
    )


__all__: list[str] = []
