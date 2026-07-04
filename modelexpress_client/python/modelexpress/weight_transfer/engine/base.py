# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Abstract weight loader adapter.

Implement this interface to plug a new inference engine (SGLang, TRT-LLM,
etc.) into the trainer pull / push workflow without modifying any other
module.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Iterator

if TYPE_CHECKING:
    import torch.nn as nn
    from ..protocol.types import TrainerTable


class WeightLoaderAdapter(ABC):
    """Interface between the weight sync machinery and one inference engine.

    The two required methods cover the two sync directions:

      - ``iter_lazy_weights``  used by PullRole to drive the engine's
        native weight loader with LazyWeight placeholders (bake pass).

      - ``iter_param_shards``  used by PushRole to enumerate live
        parameter memory addresses for publishing as an InferenceTable.

    Optional hooks allow engines to run post-load processing (e.g. FP8
    quantization) after weights land in GPU memory.
    """

    @abstractmethod
    def iter_lazy_weights(
        self,
        table: TrainerTable,
    ) -> Iterator[tuple[str, Any]]:
        """Yield (param_name, LazyWeight) pairs for the bake pass.

        The pairs must be in the order that the engine's ``load_weights()``
        method expects, with names in the engine's own naming convention.
        Use an adapter (e.g. ``adapters.moe.MoEAdapter``) if the trainer
        and engine use different naming schemes.

        Args:
            table: TrainerTable from the current broadcast step.  The adapter
                reads ``table.tensors`` to know which names and shapes to emit.
        """

    @abstractmethod
    def iter_param_shards(self, model: nn.Module) -> Iterator[tuple[str, Any]]:
        """Yield (param_name, tensor) pairs for building an InferenceTable.

        Used by PushRole to register inference worker GPU memory with NIXL
        so the trainer can WRITE directly into live parameter storage.

        Only tensors that the trainer is responsible for syncing should be
        yielded here (i.e. skip embeddings, norms, biases if the trainer
        does not hold them).
        """

    def post_pull_hook(self, model: nn.Module) -> None:
        """Called after a PULL completes, before the engine serves requests.

        Override to run quantization kernel repack, FP8 scaling, or any
        other post-load processing the engine requires.  Default: no-op.
        """

    def post_push_hook(self, model: nn.Module) -> None:
        """Called after a PUSH completes (on the inference worker side).

        Override for any necessary cache invalidation or tensor reformat.
        Default: no-op.
        """
