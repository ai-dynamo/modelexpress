# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM delta receiver: the two engine hooks against a live vLLM model.

The receiver is a client of
:class:`~modelexpress.engines.vllm.refit.delta_engine.MxWeightTransferEngine`,
which owns vLLM's update window. Both hooks here run inside that window, so
neither opens nor closes the layerwise reload.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from modelexpress.refit.receiver import (
    ModelExpressWeightReceiver,
    PreparedRevision,
    ReceiverConfig,
    ReceiverInstallError,
)

if TYPE_CHECKING:
    from vllm.config.load import LoadConfig

    from .delta_engine import MxWeightTransferEngine

logger = logging.getLogger(__name__)


class VllmWeightReceiver(ModelExpressWeightReceiver):
    """Delta receiver for a live vLLM model, driven by an MX transfer engine."""

    def __init__(
        self,
        config: ReceiverConfig,
        receiver_id: str,
        engine: MxWeightTransferEngine,
    ) -> None:
        # The engine rather than its model, because set_weight_update_target
        # retargets engine.model / engine.model_config for draft-model updates.
        # Set before super().__init__, which calls _launch_checkpoint.
        self._engine = engine
        super().__init__(config, receiver_id)

    @property
    def _load_config(self) -> LoadConfig:
        load_config = copy.copy(self._engine.vllm_config.load_config)
        try:
            load_config.load_format = "safetensors"
        except AttributeError:
            object.__setattr__(load_config, "load_format", "safetensors")
        return load_config

    def _launch_checkpoint(self) -> Path:
        from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

        model_config = self._engine.model_config
        folder, _, _ = DefaultModelLoader(self._load_config)._prepare_weights(
            model_config.model,
            None,
            model_config.revision,
            False,
            None,
        )
        return Path(folder)

    def install_prepared_checkpoint(self, prepared: PreparedRevision) -> None:
        from vllm.config import set_current_vllm_config
        from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

        try:
            # DefaultModelLoader reads the weight location off the model config,
            # so the prepared checkpoint is installed by pointing a copy at it.
            staged = copy.copy(self._engine.model_config)
            staged.model = str(prepared.path)
            staged.revision = None
            loader = DefaultModelLoader(self._load_config)
        except Exception as error:
            raise ReceiverInstallError(str(error), False) from error

        try:
            with set_current_vllm_config(self._engine.vllm_config):
                loader.load_weights(self._engine.model, staged)
        except Exception as error:
            raise ReceiverInstallError(str(error), True) from error
        logger.info(
            "[delta] installed revision %s from %s",
            prepared.target_version,
            prepared.path,
        )
