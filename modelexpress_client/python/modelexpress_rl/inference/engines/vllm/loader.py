# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RL policy for the shared vLLM loader orchestration."""

from modelexpress.engines.vllm.loader import MxModelLoader
from ...load_strategy import RLLoadStrategyChain


class RLMxModelLoader(MxModelLoader):
    def __init__(self, load_config):
        from .. import configure_engine_logging

        configure_engine_logging("vllm")
        super().__init__(load_config)

    def _validate_load_policy(self, *, is_speculative_draft: bool = False) -> None:
        if is_speculative_draft:
            raise ValueError("RL initial loading does not support speculative draft models")

    def _run_load_strategy_chain(self, model, ctx):
        return RLLoadStrategyChain.run(model, ctx)

    def download_model(self, model_config) -> None:
        """Leave acquisition to the RL cold-start policy."""
