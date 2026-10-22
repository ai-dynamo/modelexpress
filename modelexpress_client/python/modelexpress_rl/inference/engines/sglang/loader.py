# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RL policy for the shared SGLang loader orchestration."""

from modelexpress.engines.sglang.loader import MxModelLoader
from ...load_strategy import RLLoadStrategyChain


class RLMxModelLoader(MxModelLoader):
    def __init__(self, load_config):
        from .. import configure_engine_logging

        configure_engine_logging("sglang")
        super().__init__(load_config)

    def _validate_load_policy(self, transport: str) -> None:
        if transport != "nixl":
            raise ValueError("RL initial loading requires the SGLang nixl transport")

    def _run_load_strategy_chain(self, model, ctx):
        return RLLoadStrategyChain.run(model, ctx)
