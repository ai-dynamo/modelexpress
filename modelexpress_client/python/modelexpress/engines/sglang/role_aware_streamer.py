# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang ModelStreamer loader with role-scoped checkpoint shard selection."""

from __future__ import annotations

import logging

from sglang.srt.model_loader.loader import RunaiModelStreamerLoader

from .checkpoint_selection import CheckpointWeightRule, select_role_shards

logger = logging.getLogger(__name__)


class RoleAwareModelStreamerLoader(RunaiModelStreamerLoader):
    def __init__(
        self,
        load_config,
        *,
        model_uri: str,
        selector: CheckpointWeightRule,
        role: str,
        rank: int,
    ):
        super().__init__(load_config)
        self._mx_model_uri = model_uri
        self._mx_selector = selector
        self._mx_role = role
        self._mx_rank = rank

    def _prepare_weights(self, source_uri, revision):
        folder, files = super()._prepare_weights(source_uri, revision)
        if source_uri != self._mx_model_uri:
            return folder, files
        selected = select_role_shards(
            folder, files, self._mx_selector, self._mx_role
        )
        if selected is not None:
            logger.info(
                "[Worker %s] Selected %d/%d %s checkpoint shards",
                self._mx_rank, len(selected), len(files), self._mx_role,
            )
            files = selected
        return folder, files
