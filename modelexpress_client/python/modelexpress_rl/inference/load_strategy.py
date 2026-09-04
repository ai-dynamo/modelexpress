# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dedicated RL cold-start loading policy."""

from __future__ import annotations

import logging

import torch.nn as nn

from modelexpress.adapter import StrategyFailed
from modelexpress.load_strategy import execute_load_strategies
from modelexpress.load_strategy.context import LoadContext, LoadResult
from modelexpress.load_strategy.default_strategy import DefaultStrategy
from modelexpress.load_strategy.model_streamer_strategy import ModelStreamerStrategy
from modelexpress.load_strategy.rdma_strategy import RdmaStrategy

from .. import envs
from ..control import ModelExpressControlClient, WeightVersion, WeightVersionState
from ..object_storage import ObjectStorageType
from ..s3 import S3Client
from .methods import CanonicalDeltaUpdateMethod
from .plan import ObjectStorageUpdateSource
from .receiver import (
    ObjectStorageGeneratorConfig,
    _S3Version,
    bootstrap_s3_checkpoint,
)
from .version_chain import resolve_replay_chain

_MAX_REPLAY_CHAIN_LENGTH = 64
logger = logging.getLogger(__name__)


class DesiredVersionP2PStrategy(RdmaStrategy):
    """Load the desired immutable version from an existing generator."""

    name = "desired_version_p2p"

    def is_available(self, ctx: LoadContext) -> bool:
        return envs.MX_REFIT_DESIRED_VERSION_UID is not None and super().is_available(ctx)

    def load(self, result: LoadResult, ctx: LoadContext) -> LoadResult:
        desired_version_uid = envs.MX_REFIT_DESIRED_VERSION_UID
        if desired_version_uid is None:
            raise StrategyFailed("desired version is not configured", mutated=False)
        original_revision = ctx.identity.revision
        ctx.identity.revision = desired_version_uid
        try:
            return super().load(result, ctx)
        except BaseException:
            ctx.identity.revision = original_revision
            raise


class DesiredVersionS3Strategy(ModelStreamerStrategy):
    """Reconstruct the desired full-plus-delta chain and load it locally."""

    name = "desired_version_s3"

    def is_available(self, ctx: LoadContext) -> bool:
        return (
            envs.MX_REFIT_DESIRED_VERSION_UID is not None
            and envs.MX_REFIT_CHECKPOINT_DIR is not None
            and self.supports_explicit_uri(ctx)
        )

    def load(self, result: LoadResult, ctx: LoadContext) -> LoadResult:
        desired_version_uid = envs.MX_REFIT_DESIRED_VERSION_UID
        checkpoint_dir = envs.MX_REFIT_CHECKPOINT_DIR
        if desired_version_uid is None or checkpoint_dir is None:
            raise StrategyFailed(
                "desired version or checkpoint directory is not configured",
                mutated=False,
            )

        original_revision = ctx.identity.revision
        method = None
        prepared = None
        engine_loaded = False
        try:
            chain = _resolve_s3_replay_chain(ctx, desired_version_uid)
            root = chain[0]
            assert root.object_storage is not None
            s3 = S3Client()
            try:
                seed_path = bootstrap_s3_checkpoint(
                    model_name=ctx.identity.model_name,
                    version=_S3Version(
                        version_id=root.version_id,
                        base_version_id=root.base_version_id,
                        payload_format=root.payload_format,
                        uri=root.object_storage.uri,
                    ),
                    refit_checkpoint_dir=checkpoint_dir,
                    s3=s3,
                )
            finally:
                s3.close()

            method = CanonicalDeltaUpdateMethod(
                model_name=ctx.identity.model_name,
                config=ObjectStorageGeneratorConfig(
                    storage_type=ObjectStorageType.S3,
                    initial_base_version_id=root.version_id,
                    seed_checkpoint_path=seed_path,
                    refit_checkpoint_dir=checkpoint_dir,
                ),
            )
            prepared = method.prepare_chain(
                tuple(
                    (
                        version,
                        ObjectStorageUpdateSource(
                            storage=version.object_storage,
                            payload_format=version.payload_format,
                        ),
                    )
                    for version in chain
                )
            )
            with method.installation_context(prepared):
                loaded = self.load_uri(result, ctx, str(prepared.checkpoint.path))
                engine_loaded = True
            ctx.identity.revision = desired_version_uid
            return loaded
        except StrategyFailed:
            ctx.identity.revision = original_revision
            raise
        except Exception as error:
            ctx.identity.revision = original_revision
            raise StrategyFailed(str(error), mutated=engine_loaded) from error
        finally:
            if method is not None:
                try:
                    if prepared is not None:
                        method.release(prepared)
                except Exception:
                    logger.warning(
                        "failed to release RL cold-start checkpoint",
                        exc_info=True,
                    )
                try:
                    method.close()
                except Exception:
                    logger.warning(
                        "failed to close RL cold-start checkpoint receiver",
                        exc_info=True,
                    )


def _resolve_s3_replay_chain(
    ctx: LoadContext,
    target_version_uid: str,
) -> tuple[WeightVersion, ...]:
    """Resolve and validate a target back to its full HF root."""
    with ModelExpressControlClient.connect(server_url=ctx.mx_server_url) as client:
        def fetch_ready_version(version_uid: str) -> WeightVersion:
            version = client.get_weight_version(version_uid)
            if version.state is not WeightVersionState.READY:
                raise RuntimeError(f"revision {version_uid!r} is not READY")
            if version.model_name != ctx.identity.model_name:
                raise RuntimeError(
                    f"revision {version_uid!r} model_name does not match the worker"
                )
            if (
                version.object_storage is None
                or version.object_storage.storage_type is not ObjectStorageType.S3
            ):
                raise RuntimeError(f"revision {version_uid!r} has no S3 source")
            return version

        return resolve_replay_chain(
            target_version_id=target_version_uid,
            fetch_ready_version=fetch_ready_version,
            max_chain_length=_MAX_REPLAY_CHAIN_LENGTH,
        )


class RLLoadStrategyChain:
    """Load using the RL-specific cold-start fallback order."""

    @staticmethod
    def run(model: nn.Module, ctx: LoadContext) -> nn.Module:
        # A desired UID is a correctness constraint: version-agnostic fallbacks
        # could load different weights and let the worker serve the wrong version.
        if envs.MX_REFIT_DESIRED_VERSION_UID is not None:
            strategies = [
                DesiredVersionP2PStrategy(),
                DesiredVersionS3Strategy(),
            ]
        else:
            strategies = [
                ModelStreamerStrategy(),
                DefaultStrategy(),
            ]
        return execute_load_strategies(
            model,
            ctx,
            strategies,
        )


__all__ = [
    "DesiredVersionP2PStrategy",
    "DesiredVersionS3Strategy",
    "RLLoadStrategyChain",
]
