# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dedicated RL cold-start loading policy."""

from __future__ import annotations

import logging

import torch.nn as nn

from modelexpress.adapter import StrategyFailed, StrategyRecoveryError
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


def _gather_phase(
    ctx: LoadContext,
    *,
    phase: str,
    desired_version_uid: str | None,
    result: object,
) -> tuple[object, ...]:
    """Synchronize one cold-start phase and reject desired-version drift."""
    configured_version_uid = envs.MX_REFIT_DESIRED_VERSION_UID
    state = (phase, desired_version_uid, configured_version_uid, result)
    states = (
        ctx.adapter.all_gather_state(state)
        if ctx.adapter is not None
        else (state,)
    )
    for rank, peer in enumerate(states):
        if not isinstance(peer, tuple) or len(peer) != 4:
            raise StrategyRecoveryError(
                f"invalid distributed RL cold-start state from rank {rank}: {peer!r}"
            )
        peer_phase, peer_desired, peer_configured, _peer_result = peer
        if peer_phase != phase:
            raise StrategyRecoveryError(
                "distributed RL cold-start phase disagreement: "
                f"local={phase!r}, rank {rank}={peer_phase!r}"
            )
        if (
            peer_desired != desired_version_uid
            or peer_configured != desired_version_uid
        ):
            raise StrategyRecoveryError(
                "distributed RL cold-start desired-version disagreement: "
                f"pinned={desired_version_uid!r}, rank {rank} "
                f"pinned={peer_desired!r}, configured={peer_configured!r}"
            )
    return tuple(peer[3] for peer in states)


def _agree_desired_version(ctx: LoadContext) -> str | None:
    """Pin the desired version from global rank zero across all workers."""
    configured = envs.MX_REFIT_DESIRED_VERSION_UID
    desired_version_uid = (
        ctx.adapter.broadcast_state(configured)
        if ctx.adapter is not None
        else configured
    )
    if desired_version_uid is not None and not isinstance(desired_version_uid, str):
        raise StrategyRecoveryError(
            "global rank 0 broadcast an invalid RL cold-start desired version: "
            f"{desired_version_uid!r}"
        )
    _gather_phase(
        ctx,
        phase="desired_version",
        desired_version_uid=desired_version_uid,
        result=None,
    )
    return desired_version_uid


def _require_uniform_result(
    results: tuple[object, ...],
    *,
    phase: str,
) -> object:
    """Return the shared result or reject disagreement between ranks."""
    first = results[0]
    if any(result != first for result in results[1:]):
        raise StrategyRecoveryError(
            f"distributed RL cold-start result disagreement during {phase}: {results!r}"
        )
    return first


def _fetch_ready_version(
    client: ModelExpressControlClient,
    ctx: LoadContext,
    version_uid: str,
    *,
    require_s3: bool,
) -> WeightVersion:
    """Validate the control-plane record required by a load strategy."""
    version = client.get_weight_version(version_uid)
    if version.version_id != version_uid:
        raise RuntimeError(
            f"requested revision {version_uid!r} but MX returned "
            f"{version.version_id!r}"
        )
    if version.state is not WeightVersionState.READY:
        raise RuntimeError(f"revision {version_uid!r} is not READY")
    if version.model_name != ctx.identity.model_name:
        raise RuntimeError(
            f"revision {version_uid!r} model_name does not match the worker"
        )
    if require_s3 and (
        version.object_storage is None
        or version.object_storage.storage_type is not ObjectStorageType.S3
    ):
        raise RuntimeError(f"revision {version_uid!r} has no S3 source")
    return version


class DesiredVersionP2PStrategy(RdmaStrategy):
    """Load the desired immutable version from an existing generator."""

    name = "desired_version_p2p"

    def is_available(self, ctx: LoadContext) -> bool:
        """Report P2P availability only when every rank agrees."""
        desired_version_uid = ctx.desired_version_uid
        if desired_version_uid is None:
            return False
        available = super().is_available(ctx)
        results = _gather_phase(
            ctx,
            phase="p2p_available",
            desired_version_uid=desired_version_uid,
            result=available,
        )
        return bool(_require_uniform_result(results, phase="P2P availability"))

    def load(self, result: LoadResult, ctx: LoadContext) -> LoadResult:
        """Load the pinned version over P2P and reconcile rank outcomes."""
        desired_version_uid = ctx.desired_version_uid
        if desired_version_uid is None:
            raise StrategyFailed("desired version is not configured", mutated=False)
        loaded = None
        failure: BaseException | None = None
        outcome = "loaded"
        try:
            with ModelExpressControlClient.connect(
                server_url=ctx.mx_server_url
            ) as client:
                _fetch_ready_version(
                    client,
                    ctx,
                    desired_version_uid,
                    require_s3=False,
                )
            original_revision = ctx.identity.revision
            ctx.identity.revision = desired_version_uid
            try:
                loaded = super().load(result, ctx)
            except BaseException:
                # P2P mutates the identity before interruptible transfer work.
                ctx.identity.revision = original_revision
                raise
        except StrategyRecoveryError as error:
            failure = error
            outcome = "fatal"
        except StrategyFailed as error:
            failure = error
            outcome = "miss"
        except Exception as error:
            failure = StrategyFailed(str(error), mutated=False)
            outcome = "miss"

        outcomes = _gather_phase(
            ctx,
            phase="p2p_result",
            desired_version_uid=desired_version_uid,
            result=outcome,
        )
        agreed_outcome = _require_uniform_result(outcomes, phase="P2P load")
        if agreed_outcome == "fatal":
            raise StrategyRecoveryError(
                "a rank could not recover from desired-version P2P loading"
            ) from failure
        if failure is not None:
            raise failure
        assert loaded is not None
        return loaded


class DesiredVersionS3Strategy(ModelStreamerStrategy):
    """Reconstruct the desired full-plus-delta chain and load it locally."""

    name = "desired_version_s3"

    def is_available(self, ctx: LoadContext) -> bool:
        """Report S3 availability only when every rank can use it."""
        desired_version_uid = ctx.desired_version_uid
        if desired_version_uid is None:
            return False
        available = envs.MX_REFIT_CHECKPOINT_DIR is not None and (
            self.supports_explicit_uri(ctx)
        )
        results = _gather_phase(
            ctx,
            phase="s3_available",
            desired_version_uid=desired_version_uid,
            result=available,
        )
        return bool(_require_uniform_result(results, phase="S3 availability"))

    @staticmethod
    def _prepare(
        ctx: LoadContext,
        chain: tuple[WeightVersion, ...],
        checkpoint_dir: str,
    ):
        """Materialize a replay chain without activating its checkpoint."""
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
        try:
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
        except BaseException:
            method.close()
            raise
        return method, prepared

    def load(self, result: LoadResult, ctx: LoadContext) -> LoadResult:
        """Coordinate node-local reconstruction and distributed engine loading."""
        desired_version_uid = ctx.desired_version_uid
        checkpoint_dir = envs.MX_REFIT_CHECKPOINT_DIR
        if desired_version_uid is None or checkpoint_dir is None:
            raise StrategyFailed(
                "desired version or checkpoint directory is not configured",
                mutated=False,
            )

        original_revision = ctx.identity.revision
        method = None
        prepared = None
        try:
            chain = None
            chain_error = None
            try:
                chain = _resolve_s3_replay_chain(ctx, desired_version_uid)
            except BaseException as error:
                chain_error = error
            chain_result = (
                ("ready", tuple(version.version_id for version in chain))
                if chain is not None
                else ("failed",)
            )
            chain_results = _gather_phase(
                ctx,
                phase="s3_chain",
                desired_version_uid=desired_version_uid,
                result=chain_result,
            )
            agreed_chain = _require_uniform_result(
                chain_results,
                phase="S3 replay-chain resolution",
            )
            if agreed_chain[0] != "ready":
                raise StrategyRecoveryError(
                    "at least one rank failed to resolve the S3 replay chain"
                ) from chain_error
            assert chain is not None

            is_local_leader = ctx.local_rank == 0
            leader_error = None
            if is_local_leader:
                try:
                    method, prepared = self._prepare(ctx, chain, checkpoint_dir)
                except BaseException as error:
                    leader_error = error
            leader_results = _gather_phase(
                ctx,
                phase="s3_local_leader_prepared",
                desired_version_uid=desired_version_uid,
                result=(is_local_leader, leader_error is None),
            )
            if any(
                is_leader and not succeeded for is_leader, succeeded in leader_results
            ):
                raise StrategyRecoveryError(
                    "a local leader failed to prepare the S3 checkpoint"
                ) from leader_error

            follower_error = None
            if not is_local_leader:
                try:
                    method, prepared = self._prepare(ctx, chain, checkpoint_dir)
                except BaseException as error:
                    follower_error = error
            prepared_results = _gather_phase(
                ctx,
                phase="s3_all_prepared",
                desired_version_uid=desired_version_uid,
                result=leader_error is None and follower_error is None,
            )
            if not all(prepared_results):
                raise StrategyRecoveryError(
                    "at least one rank failed to attach to the S3 checkpoint"
                ) from follower_error
            assert method is not None and prepared is not None

            loaded = None
            load_error = None
            try:
                with method.installation_context(prepared, activate=False):
                    loaded = self.load_uri(
                        result,
                        ctx,
                        str(prepared.checkpoint.path),
                    )
            except BaseException as error:
                load_error = error
            load_results = _gather_phase(
                ctx,
                phase="s3_engine_loaded",
                desired_version_uid=desired_version_uid,
                result=load_error is None,
            )
            if not all(load_results):
                raise StrategyRecoveryError(
                    "at least one rank failed to load the prepared S3 checkpoint"
                ) from load_error

            activation_error = None
            if is_local_leader:
                try:
                    method.activate(prepared)
                except BaseException as error:
                    activation_error = error
            activation_results = _gather_phase(
                ctx,
                phase="s3_checkpoint_activated",
                desired_version_uid=desired_version_uid,
                result=(is_local_leader, activation_error is None),
            )
            if any(
                is_leader and not succeeded
                for is_leader, succeeded in activation_results
            ):
                raise StrategyRecoveryError(
                    "a local leader failed to activate the S3 checkpoint"
                ) from activation_error

            ctx.identity.revision = desired_version_uid
            assert loaded is not None
            return loaded
        except StrategyRecoveryError:
            ctx.identity.revision = original_revision
            raise
        except StrategyFailed:
            ctx.identity.revision = original_revision
            raise
        except Exception as error:
            ctx.identity.revision = original_revision
            raise StrategyFailed(str(error), mutated=False) from error
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
            return _fetch_ready_version(
                client,
                ctx,
                version_uid,
                require_s3=True,
            )

        return resolve_replay_chain(
            target_version_id=target_version_uid,
            fetch_ready_version=fetch_ready_version,
            max_chain_length=_MAX_REPLAY_CHAIN_LENGTH,
        )


class RLLoadStrategyChain:
    """Load using the RL-specific cold-start fallback order."""

    @staticmethod
    def run(model: nn.Module, ctx: LoadContext) -> nn.Module:
        """Execute the cold-start strategy chain for the agreed version."""
        ctx.desired_version_uid = _agree_desired_version(ctx)
        # A desired UID is a correctness constraint: version-agnostic fallbacks
        # could load different weights and let the worker serve the wrong version.
        if ctx.desired_version_uid is not None:
            strategies = [
                DesiredVersionP2PStrategy(),
                DesiredVersionS3Strategy(),
            ]
        else:
            logger.warning(
                "[Worker %s] RL initial load has no desired version; using "
                "version-agnostic ModelStreamer and engine-default fallbacks",
                ctx.global_rank,
            )
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
