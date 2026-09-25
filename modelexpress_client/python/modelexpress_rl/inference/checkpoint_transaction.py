# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: BLE001

"""Opt-in collective checkpoint installation at a caller-owned serving safe point.

Local BaseExceptions are gathered before re-raising so peers also fence.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import AbstractContextManager, ExitStack
from dataclasses import replace
from typing import Protocol

from .checkpoint_selection import CheckpointChanges
from .receiver import PreparedCheckpoint

logger = logging.getLogger(__name__)


class StagedCheckpoint(Protocol):
    def install(self) -> None:
        """Copy fully staged results into existing engine storage."""


class CheckpointTransactionError(RuntimeError):
    """The caller must keep serving fenced and recover the worker."""


def refit_checkpoint_collectively(
    prepared: PreparedCheckpoint,
    *,
    serving_version: str,
    world_size: int,
    all_gather: Callable[[object], tuple[object, ...]],
    installation_context: Callable[[], AbstractContextManager],
    prepare_partial: Callable[[PreparedCheckpoint], StagedCheckpoint | None],
    install_full: Callable[[], None],
    synchronize: Callable[[], None],
    activate: Callable[[], None],
    commit_version: Callable[[], None],
    fence: Callable[[], None],
) -> str:
    """Agree, stage, install, activate, then commit on every participating rank.

    Every rank must call this while serving is drained, the old runtime source
    is unpublished, and outstanding donor reads have finished. The gather transport must
    have a finite timeout and include the entire worker, not just one TP subgroup.
    ``fence`` must prevent inference and publication until worker recovery. No
    callbacks may publish the new runtime before this function returns. This API
    is experimental and is not called by the default warm-refit session.

    None from prepare_partial means unsupported and selects full reload on ALL
    ranks. Exceptions abort the transaction; there is no retry after mutation.
    """

    def gather(phase: str, value: object) -> tuple[object, ...]:
        state = (phase, serving_version, prepared.target_version, value)
        states = all_gather(state)
        if world_size < 1 or len(states) != world_size:
            raise CheckpointTransactionError("incomplete checkpoint rank agreement")
        values = []
        for peer in states:
            if not isinstance(peer, tuple) or len(peer) != 4 or peer[:3] != state[:3]:
                raise CheckpointTransactionError(
                    f"checkpoint rank disagreement during {phase}"
                )
            values.append(peer[3])
        return tuple(values)

    def succeeded(phase: str, error: BaseException | None) -> None:
        if not all(value is True for value in gather(phase, error is None)):
            raise CheckpointTransactionError(
                f"checkpoint transaction failed during {phase}"
            ) from error

    try:
        if not serving_version or not prepared.target_version:
            raise CheckpointTransactionError("checkpoint versions must be known")
        changes = gather("changes", prepared.changes)
        known = [item for item in changes if item is not None]
        agreed = known[0] if known else None
        if (
            not isinstance(agreed, CheckpointChanges)
            or agreed.base_version != serving_version
            or agreed.target_version != prepared.target_version
            or not isinstance(agreed.names, frozenset)
            or any(not isinstance(name, str) or not name for name in agreed.names)
            or any(item != agreed for item in known)
        ):
            logger.info(
                "Full checkpoint reload: changed metadata is missing, conflicting, "
                "or does not match the serving base and target"
            )
            agreed = None
        checkpoint = replace(prepared, changes=agreed)

        exit_error = None
        mode = "full"
        try:
            with ExitStack() as stack:
                entry_error = None
                try:
                    stack.enter_context(installation_context())
                except BaseException as error:
                    entry_error = error
                succeeded("locked", entry_error)

                staged = None
                stage_error = None
                try:
                    if agreed is not None:
                        staged = prepare_partial(checkpoint)
                except BaseException as error:
                    stage_error = error
                succeeded("staged", stage_error)
                partial = gather("selection", staged is not None)
                if all(value is True for value in partial):
                    mode = "partial"
                else:
                    if agreed is not None:
                        logger.info(
                            "Full checkpoint reload: at least one rank declined "
                            "partial installation; see rank-local capability logs"
                        )
                    staged = None

                load_error = None
                try:
                    if mode == "partial":
                        assert staged is not None
                        staged.install()
                    else:
                        install_full()
                    synchronize()
                except BaseException as error:
                    load_error = error
                succeeded("installed", load_error)
        except CheckpointTransactionError:
            raise
        except BaseException as error:
            exit_error = error
        # All shared checkpoint locks must be released before any exclusive
        # activation lock is acquired, including on co-located ranks.
        succeeded("unlocked", exit_error)

        activation_error = None
        try:
            activate()
        except BaseException as error:
            activation_error = error
        succeeded("activated", activation_error)

        commit_error = None
        try:
            commit_version()
        except BaseException as error:
            commit_error = error
        succeeded("committed", commit_error)
        return mode
    except BaseException:
        fence()
        raise
