# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RL source publication lifecycle helpers."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from modelexpress import p2p_pb2
from modelexpress.metadata.heartbeat import HeartbeatThread
from modelexpress.nixl_transfer import NixlTransferManager
from modelexpress.rl_metadata import RlSourceRole

if TYPE_CHECKING:
    from modelexpress.client import MxClientBase

logger = logging.getLogger("modelexpress.rl_publication")


@dataclass(frozen=True)
class RlPublishedSource:
    """One live RL source advertisement and its backing NIXL state."""

    mx_source_id: str
    model_version: int
    role: RlSourceRole
    worker_rank: int
    tensors: dict[str, torch.Tensor]
    manager: NixlTransferManager
    heartbeat: HeartbeatThread


def snapshot_tensors_for_retention(
    tensors: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Clone tensors so retained publications keep stable bytes after mutation."""
    snapshots = {name: tensor.detach().clone() for name, tensor in tensors.items()}
    cuda_devices = {
        snapshot.device
        for snapshot in snapshots.values()
        if snapshot.device.type == "cuda"
    }
    for device in cuda_devices:
        torch.cuda.synchronize(device)
    return snapshots


class RlPublicationStore:
    """Track live RL source publications for one local worker."""

    def __init__(
        self,
        *,
        mx_client: "MxClientBase",
        worker_id: str,
        retain_latest_k: int,
    ) -> None:
        self.mx_client = mx_client
        self.worker_id = worker_id
        self.retain_latest_k = retain_latest_k
        self.sources: list[RlPublishedSource] = []

    @property
    def current(self) -> RlPublishedSource | None:
        return self.sources[-1] if self.sources else None

    def add(self, source: RlPublishedSource) -> None:
        self.sources.append(source)

    def close_duplicate(
        self,
        *,
        role: RlSourceRole,
        model_version: int,
        worker_rank: int,
    ) -> None:
        for source in list(self.sources):
            if (
                source.role == role
                and source.model_version == model_version
                and source.worker_rank == worker_rank
            ):
                self.close(source, mark_stale=True)

    def prune(self) -> None:
        latest_by_role: dict[RlSourceRole, int] = {}
        for source in self.sources:
            latest_by_role[source.role] = max(
                latest_by_role.get(source.role, 0),
                source.model_version,
            )

        for source in list(self.sources):
            latest_version = latest_by_role[source.role]
            oldest_retained = max(0, latest_version - self.retain_latest_k + 1)
            if source.model_version < oldest_retained:
                self.close(source, mark_stale=True)

    def close_current(self, *, mark_stale: bool) -> None:
        if self.current is not None:
            self.close(self.current, mark_stale=mark_stale)

    def close_all(self, *, mark_stale: bool) -> None:
        for source in list(self.sources):
            self.close(source, mark_stale=mark_stale)

    def shutdown_all(self) -> None:
        self.close_all(mark_stale=False)

    def close(self, source: RlPublishedSource, *, mark_stale: bool) -> None:
        if source not in self.sources:
            return
        source.heartbeat.stop(mark_stale=False)
        try:
            if mark_stale:
                self.mx_client.update_status(
                    source.mx_source_id,
                    self.worker_id,
                    source.worker_rank,
                    p2p_pb2.SOURCE_STATUS_STALE,
                )
        except Exception:
            logger.warning(
                "Failed to mark ModelExpress RL source stale: source_id=%s",
                source.mx_source_id,
                exc_info=True,
            )
        source.manager.shutdown()
        self.sources.remove(source)
