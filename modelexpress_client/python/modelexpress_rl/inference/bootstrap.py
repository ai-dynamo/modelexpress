# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Early transport initialization for embedded generator runtimes."""

from __future__ import annotations

import uuid

import torch

from .nixl_staged_transfer import _NixlStagedTransfer


class ModelExpressGeneratorBootstrap:
    """Own an MX NIXL transport created before an inference engine starts."""

    def __init__(self, *, device_id: int) -> None:
        self.worker_id = uuid.uuid4().hex[:8]
        self.device_id = device_id
        self._transfer: _NixlStagedTransfer | None = _NixlStagedTransfer(
            agent_name=f"mx-refit-{self.worker_id}",
            device_id=device_id,
            device=torch.device("cuda", device_id),
            listen_port=None,
        )
        self._claimed = False

    def claim(self, *, device_id: int) -> _NixlStagedTransfer:
        """Transfer ownership to one matching generator runtime."""
        if self._claimed:
            raise RuntimeError("generator bootstrap was already claimed")
        if device_id != self.device_id:
            raise ValueError(
                "generator bootstrap device does not match the engine: "
                f"{self.device_id} != {device_id}"
            )
        transfer = self._transfer
        if transfer is None:
            raise RuntimeError("generator bootstrap was closed")
        self._claimed = True
        self._transfer = None
        return transfer

    def close(self) -> None:
        """Close the transfer unless ownership has moved to a runtime."""
        transfer = self._transfer
        self._transfer = None
        if transfer is not None:
            transfer.close()


__all__ = ["ModelExpressGeneratorBootstrap"]
