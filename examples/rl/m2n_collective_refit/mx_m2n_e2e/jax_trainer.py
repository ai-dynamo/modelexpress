# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trainer side, JAX: a sharded ``jax.Array`` tree published over M2N.

The counterpart of ``trainer.py``'s ``FsdpPublisher``, and deliberately shaped
the same way, because the two together are what show the engine boundary is
general rather than a torch interface with one other caller.

What differs from the torch side is only where the local bytes are found.
FSDP2 hands a ``DTensor`` whose ``to_local()`` is this rank's slice; JAX hands
a globally-sharded array whose ``addressable_shards[0].data`` is. Both are a
dense row-major buffer of the same extent, and on GPU the two produce
byte-identical storage for every dtype measured, which is what lets one plan
describe both sides of a mixed transfer.

What is *not* here is a runnable ``main``. The torch trainer's is exercised by
the end-to-end example on a real multi-GPU box; the JAX one has not been run
on one, and a plausible untested launcher is worth less than its absence.
"""

from __future__ import annotations

from typing import Any

from modelexpress_rl.collective import LocalParamSpec
from modelexpress_rl.collective import jax_interop


class JaxPublisher:
    """``Publisher`` over a dim-0-sharded JAX parameter tree.

    ``params`` maps each canonical parameter name onto its **global**
    ``jax.Array``. The shard is taken here rather than by the caller so the
    extent can be checked against what the plan promised.
    """

    def __init__(
        self,
        params: dict[str, Any],
        plan,
        groupings,
        mesh_size: int,
    ) -> None:
        self._plan = plan
        self._groupings = groupings
        self._specs: dict[str, LocalParamSpec] = {}
        self._arrays: list[Any] = []

        missing = [entry.name for entry in plan.bulk if entry.name not in params]
        if missing:
            raise KeyError(
                f"{len(missing)} planned parameter(s) are absent from the trainer "
                f"tree: {', '.join(missing[:5])}"
            )
        for entry in plan.bulk:
            shard = jax_interop.local_shard(
                params[entry.name],
                name=entry.name,
                expect_rows=entry.global_shape[0] // mesh_size,
            )
            self._arrays.append(shard)
            self._specs[entry.name] = LocalParamSpec(
                base=jax_interop.JaxDeviceBuffer(shard)
            )

    def capture(self):
        return self._plan

    def parameter_names(self) -> list[str]:
        return self._plan.parameter_names()

    def local_params(self) -> dict[str, LocalParamSpec]:
        return self._specs

    def start_new_round(self, version: str) -> None:
        """Wait for every shard's producing computation before the wire op.

        JAX enqueues on its own stream and ``reshard`` takes a bare device
        pointer with no stream handshake, so nothing else orders the transfer
        against the step that produced these weights. The race is measured
        rather than assumed: a consumer reading the buffer on another stream
        mid-computation sees intermediate values. One synchronize per round
        rather than per parameter, which the protocol allows because no
        parameter may change after the round opens.
        """
        jax_interop.ready(*self._arrays)

    def cleanup(self) -> None:
        self._specs.clear()
        self._arrays.clear()
