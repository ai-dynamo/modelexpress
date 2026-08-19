# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Engine boundary for the NCCL M2N collective refit path.

The plan says *what* moves. This says *where it lives locally*, and it is the
only place a training framework's or an inference engine's storage layout is
encoded. Everything else on this path is framework-neutral, which is the whole
reason the boundary is this narrow.

Deliberately shaped like the contract NeMo RL already uses, so an existing
name-to-local-tensor map ports across with its hooks intact.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    import torch

    from .types import ReshardPlan


@dataclass
class RefitCtx:
    """Handoff between one parameter's ``pre`` and ``post`` hooks.

    The wire op reads only ``buf``. ``extra`` carries whatever the engine needs
    to finish the job in ``post`` -- most commonly the slice of a fused
    parameter the received tile belongs to.
    """

    buf: Any
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class LocalParamSpec:
    """How one canonical parameter is realized on this worker.

    ``base`` is the live local tensor when it can be sent from, or received
    into, directly. ``pre`` and ``post`` cover the cases where it cannot:

    - a trainer stacking its per-expert views into one grouped tensor;
    - a generator receiving into a scratch buffer because the destination is a
      slice of a fused parameter, then copying it into place.

    Both hooks and the wire op are enqueued on the same CUDA stream, so the
    staging is ordered against the transfer without a host synchronize.
    """

    base: Any = None
    pre: Callable[[Any], RefitCtx] | None = None
    post: Callable[[RefitCtx], None] | None = None

    def enter(self) -> RefitCtx:
        """Materialize the buffer the wire op should use."""
        if self.pre is not None:
            return self.pre(self.base)
        if self.base is None:
            raise ValueError("a LocalParamSpec needs either a base tensor or a pre hook")
        return RefitCtx(buf=self.base)

    def leave(self, ctx: RefitCtx) -> None:
        """Commit the transferred bytes, if this parameter needs it."""
        if self.post is not None:
            self.post(ctx)


@runtime_checkable
class Publisher(Protocol):
    """Trainer-side engine boundary.

    The Publisher owns every framework-specific decision on this path: which
    parameters exist, how their ranks are laid out, which of them the reshard
    can carry, and where each one lives in this process. The shared core never
    infers any of it from a parameter name.
    """

    def capture(self) -> ReshardPlan:
        """Declare the plan: names, shapes, dtypes, meshes, placements, partitions."""
        ...

    def local_params(self) -> dict[str, LocalParamSpec]:
        """Map each declared name onto this rank's local storage."""
        ...

    def start_new_round(self, version: str) -> None:
        """Prepare for one refit. May stage weights into a proxy buffer."""
        ...

    def cleanup(self) -> None: ...


@runtime_checkable
class Loader(Protocol):
    """Generator-side engine boundary."""

    def capture(self) -> ReshardPlan:
        """Declare the same plan the Publisher does, from this side's view."""
        ...

    def local_params(self) -> dict[str, LocalParamSpec]:
        """Map each declared name onto live engine storage or a receive buffer."""
        ...

    def start_new_round(self, version: str) -> None: ...

    def install(self, layer_group_id: int) -> None:
        """Commit one layer group's received buffers into the live model."""
        ...

    def finish(self) -> None:
        """Refresh derived state once every group has been installed."""
        ...

    def cleanup(self) -> None: ...


def resolve_specs(
    plan: ReshardPlan,
    specs: dict[str, LocalParamSpec],
) -> None:
    """Check that every declared parameter has local storage on this worker.

    Worth failing here rather than mid-collective. The plan is the op sequence
    both sides walk in lockstep, so a name the local map does not cover means
    this rank would skip an op its peers issue -- which hangs the communicator
    rather than raising on the rank that is actually misconfigured.
    """
    missing = [name for name in plan.parameter_names() if name not in specs]
    if missing:
        raise KeyError(
            f"{len(missing)} declared parameter(s) have no local storage on this worker: "
            f"{', '.join(missing[:5])}"
        )
