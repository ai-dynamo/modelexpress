# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure-data op chain types.  No torch dependency -- safe to import anywhere."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OpSpec:
    """One recorded tensor operation in an op chain.

    Args:
        name: Tensor method name (e.g. "narrow", "view", "transpose").
        args: Positional arguments *after* self (e.g. (0, 2, 4) for narrow).
        kwargs: Keyword arguments.
    """

    name: str
    args: tuple
    kwargs: dict

    def __post_init__(self) -> None:
        object.__setattr__(self, "args", tuple(self.args))
        object.__setattr__(self, "kwargs", dict(self.kwargs))


# A sequence of ops applied left-to-right to a trainer tensor before copy_().
OpChain = tuple[OpSpec, ...]

# Closed set of ops that the weight loader is allowed to use in a copy_ path.
# Ops outside this set that appear in a bake pass will cause a loud failure
# rather than a silently wrong plan.
SUPPORTED_OPS: frozenset[str] = frozenset({
    "narrow",
    "view",
    "reshape",
    "transpose",
    "permute",
    "chunk",
    "split",
    "squeeze",
    "unsqueeze",
    "contiguous",
    "flatten",
    "__getitem__",
    "select",
    "t",
})
