# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mooncake environment promotion helpers.

ModelExpress and other Mooncake workloads in the same process (e.g.
distributed KV cache or prefill/decode disaggregation) may use different
Mooncake clusters. ModelExpress settings are exported under an ``MX_MC_*``
prefix while the native Mooncake libraries only read the ``MC_*`` names.
Instead of translating every variable by hand, ``mx_mc_env_override``
snapshots each non-empty ``MX_MC_*`` variable, copies it onto the
corresponding ``MC_*`` name for the duration of the context, and restores the
previous ``MC_*`` values (or their absence) on exit.

This is a leaf module: it imports only the standard library, so it can be
unit-tested without torch, mooncake, or the rest of the ModelExpress client.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Iterator

logger = logging.getLogger("modelexpress.mooncake_env")

_MX_MC_PREFIX = "MX_MC_"
_MC_PREFIX = "MC_"


@contextmanager
def mx_mc_env_override() -> Iterator[None]:
    """Temporarily promote ``MX_MC_*`` variables onto the native ``MC_*`` names.

    Unset or empty ``MX_MC_*`` variables leave the native ``MC_*`` values
    untouched, preserving the old ``MX_MC_* > MC_*`` fallback order. The
    original ``MC_*`` values are restored (or removed, when they did not
    exist) on exit, even if the block raises.
    """
    overridden: list[tuple[str, str, bool]] = []
    try:
        for name, value in list(os.environ.items()):
            if not name.startswith(_MX_MC_PREFIX) or not value:
                continue
            native_name = _MC_PREFIX + name[len(_MX_MC_PREFIX):]
            overridden.append(
                (
                    native_name,
                    os.environ.get(native_name, ""),
                    native_name in os.environ,
                )
            )
            os.environ[native_name] = value
            logger.debug("[Mooncake] env promote: %s -> %s", name, native_name)
        yield
    finally:
        for native_name, previous, was_set in overridden:
            if was_set:
                os.environ[native_name] = previous
            else:
                os.environ.pop(native_name, None)
            logger.debug("[Mooncake] env restore: %s", native_name)
