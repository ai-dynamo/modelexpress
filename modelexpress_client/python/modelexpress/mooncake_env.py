# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mooncake environment promotion helpers.

ModelExpress and other Mooncake workloads in the same process (e.g.
distributed KV cache or prefill/decode disaggregation) may use different
Mooncake clusters. ModelExpress settings are exported under an ``MX_MC_*``
prefix while the native Mooncake libraries only read the ``MC_*`` names.
Likewise, artifact-store etcd client settings use ``MX_ETCD_*`` while the
native client reads ``ETCD_*``. Instead of translating every variable by hand,
``mx_mc_env_override`` snapshots each non-empty prefixed variable, promotes it
onto its native name for the duration of the context, and restores the
previous native values (or their absence) on exit.

This is a leaf module: it imports only the standard library, so it can be
unit-tested without torch, mooncake, or the rest of the ModelExpress client.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Iterator

logger = logging.getLogger("modelexpress.mooncake_env")

# Mooncake uses ``MC_*`` while its etcd client reads ``ETCD_*``.
_PROMOTION_PREFIXES = (
    ("MX_MC_", "MC_"),
    ("MX_ETCD_", "ETCD_"),
)


@contextmanager
def mx_mc_env_override() -> Iterator[None]:
    """Temporarily promote ModelExpress Mooncake and etcd configuration.

    ``MX_MC_*`` promotes to ``MC_*`` and ``MX_ETCD_*`` promotes to ``ETCD_*``.
    Unset or empty prefixed variables leave the native values untouched,
    preserving the fallback order. The original native values are restored
    (or removed, when they did not exist) on exit, even if the block raises.
    """
    overridden: list[tuple[str, str, bool]] = []
    try:
        for source_prefix, native_prefix in _PROMOTION_PREFIXES:
            for name, value in list(os.environ.items()):
                if not name.startswith(source_prefix) or not value:
                    continue
                native_name = native_prefix + name[len(source_prefix):]
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
