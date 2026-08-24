# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in smoke check for a caller-provided Mooncake artifact cluster."""

from __future__ import annotations

import os
from uuid import uuid4

import pytest

from modelexpress.metadata import mooncake_artifact_cache as mc


if os.getenv("MX_RUN_MOONCAKE_INTEGRATION") != "1":
    pytest.skip(
        "set MX_RUN_MOONCAKE_INTEGRATION=1 to use a preconfigured Mooncake store",
        allow_module_level=True,
    )


@pytest.mark.slow
def test_real_mooncake_store_put_get_and_missing_key():
    prefix = f"modelexpress-test/{uuid4().hex}"
    payload = b"modelexpress-mooncake-smoke"
    with mc._store_session() as store:
        assert store.put_bytes(f"{prefix}/present", payload) == 0
        assert store.get_bytes(f"{prefix}/present", expected_size=len(payload)) == payload
        assert store.get_bytes(f"{prefix}/missing", expected_size=len(payload)) is None
        assert store.remove(f"{prefix}/present") in (0, -704)
