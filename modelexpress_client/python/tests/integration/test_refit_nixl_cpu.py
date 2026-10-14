# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Logical refit correctness through real host-memory NIXL transfers."""

import pytest

from .nixl_cpu_harness import run_case

pytestmark = pytest.mark.cpu_nixl


@pytest.mark.parametrize("publishers,receivers", [(1, 2), (2, 1), (2, 3)])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16", "float16"])
def test_resharded_updates_match_fresh_load(publishers, receivers, dtype):
    run_case(publishers=publishers, receivers=receivers, dtype=dtype)


@pytest.mark.parametrize("lifecycle", ["reallocate", "restart"])
def test_refit_after_publisher_storage_changes(lifecycle):
    run_case(publishers=2, receivers=3, lifecycle=lifecycle)


def test_unknown_peer_fails_and_valid_retry_installs_weights():
    run_case(publishers=2, receivers=1, retry=True)


def test_publisher_death_fails_with_bounded_wait_and_restart_recovers():
    run_case(publishers=2, receivers=1, publisher_death=True)


def test_nonzero_parameter_storage_offset_preserves_values_and_guards():
    run_case(publishers=2, receivers=3, storage_offset=1)
