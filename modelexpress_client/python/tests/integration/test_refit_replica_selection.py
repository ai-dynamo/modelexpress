# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Replica selection must install current values from each selected logical slot."""

import pytest

from .nixl_cpu_harness import run_replica_case

pytestmark = pytest.mark.cpu_nixl


@pytest.mark.parametrize("independent_slots", [False, True])
def test_repeated_replica_selection_installs_current_weights(independent_slots):
    run_replica_case(independent_slots=independent_slots)


def test_selected_replica_failure_then_recovery_installs_current_weights():
    run_replica_case(failure=True)
