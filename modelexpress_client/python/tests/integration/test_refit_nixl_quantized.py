# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Packed state plus scale behavior; not GPU quantizer qualification."""

import pytest

from .nixl_cpu_harness import run_case

pytestmark = pytest.mark.cpu_nixl


@pytest.mark.parametrize("publishers,receivers", [(1, 2), (2, 1), (2, 3)])
def test_packed_weights_and_scale_only_updates(publishers, receivers):
    run_case(publishers=publishers, receivers=receivers, quantized=True)
