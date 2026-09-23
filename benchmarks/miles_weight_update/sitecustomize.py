# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Install benchmark-only MILES runtime hooks in Ray worker interpreters."""

import os

if os.environ.get("MILES_BENCH_ENABLE_RUNTIME_RECEIPTS") == "1":
    from runtime_harness import install_miles_hooks

    install_miles_hooks()
