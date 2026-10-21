# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility alias for RL refit timing."""

import sys

from modelexpress_rl.refit import timing

sys.modules[__name__] = timing
