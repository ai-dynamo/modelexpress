# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Back-compat shim — the implementation moved to
``modelexpress.engines.vllm.weight_transfer`` to live alongside the
existing vLLM engine adapters. This module re-exports the public surface
so existing imports keep working.

Prefer ``from modelexpress.engines.vllm.weight_transfer import ...`` in
new code.
"""

from modelexpress.engines.vllm.weight_transfer import (  # noqa: F401
    MxInitInfo,
    MxTrainerSendArgs,
    MxUpdateInfo,
    MxWeightTransferEngine,
)
