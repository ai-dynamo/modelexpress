# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility aliases for the RL-owned reshard implementation."""

import sys
from importlib import import_module

from modelexpress_rl.refit import reshard

for _name in (
    "cuda_pool", "geometry", "receiver", "rendezvous", "slice_plan",
    "throughput", "transfer_plan", "types", "verify", "transport",
    "transport.base", "transport.nixl",
):
    sys.modules[f"{__name__}.{_name}"] = import_module(f"{reshard.__name__}.{_name}")

for _name in ("aliases", "publisher"):
    _module = import_module(f"modelexpress_rl.train.engines.megatron.{_name}")
    sys.modules[f"{__name__}.megatron_{_name}"] = _module
    setattr(reshard, f"megatron_{_name}", _module)

sys.modules[__name__] = reshard
