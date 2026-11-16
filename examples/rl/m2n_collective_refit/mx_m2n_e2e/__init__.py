# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real-framework end-to-end refit over the NCCL M2N collective path.

A trainer holding a real checkpoint, sharded by a real framework, refits a
live vLLM engine through the ModelExpress control plane. The engine boundary
is the only framework-specific code; everything else is the shipped client.
"""
