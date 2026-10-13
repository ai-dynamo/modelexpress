# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Conftest for GPU-resident real-model tests.

These tests require real CUDA and vLLM. The shared conftest leaves an installed
vLLM unchanged, so real-model integration uses the real modules.
"""
