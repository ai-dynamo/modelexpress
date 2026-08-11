# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Conftest for GPU-resident real-model tests.

This directory intentionally does NOT inherit the parent conftest vllm mocks:
loading real HuggingFace models via transformers+accelerate into CUDA is
incompatible with the MagicMock vllm shims in tests/conftest.py.
"""
