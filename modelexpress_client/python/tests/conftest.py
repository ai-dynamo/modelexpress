# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Pre-mock vLLM modules so tests can run without vLLM installed.

This conftest runs before any test module is collected, injecting mock
modules into sys.modules for all vllm.* imports used by the modelexpress
package. BaseModelLoader is a real ABC so that __abstractmethods__
validation works correctly.
"""

import sys
from abc import ABC, abstractmethod
from unittest.mock import MagicMock

import torch.nn as nn


def _maybe_mock_vllm():
    """Inject vLLM mocks into sys.modules if vLLM is not installed."""
    try:
        import vllm  # noqa: F401
        return  # vLLM is available, no mocking needed
    except ImportError:
        pass

    # Real ABC so __abstractmethods__ propagates to subclasses
    class BaseModelLoader(ABC):
        def __init__(self, load_config):
            self.load_config = load_config

        @abstractmethod
        def download_model(self, model_config) -> None:
            raise NotImplementedError

        @abstractmethod
        def load_weights(self, model: nn.Module, model_config) -> None:
            raise NotImplementedError

        def load_model(self, vllm_config, model_config):
            raise NotImplementedError

    # No-op decorator that just returns the class
    def register_model_loader(load_format):
        def _wrapper(cls):
            return cls
        return _wrapper

    # Build mock module tree
    vllm_mods = {
        "vllm": MagicMock(),
        "vllm.config": MagicMock(),
        "vllm.config.load": MagicMock(),
        "vllm.model_executor": MagicMock(),
        "vllm.model_executor.model_loader": MagicMock(),
        "vllm.model_executor.model_loader.base_loader": MagicMock(),
        "vllm.model_executor.model_loader.default_loader": MagicMock(),
        "vllm.model_executor.model_loader.dummy_loader": MagicMock(),
        "vllm.model_executor.model_loader.utils": MagicMock(),
        "vllm.utils": MagicMock(),
        "vllm.utils.torch_utils": MagicMock(),
        "vllm.distributed": MagicMock(),
    }

    # Wire up real objects where behavior matters
    vllm_mods["vllm.model_executor.model_loader.base_loader"].BaseModelLoader = BaseModelLoader
    vllm_mods["vllm.model_executor.model_loader"].register_model_loader = register_model_loader
    vllm_mods["vllm.model_executor.model_loader"].BaseModelLoader = BaseModelLoader

    # set_default_torch_dtype needs to be a real context manager
    from contextlib import contextmanager

    @contextmanager
    def _set_default_torch_dtype(dtype):
        yield

    vllm_mods["vllm.utils.torch_utils"].set_default_torch_dtype = _set_default_torch_dtype

    for mod_name, mod in vllm_mods.items():
        sys.modules[mod_name] = mod


_maybe_mock_vllm()
