# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
from types import ModuleType

import pytest
import torch

from modelexpress._nixl import load_nixl_api


@pytest.mark.parametrize("cuda", ["12.9", "13.0"])
def test_load_nixl_api_prefers_matching_cuda_package(monkeypatch, cuda) -> None:
    monkeypatch.setattr(torch.version, "cuda", cuda)
    expected = f"nixl_cu{cuda.split('.')[0]}._api"
    apis = {name: ModuleType(name) for name in ("nixl_cu12._api", "nixl_cu13._api")}
    attempted = []

    def import_module(name: str) -> ModuleType:
        attempted.append(name)
        if name in apis:
            return apis[name]
        raise ImportError(name)

    monkeypatch.setattr(importlib, "import_module", import_module)

    assert load_nixl_api() is apis[expected]
    assert attempted == [expected]


def test_load_nixl_api_does_not_try_other_cuda_major(monkeypatch) -> None:
    monkeypatch.setattr(torch.version, "cuda", "12.9")
    attempted = []

    def import_module(name):
        attempted.append(name)
        raise ModuleNotFoundError(name, name=name)

    monkeypatch.setattr(importlib, "import_module", import_module)
    assert load_nixl_api() is None
    assert attempted == ["nixl_cu12._api", "nixl._api"]


def test_load_nixl_api_falls_back_to_meta_package(monkeypatch) -> None:
    api = ModuleType("nixl._api")

    def import_module(name: str) -> ModuleType:
        if name == "nixl._api":
            return api
        raise ModuleNotFoundError(f"No module named {name}", name=name)

    monkeypatch.setattr(importlib, "import_module", import_module)

    assert load_nixl_api() is api


def test_load_nixl_api_reraises_internal_import_error(monkeypatch) -> None:
    def import_module(name: str) -> ModuleType:
        raise ImportError(f"{name} has a broken native dependency")

    monkeypatch.setattr(importlib, "import_module", import_module)

    try:
        load_nixl_api()
    except ImportError as exc:
        assert "broken native dependency" in str(exc)
    else:
        raise AssertionError("internal ImportError was swallowed")


def test_load_nixl_api_reraises_missing_dependency(monkeypatch) -> None:
    def import_module(name: str) -> ModuleType:
        raise ModuleNotFoundError("No module named libfoo", name="libfoo")

    monkeypatch.setattr(importlib, "import_module", import_module)

    try:
        load_nixl_api()
    except ModuleNotFoundError as exc:
        assert exc.name == "libfoo"
    else:
        raise AssertionError("missing dependency was swallowed")
