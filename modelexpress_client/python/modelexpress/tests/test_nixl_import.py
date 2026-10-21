# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
from types import ModuleType

from modelexpress._nixl import load_nixl_api


def test_load_nixl_api_prefers_cuda_specific_package(monkeypatch) -> None:
    api = ModuleType("nixl_cu13._api")
    attempted = []

    def import_module(name: str) -> ModuleType:
        attempted.append(name)
        if name == "nixl_cu13._api":
            return api
        raise ImportError(name)

    monkeypatch.setattr(importlib, "import_module", import_module)

    assert load_nixl_api() is api
    assert attempted == ["nixl_cu13._api"]


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
