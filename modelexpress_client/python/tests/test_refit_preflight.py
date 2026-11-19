# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def preflight(monkeypatch):
    path = (
        Path(__file__).resolve().parents[3]
        / ".agents/skills/modelexpress-rl-refit/scripts/refit_preflight.py"
    )
    spec = importlib.util.spec_from_file_location("_refit_preflight_test", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "source, expected",
    [
        ('raise ImportError("missing dependency")', "fail"),
        ('raise OSError("missing native library")', "fail"),
        ("", "fail"),
        ("def reshard(): pass", "pass"),
    ],
)
def test_m2n_checks_required_import(preflight, monkeypatch, tmp_path, source, expected):
    package = tmp_path / "nccl"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "m2n.py").write_text(source)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "nccl", raising=False)
    monkeypatch.delitem(sys.modules, "nccl.m2n", raising=False)
    monkeypatch.setattr(preflight, "loaded_nccl_version", lambda: (2, 30, 7))

    try:
        checks = {check.name: check for check in preflight.check_m2n()}
        assert checks["import:nccl.m2n"].status == expected
    finally:
        sys.modules.pop("nccl.m2n", None)
        sys.modules.pop("nccl", None)


@pytest.mark.parametrize(
    "order, strategy, desired, expected",
    [
        (None, "RL", "v1", "pass"),
        ("", "RL", "v1", "fail"),
        ("GENERATOR,GENERATOR", "RL", "v1", "fail"),
        ("GENERATOR,BOGUS", "RL", "v1", "fail"),
        ("GENERATOR,", "RL", "v1", "fail"),
        ("GENERATOR,GENERATOR", "INFERENCE", "", "fail"),
        (" generator , object_storage ", "RL", "v1", "pass"),
        ("TRAINER", "RL", "v1", "fail"),
        ("TRAINER", "RL", "", "pass"),
        ("TRAINER", "INFERENCE", "v1", "pass"),
    ],
)
def test_nixl_validates_source_order(
    preflight, monkeypatch, order, strategy, desired, expected
):
    monkeypatch.setattr(preflight, "_module_importable", lambda *_: False)
    monkeypatch.setenv("MX_LOAD_STRATEGY_CHAIN", strategy)
    monkeypatch.setenv("MX_REFIT_DESIRED_VERSION_UID", desired)
    if order is None:
        monkeypatch.delenv("MX_GENERATOR_SOURCE_ORDER", raising=False)
    else:
        monkeypatch.setenv("MX_GENERATOR_SOURCE_ORDER", order)

    checks = {check.name: check for check in preflight.check_nixl()}

    assert checks["load-strategy"].status == expected


@pytest.mark.parametrize("timeout", ["-1", "0", "nan", "inf", "-inf"])
def test_invalid_timeout_fails_before_probing(preflight, monkeypatch, capsys, timeout):
    probe = MagicMock()
    monkeypatch.setattr(preflight, "check_m2n", probe)
    monkeypatch.setattr(preflight, "check_endpoint", probe)

    with pytest.raises(SystemExit) as error:
        preflight.main(
            ["--path", "m2n", "--endpoint", "127.0.0.1:8001", f"--timeout={timeout}"]
        )

    assert error.value.code == 2
    assert "--timeout must be finite and positive" in capsys.readouterr().err
    probe.assert_not_called()


def test_valid_timeout_preserves_json_report(preflight, monkeypatch, capsys):
    monkeypatch.setattr(preflight, "check_m2n", lambda: [])
    connect = MagicMock()
    monkeypatch.setattr(preflight.socket, "create_connection", connect)

    result = preflight.main(
        ["--path", "m2n", "--endpoint", "127.0.0.1:8001", "--timeout", "0.25"]
    )

    assert result == 0
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is True
    assert report["checks"][0]["status"] == "pass"
    connect.assert_called_once_with(("127.0.0.1", 8001), timeout=0.25)
