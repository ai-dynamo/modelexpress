# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import importlib.util
import json
import struct
from types import SimpleNamespace
from pathlib import Path

import pytest
import torch


PROBE = Path(__file__).with_name("weight_diagnostics.py")
spec = importlib.util.spec_from_file_location("weight_diagnostics", PROBE)
diagnostics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostics)


@pytest.mark.parametrize("enabled", [False, True])
def test_audit_runs_after_load_and_preserves_model(monkeypatch, capsys, enabled):
    tensor = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4).T
    model = object()
    ctx = SimpleNamespace(tensors={})

    class Loader:
        def _load_model(self, vllm_config, model_config, prefix, context, is_speculative_draft, previous_model):
            tensor.add_(1)
            context.tensors = {"weight": tensor, "scalar": tensor[0, 0]}
            return model

    monkeypatch.setenv("MX_CI_WEIGHT_DIGESTS", "1" if enabled else "0")
    diagnostics.instrument_loader(Loader)
    assert Loader()._load_model(None, None, "", ctx, False) is model
    output = capsys.readouterr().out
    if not enabled:
        assert output == ""
        return
    record = {}
    for line in output.splitlines():
        record.update(json.loads(line.split("[MX-CI-WEIGHTS] ", 1)[1]))
    for name, value in {"weight": tensor, "scalar": tensor[0, 0]}.items():
        raw_bytes = b"".join(struct.pack("<f", float(x))[2:] for x in value.reshape(-1))
        digest = hashlib.sha256(raw_bytes).hexdigest()
        assert record[name] == {"shape": list(value.shape), "dtype": "torch.bfloat16", "sha256": digest}


def test_large_catalog_uses_short_independent_log_records(monkeypatch, capsys):
    tensors = {f"language_model.model.layers.{i}.weight": torch.ones(1) for i in range(427)}

    class Loader:
        def _load_model(self, *args):
            return object()

    monkeypatch.setenv("MX_CI_WEIGHT_DIGESTS", "1")
    diagnostics.instrument_loader(Loader)
    Loader()._load_model(None, None, "", SimpleNamespace(tensors=tensors), False)
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == len(tensors)
    assert all(len(line.encode()) < 4096 for line in lines)
    names = [next(iter(json.loads(line.split("[MX-CI-WEIGHTS] ", 1)[1]))) for line in lines]
    assert names == sorted(tensors)


def test_installer_patches_the_installed_module_for_worker_subprocesses(monkeypatch, tmp_path):
    import os
    import runpy
    import subprocess
    import sys
    import sysconfig

    directory = tmp_path / "modelexpress/engines/vllm"
    directory.mkdir(parents=True)
    loader = directory / "loader.py"
    loader.write_text("class MxModelLoader:\n    def _load_model(self, *args): return 42\n")
    for package in (directory, directory.parent, directory.parent.parent):
        (package / "__init__.py").touch()
    monkeypatch.setattr(sysconfig, "get_path", lambda _: str(tmp_path))
    runpy.run_path(str(PROBE), run_name="__main__")
    assert (directory / "_ci_weight_diagnostics.py").read_bytes() == PROBE.read_bytes()
    assert "instrument_loader(MxModelLoader)" in loader.read_text()
    result = subprocess.run(
        [sys.executable, "-c", "from modelexpress.engines.vllm.loader import MxModelLoader; assert MxModelLoader()._load_model(None, None, '', None, False) == 42"],
        cwd=tmp_path, env={**os.environ, "PYTHONPATH": str(tmp_path), "MX_CI_WEIGHT_DIGESTS": "0"},
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


def test_draft_temporary_embedding_is_not_part_of_transfer_audit(monkeypatch, capsys):
    class Loader:
        def _load_model(self, vllm_config, model_config, prefix, ctx, is_speculative_draft, previous_model):
            ctx.tensors = {"__mx_draft__.layer.weight": torch.tensor([1, 2], dtype=torch.bfloat16)}
            return SimpleNamespace(temporary_embedding=previous_model)

    monkeypatch.setenv("MX_CI_WEIGHT_DIGESTS", "1")
    diagnostics.instrument_loader(Loader)
    for temporary in (torch.ones(4), torch.zeros(4)):
        Loader()._load_model(None, None, "", SimpleNamespace(tensors={}), True, temporary)
    records = [json.loads(line.split("[MX-CI-WEIGHTS] ", 1)[1]) for line in capsys.readouterr().out.splitlines()]
    assert records[0] == records[1]
    assert list(records[0]) == ["__mx_draft__.layer.weight"]
