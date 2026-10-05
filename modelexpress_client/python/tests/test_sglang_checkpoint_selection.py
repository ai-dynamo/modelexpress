# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""File selection for the two passes of a shared SGLang checkpoint."""

import json
import fnmatch
import sys
from types import SimpleNamespace
from pathlib import Path

from modelexpress.engines.sglang.draft_weights import Qwen35MtpWeights


def select_role_shards(*args):
    from modelexpress.engines.sglang.checkpoint_selection import select_role_shards

    return select_role_shards(*args)


def test_qwen35_selects_main_and_draft_shards_without_discarding_mixed_shards(
    tmp_path,
):
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.layers.0.weight": "main.safetensors",
                    "model.layers.1.weight": "mixed.safetensors",
                    "mtp.fc.weight": "mixed.safetensors",
                    "mtp.layers.0.weight": "draft.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )
    files = [
        str(tmp_path / "main.safetensors"),
        str(tmp_path / "mixed.safetensors"),
        str(tmp_path / "draft.safetensors"),
    ]
    selector = Qwen35MtpWeights()

    assert select_role_shards(str(tmp_path), files, selector, "main") == files[:2]
    assert select_role_shards(str(tmp_path), files, selector, "draft") == files[1:]


def test_qwen35_missing_or_incomplete_index_falls_back_to_all_shards(tmp_path):
    files = [str(tmp_path / "main.safetensors"), str(tmp_path / "draft.safetensors")]
    selector = Qwen35MtpWeights()

    assert select_role_shards(str(tmp_path), files, selector, "draft") is None

    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.layers.0.weight": "main.safetensors"}}),
        encoding="utf-8",
    )
    assert select_role_shards(str(tmp_path), files, selector, "main") is None

    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"mtp.fc.weight": "missing.safetensors"}}),
        encoding="utf-8",
    )
    assert select_role_shards(str(tmp_path), files, selector, "draft") is None


def test_qwen35_draft_without_matching_tensor_falls_back(tmp_path):
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.layers.0.weight": "main.safetensors"}}),
        encoding="utf-8",
    )
    files = [str(tmp_path / "main.safetensors")]

    assert select_role_shards(
        str(tmp_path), files, Qwen35MtpWeights(), "draft"
    ) is None


def test_object_store_reads_only_index_before_selecting_shards(monkeypatch):
    calls = []

    def pull_files(uri, directory, allow_pattern):
        calls.append((uri, allow_pattern))
        if not any(
            fnmatch.fnmatch("model/model.safetensors.index.json", pattern)
            for pattern in allow_pattern
        ):
            return
        Path(directory, "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {
                "model.layers.0.weight": "main.safetensors",
                "mtp.fc.weight": "draft.safetensors",
            }}), encoding="utf-8",
        )

    monkeypatch.setitem(sys.modules, "runai_model_streamer", SimpleNamespace(
        pull_files=pull_files,
    ))
    files = [
        "s3://bucket/model/main.safetensors",
        "s3://bucket/model/draft.safetensors",
    ]

    assert select_role_shards(
        "s3://bucket/model", files, Qwen35MtpWeights(), "draft"
    ) == files[1:]
    assert calls == [
        ("s3://bucket/model", ["model.safetensors.index.json", "*/model.safetensors.index.json"])
    ]


def test_object_store_index_read_error_keeps_sglang_full_load(monkeypatch):
    def pull_files(*args, **kwargs):
        raise RuntimeError("object store index unavailable")

    monkeypatch.setitem(
        sys.modules,
        "runai_model_streamer",
        SimpleNamespace(pull_files=pull_files),
    )
    files = ["s3://bucket/model/main.safetensors"]
    assert select_role_shards(
        "s3://bucket/model", files, Qwen35MtpWeights(), "main"
    ) is None
