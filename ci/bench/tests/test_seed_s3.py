# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Publish seed manifests only after the pinned snapshot uploads are verified."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("wrong_size", [False, True])
def test_seed_upload_verifies_objects_before_publishing_manifest(
    tmp_path, monkeypatch, wrong_size
):
    config = {
        "model": "pinned/model",
        "revision": "a" * 40,
        "bucket": "mx-refit",
        "seed_prefix": "seed/",
        "storage": {"endpoint_url": "http://seaweedfs:9000", "region": "us-east-1"},
    }
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    (tmp_path / ".cache").mkdir()
    (tmp_path / ".cache/token").write_text("must not upload")
    objects = {}

    def download(model, **kwargs):
        assert model == config["model"] and kwargs["revision"] == config["revision"]
        assert "local_dir" not in kwargs
        return str(tmp_path)

    def upload(path, bucket, key, **kwargs):
        assert kwargs.get("Config", {}).get("preferred_transfer_client") == "classic"
        assert bucket == config["bucket"]
        objects[key] = Path(path).read_bytes()

    client = SimpleNamespace(
        create_bucket=lambda **kwargs: None,
        upload_file=upload,
        head_object=lambda **kwargs: {
            "ContentLength": len(objects[kwargs["Key"]]) + int(wrong_size)
        },
        put_object=lambda **kwargs: objects.update({kwargs["Key"]: kwargs["Body"]}),
    )
    for name, module in {
        "boto3": SimpleNamespace(client=lambda *args, **kwargs: client),
        "boto3.s3.transfer": SimpleNamespace(TransferConfig=lambda **kwargs: kwargs),
        "botocore.config": SimpleNamespace(Config=lambda **kwargs: kwargs),
        "huggingface_hub": SimpleNamespace(snapshot_download=download),
        "harness.config": SimpleNamespace(load_config=lambda: config),
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    path = Path(__file__).resolve().parents[1] / "scenarios/delta/seed_s3.py"
    spec = importlib.util.spec_from_file_location("seed_s3_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if wrong_size:
        with pytest.raises(RuntimeError, match="size mismatch"):
            module.main()
        assert "seed/snapshot-manifest.json" not in objects
    else:
        module.main()
        manifest = json.loads(objects.pop("seed/snapshot-manifest.json"))
        assert manifest["revision"] == config["revision"]
        assert {file["name"] for file in manifest["files"]} == {
            "config.json",
            "model.safetensors",
        }
        assert all(not key.startswith("seed/.cache") for key in objects)
