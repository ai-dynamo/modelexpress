# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Seed an isolated CI S3 fixture from the pinned Hugging Face snapshot."""

import json
from pathlib import Path

import boto3
from botocore.config import Config
from harness.config import load_config
from huggingface_hub import snapshot_download


def main():
    config = load_config()
    if not config["storage"]["endpoint_url"]:
        raise ValueError("CI seed upload requires an explicit endpoint")
    root = Path(
        snapshot_download(
            config["model"],
            revision=config["revision"],
            local_dir="/tmp/mx-seed",
            allow_patterns=[
                "*.json",
                "*.safetensors",
                "*.txt",
                "*.model",
                "*.tiktoken",
                "*.jinja",
                "*.py",
            ],
        )
    )
    s3 = boto3.client(
        "s3",
        endpoint_url=config["storage"]["endpoint_url"],
        region_name=config["storage"]["region"],
        config=Config(s3={"addressing_style": "path"}),
    )
    s3.create_bucket(Bucket=config["bucket"])
    files = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if not path.is_file() or relative.parts[0] == ".cache":
            continue
        name, size = relative.as_posix(), path.stat().st_size
        key = config["seed_prefix"] + name
        s3.upload_file(str(path), config["bucket"], key)
        if s3.head_object(Bucket=config["bucket"], Key=key)["ContentLength"] != size:
            raise RuntimeError("Seed upload size mismatch: " + name)
        files.append({"name": name, "bytes": size})
    s3.put_object(
        Bucket=config["bucket"],
        Key=config["seed_prefix"] + "snapshot-manifest.json",
        Body=json.dumps(
            {"model": config["model"], "revision": config["revision"], "files": files}
        ).encode(),
    )
    print(f"Uploaded pinned seed snapshot: {len(files)} files", flush=True)


if __name__ == "__main__":
    main()
