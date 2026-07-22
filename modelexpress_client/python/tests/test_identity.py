# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

from modelexpress.rl.identity import content_digest


def _write_version(version_dir, shard_bytes):
    version_dir.mkdir()
    (version_dir / "model-00000-of-00001.safetensors").write_bytes(shard_bytes)
    index = {
        "metadata": {
            "version": "000001",
            "base_version": "000000",
            "checksums": {"model.weight": "same-target-state-checksum"},
        },
        "weight_map": {
            "model.weight": "model-00000-of-00001.safetensors",
        },
    }
    (version_dir / "model.safetensors.index.json").write_text(
        json.dumps(index, sort_keys=True)
    )


def test_content_digest_hashes_delta_bytes_not_target_state_checksums(tmp_path):
    first = tmp_path / "first" / "weight_v000001"
    second = tmp_path / "second" / "weight_v000001"
    first.parent.mkdir()
    second.parent.mkdir()
    _write_version(first, b"xor-delta-from-base-a")
    _write_version(second, b"xor-delta-from-base-b")

    first_digest = content_digest(str(first))

    assert first_digest == content_digest(str(first))
    assert first_digest != content_digest(str(second))
    assert first_digest == first_digest.lower()
