# SPDX-License-Identifier: Apache-2.0

import torch
from modelexpress.refit.source.canonical import FilesystemCanonicalBaseStore
from safetensors import safe_open

from modelexpress.refit.receiver import (
    materialize_snapshot_to_safetensors,
)


def test_materialize_snapshot_writes_bounded_hf_safetensors(tmp_path):
    store = FilesystemCanonicalBaseStore(tmp_path / "canonical")
    writer = store.begin_snapshot("2")
    writer.add_tensor("model.b", torch.tensor([3.0], dtype=torch.float32))
    writer.add_tensor("model.a", torch.tensor([1, 2], dtype=torch.int64))
    snapshot = writer.finalize()

    target = materialize_snapshot_to_safetensors(store, snapshot, tmp_path / "prepared")

    assert target == tmp_path / "prepared"
    with safe_open(target / "model.safetensors", framework="pt", device="cpu") as f:
        assert sorted(f.keys()) == ["model.a", "model.b"]
        assert torch.equal(f.get_tensor("model.a"), torch.tensor([1, 2]))
        assert torch.equal(f.get_tensor("model.b"), torch.tensor([3.0]))
