# SPDX-License-Identifier: Apache-2.0

import torch
from modelexpress.refit.source.canonical import FilesystemCanonicalBaseStore
from safetensors.torch import save_file

from modelexpress.refit.receiver import (
    seed_base_from_safetensors,
)


def test_seed_base_from_safetensors_is_deterministic_and_bounded(tmp_path):
    first = tmp_path / "model-00001-of-00002.safetensors"
    second = tmp_path / "model-00002-of-00002.safetensors"
    save_file({"model.b": torch.tensor([2.0])}, first)
    save_file({"model.a": torch.tensor([1, 3], dtype=torch.int64)}, second)
    store = FilesystemCanonicalBaseStore(tmp_path / "base")

    snapshot = seed_base_from_safetensors(store, "1", [first, second])

    assert [item.name for item in snapshot.tensors] == ["model.a", "model.b"]
    assert torch.equal(store.read_tensor(snapshot, "model.a"), torch.tensor([1, 3]))
    assert store.open_snapshot("1") == snapshot
