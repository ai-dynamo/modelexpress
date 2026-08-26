# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from modelexpress.refit.reshard.rendezvous import unwrap_rendezvous_blob
from modelexpress.refit.reshard.verify import tensor_digest
from modelexpress_rl.train.engines.fsdp.publisher import (
    LocalTensorShard,
    build_fsdp_reshard_manifest,
    capture_local_shards,
)


class _Manager:
    agent_name = "trainer-r0"
    nixl_metadata = b"agent-metadata"
    listen_port = 19000


class _ServedOn:
    """A registered served buffer on a device this test host does not have."""

    dtype = torch.bfloat16

    def __init__(self, device: str) -> None:
        self.device = torch.device(device)

    def is_contiguous(self) -> bool:
        return True

    def data_ptr(self) -> int:
        return 0x1000

    def element_size(self) -> int:
        return 2


def test_capture_publishes_full_copies_and_skips_non_float():
    state_dict = {
        "w": torch.arange(8, dtype=torch.float32).reshape(2, 4),
        "b": torch.ones(4, dtype=torch.bfloat16),
        "step": torch.arange(4, dtype=torch.long),  # non-float: skipped
    }

    shards = capture_local_shards(state_dict)

    by_name = {s.name: s for s in shards}
    assert set(by_name) == {"w", "b"}
    for shard in shards:
        # Plain (non-DTensor) tensors are held in full by every rank.
        assert shard.shard_offset == (0,) * len(shard.global_shape)
        assert shard.local_shape == shard.global_shape
        assert shard.staging_tensor is None
    assert by_name["w"].global_shape == (2, 4)


def test_capture_gives_a_scalar_tensor_a_rank_zero_offset():
    state_dict = {"loss_scale": torch.tensor(1.0, dtype=torch.float32)}

    (shard,) = capture_local_shards(state_dict)

    assert shard.global_shape == ()
    assert shard.shard_offset == ()
    assert shard.local_shape == ()


def test_build_manifest_describes_the_served_buffer():
    tensor = torch.zeros(2, 4, dtype=torch.bfloat16)
    shard = LocalTensorShard(
        name="w",
        global_shape=(2, 4),
        shard_offset=(0, 0),
        local_shape=(2, 4),
        source_tensor=tensor,
    )

    blob = build_fsdp_reshard_manifest(
        manager=_Manager(), shards=[shard], metadata_endpoint="host:1234"
    )

    payload = unwrap_rendezvous_blob(blob)
    assert payload.agent_metadata == b"agent-metadata"
    (published,) = payload.tensors
    assert published.name == "w"
    assert published.dtype == "torch.bfloat16"
    assert published.elsize == 2
    assert tuple(published.full_shape) == (2, 4)
    (pshard,) = published.shards
    assert pshard.agent_name == "trainer-r0"
    assert pshard.addr == tensor.data_ptr()
    assert tuple(pshard.shard_offset) == (0, 0)
    assert tuple(pshard.shape) == (2, 4)


@pytest.mark.parametrize(
    ("device", "expected_device_id"),
    [("cuda:3", 3), ("xpu:2", 2), ("xpu:0", 0), ("cpu", 0)],
)
def test_build_manifest_publishes_the_served_device_ordinal(device, expected_device_id):
    """The published ordinal is the device the served buffer actually lives on.

    A receiver feeds this ordinal straight into the remote NIXL descriptor
    (``reshard/transport/nixl.py::_ranges_for``), so dropping it for a non-CUDA
    device points peers at device 0 while the buffer sits elsewhere.
    """
    shard = LocalTensorShard("w", (2, 4), (0, 0), (2, 4), _ServedOn(device))

    payload = unwrap_rendezvous_blob(
        build_fsdp_reshard_manifest(
            manager=_Manager(), shards=[shard], metadata_endpoint="host:1234"
        )
    )

    (published,) = payload.tensors
    (pshard,) = published.shards
    assert pshard.device_id == expected_device_id


def test_build_manifest_publishes_enabled_verification_digest(monkeypatch):
    monkeypatch.setenv("MX_RESHARD_PUBLISH_DIGEST", "1")
    tensor = torch.arange(8, dtype=torch.bfloat16).reshape(2, 4)
    shard = LocalTensorShard("w", (2, 4), (0, 0), (2, 4), tensor)

    payload = unwrap_rendezvous_blob(
        build_fsdp_reshard_manifest(
            manager=_Manager(), shards=[shard], metadata_endpoint="host:1234"
        )
    )

    assert payload.tensors[0].shards[0].digest == tensor_digest(tensor)


def test_build_manifest_groups_multiple_shards_under_one_tensor():
    top = torch.zeros(2, 4, dtype=torch.bfloat16)
    bottom = torch.zeros(2, 4, dtype=torch.bfloat16)
    shards = [
        LocalTensorShard("w", (4, 4), (0, 0), (2, 4), top),
        LocalTensorShard("w", (4, 4), (2, 0), (2, 4), bottom),
    ]

    payload = unwrap_rendezvous_blob(
        build_fsdp_reshard_manifest(
            manager=_Manager(), shards=shards, metadata_endpoint="host:1234"
        )
    )

    (published,) = payload.tensors
    offsets = sorted(tuple(s.shard_offset) for s in published.shards)
    assert offsets == [(0, 0), (2, 0)]


def test_build_manifest_rejects_non_contiguous_served_tensor():
    strided = torch.zeros(4, 4, dtype=torch.bfloat16)[:, ::2]
    shard = LocalTensorShard("w", (4, 2), (0, 0), (4, 2), strided)

    with pytest.raises(ValueError, match="contiguous"):
        build_fsdp_reshard_manifest(
            manager=_Manager(), shards=[shard], metadata_endpoint="host:1234"
        )


def test_build_manifest_rejects_an_invalid_address():
    meta = torch.zeros(2, 4, dtype=torch.bfloat16, device="meta")
    shard = LocalTensorShard("w", (2, 4), (0, 0), (2, 4), meta)

    with pytest.raises(ValueError, match="invalid address"):
        build_fsdp_reshard_manifest(
            manager=_Manager(), shards=[shard], metadata_endpoint="host:1234"
        )


def test_build_manifest_requires_host_port_endpoint():
    shard = LocalTensorShard(
        "w", (2, 4), (0, 0), (2, 4), torch.zeros(2, 4, dtype=torch.bfloat16)
    )

    with pytest.raises(ValueError, match="host:port"):
        build_fsdp_reshard_manifest(
            manager=_Manager(), shards=[shard], metadata_endpoint="no-port"
        )
