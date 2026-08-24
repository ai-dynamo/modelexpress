# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from modelexpress_rl.train.adapter import TrainerStagingMode, WeightPayloadFormat
from modelexpress_rl.train.engines.fsdp.adapter import FSDPTrainerAdapter

ADAPTER = "modelexpress_rl.train.engines.fsdp.adapter"


class _Manager:
    agent_name = "trainer-r0"
    nixl_metadata = b"agent-metadata"
    listen_port = 19000

    def __init__(self):
        self.registered = []

    def register_tensors(self, tensors):
        self.registered.append(dict(tensors))
        return self.nixl_metadata


@pytest.fixture
def dist_ready(monkeypatch):
    monkeypatch.setattr(f"{ADAPTER}.dist.is_available", lambda: True)
    monkeypatch.setattr(f"{ADAPTER}.dist.is_initialized", lambda: True)
    monkeypatch.setattr(f"{ADAPTER}.dist.get_rank", lambda: 0)


def _adapter(manager=None):
    return FSDPTrainerAdapter(
        manager=manager or _Manager(), nixl_metadata_endpoint="host:1234"
    )


def _stage(adapter, state_dict, mode=TrainerStagingMode.IN_PLACE):
    return adapter.stage_shard(
        tensors=state_dict,
        staging_mode=mode,
        payload_format=WeightPayloadFormat.FULL_TENSOR,
    )


def test_requires_initialized_distributed_engine(monkeypatch):
    monkeypatch.setattr(f"{ADAPTER}.dist.is_available", lambda: True)
    monkeypatch.setattr(f"{ADAPTER}.dist.is_initialized", lambda: False)

    with pytest.raises(RuntimeError, match="distributed process group"):
        _adapter()


def test_source_slot_id_is_rank_stamped(dist_ready):
    assert _adapter().source_slot_id == "publisher:global-rank:0"


def test_bind_tensors_validates_state_dict_and_returns_rank_slot(dist_ready):
    adapter = _adapter()

    assert adapter.bind_tensors({"w": torch.ones(2, 4)}) == (
        "publisher:global-rank:0"
    )
    with pytest.raises(TypeError, match="state_dict"):
        adapter.bind_tensors([torch.ones(2, 4)])


def test_in_place_stage_registers_once(dist_ready):
    manager = _Manager()
    adapter = _adapter(manager)
    state_dict = {"w": torch.ones(2, 4, dtype=torch.bfloat16)}

    staged = _stage(adapter, state_dict)

    assert staged.manifest.tensor_count == 1
    assert staged.manifest.total_bytes == 2 * 4 * 2  # bf16 elsize
    assert staged.manifest.transport == "NIXL"
    staged.publish_ready.wait()  # IN_PLACE performs no copy: no-op

    # Re-staging the same weights must not re-register (setup is one-time).
    _stage(adapter, state_dict)
    assert len(manager.registered) == 1


def test_in_place_rejects_a_moved_source(dist_ready):
    adapter = _adapter()
    _stage(adapter, {"w": torch.ones(2, 4, dtype=torch.bfloat16)})

    # Same name/shape/dtype but fresh storage: the registered address is stale.
    with pytest.raises(NotImplementedError, match="source storage moved"):
        _stage(adapter, {"w": torch.ones(2, 4, dtype=torch.bfloat16)})


def test_stage_rejects_a_changed_tensor_set(dist_ready):
    adapter = _adapter()
    state_dict = {"w": torch.ones(2, 4, dtype=torch.bfloat16)}
    _stage(adapter, state_dict)

    state_dict["b"] = torch.ones(4, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="tensor set changed"):
        _stage(adapter, state_dict)


def test_stage_rejects_a_changed_shard_geometry(dist_ready):
    adapter = _adapter()
    _stage(adapter, {"w": torch.ones(2, 4, dtype=torch.bfloat16)})

    # Same name, different local shape: geometry must stay fixed after initialize.
    with pytest.raises(ValueError, match="shard geometry changed"):
        _stage(adapter, {"w": torch.ones(4, 4, dtype=torch.bfloat16)})


def test_in_place_requires_wire_dtype_source(dist_ready):
    adapter = _adapter()
    with pytest.raises(NotImplementedError, match="use COPY_TO_DEVICE to cast"):
        _stage(adapter, {"w": torch.ones(2, 4, dtype=torch.float32)})


def test_in_place_requires_contiguous_source(dist_ready):
    adapter = _adapter()
    strided = torch.ones(4, 4, dtype=torch.bfloat16).t()
    with pytest.raises(NotImplementedError, match="contiguous"):
        _stage(adapter, {"w": strided})


def test_staging_mode_cannot_change_after_initialize(dist_ready):
    adapter = _adapter()
    _stage(adapter, {"w": torch.ones(2, 4, dtype=torch.bfloat16)})

    with pytest.raises(ValueError, match="initialized for"):
        _stage(
            adapter,
            {"w": torch.ones(2, 4, dtype=torch.bfloat16)},
            mode=TrainerStagingMode.COPY_TO_DEVICE,
        )


def test_unsupported_staging_mode_is_rejected(dist_ready):
    adapter = _adapter()
    with pytest.raises(NotImplementedError, match="COPY_TO_HOST"):
        _stage(
            adapter,
            {"w": torch.ones(2, 4, dtype=torch.bfloat16)},
            mode=TrainerStagingMode.COPY_TO_HOST,
        )


def test_unsupported_payload_format_is_rejected(dist_ready):
    adapter = _adapter()
    with pytest.raises(NotImplementedError, match="XOR_DELTA"):
        adapter.stage_shard(
            tensors={"w": torch.ones(2, 4, dtype=torch.bfloat16)},
            staging_mode=TrainerStagingMode.IN_PLACE,
            payload_format=WeightPayloadFormat.XOR_DELTA,
        )


def test_non_dict_tensors_is_rejected(dist_ready):
    adapter = _adapter()
    with pytest.raises(TypeError, match="state_dict"):
        _stage(adapter, [torch.ones(2, 4, dtype=torch.bfloat16)])
