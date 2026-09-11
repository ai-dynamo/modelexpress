# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext

import pytest
import torch
from modelexpress.refit.reshard.rendezvous import unwrap_rendezvous_blob
from modelexpress_rl.train.adapter import TrainerStagingMode, WeightPayloadFormat
from modelexpress_rl.train.context import FSDPTrainerContext
from modelexpress_rl.train.engines import _create_trainer_adapter
from modelexpress_rl.train.engines.fsdp.adapter import FSDPTrainerAdapter

ADAPTER = "modelexpress_rl.train.engines.fsdp.adapter"


def test_copy_preserves_per_tensor_dtype_and_warm_buffers(dist_ready, monkeypatch):
    monkeypatch.setattr(f"{ADAPTER}.classic_cuda_alloc", nullcontext)
    manager = _Manager()
    overrides = {"bias": torch.float32}
    adapter = _create_trainer_adapter(
        FSDPTrainerContext(wire_dtype_overrides=overrides),
        manager=manager,
        nixl_metadata_endpoint="host:1234",
    )
    # Use values that BF16 cannot represent exactly; restoring FP32 must not hide rounding.
    bias = torch.tensor([1.000123, -2.0031, 0.015540123], dtype=torch.float32)
    weights = torch.tensor([1.000123, 2.0031], dtype=torch.float32)
    state = {"bias": bias, "weights": weights}
    first = _stage(adapter, state, TrainerStagingMode.COPY_TO_DEVICE)
    first.publish_ready.wait()
    served = {
        key.split("__", 3)[-1]: value for key, value in manager.registered[0].items()
    }
    assert served["bias"].dtype == torch.float32
    assert torch.equal(served["bias"], bias)
    assert served["weights"].dtype == torch.bfloat16
    assert torch.equal(served["weights"], weights.bfloat16())
    assert first.manifest.total_bytes == bias.numel() * 4 + weights.numel() * 2
    published = {t.name: t for t in unwrap_rendezvous_blob(first.manifest.data).tensors}
    assert (published["bias"].dtype, published["bias"].elsize) == ("torch.float32", 4)
    assert (published["weights"].dtype, published["weights"].elsize) == (
        "torch.bfloat16",
        2,
    )
    addresses = {name: value.data_ptr() for name, value in served.items()}
    overrides["bias"] = torch.bfloat16  # Caller mutation cannot change a bound policy.
    bias.add_(0.000321)
    second = _stage(adapter, state, TrainerStagingMode.COPY_TO_DEVICE)
    second.publish_ready.wait()
    assert torch.equal(served["bias"], bias)
    assert second.manifest.total_bytes == first.manifest.total_bytes
    assert {name: value.data_ptr() for name, value in served.items()} == addresses
    assert len(manager.registered) == 1


def test_in_place_accepts_fp32_override(dist_ready):
    adapter = FSDPTrainerAdapter(
        manager=_Manager(),
        nixl_metadata_endpoint="host:1234",
        wire_dtype_overrides={"bias": torch.float32},
    )
    staged = _stage(
        adapter,
        {"bias": torch.tensor([1.000123]), "w": torch.ones(2, dtype=torch.bfloat16)},
    )
    assert staged.manifest.total_bytes == 8


def test_multiple_dtype_overrides_include_fp16_and_scalar(dist_ready, monkeypatch):
    monkeypatch.setattr(f"{ADAPTER}.classic_cuda_alloc", nullcontext)
    manager = _Manager()
    adapter = FSDPTrainerAdapter(
        manager=manager,
        nixl_metadata_endpoint="host:1234",
        wire_dtype_overrides={"projection": torch.float16, "scale": torch.float32},
    )
    state = {
        "projection": torch.ones(2, 3),
        "scale": torch.tensor(1.000123),
        "weights": torch.ones(4),
    }
    staged = _stage(adapter, state, TrainerStagingMode.COPY_TO_DEVICE)
    served = {
        key.split("__", 3)[-1]: value for key, value in manager.registered[0].items()
    }
    assert served["projection"].dtype == torch.float16
    assert served["weights"].dtype == torch.bfloat16
    assert torch.equal(served["scale"], state["scale"])
    assert staged.manifest.total_bytes == 6 * 2 + 4 + 4 * 2


def test_copy_rejects_source_dtype_change_before_writing(dist_ready, monkeypatch):
    monkeypatch.setattr(f"{ADAPTER}.classic_cuda_alloc", nullcontext)
    manager = _Manager()
    adapter = FSDPTrainerAdapter(
        manager=manager,
        nixl_metadata_endpoint="host:1234",
        wire_dtype_overrides={"bias": torch.float32},
    )
    _stage(
        adapter, {"bias": torch.tensor([1.000123])}, TrainerStagingMode.COPY_TO_DEVICE
    )
    arena = next(iter(manager.registered[0].values()))
    before = arena.clone()
    with pytest.raises(ValueError, match="dtype changed"):
        _stage(
            adapter,
            {"bias": torch.tensor([9.0], dtype=torch.bfloat16)},
            TrainerStagingMode.COPY_TO_DEVICE,
        )
    assert torch.equal(arena, before)


def test_rejects_unknown_override_name(dist_ready):
    adapter = FSDPTrainerAdapter(
        manager=_Manager(),
        nixl_metadata_endpoint="host:1234",
        wire_dtype_overrides={"misspelled": torch.float32},
    )
    with pytest.raises(ValueError, match="override.*state_dict"):
        adapter.bind_tensors({"bias": torch.ones(2)})


@pytest.mark.parametrize("dtype", [torch.int32, torch.float8_e4m3fn, "float32"])
def test_rejects_unsupported_wire_dtype(dist_ready, dtype):
    with pytest.raises(ValueError, match="wire dtype"):
        FSDPTrainerAdapter(
            manager=_Manager(),
            nixl_metadata_endpoint="host:1234",
            wire_dtype_overrides={"bias": dtype},
        )


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

    assert adapter.bind_tensors({"w": torch.ones(2, 4)}) == ("publisher:global-rank:0")
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
