# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext
from unittest.mock import Mock

import pytest
import torch
from modelexpress.refit.reshard.rendezvous import unwrap_rendezvous_blob
from modelexpress_rl.train.adapter import TrainerStagingMode, WeightPayloadFormat
from modelexpress_rl.train.context import FSDPTrainerContext
from modelexpress_rl.train.engines import _create_trainer_adapter
from modelexpress_rl.train.engines.fsdp.adapter import FSDPTrainerAdapter

ADAPTER = "modelexpress_rl.train.engines.fsdp.adapter"


def _mock_host_allocation_without_cuda(monkeypatch):
    """Exercise snapshot logic on CPU; the CUDA test checks real pinned copies."""
    if torch.cuda.is_available():
        return
    empty = torch.empty

    def allocate(size, *, dtype, device, pin_memory):
        assert device == "cpu"
        assert pin_memory is True
        return empty(size, dtype=dtype, device=device)

    monkeypatch.setattr(f"{ADAPTER}.torch.cuda.is_available", lambda: True)
    monkeypatch.setattr(f"{ADAPTER}.torch.empty", allocate)


@pytest.mark.parametrize(
    "mode", [TrainerStagingMode.COPY_TO_HOST, TrainerStagingMode.COPY_TO_DEVICE]
)
def test_copy_preserves_per_tensor_dtype_and_warm_buffers(
    dist_ready, monkeypatch, mode
):
    monkeypatch.setattr(f"{ADAPTER}.classic_cuda_alloc", nullcontext)
    if mode is TrainerStagingMode.COPY_TO_HOST:
        _mock_host_allocation_without_cuda(monkeypatch)
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
    first = _stage(adapter, state, mode)
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
    second = _stage(adapter, state, mode)
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


def test_warm_stage_reuses_structural_manifest(dist_ready, monkeypatch):
    monkeypatch.delenv("MX_RESHARD_PUBLISH_DIGEST", raising=False)
    adapter = _adapter()
    state_dict = {"w": torch.ones(2, 4, dtype=torch.bfloat16)}

    first = _stage(adapter, state_dict)
    state_dict["w"].fill_(2)
    second = _stage(adapter, state_dict)

    assert second.manifest is first.manifest


def test_digest_mode_rebuilds_version_manifest(dist_ready, monkeypatch):
    monkeypatch.setenv("MX_RESHARD_PUBLISH_DIGEST", "1")
    adapter = _adapter()
    state_dict = {"w": torch.ones(2, 4, dtype=torch.bfloat16)}

    first = _stage(adapter, state_dict)
    state_dict["w"].fill_(2)
    second = _stage(adapter, state_dict)

    assert second.manifest.data != first.manifest.data


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
    with pytest.raises(NotImplementedError, match="use COPY_TO_HOST to cast"):
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
    with pytest.raises(NotImplementedError, match="WRITE_TO_STORAGE"):
        _stage(
            adapter,
            {"w": torch.ones(2, 4, dtype=torch.bfloat16)},
            mode=TrainerStagingMode.WRITE_TO_STORAGE,
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


def test_host_staging_without_cuda_rejects_before_allocation(dist_ready, monkeypatch):
    manager = _Manager()
    adapter = _adapter(manager)
    source = torch.ones(2, dtype=torch.float32)
    allocate = Mock(side_effect=AssertionError("must reject before allocation"))
    monkeypatch.setattr(f"{ADAPTER}.torch.cuda.is_available", lambda: False)
    monkeypatch.setattr(f"{ADAPTER}.torch.empty", allocate)

    with pytest.raises(RuntimeError, match="COPY_TO_HOST staging requires CUDA"):
        _stage(adapter, {"w": source}, TrainerStagingMode.COPY_TO_HOST)

    allocate.assert_not_called()
    assert manager.registered == []


def test_host_snapshot_survives_source_mutation_and_rematerialization(
    dist_ready, monkeypatch
):
    _mock_host_allocation_without_cuda(monkeypatch)
    manager = _Manager()
    adapter = _adapter(manager)
    source = torch.arange(12, dtype=torch.float32).reshape(3, 4).t()
    staged = _stage(adapter, {"w": source}, TrainerStagingMode.COPY_TO_HOST)
    staged.publish_ready.wait()
    (served,) = staged.buffer_owner
    assert served.device.type == "cpu"
    assert served.is_contiguous()
    assert torch.equal(served, source.bfloat16())
    snapshot = served.clone()
    address = served.data_ptr()
    source.fill_(-1)
    assert torch.equal(served, snapshot)
    for value in (7, 11):
        next_source = torch.full((4, 3), value, dtype=torch.float32)
        next_stage = _stage(
            adapter, {"w": next_source}, TrainerStagingMode.COPY_TO_HOST
        )
        next_stage.publish_ready.wait()
        assert next_stage.manifest is staged.manifest
        assert served.data_ptr() == address
        assert torch.equal(served, next_source.bfloat16())
    assert len(manager.registered) == 1
    (published,) = unwrap_rendezvous_blob(staged.manifest.data).tensors
    assert published.shards[0].memory_type == "DRAM"


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA pinned host copies"
)
def test_cuda_host_snapshot_fences_nondefault_stream_and_preserves_fp32(dist_ready):
    manager = _Manager()
    adapter = FSDPTrainerAdapter(
        manager=manager,
        nixl_metadata_endpoint="host:1234",
        wire_dtype_overrides={"bias": torch.float32},
    )
    stream = torch.cuda.Stream()
    addresses = None
    for value in (1.000123, 2.000321, 3.001234):
        with torch.cuda.stream(stream):
            state = {
                "w": torch.full(
                    (128, 256), value, device="cuda", dtype=torch.float32
                ).t(),
                "bias": torch.tensor([value], device="cuda", dtype=torch.float32),
            }
            staged = _stage(adapter, state, TrainerStagingMode.COPY_TO_HOST)
        staged.publish_ready.wait()
        served = {k.split("__", 3)[-1]: v for k, v in manager.registered[0].items()}
        assert all(t.is_pinned() and t.device.type == "cpu" for t in served.values())
        assert torch.equal(served["w"], torch.full((256, 128), value).bfloat16())
        assert torch.equal(served["bias"], torch.tensor([value], dtype=torch.float32))
        current = {k: t.data_ptr() for k, t in served.items()}
        if addresses is not None:
            assert current == addresses
        addresses = current
    assert len(manager.registered) == 1
