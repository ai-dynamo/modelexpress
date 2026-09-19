# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext

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


class _FakeEvent:
    def __init__(self):
        self.recorded_on = None
        self.waits = 0

    def record(self, stream):
        self.recorded_on = stream

    def synchronize(self):
        self.waits += 1


class _FakeDeviceModule:
    """Stand-in for torch.cuda / torch.xpu that records how it was driven."""

    def __init__(self):
        self.requested = []
        self.streams_for = []
        self.events = []

    def Event(self):
        event = _FakeEvent()
        self.events.append(event)
        return event

    def current_stream(self, device=None):
        self.streams_for.append(device)
        return f"stream:{device}"


def _unexpected_cuda_call(*args, **kwargs):
    raise AssertionError("the arena fence must not touch the process CUDA device")


@pytest.fixture
def device_module(monkeypatch):
    module = _FakeDeviceModule()

    def get_device_module(device):
        module.requested.append(device)
        return module

    monkeypatch.setattr(f"{ADAPTER}.classic_cuda_alloc", nullcontext)
    monkeypatch.setattr(torch, "get_device_module", get_device_module)
    return module


def test_copy_to_device_fences_the_arena_device(dist_ready, device_module):
    adapter = _adapter()

    staged = _stage(
        adapter,
        {"w": torch.ones(2, 4, dtype=torch.bfloat16)},
        mode=TrainerStagingMode.COPY_TO_DEVICE,
    )

    # The fence comes from the arena's own device, and is recorded on that
    # device's stream rather than whatever stream the process happens to be on.
    assert device_module.requested == [torch.device("cpu")]
    assert device_module.streams_for == [torch.device("cpu")]
    (event,) = device_module.events
    assert event.recorded_on == "stream:cpu"

    assert event.waits == 0
    staged.publish_ready.wait()
    assert event.waits == 1


def test_copy_to_device_fence_ignores_the_process_cuda_device(
    dist_ready, device_module, monkeypatch
):
    # The previous fence read torch.cuda.current_stream() with no device, so a
    # CUDA-capable process fenced its current device instead of the arena's.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_stream", _unexpected_cuda_call)
    monkeypatch.setattr(torch.cuda, "Event", _unexpected_cuda_call)

    adapter = _adapter()
    staged = _stage(
        adapter,
        {"w": torch.ones(2, 4, dtype=torch.bfloat16)},
        mode=TrainerStagingMode.COPY_TO_DEVICE,
    )
    staged.publish_ready.wait()

    assert device_module.requested == [torch.device("cpu")]


def test_copy_to_device_rejects_arenas_spanning_devices(dist_ready, device_module):
    manager = _Manager()
    adapter = _adapter(manager)

    # One rank serves one device: a shard set that lands arenas on two devices
    # cannot be covered by a single fence.
    with pytest.raises(NotImplementedError, match="one device per rank"):
        _stage(
            adapter,
            {
                "w": torch.ones(2, 4, dtype=torch.bfloat16),
                "b": torch.ones(4, dtype=torch.bfloat16, device="meta"),
            },
            mode=TrainerStagingMode.COPY_TO_DEVICE,
        )

    # Rejection must precede registration: nothing may be bound to the manager,
    # which is configured for one device.
    assert manager.registered == []


class _StubArena:
    """Only what the device check reads, so ordinals need no real hardware."""

    def __init__(self, device):
        self.device = torch.device(device)


def test_arena_device_rejects_two_ordinals_of_one_family():
    arenas = [_StubArena("cuda:0"), _StubArena("cuda:1")]

    with pytest.raises(NotImplementedError, match="cuda:0.*cuda:1"):
        FSDPTrainerAdapter._arena_device(arenas)


def test_arena_device_accepts_one_repeated_ordinal():
    arenas = [_StubArena("cuda:1"), _StubArena("cuda:1")]

    assert FSDPTrainerAdapter._arena_device(arenas) == torch.device("cuda:1")
