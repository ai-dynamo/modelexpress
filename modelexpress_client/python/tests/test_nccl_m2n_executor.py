# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""No-GPU failure-atomicity and stream-ordering tests for nccl_m2n.

The fake validates MX enqueue order and token propagation, not M2N's real CUDA
ready/done event bridge; that requires a GPU integration test.
"""

from __future__ import annotations

import pytest

from modelexpress.weight_transfer.planner.mesh import REPLICATE
from modelexpress.weight_transfer.transport import _nccl_m2n_bind as binding
from modelexpress.weight_transfer.transport.nccl_m2n_executor import (
    NcclM2nExecutor,
    ReshardParam,
)


class FakeM2N:
    def __init__(
        self,
        payloads: list[bytes],
        *,
        fail_reshard_at: int | None = None,
        fail_live_copy_at: int | None = None,
        fail_stream_sync: bool = False,
    ) -> None:
        self.payloads = payloads
        self.fail_reshard_at = fail_reshard_at
        self.fail_live_copy_at = fail_live_copy_at
        self.fail_stream_sync = fail_stream_sync
        self.events: list[tuple] = []
        self.live_ptrs: set[int] = set()
        self.reshard_calls = 0
        self.live_copy_calls = 0
        self._next_ptr = 0x1000
        self._memory: dict[int, bytearray] = {}

    def allocate_live(self, contents: bytes) -> int:
        ptr = self._allocate(len(contents))
        self._memory[ptr][:] = contents
        self.live_ptrs.add(ptr)
        return ptr

    def read(self, ptr: int, nbytes: int) -> bytes:
        buf, offset = self._find(ptr, nbytes)
        return bytes(buf[offset : offset + nbytes])

    def _allocate(self, nbytes: int) -> int:
        ptr = self._next_ptr
        self._next_ptr += nbytes + 0x100
        self._memory[ptr] = bytearray(nbytes)
        return ptr

    def _find(self, ptr: int, nbytes: int) -> tuple[bytearray, int]:
        for base, buf in self._memory.items():
            offset = ptr - base
            if 0 <= offset and offset + nbytes <= len(buf):
                return buf, offset
        raise AssertionError(f"unknown pointer range: ptr={ptr:#x}, nbytes={nbytes}")

    def init(self, max_cta: int | None = None) -> None:
        self.events.append(("init", max_cta))

    def set_device(self, device_id: int) -> None:
        self.events.append(("set_device", device_id))

    def finalize(self) -> None:
        self.events.append(("finalize",))

    def mem_alloc(self, nbytes: int) -> int:
        ptr = self._allocate(nbytes)
        self.events.append(("alloc", ptr, nbytes))
        return ptr

    def mem_free(self, ptr: int) -> None:
        self.events.append(("free", ptr))
        del self._memory[ptr]

    def window_register(self, comm: int, ptr: int, nbytes: int) -> int:
        self.events.append(("window_register", ptr, nbytes))
        return ptr + 1

    def window_deregister(self, comm: int, window: int) -> None:
        self.events.append(("window_deregister", window))

    def memcpy_dtod_async(
        self, dst_ptr: int, src_ptr: int, nbytes: int, stream: int
    ) -> None:
        self.events.append(("copy_async", dst_ptr, src_ptr, nbytes, stream))
        if dst_ptr in self.live_ptrs:
            self.live_copy_calls += 1
            if self.live_copy_calls == self.fail_live_copy_at:
                raise RuntimeError("injected live-copy failure")
        src, src_offset = self._find(src_ptr, nbytes)
        dst, dst_offset = self._find(dst_ptr, nbytes)
        dst[dst_offset : dst_offset + nbytes] = src[src_offset : src_offset + nbytes]

    def stream_synchronize(self, stream: int) -> None:
        self.events.append(("stream_synchronize", stream))
        if self.fail_stream_sync:
            raise RuntimeError("injected stream-sync failure")

    def comm_get_async_error(self, comm: int) -> int:
        return binding.ncclSuccess

    def reshard(self, comm, window, src, dst, stream=0) -> None:
        call = self.reshard_calls
        self.reshard_calls += 1
        self.events.append(("reshard", call, stream))
        if call == self.fail_reshard_at:
            raise RuntimeError("injected reshard failure")
        if dst.dataPtr:
            payload = self.payloads[call]
            buf, offset = self._find(dst.dataPtr, len(payload))
            buf[offset : offset + len(payload)] = payload

    def comm_destroy(self, comm: int) -> None:
        self.events.append(("comm_destroy", comm))


def _executor(
    fake: FakeM2N, *, rank: int = 1, stream: int = 0
) -> NcclM2nExecutor:
    return NcclM2nExecutor(
        fake,
        comm=7,
        rank=rank,
        tp_src=1,
        tp_dst=1,
        device_id=0,
        stream=stream,
    )


def _params(fake: FakeM2N) -> tuple[list[ReshardParam], list[int]]:
    old_values = [b"old0", b"old"]
    ptrs = [fake.allocate_live(value) for value in old_values]
    params = [
        ReshardParam(
            name=f"p{index}",
            global_shape=(len(value),),
            ndims=1,
            shard_dim=REPLICATE,
            dtype_nccl=binding.ncclUint8,
            local_ptr=ptr,
            local_nbytes=len(value),
        )
        for index, (ptr, value) in enumerate(zip(ptrs, old_values, strict=True))
    ]
    return params, ptrs


def test_destination_commits_only_after_complete_version_is_staged():
    fake = FakeM2N([b"new0", b"new"])
    params, _ = _params(fake)
    executor = _executor(fake)

    executor.execute(params, window_bytes=4)

    assert [
        fake.read(param.local_ptr, param.local_nbytes) for param in params
    ] == [b"new0", b"new"]
    second_reshard = max(i for i, event in enumerate(fake.events) if event[0] == "reshard")
    first_live_copy = min(
        i
        for i, event in enumerate(fake.events)
        if event[0] == "copy_async" and event[1] in fake.live_ptrs
    )
    assert second_reshard < first_live_copy
    pipeline = [
        event
        for event in fake.events
        if event[0] in ("copy_async", "reshard", "stream_synchronize")
    ]
    assert [event[0] for event in pipeline] == [
        "reshard",
        "copy_async",
        "reshard",
        "copy_async",
        "stream_synchronize",
        "copy_async",
        "copy_async",
        "stream_synchronize",
    ]
    assert all(event[-1] == 0 for event in pipeline)


def test_reshard_failure_leaves_complete_live_version_unchanged():
    fake = FakeM2N([b"new0", b"new"], fail_reshard_at=1)
    params, _ = _params(fake)
    executor = _executor(fake)

    with pytest.raises(RuntimeError, match="injected reshard failure"):
        executor.execute(params, window_bytes=4)

    assert [
        fake.read(param.local_ptr, param.local_nbytes) for param in params
    ] == [b"old0", b"old"]
    assert fake.live_copy_calls == 0
    stream_sync = max(
        i for i, event in enumerate(fake.events) if event[0] == "stream_synchronize"
    )
    window_release = min(
        i for i, event in enumerate(fake.events) if event[0] == "window_deregister"
    )
    assert stream_sync < window_release


def test_commit_failure_poison_executor_until_reinitialized():
    fake = FakeM2N([b"new0", b"new"], fail_live_copy_at=2)
    params, _ = _params(fake)
    executor = _executor(fake)

    with pytest.raises(RuntimeError, match="serving must remain stopped"):
        executor.execute(params, window_bytes=4)

    reshard_calls = fake.reshard_calls
    with pytest.raises(RuntimeError, match="unusable after a failed model commit"):
        executor.execute(params, window_bytes=4)
    assert fake.reshard_calls == reshard_calls


def test_default_stream_token_orders_source_pipeline():
    fake = FakeM2N([])
    params, _ = _params(fake)
    executor = _executor(fake, rank=0)

    executor.execute(params, window_bytes=4)

    pipeline = [
        event
        for event in fake.events
        if event[0] in ("copy_async", "reshard", "stream_synchronize")
    ]
    assert [event[0] for event in pipeline] == [
        "copy_async",
        "reshard",
        "copy_async",
        "reshard",
        "stream_synchronize",
    ]
    assert all(event[-1] == 0 for event in pipeline)

    executor.teardown()


def test_caller_explicit_stream_is_used():
    fake = FakeM2N([])
    params, _ = _params(fake)
    caller_stream = 0xBEEF
    executor = _executor(fake, rank=0, stream=caller_stream)

    executor.execute(params, window_bytes=4)
    executor.teardown()

    pipeline = [
        event
        for event in fake.events
        if event[0] in ("copy_async", "reshard", "stream_synchronize")
    ]
    assert all(event[-1] == caller_stream for event in pipeline)
    assert [event for event in fake.events if event[0] == "set_device"] == [
        ("set_device", 0),
        ("set_device", 0),
        ("set_device", 0),
    ]


def test_failed_stream_drain_retains_in_flight_resources_and_disables_executor():
    fake = FakeM2N([b"new0", b"new"], fail_stream_sync=True)
    params, _ = _params(fake)
    executor = _executor(fake)

    with pytest.raises(RuntimeError, match="injected stream-sync failure"):
        executor.execute(params, window_bytes=4)

    assert not any(event[0] == "window_deregister" for event in fake.events)
    assert not any(event[0] == "free" for event in fake.events)
    with pytest.raises(RuntimeError, match="stream could not be drained"):
        executor.execute(params, window_bytes=4)
