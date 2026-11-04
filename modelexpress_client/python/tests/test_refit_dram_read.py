# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""post_read_batch may target pinned host memory while reading remote VRAM."""

from types import SimpleNamespace

from modelexpress.nixl_transfer import NIXL_DRAM_MEM_TYPE, NixlTransferManager


class _Agent:
    def __init__(self):
        self.prepped = []  # (agent_name, xfer_list, mem_type)

    def prep_xfer_dlist(self, *, agent_name, xfer_list, mem_type, backends):
        self.prepped.append((agent_name, list(xfer_list), mem_type))
        return ("prepped", agent_name, mem_type)

    def make_prepped_xfer(self, **kwargs):
        return ("handle", kwargs["local_xfer_side"], kwargs["remote_xfer_side"])

    def transfer(self, handle):
        pass


def _manager():
    mgr = object.__new__(NixlTransferManager)
    mgr._agent = _Agent()
    mgr._backends = ["UCX"]
    mgr._device_id = 3
    mgr._accelerator_backend = SimpleNamespace(nixl_mem_type="VRAM")
    return mgr


def test_local_mem_type_defaults_to_remote_type():
    mgr = _manager()
    posted = mgr.post_read_batch("trainer", [(0x1000, 0x2000, 64, 7)])
    assert posted is not None
    remote, local = mgr._agent.prepped
    assert remote == ("trainer", [(0x1000, 64, 7)], "VRAM")
    assert local == ("", [(0x2000, 64, 3)], "VRAM")


def test_dram_local_type_uses_host_device_zero():
    mgr = _manager()
    mgr.post_read_batch(
        "trainer", [(0x1000, 0x2000, 64, 7)], local_mem_type=NIXL_DRAM_MEM_TYPE
    )
    remote, local = mgr._agent.prepped
    assert remote == ("trainer", [(0x1000, 64, 7)], "VRAM")
    assert local == ("", [(0x2000, 64, 0)], NIXL_DRAM_MEM_TYPE)


def test_execute_read_batch_forwards_local_mem_type(monkeypatch):
    mgr = _manager()
    seen = {}

    def post(remote_agent_name, ranges, mem_type=None, local_mem_type=None):
        seen["local"] = local_mem_type
        return None

    monkeypatch.setattr(mgr, "post_read_batch", post)
    assert mgr.execute_read_batch(
        "trainer", [(1, 2, 3, 4)], local_mem_type=NIXL_DRAM_MEM_TYPE
    ) == (0, 0, 0.0)
    assert seen["local"] == NIXL_DRAM_MEM_TYPE
