# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Invariants for the SGLang RDMA receive path.

A receiving worker runs

    before_rdma_receive -> discover_tensors -> register -> [RDMA lands]
      -> after_rdma_receive -> publish weight_info

and then serves as a source for the next worker. ``weight_info`` snapshots
``tensor.data_ptr()`` at registration time (engines/sglang/loader.py), and
nixl_transfer.py documents the matching requirement: the registered tensors
must remain the same objects the model computes with.

Anything ``after_rdma_receive`` runs that reallocates a manifest tensor
breaks that pairing. These tests pin both halves: the value-derived state
must be rebuilt from the received weights, and rebuilding it must not move
any tensor that was already registered.

The model mirrors the structure of SGLang's Kimi-K3 ``post_load_weights``
(kimi-k3 branch): MLA absorption into w_kc/w_vc, the horizontally-fused
front merge, and the ``_attn_res_cw_cache`` value cache that ``get_cw``
short-circuits on. Only the accelerator-device predicate is stubbed; the
adapter code under test is the real one.
"""

import contextlib
import sys
import types

import pytest
import torch
from torch import nn

_loader = types.ModuleType("sglang.srt.model_loader.loader")
_loader.device_loading_context = contextlib.contextmanager(
    lambda module, device: iter([None])
)
for _name, _mod in (
    ("sglang", types.ModuleType("sglang")),
    ("sglang.srt", types.ModuleType("sglang.srt")),
    ("sglang.srt.model_loader", types.ModuleType("sglang.srt.model_loader")),
    ("sglang.srt.model_loader.loader", _loader),
):
    sys.modules.setdefault(_name, _mod)

from modelexpress.engines.sglang.adapter import SglangAdapter  # noqa: E402
from modelexpress.load_strategy.context import LoadResult  # noqa: E402
from modelexpress.tensor_utils import capture_tensor_attrs  # noqa: E402


class _CpuAsAccel:
    """Backend interface with CPU tensors treated as accelerator memory."""

    torch_device_type = "cpu"

    def is_accel_tensor(self, tensor: torch.Tensor) -> bool:
        return tensor.device.type == "cpu"

    def supports_rdma_p2p(self) -> bool:
        return True

    def supports_pool_reg(self) -> bool:
        return True


def _merge_weights_as_views(mods):
    """Transcribed from kimi_k3.py _merge_weights_as_views."""
    ws = [m.weight.data for m in mods]
    sizes = [w.shape[0] for w in ws]
    merged = torch.cat(ws, dim=0).contiguous()
    off = 0
    for m, n in zip(mods, sizes):
        m.weight.data = merged[off : off + n]
        off += n
    return merged


def _get_cw(proj, norm, dtype=torch.float32):
    """Transcribed from sglang attn_residual.get_cw."""
    cache = getattr(proj, "_attn_res_cw_cache", None)
    if cache is None:
        cache = {}
        proj._attn_res_cw_cache = cache
    cw = cache.get(dtype)
    if cw is None:
        cw = (norm.weight.float() * proj.weight.squeeze().float()).contiguous()
        cw = cache[dtype] = cw.to(dtype)
    return cw


class _MLAAttention(nn.Module):
    def __init__(self, heads=2, qk_nope=4, v_head=4, kv_lora=8):
        super().__init__()
        self.qk_nope_head_dim = qk_nope
        self.v_head_dim = v_head
        self.kv_b_proj = nn.Linear(kv_lora, heads * (qk_nope + v_head), bias=False)

    def absorb(self):
        w_kc, w_vc = self.kv_b_proj.weight.unflatten(
            0, (-1, self.qk_nope_head_dim + self.v_head_dim)
        ).split([self.qk_nope_head_dim, self.v_head_dim], dim=1)
        self.w_kc = w_kc.transpose(1, 2).contiguous().transpose(1, 2)
        self.w_vc = w_vc.contiguous().transpose(1, 2)


class _MoE(nn.Module):
    def __init__(self, h=8):
        super().__init__()
        self.gate = nn.Linear(h, 4, bias=False)
        self.routed_expert_down_proj = nn.Linear(h, 6, bias=False)
        self._front_w = None

    def merge_front(self):
        self._front_w = _merge_weights_as_views(
            [self.gate, self.routed_expert_down_proj]
        )


class _Layer(nn.Module):
    def __init__(self, h=8):
        super().__init__()
        self.self_attn = _MLAAttention(kv_lora=h)
        self.mlp = _MoE(h)
        self.res_proj = nn.Linear(h, 1, bias=False)
        self.res_norm = nn.LayerNorm(h)


class _K3Like(nn.Module):
    def __init__(self, h=8, layers=2):
        super().__init__()
        self.layers = nn.ModuleList([_Layer(h) for _ in range(layers)])

    def post_load_weights(self):
        for layer in self.layers:
            layer.self_attn.absorb()
            _get_cw(layer.res_proj, layer.res_norm, dtype=torch.bfloat16)
            _get_cw(layer.res_proj, layer.res_norm)
            layer.mlp.merge_front()


def _adapter():
    adapter = SglangAdapter(
        load_config=types.SimpleNamespace(),
        model_config=types.SimpleNamespace(),
        device_config=types.SimpleNamespace(device="cuda"),
    )
    adapter.accelerator_backend = _CpuAsAccel()
    adapter.target_device = torch.device("cpu")
    return adapter


@pytest.fixture
def received():
    """Drive a receive to just past after_rdma_receive.

    Returns (adapter, result, registered, published) where ``registered``
    holds the tensor objects handed to register_memory and ``published``
    holds the (addr, numel, element_size) triples snapshotted alongside
    them, exactly as loader.py builds weight_info.
    """
    torch.manual_seed(0)
    adapter = _adapter()
    model = _K3Like()
    result = LoadResult(value=model, model=model)

    result = adapter.before_rdma_receive(result)
    registered = adapter.discover_tensors(result)
    published = {
        name: (t.data_ptr(), t.numel(), t.element_size())
        for name, t in registered.items()
    }

    # The bytes that land come from a source that ran the same derivation on
    # real weights, so derived tensors are self-consistent with their inputs.
    torch.manual_seed(1234)
    source = _K3Like()
    with capture_tensor_attrs(adapter.accelerator_backend):
        source.post_load_weights()
    src = adapter.discover_tensors(LoadResult(value=source, model=source))
    assert set(src) == set(registered), "source/target manifest asymmetry"
    with torch.no_grad():
        for name, tensor in registered.items():
            tensor.copy_(src[name])

    result = adapter.after_rdma_receive(result)
    return adapter, result, registered, published


def test_value_derived_cache_matches_the_received_weights(received):
    """cw must equal a fresh derivation from the weights that landed.

    Satisfied by manifest coverage: discover_tensors adopts the dict-held
    entries, so the transfer writes the source's cw straight into them. No
    post-receive derivation runs, so nothing can double-transform a weight
    or move a registered tensor.
    """
    _, result, _, _ = received
    for layer in result.model.layers:
        cache = getattr(layer.res_proj, "_attn_res_cw_cache", None)
        assert cache, "value-derived cache missing after receive"
        expected = (
            layer.res_norm.weight.float() * layer.res_proj.weight.squeeze().float()
        ).contiguous()
        for dtype, cached in cache.items():
            assert torch.equal(cached, expected.to(dtype)), (
                f"cw for {dtype} was not rebuilt from the received weights"
            )


def test_registered_tensors_keep_their_addresses(received):
    """No manifest tensor may move after it has been registered.

    weight_info is snapshotted at registration and published unchanged on
    the success path (engines/sglang/loader.py) -- the loader never
    re-discovers. A tensor that moves leaves the published entry pointing
    at an orphaned copy that the model no longer computes with.
    """
    adapter, result, _, published = received
    live = adapter.discover_tensors(result)
    moved = {
        name: (addr, live[name].data_ptr())
        for name, (addr, _, _) in published.items()
        if name in live and live[name].data_ptr() != addr
    }
    assert not moved, (
        f"{len(moved)} of {len(published)} registered tensors moved after "
        f"after_rdma_receive: {sorted(moved)}"
    )


def test_registered_tensors_are_the_tensors_the_model_uses(received):
    """The registered objects must still be reachable from the module tree.

    nixl_transfer.register_tensors documents this pairing: the registered
    dict must hold the same objects as the live parameters/buffers, or a
    later RDMA read serves memory the model has stopped updating.
    """
    adapter, result, registered, _ = received
    live = adapter.discover_tensors(result)
    detached = [
        name
        for name, tensor in registered.items()
        if name in live and live[name].data_ptr() != tensor.data_ptr()
    ]
    assert not detached, (
        f"registered tensors no longer paired with live model tensors: "
        f"{sorted(detached)}"
    )
