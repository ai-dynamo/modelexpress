# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang manifest coverage for weight-derived tensors held in containers.

``capture_tensor_attrs`` only promotes bare Tensor attributes assigned on an
``nn.Module``. Weight-derived tensors that live inside a dict/tuple, or on a
non-Module object such as a quant method, stay invisible to
``named_parameters()``/``named_buffers()`` and so never reach the RDMA
manifest. On a target they keep whatever was derived from random-init
weights, which passes every manifest check and then degrades inference.

Kimi-K3 has two of these: ``_attn_res_cw_cache`` (dict, read by ``get_cw``)
and ``_k3_fused_decode_args`` (tuple, consumed by the fused KDA decode
kernel). ``discover_tensors`` calls ``adopt_hidden_tensors`` so both are
registered and transferred, which keeps the receive path on its original
contract: transfer the final state, derive nothing afterwards.
"""

import contextlib
import sys
import types

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


class _CpuAsAccel:
    """Backend interface with CPU tensors treated as accelerator memory."""

    torch_device_type = "cpu"

    def is_accel_tensor(self, tensor: torch.Tensor) -> bool:
        return tensor.device.type == "cpu"

    def supports_rdma_p2p(self) -> bool:
        return True

    def supports_pool_reg(self) -> bool:
        return True


class _QuantMethod:
    """Non-Module object holding a derived tensor, as SGLang quant methods do."""

    def __init__(self, swizzled: torch.Tensor):
        self.swizzled_weight = swizzled


class _Layer(nn.Module):
    def __init__(self, h=8):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(h, h))
        # dict-held, keyed by dtype (cf. SGLang get_cw / _attn_res_cw_cache)
        self._attn_res_cw_cache = {
            torch.float32: torch.randn(h),
            torch.bfloat16: torch.randn(h).to(torch.bfloat16),
        }
        # tuple-held, mixed tensors and a scalar (cf. _k3_fused_decode_args)
        self._k3_fused_decode_args = (torch.randn(4, h), torch.randn(h), 1e-5)
        self.quant_method = _QuantMethod(torch.randn(h, h))


def _adapter() -> SglangAdapter:
    adapter = SglangAdapter(
        load_config=types.SimpleNamespace(),
        model_config=types.SimpleNamespace(),
        device_config=types.SimpleNamespace(device="cuda"),
    )
    adapter.accelerator_backend = _CpuAsAccel()
    adapter.target_device = torch.device("cpu")
    return adapter


def _discover(model):
    return _adapter().discover_tensors(LoadResult(value=model, model=model))


def test_container_held_tensors_reach_the_manifest():
    model = _Layer()
    tensors = _discover(model)

    # collect_module_tensors yields tensor.data, a fresh object over the same
    # storage, so identity is by address rather than by id().
    hidden = {
        model._attn_res_cw_cache[torch.float32].data_ptr(),
        model._attn_res_cw_cache[torch.bfloat16].data_ptr(),
        model._k3_fused_decode_args[0].data_ptr(),
        model._k3_fused_decode_args[1].data_ptr(),
        model.quant_method.swizzled_weight.data_ptr(),
    }
    covered = {t.data_ptr() for t in tensors.values()}
    assert hidden <= covered, (
        f"{len(hidden - covered)} weight-derived tensors missing from the "
        f"manifest: {sorted(tensors)}"
    )


def test_manifest_entry_aliases_the_container_entry():
    """An RDMA write into the manifest entry must be visible via the container.

    adopt_hidden_tensors registers the same tensor object rather than a copy,
    so the dict entry and the buffer share storage. Without that, a transfer
    would land in memory the engine never reads.
    """
    model = _Layer()
    tensors = _discover(model)

    cached = model._attn_res_cw_cache[torch.float32]
    entry = next(t for t in tensors.values() if t.data_ptr() == cached.data_ptr())

    with torch.no_grad():  # stand in for the RDMA write
        entry.copy_(torch.full_like(entry, 4.25))

    assert torch.equal(
        model._attn_res_cw_cache[torch.float32],
        torch.full_like(cached, 4.25),
    ), "manifest entry does not alias the cached tensor"


def test_non_tensor_container_members_are_ignored():
    """The scalar in _k3_fused_decode_args must not become a manifest entry.

    Counting the adopted entries rather than type-checking them: the
    collector returns tensors by construction, so an isinstance check would
    pass whether or not the scalar was adopted.
    """
    model = _Layer()
    tensors = _discover(model)

    adopted = [n for n in tensors if "_k3_fused_decode_args" in n]
    tensor_members = sum(
        1 for m in model._k3_fused_decode_args if isinstance(m, torch.Tensor)
    )
    assert len(model._k3_fused_decode_args) > tensor_members, (
        "fixture must hold a non-tensor member for this test to mean anything"
    )
    assert len(adopted) == tensor_members, (
        f"expected {tensor_members} adopted entries, got {len(adopted)}: {adopted}"
    )


def test_parameters_are_not_duplicated():
    """Aliases of already-registered tensors are skipped by data_ptr dedup."""
    model = _Layer()
    model._alias_of_weight = (model.weight.data,)
    tensors = _discover(model)

    ptr = model.weight.data_ptr()
    assert sum(1 for t in tensors.values() if t.data_ptr() == ptr) == 1
