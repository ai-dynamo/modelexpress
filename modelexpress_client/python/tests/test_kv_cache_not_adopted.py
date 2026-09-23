# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime KV/state cache must stay out of the RDMA manifest.

``adopt_hidden_tensors`` scans plain Python attributes for accelerator tensors
that post-load processing created outside ``register_buffer``, so they reach the
target. vLLM's Mamba/GDN layers keep their conv/ssm state in exactly that shape
of object -- ``mamba/abstract.py``: ``self.kv_cache = tuple(states)`` -- so the
scan used to adopt it and publish it like a weight.

On a restored Qwen3-Next-80B replica that was fatal, not merely wasteful. The
state tensors are non-contiguous views into the KV-cache arena, so
``collect_module_tensors`` widened each to its full underlying storage: 1.48 GiB
per layer against a 68 MB tensor, 12 of them, +18.6 GB on the manifest. The
arena is multi-handle VMM memory, and ``ibv_reg_mr`` rejects a range spanning
``cuMemCreate`` boundaries:

    ibv_reg_mr(address=0x2880000000, length=1550843904, ...) failed: Bad address

which failed the whole registration, then the RDMA strategy, then the fallback
rebuild OOMed. The address and length matched
``model.layers.0.linear_attn._mx_kv_cache_0_t.__storage`` exactly.

Cold loads never hit it: the attribute still holds placeholder empty tensors,
which are skipped for having no elements. It needs a restore, where weights are
re-registered with the cache already populated.

Run: pytest tests/test_kv_cache_not_adopted.py
"""

import torch
import torch.nn as nn

from modelexpress.tensor_utils import adopt_hidden_tensors


class FakeBackend:
    """Treat every tensor as an accelerator tensor so this runs without a GPU."""

    def is_accel_tensor(self, tensor: torch.Tensor) -> bool:
        return isinstance(tensor, torch.Tensor)


class GdnLikeLayer(nn.Module):
    """Mirrors the vLLM shape: state in a plain tuple attribute."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(4))
        # mamba/abstract.py: self.kv_cache = tuple(states)
        self.kv_cache = (torch.zeros(8), torch.zeros(8))


class QuantLikeLayer(nn.Module):
    """The case adoption exists for: weight-derived tensors on a plain object."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(4))
        self.quant_state = (torch.zeros(8),)


def _buffer_names(model: nn.Module) -> set[str]:
    return {name for name, _ in model.named_buffers()}


def test_kv_cache_state_is_not_adopted():
    layer = GdnLikeLayer()
    adopted = adopt_hidden_tensors(layer, accelerator_backend=FakeBackend())

    assert adopted == 0
    assert not any("kv_cache" in name for name in _buffer_names(layer))


def test_weight_derived_tensors_are_still_adopted():
    """The exclusion must be narrow -- this is what the scan is for."""
    layer = QuantLikeLayer()
    adopted = adopt_hidden_tensors(layer, accelerator_backend=FakeBackend())

    assert adopted == 1
    assert "_mx_quant_state_0_t" in _buffer_names(layer)


def test_exclusion_is_per_attribute_not_per_module():
    """A layer carrying both must keep the weight-derived half."""

    class Both(nn.Module):
        def __init__(self):
            super().__init__()
            self.kv_cache = (torch.zeros(8),)
            self.quant_state = (torch.zeros(8),)

    layer = Both()
    adopted = adopt_hidden_tensors(layer, accelerator_backend=FakeBackend())

    names = _buffer_names(layer)
    assert adopted == 1
    assert "_mx_quant_state_0_t" in names
    assert not any("kv_cache" in name for name in names)


def test_empty_placeholder_cache_is_a_no_op():
    """Cold-load shape: the attribute exists but holds empty tensors."""

    class ColdLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.kv_cache = (torch.tensor([]), torch.tensor([]))

    layer = ColdLayer()
    assert adopt_hidden_tensors(layer, accelerator_backend=FakeBackend()) == 0


def test_compiled_fn_example_inputs_are_not_adopted():
    """A reload discovers tensors on a compiled model; torch.compile's example
    inputs hang off it and are not weights."""

    class Artifacts:
        def __init__(self):
            self.example_inputs = [None, torch.zeros(8, dtype=torch.int32)]

    class CompiledFn:
        def __init__(self):
            self._artifacts = Artifacts()

    class CompiledModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(4))
            self.aot_compiled_fn = CompiledFn()

    model = CompiledModel()
    assert adopt_hidden_tensors(model, accelerator_backend=FakeBackend()) == 0
    assert not any("aot_compiled_fn" in name for name in _buffer_names(model))
