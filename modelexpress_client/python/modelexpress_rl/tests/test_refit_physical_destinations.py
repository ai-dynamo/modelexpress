# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from itertools import product
import pytest
import torch
from modelexpress_rl.refit.reshard import geometry
from modelexpress_rl.refit.reshard.geometry import build_lazy_weights, capture_weights

class PhysicalViewModel(torch.nn.Module):
    def __init__(self, *, step=1, registry="buffer", stamped=False, start=0, size=6):
        super().__init__()
        self.backing = torch.full((24,), -17.0)
        value = self.backing.as_strided((2, 3), (step, 2 * step), 3)
        if registry == "parameter":
            self.bias = torch.nn.Parameter(value, requires_grad=False)
        else:
            self.register_buffer("bias", value)
        self.step, self.start, self.size = step, start, size
        self.stamped = stamped
        self.capture_calls = 0
        self.bias.weight_loader = self.copy_view

    def copy_view(self, target, source):
        target.as_strided(
            (self.size,), (self.step,), target.storage_offset() + self.start * self.step
        ).copy_(source)

    def load_weights(self, weights):
        self.capture_calls += 1
        for _, source in weights:
            if self.stamped:
                self.bias.weight_loader(self.bias, source)
            else:
                self.copy_view(self.bias, source)


def test_legacy_destination_reconstruction_keeps_canonical_mapping():
    model = PhysicalViewModel(stamped=True)
    captured = capture_weights(
        model, build_lazy_weights([("weight", torch.float32, (6,))])
    )
    assert captured.unsupported == ["weight"]
    assert "canonical owner coordinates" in captured.unsupported_reasons["weight"]


@pytest.mark.parametrize("stamped", [False, True])
@pytest.mark.parametrize("offset", [-1, 5])
def test_physical_capture_rejects_views_outside_owner_within_same_allocation(
    stamped, offset
):
    model = PhysicalViewModel(stamped=stamped, start=offset, size=2)
    before = model.backing.clone()
    captured = capture_weights(
        model, build_lazy_weights([("weight", torch.float32, (2,))]), source_replay=True
    )
    assert captured.unsupported == ["weight"]
    assert "bounds" in captured.unsupported_reasons["weight"]
    assert torch.equal(model.backing, before)


@pytest.mark.parametrize("stamped", [False, True])
def test_physical_capture_rejects_writes_into_owner_storage_gaps(stamped):
    class Model(PhysicalViewModel):
        def copy_view(self, target, source):
            target.as_strided((2,), (1,), target.storage_offset()).copy_(source)

    model = Model(step=2, stamped=stamped)
    captured = capture_weights(
        model, build_lazy_weights([("weight", torch.float32, (2,))]), source_replay=True
    )
    assert captured.unsupported == ["weight"]


def test_physical_owner_resolution_distinguishes_interleaved_storage():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            backing = torch.zeros(8)
            self.register_buffer("even", backing[::2])
            self.register_buffer("odd", backing[1::2])

        def load_weights(self, weights):
            for _, source in weights:
                self.even[2:3].copy_(source)

    captured = capture_weights(
        Model(),
        build_lazy_weights([("weight", torch.float32, (1,))]),
        source_replay=True,
    )
    assert not captured.unsupported and not captured.unattributed
    assert captured.copies[0].param_name == "even"
    assert captured.copies[0].dest_offset == 4


def test_physical_owner_resolution_rejects_ambiguous_views():
    model = PhysicalViewModel(start=1, size=4)
    model.register_buffer("alias", model.backing[4:8])
    captured = capture_weights(
        model, build_lazy_weights([("weight", torch.float32, (4,))]), source_replay=True
    )
    assert captured.unsupported == ["weight"]
    assert "ambiguous" in captured.unsupported_reasons["weight"]


@pytest.mark.parametrize(
    "owner_shape,owner_stride",
    [
        ((2, 3), (1, 2)),
        ((2, 3), (2, 4)),
        ((3, 2), (7, 2)),
        ((2, 2, 2), (20, 5, 1)),
        ((0, 3), (3, 1)),
        ((), ()),
    ],
)
def test_physical_bounds_match_element_address_ownership(owner_shape, owner_stride):
    from modelexpress_rl.refit.reshard.types import UnsupportedReshard

    owned = {
        sum(i * step for i, step in zip(index, owner_stride))
        for index in product(*(range(size) for size in owner_shape))
    }
    span = max(owned, default=-1) + 1
    for shape in ((), (0,), (1,), (2,), (4,), (2, 3)):
        for stride in product(range(4), repeat=len(shape)):
            for offset in range(-1, span + 2):
                addressed = {
                    offset + sum(i * step for i, step in zip(index, stride))
                    for index in product(*(range(size) for size in shape))
                }
                expected = 0 <= offset <= span and addressed <= owned
                try:
                    geometry._validate_physical_destination(
                        shape, stride, offset, owner_shape, owner_stride
                    )
                    accepted = True
                except UnsupportedReshard:
                    accepted = False
                assert accepted == expected, (shape, stride, offset)
