# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from modelexpress.refit_trainer_step import (
    trainer_step_replacement_tensor,
    trainer_step_source_provenance,
    trainer_step_tensor_for_range,
)


def _range_slices(tensor_range):
    return tuple(slice(start, end) for start, end in tensor_range)


def test_trainer_step_source_range_matches_full_replacement_slice():
    shape = (7, 4)
    tensor_range = ((2, 6), (0, 4))

    full = trainer_step_replacement_tensor(
        shape,
        dtype=torch.float32,
        device="cpu",
    )
    source = trainer_step_tensor_for_range(
        shape,
        tensor_range,
        dtype=torch.float32,
        device="cpu",
    )

    assert source.shape == (4, 4)
    torch.testing.assert_close(source, full[_range_slices(tensor_range)])


def test_trainer_step_changes_parameter_from_initial_values():
    shape = (5, 3)
    tensor_range = ((0, 5), (0, 3))

    initial = trainer_step_tensor_for_range(
        shape,
        tensor_range,
        dtype=torch.float32,
        device="cpu",
        step_count=0,
    )
    after_step = trainer_step_tensor_for_range(
        shape,
        tensor_range,
        dtype=torch.float32,
        device="cpu",
        step_count=1,
    )

    assert not torch.allclose(initial, after_step)


def test_trainer_step_preserves_requested_dtype():
    payload = trainer_step_replacement_tensor(
        (6, 4),
        dtype=torch.bfloat16,
        device="cpu",
    )

    assert payload.dtype == torch.bfloat16
    assert payload.shape == (6, 4)


def test_trainer_step_provenance_is_explicit():
    provenance = trainer_step_source_provenance(step_count=2, learning_rate=0.25)

    assert provenance["source_payload_generator"] == "torch.optim.SGD"
    assert provenance["optimizer_step_count"] == 2
    assert provenance["learning_rate"] == 0.25
    assert provenance["optimizer_step_publisher_used"] is True
    assert provenance["synthetic_training_objective_used"] is True
    assert provenance["static_replacement_formula_used"] is False
    assert provenance["real_rl_training_loop_used"] is False
