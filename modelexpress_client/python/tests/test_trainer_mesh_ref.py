# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import FrozenInstanceError

import pytest
from modelexpress_rl import (
    TrainerMeshRef,
    WeightPayloadFormat,
    WeightVersion,
    WeightVersionState,
)
from modelexpress_rl.control import TrainerMesh


@pytest.mark.parametrize("mesh_id", ["", " \t\n"])
def test_trainer_mesh_ref_rejects_blank_mesh_id(mesh_id: str) -> None:
    with pytest.raises(ValueError, match="mesh_id"):
        TrainerMeshRef(mesh_id, 1)


@pytest.mark.parametrize("mesh_id", [None, 1])
def test_trainer_mesh_ref_rejects_non_string_mesh_id(mesh_id) -> None:
    with pytest.raises(TypeError, match="mesh_id"):
        TrainerMeshRef(mesh_id, 1)


@pytest.mark.parametrize("generation", [True, False, 0, -1, 2**64])
def test_trainer_mesh_ref_rejects_invalid_generation(generation) -> None:
    with pytest.raises(ValueError, match="generation"):
        TrainerMeshRef("mesh-a", generation)


def test_trainer_mesh_ref_is_immutable_and_exported() -> None:
    ref = TrainerMeshRef("mesh-a", 1)
    with pytest.raises(FrozenInstanceError):
        ref.mesh_id = "mesh-b"


def test_trainer_mesh_exposes_ref() -> None:
    mesh = TrainerMesh("mesh-a", "model-a", 7, {})
    assert mesh.ref == TrainerMeshRef("mesh-a", 7)


@pytest.mark.parametrize(
    ("mesh_id", "generation", "expected"),
    [("mesh-a", 7, TrainerMeshRef("mesh-a", 7)), (None, 0, None)],
)
def test_weight_version_exposes_optional_mesh_ref(mesh_id, generation, expected) -> None:
    version = WeightVersion(
        version_id="version-a",
        model_name="model-a",
        payload_format=WeightPayloadFormat.FULL_TENSOR,
        layout_signature="",
        state=WeightVersionState.READY,
        created_at_unix_ms=0,
        trainer_mesh_generation=generation,
        trainer_mesh_id=mesh_id,
    )
    assert version.mesh_ref == expected
