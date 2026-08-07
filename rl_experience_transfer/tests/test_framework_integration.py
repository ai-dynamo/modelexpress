# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pinned-source native conversion and framework-pipeline component tests."""

from __future__ import annotations

import os
from itertools import product
from pathlib import Path

import pytest

from examples.framework_pipeline import (
    install_framework_source_shims,
    run_framework_pipeline,
)
from examples.framework_roundtrip import (
    FRAMEWORKS,
    run_framework_case,
    supports_conversion,
)

if os.environ.get("RLXFER_RUN_FRAMEWORK_INTEGRATION") != "1":
    pytest.skip(
        "set RLXFER_RUN_FRAMEWORK_INTEGRATION=1 in the isolated framework environment",
        allow_module_level=True,
    )

pytestmark = [
    pytest.mark.integration,
    pytest.mark.e2e,
    pytest.mark.nemo_rl,
    pytest.mark.prime_rl,
    pytest.mark.slime,
    pytest.mark.miles,
]


@pytest.fixture(scope="module", autouse=True)
def source_import_shims() -> None:
    """Bypass unrelated framework launch stacks while loading pinned native types."""

    install_framework_source_shims()


@pytest.mark.parametrize(
    ("producer_framework", "consumer_framework"),
    product(FRAMEWORKS, repeat=2),
)
def test_native_framework_conversion_matrix(
    tmp_path: Path,
    producer_framework: str,
    consumer_framework: str,
) -> None:
    result = run_framework_case(producer_framework, consumer_framework, tmp_path / "queue")

    assert result["result"] == "PASSED"
    expected = (
        "converted"
        if supports_conversion(producer_framework, consumer_framework)
        else ("rejected_as_unsafe")
    )
    assert result["expected_outcome"] == expected
    if expected == "converted":
        assert result["acknowledged"] is True
        assert result["gradient_finite"] is True
        assert result["gradient_present"] is True
        assert result["parameter_changed"] is True
    else:
        assert result["rejection"]


@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_framework_pipeline_components(tmp_path: Path, framework: str) -> None:
    result = run_framework_pipeline(framework, tmp_path / framework)

    assert result["result"] == "PASSED"
    assert result["acknowledged"] is True
    assert result["gradient_finite"] is True
    assert result["gradient_present"] is True
    assert result["parameter_changed"] is True
    assert result["rollout_source"]
    assert result["trainer_source"]
