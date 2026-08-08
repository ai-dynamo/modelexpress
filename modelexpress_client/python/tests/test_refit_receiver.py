# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from modelexpress.refit.receiver import (
    ModelExpressWeightReceiver,
    ReceiverInstallError,
)


def prepared(tmp_path: Path, version="1", digest="sha256:target"):
    path = tmp_path / version
    path.mkdir()
    return {"version": version, "digest": digest, "path": path}


def receiver(tmp_path, *, install_target=lambda _target: None):
    target = prepared(tmp_path)
    return (
        ModelExpressWeightReceiver(
            receiver_id="receiver",
            model_id="model",
            installed_version="0",
            installed_digest="sha256:base",
            prepare_target=lambda *_args: target,
            install_target=install_target,
        ),
        target,
    )


def test_prepare_and_install_advance_exact_identity(tmp_path):
    installed = []
    value, target = receiver(tmp_path, install_target=installed.append)

    value.start_weight_update("1")
    result = value.update_weights()

    assert installed == [target]
    assert result.success
    assert result.installed_version == "1"
    assert result.target_digest == "sha256:target"
    assert value.status().installed_version == "1"


def test_prepare_does_not_mutate_live_weights(tmp_path):
    installed = []
    value, _target = receiver(tmp_path, install_target=installed.append)

    value.start_weight_update("1")

    assert installed == []
    assert value.status().installed_version == "0"


def test_prewrite_install_failure_is_failed(tmp_path):
    def fail(_target):
        raise ReceiverInstallError("setup failed", mutation_started=False)

    value, _target = receiver(tmp_path, install_target=fail)
    value.start_weight_update("1")

    result = value.update_weights()

    assert not result.success
    assert result.state.name == "FAILED"
    assert value.status().installed_version == "0"


def test_postwrite_install_failure_is_poisoned(tmp_path):
    def fail(_target):
        raise ReceiverInstallError("load failed", mutation_started=True)

    value, _target = receiver(tmp_path, install_target=fail)
    value.start_weight_update("1")

    result = value.update_weights()

    assert not result.success
    assert result.state.name == "POISONED"
    with pytest.raises(RuntimeError, match="poisoned"):
        value.start_weight_update("2")


def test_complete_model_only(tmp_path):
    value, _target = receiver(tmp_path)
    value.start_weight_update("1")

    with pytest.raises(ValueError, match="complete-model"):
        value.update_weights(layers=("model.layer",))
