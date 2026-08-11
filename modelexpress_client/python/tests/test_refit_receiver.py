# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from modelexpress.refit.api import ReceiverRevisionState
from modelexpress.refit.receiver import (
    ModelExpressWeightReceiver,
    PreparedRevision,
    ReceiverConfig,
    ReceiverInstallError,
)


class Receiver(ModelExpressWeightReceiver):
    def __init__(self, tmp_path: Path):
        self.targets = {}
        loader = SimpleNamespace(
            _prepare_weights=lambda *_args: (tmp_path / "launch", None, None)
        )
        super().__init__(
            ReceiverConfig(
                model_id="model",
                catalog_endpoint="mx:8001",
                initial_version="0",
                preparation_cache_dir=tmp_path / "cache",
            ),
            "receiver",
            SimpleNamespace(
                loader=loader,
                model_config=SimpleNamespace(
                    model_path=str(tmp_path / "launch"),
                    revision=None,
                ),
            ),
        )
        self.installed_digest = "sha256:base"
        self.state = ReceiverRevisionState.VERIFIED
        self.installed = []
        self.install_error = None
        receiver = self

        class Checkpoint:
            def prepare(self, version, _installed_version, _installed_digest):
                target = receiver.targets[version]
                if isinstance(target, Exception):
                    raise target
                return target

            def installation(self, _prepared):
                return nullcontext()

        self.checkpoint = Checkpoint()

    def install_prepared_checkpoint(self, prepared):
        if self.install_error is not None:
            raise self.install_error
        self.installed.append(prepared)


def prepared(tmp_path: Path, version="1", digest="sha256:target", metrics=None):
    path = tmp_path / version
    path.mkdir(exist_ok=True)
    return PreparedRevision(version, digest, path, metrics or {})


def receiver(tmp_path):
    value = Receiver(tmp_path)
    target = prepared(tmp_path)
    value.targets["1"] = target
    return value, target


def test_prepare_and_install_advance_exact_identity(tmp_path):
    value, target = receiver(tmp_path)

    value.start_weight_update("1")
    result = value.update_weights()

    assert value.installed == [target]
    assert result.success
    assert result.installed_version == "1"
    assert result.target_digest == "sha256:target"
    assert value.status().installed_version == "1"


def test_receiver_metrics_are_drained_per_phase(tmp_path):
    value, _target = receiver(tmp_path)

    value.start_weight_update("1")
    prepare_metrics = value.pop_metrics()

    assert prepare_metrics["perf/mx_receive_prepare_time"] >= 0
    assert value.pop_metrics() == {}

    value.update_weights()
    install_metrics = value.pop_metrics()

    assert install_metrics["perf/mx_receive_install_time"] >= 0
    assert value.pop_metrics() == {}


def test_failed_prepare_replaces_stale_metrics(tmp_path):
    value, _target = receiver(tmp_path)
    value.targets["1"] = prepared(
        tmp_path,
        metrics={"perf/stale": 1.0},
    )
    value.targets["2"] = RuntimeError("download failed")
    value.start_weight_update("1")

    with pytest.raises(RuntimeError, match="download failed"):
        value.start_weight_update("2")

    metrics = value.pop_metrics()
    assert metrics["perf/mx_receive_prepare_time"] >= 0
    assert "perf/stale" not in metrics
    assert value.prepared_identity is None
    assert value.pop_metrics() == {}


def test_prepare_does_not_mutate_live_weights(tmp_path):
    value, _target = receiver(tmp_path)

    value.start_weight_update("1")

    assert value.installed == []
    assert value.status().installed_version == "0"


def test_prewrite_install_failure_is_failed(tmp_path):
    value, _target = receiver(tmp_path)
    value.install_error = ReceiverInstallError("setup failed", False)
    value.start_weight_update("1")

    result = value.update_weights()

    assert not result.success
    assert result.state.name == "FAILED"
    assert value.status().installed_version == "0"


def test_postwrite_install_failure_is_poisoned(tmp_path):
    value, _target = receiver(tmp_path)
    value.install_error = ReceiverInstallError("load failed", True)
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
