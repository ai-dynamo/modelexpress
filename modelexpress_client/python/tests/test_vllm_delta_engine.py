# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The vLLM delta transfer engine and the receiver it drives as a client."""

import sys
from dataclasses import fields
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from tests.vllm_weight_transfer_fake import install_weight_transfer_fake

install_weight_transfer_fake()

from modelexpress.engines.vllm.refit import delta_engine  # noqa: E402
from modelexpress.engines.vllm.refit.delta_engine import (  # noqa: E402
    MX_WEIGHT_TRANSFER_BACKEND,
    MxDeltaInitInfo,
    MxDeltaUpdateInfo,
    MxWeightTransferEngine,
)  # noqa: E402
from modelexpress.engines.vllm.refit.delta_receiver import (  # noqa: E402
    VllmWeightReceiver,
)  # noqa: E402
from modelexpress.refit.api import (  # noqa: E402
    ReceiverRevisionState,
    WeightUpdateResult,
)
from modelexpress.refit.receiver import (  # noqa: E402
    PreparedRevision,
    ReceiverConfig,
    ReceiverInstallError,
)  # noqa: E402

INIT_INFO = {
    "model_id": "model",
    "catalog_endpoint": "mx:8001",
    "initial_version": "0",
    "preparation_cache_dir": "/tmp/mx-cache",
    "ready_timeout_seconds": 5.0,
    "s3_endpoint_url": "http://minio:9000",
}
PREPARED = PreparedRevision("7", "sha256:target", "/cache/checkpoint", {})


class Receiver:
    """Stands in for the delta receiver, recording the protocol it is driven by."""

    def __init__(self, **kwargs):
        self.config = kwargs["config"]
        self.receiver_id = kwargs["receiver_id"]
        self.engine = kwargs["engine"]
        self.installed_version = self.config.initial_version
        self.calls: list = []
        self.state = ReceiverRevisionState.VERIFIED
        self.prepare_error: Exception | None = None
        self.install_state: ReceiverRevisionState | None = None
        self.metrics: dict[str, float] = {}
        # A cold prepare by default: this rank downloaded and XORed the revision.
        self.prepare_metrics = {
            "perf/mx_receive_prepare_time": 3.0,
            "perf/mx_receive_delta_index_download": 0.5,
            "perf/mx_receive_pool": 2.0,
        }
        self._pending: str | None = None

    def initialize(self):
        self.calls.append("initialize")

    def start_weight_update(self, version):
        self.calls.append(("start_weight_update", version))
        self.metrics = dict(self.prepare_metrics)
        if self.prepare_error is not None:
            raise self.prepare_error
        self._pending = version

    def update_weights(self, layers=None, defer_verification=False):
        self.calls.append("update_weights")
        self.metrics = {"perf/mx_receive_install_time": 1.0}
        if self.install_state is not None:
            self.state = self.install_state
            return WeightUpdateResult(
                success=False,
                receiver_id=self.receiver_id,
                installed_version=self.installed_version,
                state=self.install_state,
                detail="install blew up",
            )
        if defer_verification:
            self.state = ReceiverRevisionState.BYTES_RECEIVED
            return WeightUpdateResult(
                success=True,
                receiver_id=self.receiver_id,
                installed_version=self.installed_version,
                state=self.state,
                target_digest="sha256:target",
            )
        self.installed_version = self._pending
        return WeightUpdateResult(
            success=True,
            receiver_id=self.receiver_id,
            installed_version=self.installed_version,
            state=ReceiverRevisionState.VERIFIED,
        )

    def mark_verified(self):
        self.calls.append("mark_verified")
        self.installed_version = self._pending
        self.state = ReceiverRevisionState.VERIFIED
        return WeightUpdateResult(
            success=True,
            receiver_id=self.receiver_id,
            installed_version=self.installed_version,
            state=self.state,
            target_digest="sha256:target",
        )

    def mark_poisoned(self, detail):
        self.calls.append("mark_poisoned")
        self.state = ReceiverRevisionState.POISONED
        return WeightUpdateResult(
            success=False,
            receiver_id=self.receiver_id,
            installed_version=self.installed_version,
            state=self.state,
            detail=detail,
        )

    def pop_metrics(self):
        metrics, self.metrics = self.metrics, {}
        return metrics

    def status(self):
        return SimpleNamespace(state=self.state)


class Loader:
    """Recording stand-in for vLLM's DefaultModelLoader."""

    def __init__(self, load_config):
        self.load_config = load_config
        self.calls: list = []

    def _prepare_weights(self, path, subfolder, revision, fall_back_to_pt, overrides):
        self.calls.append(("prepare", path, revision, fall_back_to_pt))
        return path, ["model.safetensors"], True

    def load_weights(self, model, model_config):
        self.calls.append(("load", model, model_config.model))


@pytest.fixture
def model():
    return nn.Linear(2, 2)


@pytest.fixture
def vllm_config():
    return SimpleNamespace(
        parallel_config=SimpleNamespace(rank=3, tensor_parallel_size=4),
        model_config=SimpleNamespace(model="/weights/launch", revision=None),
        load_config=SimpleNamespace(load_format="modelexpress"),
    )


@pytest.fixture
def reload_calls(monkeypatch):
    """Record vLLM's update window instead of opening a real one."""
    calls: list[str] = []
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.model_loader.reload",
        SimpleNamespace(
            initialize_layerwise_reload=lambda _model: calls.append("initialize"),
            finalize_layerwise_reload=lambda _model, _config: calls.append("finalize"),
        ),
    )
    return calls


@pytest.fixture
def loaders(monkeypatch):
    """Install a recording DefaultModelLoader; yields every instance built."""
    built: list[Loader] = []

    def build(load_config):
        built.append(Loader(load_config))
        return built[-1]

    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.model_loader.default_loader",
        SimpleNamespace(DefaultModelLoader=build),
    )
    return built


@pytest.fixture
def bare_engine(monkeypatch, model, vllm_config, reload_calls):
    monkeypatch.setattr(
        delta_engine,
        "build_delta_receiver",
        lambda _backend, **kwargs: Receiver(**kwargs),
    )
    return MxWeightTransferEngine(
        SimpleNamespace(backend=MX_WEIGHT_TRANSFER_BACKEND),
        vllm_config,
        torch.device("cpu"),
        model,
    )


@pytest.fixture
def engine(bare_engine):
    bare_engine.init_transfer_engine(bare_engine.parse_init_info(INIT_INFO))
    return bare_engine


def timing_records(caplog) -> list[str]:
    return [line for line in caplog.text.splitlines() if "MX_REFIT_TIMING" in line]


def test_engine_is_a_complete_weight_transfer_engine():
    assert not MxWeightTransferEngine.__abstractmethods__


def test_trainer_push_is_rejected():
    with pytest.raises(NotImplementedError, match="receiver-pulled"):
        MxWeightTransferEngine.trainer_send_weights(iter(()), {})


def test_engine_refuses_draft_model_updates():
    # One receiver is bound to one catalog model id and one local checkpoint.
    assert MxWeightTransferEngine.supports_draft_weight_update is False


def test_init_builds_and_initializes_the_receiver(engine):
    receiver = engine.receiver

    assert receiver.calls == ["initialize"]
    assert receiver.config.model_id == "model"
    assert receiver.config.s3_endpoint_url == "http://minio:9000"
    assert receiver.config.ready_timeout_seconds == 5.0
    assert receiver.receiver_id.endswith(":3")
    assert receiver.engine is engine


def test_init_info_carries_exactly_what_the_receiver_config_needs():
    assert {field.name for field in fields(MxDeltaInitInfo)} == {
        field.name for field in fields(ReceiverConfig)
    }


def test_init_info_rejects_an_unknown_field(bare_engine):
    with pytest.raises(ValueError):
        bare_engine.parse_init_info({**INIT_INFO, "surprise": 1})


def test_version_travels_in_update_info_through_vllms_entry_point(engine, reload_calls):
    engine.start_weight_update()
    engine.update_weights({"version": "7"})
    engine.finish_weight_update()

    assert engine.receiver.calls == [
        "initialize",
        ("start_weight_update", "7"),
        "update_weights",
        "mark_verified",
    ]
    assert engine.receiver.installed_version == "7"
    assert reload_calls == ["initialize", "finalize"]


def test_the_window_spans_prepare_and_install(engine, reload_calls):
    engine.start_weight_update()
    assert reload_calls == ["initialize"]

    engine.receive_weights(MxDeltaUpdateInfo(version="7"))
    assert reload_calls == ["initialize"]

    engine.finish_weight_update()
    assert reload_calls == ["initialize", "finalize"]


def test_receiver_is_verified_only_after_vllm_finishes(engine):
    engine.start_weight_update()
    engine.receive_weights(MxDeltaUpdateInfo(version="7"))

    assert engine.receiver.state is ReceiverRevisionState.BYTES_RECEIVED
    assert engine.receiver.installed_version == "0"

    engine.finish_weight_update()

    assert engine.receiver.state is ReceiverRevisionState.VERIFIED
    assert engine.receiver.installed_version == "7"
    assert engine.receiver.calls[-1] == "mark_verified"


def test_vllm_finalization_failure_poisons_receiver(engine, monkeypatch):
    engine.start_weight_update()
    engine.receive_weights(MxDeltaUpdateInfo(version="7"))
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.model_loader.reload",
        SimpleNamespace(
            finalize_layerwise_reload=lambda *_args: (_ for _ in ()).throw(
                RuntimeError("finalize failed")
            )
        ),
    )

    with pytest.raises(RuntimeError, match="finalize failed"):
        engine.finish_weight_update()

    assert engine.receiver.state is ReceiverRevisionState.POISONED
    assert engine.receiver.installed_version == "0"
    assert engine.receiver.calls[-1] == "mark_poisoned"


def test_update_before_initialization_is_refused(bare_engine):
    with pytest.raises(RuntimeError, match="init_transfer_engine"):
        bare_engine.receive_weights(MxDeltaUpdateInfo(version="7"))


@pytest.mark.parametrize(
    "state", [ReceiverRevisionState.FAILED, ReceiverRevisionState.POISONED]
)
def test_failed_install_raises_and_keeps_the_state_queryable(engine, state):
    engine.receiver.install_state = state
    engine.start_weight_update()

    with pytest.raises(RuntimeError, match=state.value):
        engine.receive_weights(MxDeltaUpdateInfo(version="7"))

    assert engine.receiver.status().state is state


def test_a_failed_prepare_does_not_leave_its_timings_behind(engine):
    engine.receiver.prepare_error = RuntimeError("download failed")
    engine.start_weight_update()

    with pytest.raises(RuntimeError, match="download failed"):
        engine.receive_weights(MxDeltaUpdateInfo(version="7"))

    assert engine.receiver.metrics == {}


def test_timings_are_emitted_once_per_update(engine, caplog):
    with caplog.at_level("INFO", logger=delta_engine.logger.name):
        engine.start_weight_update()
        engine.receive_weights(MxDeltaUpdateInfo(version="7"))
        engine.finish_weight_update()

    records = timing_records(caplog)
    assert len(records) == 1
    assert '"version":"7"' in records[0]
    assert '"rank":3' in records[0]
    assert '"cold_warm":"cold"' in records[0]


def test_a_revision_another_rank_prepared_is_reported_warm(engine, caplog):
    # The receiver's early return: the checkpoint on this node already holds the
    # revision, so no index and no bucket is fetched.
    engine.receiver.prepare_metrics = {
        "perf/mx_receive_prepare_time": 0.1,
        "perf/mx_receive_delta_index_download": 0.0,
        "perf/mx_receive_pool": 0.0,
    }
    engine.start_weight_update()

    with caplog.at_level("INFO", logger=delta_engine.logger.name):
        engine.receive_weights(MxDeltaUpdateInfo(version="7"))
        engine.finish_weight_update()

    assert '"cold_warm":"warm"' in caplog.text


def test_shutdown_releases_the_receiver(engine):
    engine.shutdown()

    assert engine.receiver is None


def test_receiver_resolves_the_launch_checkpoint_through_vllm(bare_engine, loaders):
    receiver = VllmWeightReceiver(ReceiverConfig(**INIT_INFO), "host:3", bare_engine)

    assert str(receiver.launch_checkpoint) == "/weights/launch"
    assert loaders[0].calls == [("prepare", "/weights/launch", None, False)]
    # A worker booted on --load-format modelexpress carries a format
    # DefaultModelLoader rejects.
    assert loaders[0].load_config.load_format == "safetensors"


def test_receiver_installs_from_the_prepared_path_not_the_launch_path(
    bare_engine, loaders
):
    receiver = VllmWeightReceiver(ReceiverConfig(**INIT_INFO), "host:3", bare_engine)

    receiver.install_prepared_checkpoint(PREPARED)

    assert loaders[-1].calls[-1] == ("load", bare_engine.model, "/cache/checkpoint")
    assert bare_engine.model_config.model == "/weights/launch"


def test_receiver_installs_into_the_engines_current_target(bare_engine, loaders):
    receiver = VllmWeightReceiver(ReceiverConfig(**INIT_INFO), "host:3", bare_engine)
    draft = nn.Linear(2, 2)
    bare_engine.set_weight_update_target(
        draft, SimpleNamespace(model="/weights/draft", revision=None)
    )

    receiver.install_prepared_checkpoint(PREPARED)

    assert loaders[-1].calls[-1] == ("load", draft, "/cache/checkpoint")


@pytest.mark.parametrize(
    "failing, mutation_started", [("setup", False), ("load", True)]
)
def test_install_failure_reports_whether_weights_were_touched(
    bare_engine, loaders, monkeypatch, failing, mutation_started
):
    receiver = VllmWeightReceiver(ReceiverConfig(**INIT_INFO), "host:3", bare_engine)

    class Failing(Loader):
        def load_weights(self, model, model_config):
            raise RuntimeError("load failed")

    def build(load_config):
        if failing == "setup":
            raise RuntimeError("loader setup failed")
        return Failing(load_config)

    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.model_loader.default_loader",
        SimpleNamespace(DefaultModelLoader=build),
    )

    with pytest.raises(ReceiverInstallError) as error:
        receiver.install_prepared_checkpoint(PREPARED)

    assert error.value.mutation_started is mutation_started


class Bare(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2))
        self.workspace = torch.zeros(4)


class Mla(nn.Module):
    """Minimal MLA geometry: kv_b_proj is (heads * (qk_nope + v), kv_lora_rank)."""

    def __init__(self):
        super().__init__()
        self.num_heads = 1
        self.qk_nope_head_dim = 2
        self.v_head_dim = 2
        self.kv_b_proj = nn.Linear(3, 4, bias=False)
        self.W_UV = torch.zeros(2, 1, 3)
        self.W_UK_T = torch.zeros(2, 3, 1)


def test_bare_attribute_tensors_keep_their_graph_bound_storage(engine):
    module = Bare()
    engine.model = module
    boot = module.workspace

    engine.start_weight_update()
    engine.receive_weights(MxDeltaUpdateInfo(version="7"))
    module.workspace = torch.full((4,), 5.0)
    engine.finish_weight_update()

    assert module.workspace is boot
    assert torch.equal(module.workspace, torch.full((4,), 5.0))


def test_a_bare_attribute_that_changes_shape_is_reported(engine, caplog):
    module = Bare()
    engine.model = module
    boot = module.workspace

    engine.start_weight_update()
    engine.receive_weights(MxDeltaUpdateInfo(version="7"))
    module.workspace = torch.zeros(8)
    with caplog.at_level("ERROR"):
        engine.finish_weight_update()

    assert module.workspace is boot
    assert "changed shape/dtype across refit" in caplog.text


def test_mla_absorbed_weights_are_recomputed_in_place(engine):
    module = Mla()
    engine.model = module
    with torch.no_grad():
        module.kv_b_proj.weight.copy_(torch.arange(12, dtype=torch.float32).view(4, 3))
    uv_ptr = module.W_UV.data_ptr()
    uk_ptr = module.W_UK_T.data_ptr()

    engine.start_weight_update()
    engine.receive_weights(MxDeltaUpdateInfo(version="7"))
    engine.finish_weight_update()

    assert module.W_UV.data_ptr() == uv_ptr
    assert module.W_UK_T.data_ptr() == uk_ptr
    assert module.W_UV.abs().sum() > 0
