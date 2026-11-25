# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The vLLM adapter with a stage process, driven as vLLM's worker drives it.

vLLM has no staging step: the trainer pauses generation, then the worker calls
``start_weight_update``, ``update_weights`` and ``finish_weight_update``. The
adapter stages and installs inside ``update_weights``. With a stage process the
download and decode run in that process while the update waits for it. Behind
the adapter are the real generator client, update session and node checkpoint
store; the engine installer, object storage and the MX server are stand-ins.
"""

import logging
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

import modelexpress_rl.inference.engines as engines_module
from modelexpress_rl.inference.engines.vllm import weight_transfer_engine
from modelexpress_rl.inference.engines.vllm.weight_transfer_engine import (
    ModelExpressWeightTransferEngine,
)
from modelexpress_rl.inference.plan import (
    EngineCapabilities,
    EngineInstaller,
    PreparedCheckpointArtifact,
)
from modelexpress_rl.inference.runtime import EngineRuntime
from tests import test_refit_stage_process as stage_process_tests
from tests.helpers import refit_stage_process as fakes
from tests.test_refit_stage_process import (
    MX_MODEL,
    STORE_KEY,
    _alive,
    _wait_until,
)
from tests.test_vllm_weight_transfer_engine import _engine

# The fake MX server, object store and node store of the stage-process tests;
# pytest finds a fixture among the attributes of the module that requests it.
harness = stage_process_tests.harness

_VERSIONS = ("v1", "v2", "v3", "v4")
_stage_process_modes = pytest.mark.parametrize(
    "stage_process", [True, False], ids=["stage-process", "in-process"]
)


class _Worker:
    """vLLM's worker around one weight transfer engine: start, update, finish."""

    def __init__(self, engine: ModelExpressWeightTransferEngine) -> None:
        self.engine = engine
        self._update_active = False

    def init_weight_transfer_engine(self, init_info: dict) -> None:
        self.engine.init_transfer_engine(self.engine.init_info_cls(**init_info))

    def start_weight_update(self) -> None:
        if self._update_active:
            raise RuntimeError("a weight update is already active")
        try:
            self.engine.start_weight_update()
        except BaseException:
            self.engine.reset_weight_update_target()
            raise
        self._update_active = True

    def update_weights(self, update_info: dict) -> None:
        if not self._update_active:
            raise RuntimeError("start_weight_update must be called first")
        try:
            self.engine.update_weights(update_info)
        except BaseException:
            self._end_weight_update()
            raise

    def finish_weight_update(self) -> None:
        if not self._update_active:
            raise RuntimeError("finish_weight_update without a start")
        self.engine.finish_weight_update()
        self._end_weight_update()

    def _end_weight_update(self) -> None:
        self._update_active = False
        self.engine.reset_weight_update_target()

    def shutdown(self) -> None:
        self.engine.shutdown()


class _Installer(EngineInstaller):
    """Record what the engine would load: the prepared checkpoint's content."""

    def __init__(self) -> None:
        self.installed: list[str] = []

    @property
    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            artifact_types=frozenset({PreparedCheckpointArtifact})
        )

    def install(self, prepared):
        self.installed.append(fakes.digest_checkpoint(prepared.checkpoint.path))
        # Like vLLM's installer: the staging metrics travel with the install time.
        return {**prepared.metrics, "perf/mx_receive_install_time": 0.0}


class _Rig:
    """One engine rank: the MX adapter and client behind a vLLM-shaped worker."""

    def __init__(self, harness, monkeypatch, *, stage_process: bool) -> None:
        self.harness = harness
        self.installer = _Installer()
        monkeypatch.setattr(
            engines_module,
            "_create_engine_runtime",
            lambda _context: EngineRuntime(
                model_name=STORE_KEY, installer=self.installer
            ),
        )
        vllm_config = SimpleNamespace(
            parallel_config=SimpleNamespace(),
            model_config=SimpleNamespace(model=STORE_KEY),
        )
        self.engine = ModelExpressWeightTransferEngine(
            SimpleNamespace(), vllm_config, torch.device("cpu"), torch.nn.Linear(2, 2)
        )
        self.worker = _Worker(self.engine)
        self.worker.init_weight_transfer_engine(
            {
                "model_name": MX_MODEL,
                "initial_base_version_id": "v0",
                "seed_checkpoint_path": str(harness.seed),
                "refit_checkpoint_dir": str(harness.refit),
                "refit_checkpoint_max_size_gb": 1,
                "server_url": harness.address,
                "object_storage_type": "S3",
                "registration_ttl_seconds": 600,
                "rpc_timeout_seconds": 10.0,
                "stage_process": stage_process,
            }
        )
        self.client = self.engine._client
        harness.clients.append(self.client)
        # The object-storage reader this rank's client made for itself.
        self._reader = harness.parent_readers[-1]
        # (version id, thread name) of every client.stage_weight call, and the
        # metrics each returned.
        self.stage_calls: list[tuple[str, str]] = []
        self.stage_metrics: list[dict[str, float]] = []
        stage_weight = self.client.stage_weight

        def recording_stage_weight(*, version):
            self.stage_calls.append(
                (version.version_id, threading.current_thread().name)
            )
            handle = stage_weight(version=version)
            self.stage_metrics.append(handle.metrics)
            return handle

        self.client.stage_weight = recording_stage_weight

    @property
    def parent_reads(self) -> list[tuple[str, str]]:
        """Object-storage reads this rank made itself, not its stage process."""
        return self._reader.reads

    def update(self, version_id: str) -> None:
        self.worker.start_weight_update()
        self.worker.update_weights({"version_id": version_id})
        self.worker.finish_weight_update()

    def installed(self) -> list[str]:
        """Versions installed so far, by the content the engine loaded."""
        versions = {digest: name for name, digest in self.harness.digests.items()}
        return [
            versions.get(digest, "unknown content")
            for digest in self.installer.installed
        ]


@pytest.fixture
def vllm(harness, monkeypatch):
    """Build engine ranks on the harness store and shut them down afterwards."""
    rigs: list[_Rig] = []

    def build(*, stage_process: bool) -> _Rig:
        rig = _Rig(harness, monkeypatch, stage_process=stage_process)
        rigs.append(rig)
        return rig

    try:
        yield build
    finally:
        for rig in rigs:
            rig.worker.shutdown()


@_stage_process_modes
def test_update_stages_and_installs_inside_the_update(
    vllm, harness, stage_process, caplog
):
    rig = vllm(stage_process=stage_process)
    main_thread = threading.current_thread().name

    with caplog.at_level(logging.INFO):
        for version_id in _VERSIONS:
            rig.update(version_id)

    assert rig.installed() == list(_VERSIONS)
    # Staging stays a blocking step of the update call.
    assert rig.stage_calls == [(version_id, main_thread) for version_id in _VERSIONS]
    if stage_process:
        # The stage process did the reading, inside each update.
        assert rig.parent_reads == []
        assert all(
            metrics["perf/mx_stage_process_time"] > 0 for metrics in rig.stage_metrics
        )
        # The update reports the stage process's download and apply times, not
        # the zeros of this process, which found the version already staged.
        for name in ("perf/mx_receive_delta_download", "perf/mx_receive_delta_apply"):
            prefix = f"ModelExpress receiver metric {name}="
            reported = [
                float(record.getMessage().removeprefix(prefix))
                for record in caplog.records
                if record.getMessage().startswith(prefix)
            ]
            assert reported == [metrics[name] for metrics in rig.stage_metrics]
            assert any(value > 0 for value in reported)
    else:
        assert rig.parent_reads
        assert harness.procs == []
    _wait_until(lambda: not harness.service.leases, "leases to be released")


def test_two_ranks_update_each_version(vllm, harness):
    ranks = [vllm(stage_process=True) for _ in range(2)]

    for version_id in ("v1", "v2"):
        for rig in ranks:
            rig.worker.start_weight_update()
        for rig in ranks:
            rig.worker.update_weights({"version_id": version_id})
        for rig in ranks:
            rig.worker.finish_weight_update()

    for rig in ranks:
        assert rig.installed() == ["v1", "v2"]
        assert rig.parent_reads == []
    # One stage process downloaded each delta; the other found it prepared.
    pids = {rig.client._stage_process.pid for rig in ranks}
    assert len(pids) == 2
    shards = [
        uri
        for pid, reads in fakes.reads_by_pid(harness.root).items()
        for _kind, uri in reads
        if pid in pids and uri.endswith(".safetensors")
    ]
    assert len(shards) == len(set(shards)) > 0
    _wait_until(lambda: not harness.service.leases, "leases to be released")


def test_shutdown_stops_the_stage_process(vllm, harness):
    rig = vllm(stage_process=True)
    pid = rig.client._stage_process.pid
    rig.update("v1")

    rig.worker.shutdown()
    rig.worker.shutdown()

    assert rig.installed() == ["v1"]
    _wait_until(lambda: not harness.service.leases, "leases to be released")
    _wait_until(lambda: not _alive(pid), "the stage process to exit")


@pytest.mark.parametrize("stage_process", [None, True, False])
def test_init_info_carries_the_stage_process_switch(monkeypatch, stage_process):
    engine, client = _engine(monkeypatch, initialize=False)
    initialize = MagicMock(return_value=client)
    monkeypatch.setattr(
        weight_transfer_engine.ModelExpressGeneratorClient, "initialize", initialize
    )
    init_info = {} if stage_process is None else {"stage_process": stage_process}

    engine.init_transfer_engine(engine.init_info_cls(**init_info))

    assert initialize.call_args.args[0].stage_process is stage_process
