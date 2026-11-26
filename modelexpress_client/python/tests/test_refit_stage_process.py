# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage process tests with real worker subprocesses.

Object storage is a directory read through a file-backed reader, in the client
and in every worker. The MX server is an in-process RefitService on localhost.
Checkpoints are a few hundred kilobytes.
"""

import dataclasses
import json
import logging
import os
import signal
import subprocess
import threading
import time
from collections import Counter
from concurrent import futures
from pathlib import Path
from types import SimpleNamespace

import grpc
import numpy as np
import pytest
import torch

import modelexpress_rl.inference.client as client_module
import modelexpress_rl.inference.methods.canonical_delta as canonical_delta_module
import modelexpress_rl.inference.stage_process as stage_process_module
import modelexpress_rl.inference.stage_worker as stage_worker_module
from modelexpress_rl import (
    ModelExpressGeneratorClient,
    ModelExpressGeneratorConfig,
    ObjectStorageGeneratorConfig,
    ObjectStorageType,
    WeightSource,
    WeightVersionRef,
    envs,
    refit_pb2,
    refit_pb2_grpc,
)
from modelexpress_rl.inference.checkpoint_store import (
    CheckpointState,
    LocalCheckpointStore,
)
from modelexpress_rl.inference.engines.sglang import SglangGeneratorContext
from modelexpress_rl.inference.engines.sglang.installer import _SglangInstaller
from modelexpress_rl.inference.engines.stage_only import StageOnlyGeneratorContext
from modelexpress_rl.inference.plan import (
    EngineCapabilities,
    PreparedCheckpointArtifact,
    SourceResolver,
)
from modelexpress_rl.inference.stage_process import StageProcessError
from tests.helpers import refit_stage_process as fakes

MX_MODEL = "mx-logical-model"
# The store key is the engine's model name, which differs from the MX name.
STORE_KEY = "org/synthetic-model"
# Upper bound for anything a test waits on; nothing here normally takes that long.
WAIT_SECONDS = 60.0
_STAGE_PROCESS_SETTINGS = [
    name for name in envs.environment_variables if "STAGE_PROCESS" in name
]


def _wait_until(condition, what: str, timeout: float = WAIT_SECONDS) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError(f"timed out waiting for {what}")
        time.sleep(0.01)


def _alive(pid: int) -> bool:
    """Whether ``pid`` is a running process (a zombie is not)."""
    try:
        with open(f"/proc/{pid}/stat") as stat:
            return stat.read().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return False


class _RefitService(refit_pb2_grpc.RefitServiceServicer):
    """The RefitService calls a generator makes: versions, registration, leases."""

    def __init__(self) -> None:
        self.versions: dict[str, refit_pb2.WeightVersion] = {}
        self.workers: dict[str, refit_pb2.WorkerRegistration] = {}
        # lease id -> (version id, worker id)
        self.leases: dict[str, tuple[str, str]] = {}
        self.leased: list[tuple[str, str]] = []
        self.calls: Counter = Counter()
        self._lock = threading.Lock()

    def GetWeightVersion(self, request, context):
        with self._lock:
            self.calls["GetWeightVersion"] += 1
            version = self.versions.get(request.uid)
        if version is None:
            context.abort(grpc.StatusCode.NOT_FOUND, "version not found")
        return refit_pb2.GetWeightVersionResponse(version=version)

    def RegisterWorker(self, request, context):
        with self._lock:
            self.calls["RegisterWorker"] += 1
            self.workers[request.worker.worker_id] = request.worker
        return refit_pb2.RegisterWorkerResponse(worker=request.worker)

    def RegisterVersionLease(self, request, context):
        with self._lock:
            self.calls["RegisterVersionLease"] += 1
            worker = self.workers.get(request.worker_id)
            accepted = (
                worker is not None and worker.role == refit_pb2.WORKER_ROLE_GENERATOR
            )
            lease_id = f"lease-{request.worker_id}-{request.version_id}"
            if accepted:
                self.leases[lease_id] = (request.version_id, request.worker_id)
                self.leased.append((request.version_id, request.worker_id))
        if not accepted:
            context.abort(
                grpc.StatusCode.FAILED_PRECONDITION, "generator not registered"
            )
        return refit_pb2.RegisterVersionLeaseResponse(
            lease=refit_pb2.VersionLease(
                lease_id=lease_id,
                version_id=request.version_id,
                worker_id=request.worker_id,
                expires_at_unix_ms=int(time.time() * 1000) + 60_000,
            )
        )

    def DeleteVersionLease(self, request, context):
        with self._lock:
            self.calls["DeleteVersionLease"] += 1
            deleted = self.leases.pop(request.lease_id, None) is not None
        return refit_pb2.DeleteVersionLeaseResponse(deleted=deleted)


class _Harness:
    """A fake MX server, a fake object store with versions v0..v4, and cleanup.

    v0 is the launch seed, v1 and v2 are deltas on it, v3 is a full checkpoint
    and v4 a delta on v3.
    """

    def __init__(self, tmp_path: Path) -> None:
        self.root = tmp_path / "storage"
        self.seed = tmp_path / "seed"
        self.refit = tmp_path / "refit"
        self.root.mkdir()
        self.service = _RefitService()
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
        port = self.server.add_insecure_port("127.0.0.1:0")
        refit_pb2_grpc.add_RefitServiceServicer_to_server(self.service, self.server)
        self.server.start()
        self.address = f"127.0.0.1:{port}"
        self.clients: list[ModelExpressGeneratorClient] = []
        self.configs: list[ModelExpressGeneratorConfig] = []
        self.parent_readers: list[fakes.FileReader] = []
        # Every worker process the tests' clients started.
        self.procs: list[subprocess.Popen] = []

        rng = np.random.default_rng(0)
        weight_map = fakes.layout()
        tensors = {
            name: rng.standard_normal(8192).astype(np.float32) for name in weight_map
        }
        self.tensors = {"v0": tensors}
        fakes.write_checkpoint(self.seed, tensors, weight_map, checksums=False)
        (self.seed / "config.json").write_text(json.dumps({"model_type": "test"}))
        self._register(
            "v0", fakes.publish_full(self.root, "bucket/v0", tensors, weight_map)
        )
        lineage = (("v1", "v0"), ("v2", "v1"), ("v3", None), ("v4", "v3"))
        for version_id, base in lineage:
            previous = self.tensors[base or "v2"]
            tensors = fakes.mutate(previous, rng)
            self.tensors[version_id] = tensors
            prefix = f"bucket/{version_id}"
            if base is None:
                uri = fakes.publish_full(self.root, prefix, tensors, weight_map)
            else:
                uri = fakes.publish_delta(
                    self.root, prefix, previous, tensors, weight_map
                )
            self._register(version_id, uri, base)
        self.digests = {
            version_id: fakes.digest_tensors(tensors)
            for version_id, tensors in self.tensors.items()
        }

    def _register(self, version_id: str, uri: str, base: str | None = None) -> None:
        version = refit_pb2.WeightVersion(
            uid=version_id,
            model_name=MX_MODEL,
            payload_format=(
                refit_pb2.WEIGHT_PAYLOAD_FORMAT_FULL_HF_CHECKPOINT
                if base is None
                else refit_pb2.WEIGHT_PAYLOAD_FORMAT_XOR_DELTA
            ),
            layout_signature="layout-a",
            state=refit_pb2.WEIGHT_VERSION_STATE_READY,
            object_storage=refit_pb2.ObjectStorageSource(
                storage_type=refit_pb2.OBJECT_STORAGE_TYPE_S3,
                uri=uri,
            ),
        )
        if base is not None:
            version.base_version_id = base
        self.service.versions[version_id] = version

    def object_storage(self, *, seed: bool = True) -> ObjectStorageGeneratorConfig:
        return ObjectStorageGeneratorConfig(
            storage_type=ObjectStorageType.S3,
            initial_base_version_id="v0",
            seed_checkpoint_path=str(self.seed) if seed else None,
            refit_checkpoint_dir=str(self.refit),
            refit_checkpoint_max_size_gb=1,
        )

    def client(self, *, worker_id: str = "generator-0", seed: bool = True, **config):
        """Initialize a real generator client behind a stand-in SGLang runner."""
        runner = SimpleNamespace(
            model=object(),
            device="cpu",
            model_config=SimpleNamespace(
                model_path=STORE_KEY, revision=None, dtype=torch.float32
            ),
            server_args=SimpleNamespace(
                download_dir=None, model_loader_extra_config=None
            ),
            installed=[],
        )
        config.setdefault("stage_process", True)
        self.configs.append(
            ModelExpressGeneratorConfig(
                engine_context=SglangGeneratorContext(runner),
                model_name=MX_MODEL,
                worker_id=worker_id,
                server_url=self.address,
                registration_ttl_seconds=600,
                rpc_timeout_seconds=10.0,
                object_storage=self.object_storage(seed=seed),
                **config,
            )
        )
        client = ModelExpressGeneratorClient.initialize(self.configs[-1])
        self.clients.append(client)
        return client, runner

    def head(self):
        """The node checkpoint store's state: (status, version)."""
        state = LocalCheckpointStore(root=self.refit, model_name=STORE_KEY).state()
        return None if state is None else (state.status, state.version)

    def shard_reads(self, pid: int) -> list[str]:
        """Weight files that process ``pid`` read from object storage."""
        reads = fakes.reads_by_pid(self.root).get(pid, [])
        return [uri for _kind, uri in reads if uri.endswith(".safetensors")]

    def hold(self) -> None:
        """Make the next weight-file read block, in the middle of a staging step."""
        (self.root / "holding").unlink(missing_ok=True)
        (self.root / "hold").touch()

    def unhold(self) -> None:
        (self.root / "hold").unlink(missing_ok=True)

    def wait_holding(self) -> None:
        _wait_until((self.root / "holding").exists, "a weight-file read to start")

    def close(self) -> None:
        self.unhold()
        for client in self.clients:
            client.close()
        for proc in self.procs:
            try:
                proc.wait(timeout=WAIT_SECONDS)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=WAIT_SECONDS)
        self.server.stop(0)


@pytest.fixture
def harness(tmp_path, monkeypatch):
    for name in (
        *_STAGE_PROCESS_SETTINGS,
        "MX_MODEL_NAME_OVERRIDE",
        "MX_GENERATOR_SOURCE_ORDER",
        "MX_REFIT_FULL_STREAMING",
        "MX_REFIT_DELTA_WORKERS",
        "MX_REFIT_DOWNLOAD_WORKERS",
        "MX_S3_DOWNLOAD_WORKERS",
        fakes.STALL_ENV,
    ):
        monkeypatch.delenv(name, raising=False)
    harness = _Harness(tmp_path)
    # Workers inherit the environment, and with it the fake object store.
    monkeypatch.setenv(fakes.READER_ROOT_ENV, str(harness.root))
    monkeypatch.setattr(
        stage_process_module, "_WORKER_MODULE", "tests.helpers.stage_worker_entry"
    )

    def parent_reader(**_kwargs):
        reader = fakes.FileReader(harness.root)
        harness.parent_readers.append(reader)
        return reader

    monkeypatch.setattr(canonical_delta_module, "S3Client", parent_reader)

    def install(self, prepared):
        self._model_runner.installed.append(fakes.digest_checkpoint(prepared.path))

    monkeypatch.setattr(_SglangInstaller, "_install_checkpoint", install)

    class RecordingPopen(subprocess.Popen):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            harness.procs.append(self)

    monkeypatch.setattr(subprocess, "Popen", RecordingPopen)
    try:
        yield harness
    finally:
        harness.close()


def _stage_apply(harness, client, runner, version_id: str) -> dict[str, float]:
    handle = client.stage_weight(version=WeightVersionRef(version_id))
    metrics = handle.metrics
    client.apply_weight(handle)
    assert handle.applied
    assert runner.installed[-1] == harness.digests[version_id]
    handle.release()
    return metrics


def test_staging_step_runs_in_the_worker(harness):
    client, runner = harness.client()
    worker_pid = client._stage_process.pid
    assert worker_pid != os.getpid() and _alive(worker_pid)

    handle = client.stage_weight(version=WeightVersionRef("v1"))

    # The worker read the delta. This process only adopted the result.
    assert harness.shard_reads(worker_pid)
    assert harness.parent_readers[0].reads == []
    assert harness.head() == (CheckpointState.READY, "v1")
    # The handle reports what the worker measured.
    metrics = handle.metrics
    assert metrics["perf/mx_stage_process_time"] > 0
    assert metrics["perf/mx_receive_delta_apply"] > 0
    assert "perf/mx_stage_process_restarts" not in metrics
    # Only the worker leased the version, and it holds the lease for this
    # client's handle until that is installed.
    assert set(harness.service.leased) == {("v1", "generator-0-stage")}
    assert set(harness.service.leases.values()) == {("v1", "generator-0-stage")}

    client.apply_weight(handle)
    assert runner.installed == [harness.digests["v1"]]
    _wait_until(lambda: not harness.service.leases, "the worker to release its lease")
    handle.release()


class _NoMxServer:
    """Stands in for a client's MX server stub and fails every call."""

    def __getattr__(self, name):
        raise AssertionError(f"the engine's client called the MX server: {name}")


def test_engine_client_asks_the_mx_server_nothing(harness):
    client, runner = harness.client()
    calls = harness.service.calls.copy()
    stub, client._stub = client._stub, _NoMxServer()
    try:
        for version_id in ("v1", "v2", "v3", "v4"):
            _stage_apply(harness, client, runner, version_id)
        # Staging the version the engine already serves only checks it with
        # the MX server, which the worker does too.
        handle = client.stage_weight(version=WeightVersionRef("v4"))
        assert handle.applied
        handle.release()
    finally:
        client._stub = stub

    made = harness.service.calls - calls
    # One staging step's worth of calls, all from the worker.
    assert made["RegisterVersionLease"] == 4
    assert made["GetWeightVersion"] > 0
    assert {worker for _version, worker in harness.service.leased} == {
        "generator-0-stage"
    }
    _wait_until(lambda: not harness.service.leases, "leases to be released")


def test_every_kind_of_update_is_staged_in_the_worker(harness):
    client, runner = harness.client()

    # Two deltas, a full checkpoint, and a delta on that.
    for version_id in ("v1", "v2", "v3", "v4"):
        metrics = _stage_apply(harness, client, runner, version_id)
        assert metrics["perf/mx_stage_process_time"] > 0
        assert "perf/mx_stage_process_restarts" not in metrics

    assert harness.parent_readers[0].reads == []
    assert len(harness.procs) == 1
    _wait_until(lambda: not harness.service.leases, "leases to be released")


def test_skipped_version_is_replayed_in_the_worker(harness):
    client, runner = harness.client()

    _stage_apply(harness, client, runner, "v2")

    worker_reads = harness.shard_reads(client._stage_process.pid)
    assert any("bucket/v1" in uri for uri in worker_reads)
    assert any("bucket/v2" in uri for uri in worker_reads)
    assert harness.parent_readers[0].reads == []


def test_recovery_after_a_failed_install_is_staged_in_the_worker(harness, monkeypatch):
    client, runner = harness.client()
    install = _SglangInstaller._install_checkpoint

    def fail(self, prepared):
        raise RuntimeError("injected install failure")

    monkeypatch.setattr(_SglangInstaller, "_install_checkpoint", fail)
    handle = client.stage_weight(version=WeightVersionRef("v1"))
    with pytest.raises(RuntimeError, match="injected install failure"):
        client.apply_weight(handle)
    monkeypatch.setattr(_SglangInstaller, "_install_checkpoint", install)

    # The engine may now hold part of v1. Going back to v0, the version it
    # served last, is a real update: the worker stages it and holds its lease.
    restore = client.stage_weight(version=WeightVersionRef("v0"))

    assert list(harness.service.leases.values()) == [("v0", "generator-0-stage")]
    assert harness.shard_reads(os.getpid()) == []
    client.apply_weight(restore)
    assert runner.installed == [harness.digests["v0"]]
    restore.release()
    _wait_until(lambda: not harness.service.leases, "leases to be released")


# Streaming applies to a full checkpoint published without checksums, which
# only its index tells: the MX server's record is the same for both.
@pytest.mark.parametrize("checksums", [False, True])
def test_version_read_at_install_is_staged_by_the_client(
    harness, monkeypatch, checksums
):
    monkeypatch.setenv("MX_REFIT_FULL_STREAMING", "true")
    capabilities = EngineCapabilities(
        artifact_types=frozenset({PreparedCheckpointArtifact}),
        streamed_checkpoints=True,
    )
    monkeypatch.setattr(
        _SglangInstaller, "capabilities", property(lambda self: capabilities)
    )
    uri = f"s3://bucket/v5/{fakes.INDEX_NAME}"
    fakes.write_checkpoint(
        fakes.object_path(harness.root, uri).parent,
        harness.tensors["v4"],
        fakes.layout(),
        checksums=checksums,
    )
    harness._register("v5", uri)
    client, _ = harness.client()
    worker_pid = client._stage_process.pid

    handle = client.stage_weight(version=WeightVersionRef("v5"))

    streamed = not checksums
    assert handle.is_streamed is streamed
    # A streamed version is read during its install, in this process. Nothing
    # is staged ahead, so the worker reads no weights and this client holds
    # the lease. Any other full checkpoint is staged by the worker.
    assert bool(harness.shard_reads(worker_pid)) is not streamed
    assert harness.shard_reads(os.getpid()) == []
    holder = "generator-0" if streamed else "generator-0-stage"
    assert list(harness.service.leases.values()) == [("v5", holder)]
    handle.release()
    _wait_until(lambda: not harness.service.leases, "leases to be released")


def test_store_without_a_seed_is_built_by_the_worker(harness):
    # Without a seed, the first full checkpoint builds the checkpoint cache.
    client, runner = harness.client(seed=False)

    _stage_apply(harness, client, runner, "v3")
    # The cache now has a verified head and no initial checkpoint. A worker
    # that starts later joins it as it is.
    os.kill(client._stage_process.pid, signal.SIGKILL)
    _wait_until(lambda: len(harness.procs) == 2, "a new worker to start")
    metrics = _stage_apply(harness, client, runner, "v4")

    assert "perf/mx_stage_process_restarts" not in metrics
    assert client._stage_process.pid == harness.procs[1].pid
    assert all(harness.shard_reads(proc.pid) for proc in harness.procs)
    assert harness.shard_reads(os.getpid()) == []
    assert len(harness.procs) == 2


def test_two_ranks_share_one_store(harness):
    ranks = [harness.client(worker_id=f"generator-{rank}") for rank in range(2)]
    pids = {client._stage_process.pid for client, _ in ranks}
    assert len(pids) == 2

    with futures.ThreadPoolExecutor(2) as pool:
        handles = list(
            pool.map(
                lambda rank: rank[0].stage_weight(version=WeightVersionRef("v1")),
                ranks,
            )
        )

    for (client, runner), handle in zip(ranks, handles):
        client.apply_weight(handle)
        assert runner.installed == [harness.digests["v1"]]
        handle.release()
    # One worker read the delta and the other found it staged.
    shards = [uri for pid in pids for uri in harness.shard_reads(pid)]
    assert shards and len(shards) == len(set(shards))
    assert all(reader.reads == [] for reader in harness.parent_readers)


def test_failure_in_the_worker_is_the_stage_failure(harness):
    client, runner = harness.client(max_transfer_attempts=1)
    fakes.fail_reads(harness.root, "bucket/v1")

    with pytest.raises(StageProcessError, match="'v1'.*injected failure"):
        client.stage_weight(version=WeightVersionRef("v1"))

    # This process did not stage the version itself, and the worker carries on.
    assert harness.parent_readers[0].reads == []
    metrics = _stage_apply(harness, client, runner, "v1")
    assert "perf/mx_stage_process_restarts" not in metrics
    assert len(harness.procs) == 1


def test_killed_worker_is_replaced_and_the_request_sent_again(harness):
    client, runner = harness.client()
    worker_pid = client._stage_process.pid
    harness.hold()

    with futures.ThreadPoolExecutor(1) as pool:
        staging = pool.submit(client.stage_weight, version=WeightVersionRef("v1"))
        harness.wait_holding()
        os.kill(worker_pid, signal.SIGKILL)
        harness.unhold()
        handle = staging.result(WAIT_SECONDS)

    # A new worker staged it. This process still read no weights.
    new_pid = client._stage_process.pid
    assert new_pid != worker_pid and len(harness.procs) == 2
    assert handle.metrics["perf/mx_stage_process_restarts"] == 1.0
    assert harness.shard_reads(new_pid)
    assert harness.parent_readers[0].reads == []
    client.apply_weight(handle)
    assert runner.installed == [harness.digests["v1"]]
    handle.release()
    # The replacement takes the next request without another start.
    metrics = _stage_apply(harness, client, runner, "v2")
    assert "perf/mx_stage_process_restarts" not in metrics
    assert len(harness.procs) == 2


def test_worker_that_exits_between_requests_is_replaced_at_once(harness):
    client, runner = harness.client()
    _stage_apply(harness, client, runner, "v1")

    os.kill(client._stage_process.pid, signal.SIGKILL)

    # With no request: the client sees the exit and starts another worker.
    _wait_until(lambda: len(harness.procs) == 2, "a new worker to start")
    metrics = _stage_apply(harness, client, runner, "v2")
    # The request found it ready.
    assert "perf/mx_stage_process_restarts" not in metrics
    assert client._stage_process.pid == harness.procs[1].pid
    assert harness.shard_reads(harness.procs[1].pid)
    assert harness.parent_readers[0].reads == []
    assert len(harness.procs) == 2


def test_workers_that_keep_exiting_are_not_replaced_forever(harness):
    client, runner = harness.client()
    stage_process = client._stage_process
    attempts = stage_process_module._ATTEMPTS

    def idle(count):
        return (
            len(harness.procs) == count
            and not stage_process._lock.locked()
            and stage_process.pid == harness.procs[-1].pid
        )

    # Kill every worker once it is ready, with no request in between.
    for count in range(1, attempts + 2):
        _wait_until(lambda: idle(count), f"worker {count} to be ready")
        harness.procs[-1].kill()
    _wait_until(
        lambda: stage_process.pid is None and not stage_process._lock.locked(),
        "the last worker to be dropped",
    )

    # The one from init and its replacements; the last exit started none.
    assert len(harness.procs) == attempts + 1
    # A request still starts one.
    metrics = _stage_apply(harness, client, runner, "v1")
    assert metrics["perf/mx_stage_process_restarts"] == 1.0
    assert harness.parent_readers[0].reads == []


def test_request_fails_once_every_attempt_lost_its_worker(harness):
    client, runner = harness.client()
    attempts = stage_process_module._ATTEMPTS

    with futures.ThreadPoolExecutor(1) as pool:
        harness.hold()
        staging = pool.submit(client.stage_weight, version=WeightVersionRef("v1"))
        for _ in range(attempts):
            harness.wait_holding()
            # Clear the marker for the next worker, then kill this one.
            harness.hold()
            harness.procs[-1].kill()
        with pytest.raises(StageProcessError, match=f"'v1' failed in {attempts}"):
            staging.result(WAIT_SECONDS)

    # This process did not stage the version instead.
    assert harness.parent_readers[0].reads == []
    # Another worker is started at once, and it takes the next request.
    _wait_until(lambda: len(harness.procs) == attempts + 1, "a new worker to start")
    harness.unhold()
    metrics = _stage_apply(harness, client, runner, "v1")
    assert "perf/mx_stage_process_restarts" not in metrics
    assert len(harness.procs) == attempts + 1


def test_stage_that_runs_out_of_time_fails_without_another_try(harness, monkeypatch):
    monkeypatch.setenv("MX_REFIT_STAGE_PROCESS_TIMEOUT_SECONDS", "1.0")
    client, runner = harness.client()
    harness.hold()

    with pytest.raises(StageProcessError, match="within 1 s and was killed"):
        client.stage_weight(version=WeightVersionRef("v1"))

    assert harness.procs[0].poll() == -signal.SIGKILL
    assert harness.parent_readers[0].reads == []
    # The killed worker is replaced at once, and the request is not sent again.
    _wait_until(lambda: len(harness.procs) == 2, "a new worker to start")
    assert harness.shard_reads(harness.procs[1].pid) == []
    harness.unhold()
    metrics = _stage_apply(harness, client, runner, "v1")
    assert "perf/mx_stage_process_restarts" not in metrics
    assert len(harness.procs) == 2


def test_close_during_a_stage_fails_it_and_lets_the_worker_finish(harness, monkeypatch):
    monkeypatch.setenv("MX_REFIT_STAGE_PROCESS_CLOSE_GRACE_SECONDS", "0.2")
    client, _ = harness.client()
    worker_pid = client._stage_process.pid
    harness.hold()

    with futures.ThreadPoolExecutor(1) as pool:
        staging = pool.submit(client.stage_weight, version=WeightVersionRef("v1"))
        harness.wait_holding()
        client.close()
        with pytest.raises(StageProcessError, match="closed"):
            staging.result(WAIT_SECONDS)

    # The worker is not killed in the middle of its staging step.
    assert _alive(worker_pid)
    harness.unhold()
    _wait_until(lambda: not _alive(worker_pid), "the worker to exit")
    assert harness.head() == (CheckpointState.READY, "v1")


# A grace period shorter than the time the request has left, and a longer one.
@pytest.mark.parametrize("grace", ["0.1", "30"])
def test_close_leaves_a_stage_its_time_limit_and_no_longer(harness, monkeypatch, grace):
    monkeypatch.setenv("MX_REFIT_STAGE_PROCESS_CLOSE_GRACE_SECONDS", grace)
    monkeypatch.setenv("MX_REFIT_STAGE_PROCESS_TIMEOUT_SECONDS", "3.0")
    client, _ = harness.client()
    worker = harness.procs[0]
    harness.hold()

    with futures.ThreadPoolExecutor(1) as pool:
        staging = pool.submit(client.stage_weight, version=WeightVersionRef("v1"))
        harness.wait_holding()
        closing = time.monotonic()
        client.close()
        with pytest.raises(StageProcessError, match="closed"):
            staging.result(WAIT_SECONDS)

    # The read stays blocked. Nobody waits for the reply any more, and the
    # worker is still killed when the request's time is up, and reaped.
    _wait_until(lambda: worker.poll() == -signal.SIGKILL, "the worker to be killed")
    # The grace period did not add to the request's time limit.
    assert time.monotonic() - closing < 15
    store = LocalCheckpointStore(root=harness.refit, model_name=STORE_KEY)
    with store.preparation_locked(blocking=False) as free:
        # Other ranks on the node can prepare versions again.
        assert free
    assert len(harness.procs) == 1


def test_close_stops_the_worker(harness):
    client, runner = harness.client()
    worker_pid = client._stage_process.pid
    _stage_apply(harness, client, runner, "v1")

    stage_process = client._stage_process
    client.close()
    client.close()

    _wait_until(lambda: not _alive(worker_pid), "the worker to exit")
    _wait_until(lambda: not harness.service.leases, "leases to be released")
    # A closed stage process starts no new worker.
    with pytest.raises(StageProcessError, match="closed"):
        stage_process.stage("v2", serving_version_id="v1", engine_state="READY")
    assert len(harness.procs) == 1


def test_worker_exits_when_its_client_is_gone(harness):
    client, _ = harness.client()
    stage_process = client._stage_process
    worker_pid = stage_process.pid

    # What the worker sees when the client process dies: end of file.
    stage_process._conn.close()

    _wait_until(lambda: not _alive(worker_pid), "the worker to exit")
    assert harness.procs[0].wait(WAIT_SECONDS) == 0


def test_worker_starts_no_request_once_termination_is_asked():
    draining = threading.Event()
    staged = []

    class Connection:
        def poll(self, _timeout=None):
            # SIGTERM arrives while the idle worker waits, and then a request.
            draining.set()
            return True

        def recv(self):
            return {
                "op": "stage",
                "version_id": "v1",
                "serving_version_id": "v0",
                "engine_state": "READY",
            }

    stage_worker_module._serve(
        SimpleNamespace(_stage_weight=staged.append),
        Connection(),
        parent_pid=os.getppid(),
        draining=draining,
    )

    assert staged == []


def test_worker_that_cannot_start_fails_client_init(harness, monkeypatch):
    monkeypatch.setattr(
        stage_process_module, "_WORKER_MODULE", "tests.helpers.no_such_module"
    )
    attempts = stage_process_module._ATTEMPTS

    with pytest.raises(
        StageProcessError,
        match=f"failed in {attempts} attempts: it exited with code 1 before",
    ):
        harness.client()

    assert [proc.poll() for proc in harness.procs] == [1] * attempts


def test_worker_that_never_becomes_ready_is_killed(harness, monkeypatch):
    monkeypatch.setenv(fakes.STALL_ENV, "1")
    monkeypatch.setenv("MX_REFIT_STAGE_PROCESS_START_TIMEOUT_SECONDS", "0.5")

    with pytest.raises(StageProcessError, match="it was not ready within 0.5 s"):
        harness.client()

    killed = [-signal.SIGKILL] * stage_process_module._ATTEMPTS
    assert [proc.poll() for proc in harness.procs] == killed


def test_start_is_tried_again_until_a_worker_is_ready(harness, monkeypatch):
    recording = subprocess.Popen

    class FirstStartFails(recording):
        def __init__(self, args, **kwargs) -> None:
            if not harness.procs:
                args = [args[0], "-m", "tests.helpers.no_such_module", *args[3:]]
            super().__init__(args, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", FirstStartFails)

    client, runner = harness.client()

    assert [proc.poll() for proc in harness.procs] == [1, None]
    metrics = _stage_apply(harness, client, runner, "v1")
    assert "perf/mx_stage_process_restarts" not in metrics
    assert harness.parent_readers[0].reads == []


def test_stage_process_is_off_unless_asked_for(harness, monkeypatch):
    client, _ = harness.client(stage_process=None)
    assert client._stage_process is None and harness.procs == []

    monkeypatch.setenv("MX_REFIT_STAGE_PROCESS", "true")
    client, _ = harness.client(worker_id="generator-1", stage_process=None)
    assert client._stage_process is not None
    client, _ = harness.client(worker_id="generator-2", stage_process=False)
    assert client._stage_process is None
    assert len(harness.procs) == 1


class _NoPeerResolver(SourceResolver):
    """A generator-peer source that never has the version."""

    kind = WeightSource.GENERATOR

    def supports(self, version) -> bool:
        return False

    def payload_format(self, version):
        return version.payload_format

    def candidates(self, version):
        return iter(())


def test_mixed_source_order_gets_no_worker(harness, monkeypatch, caplog):
    initialize_runtime = client_module.initialize_generator_runtime

    def runtime_with_peer_first(**kwargs):
        runtime = initialize_runtime(**kwargs)
        session = runtime.session
        session._resolvers = (_NoPeerResolver(), *session._resolvers)
        return runtime

    monkeypatch.setattr(
        client_module, "initialize_generator_runtime", runtime_with_peer_first
    )

    with caplog.at_level(logging.INFO, logger="modelexpress_rl.inference.client"):
        client, runner = harness.client()

    assert client._stage_process is None
    assert "object storage is not the only source (GENERATOR, OBJECT_STORAGE)" in (
        caplog.text
    )
    _stage_apply(harness, client, runner, "v1")
    assert harness.procs == []


def test_worker_is_configured_as_the_client_without_the_engine(harness, monkeypatch):
    worker_configs = []

    def no_worker(config):
        worker_configs.append(config)
        return SimpleNamespace(close=lambda: None)

    monkeypatch.setattr(client_module, "StageProcess", no_worker)

    harness.client()

    [worker_config] = worker_configs
    [client_config] = harness.configs
    changed = {
        field.name: getattr(worker_config, field.name)
        for field in dataclasses.fields(worker_config)
        if getattr(worker_config, field.name) != getattr(client_config, field.name)
    }
    assert set(changed) == {
        "engine_context",
        "worker_id",
        "source_order",
        "object_storage",
        "stage_process",
    }
    assert isinstance(changed["engine_context"], StageOnlyGeneratorContext)
    assert changed["engine_context"].model_name == STORE_KEY
    assert changed["worker_id"] == "generator-0-stage"
    assert changed["source_order"] == (WeightSource.OBJECT_STORAGE,)
    assert changed["stage_process"] is False
    # The worker joins the client's checkpoint cache; it has no launch seed.
    assert changed["object_storage"] == dataclasses.replace(
        client_config.object_storage, seed_checkpoint_path=None
    )


def test_stage_process_knows_nothing_about_how_versions_are_staged():
    for module in (stage_process_module, stage_worker_module):
        source = Path(module.__file__).read_text()
        for name in (
            "methods",
            "receiver",
            "checkpoint",
            "version_chain",
            "session",
            "runtime",
            "object_storage",
        ):
            assert name not in source, f"{module.__name__} mentions {name}"
    client_source = Path(client_module.__file__).read_text()
    assert "from .methods" not in client_source
    assert "CanonicalDelta" not in client_source
