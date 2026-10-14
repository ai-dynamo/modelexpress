# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fresh interpreters keep unit-test engine/NIXL mocks out of integration tests."""

import json
import os
from pathlib import Path
import selectors
import subprocess
import sys
import tempfile
import time

import pytest

WORKER = Path(__file__).with_name("nixl_cpu_worker.py")
PACKAGE = Path(__file__).resolve().parents[2]


class Worker:
    def __init__(self, role, config, env, directory):
        self.log = tempfile.TemporaryFile(mode="w+")
        self.pending = b""
        self.process = subprocess.Popen(
            [sys.executable, str(WORKER), role, json.dumps(config)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
            text=True, bufsize=1, env=env, cwd=directory,
        )

    def send(self, request):
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()

    def receive(self):
        deadline = time.monotonic() + 45
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdout, selectors.EVENT_READ)
            while time.monotonic() < deadline:
                if b"\n" not in self.pending:
                    if not selector.select(max(0, deadline - time.monotonic())):
                        break
                    chunk = os.read(self.process.stdout.fileno(), 65536)
                    if not chunk:
                        break
                    self.pending += chunk
                    continue
                line, self.pending = self.pending.split(b"\n", 1)
                if line.startswith(b"MX_TEST_RESULT "):
                    result = json.loads(line.removeprefix(b"MX_TEST_RESULT "))
                    if "error" in result:
                        self.log.seek(0)
                        raise AssertionError(result["error"] + self.log.read())
                    return result
        self.log.seek(0)
        raise AssertionError(f"Worker failed or timed out: {self.log.read()}")

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5)
        self.process.stdin.close()
        self.process.stdout.close()
        self.log.close()


def run_case(**options):
    required = os.getenv("MX_REQUIRE_CPU_NIXL") == "1"
    if not required and os.getenv("MX_RUN_CPU_NIXL") != "1":
        pytest.skip("set MX_RUN_CPU_NIXL=1 to run real CPU/NIXL integration")
    env = dict(os.environ, PYTHONPATH=str(PACKAGE), UCX_TLS=os.getenv("UCX_TLS", "tcp,self"),
               UCX_NET_DEVICES="lo", CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1")
    probe = subprocess.run(
        [sys.executable, "-c", "import nixl._api; import torch"],
        env=env, capture_output=True, text=True, timeout=30,
    )
    if probe.returncode:
        if required:
            pytest.fail(probe.stderr)
        pytest.skip(f"real NIXL unavailable: {probe.stderr}")
    config = dict(publishers=1, receivers=1, dtype="float32")
    config.update(options)
    workers = []
    with tempfile.TemporaryDirectory(prefix="mx-nixl-test-") as directory:
        try:
            sources = []
            for rank in range(config["publishers"]):
                worker = Worker("source", dict(config, rank=rank), env, directory)
                workers.append(worker)
                sources.append(worker)
            receivers = []
            for rank in range(config["receivers"]):
                worker = Worker("receiver", dict(config, rank=rank), env, directory)
                workers.append(worker)
                receivers.append(worker)
            for version in (0, 1, 2, 2):
                if config.get("publisher_death") and version == 1:
                    sources[0].process.kill()
                    sources[0].process.wait(timeout=5)
                    for receiver in receivers:
                        receiver.send({"version": version, "sources": metadata, "publisher_died": True})
                    for receiver in receivers:
                        assert receiver.receive()["validated"]
                    continue
                if (config.get("lifecycle") == "restart" and version == 1) or (config.get("publisher_death") and version == 2 and sources[0].process.poll() is not None):
                    for worker in sources:
                        worker.close()
                        workers.remove(worker)
                    sources = []
                    for rank in range(config["publishers"]):
                        worker = Worker("source", dict(config, rank=rank), env, directory)
                        workers.append(worker)
                        sources.append(worker)
                for source in sources:
                    source.send({"version": version})
                metadata = [source.receive() for source in sources]
                for receiver in receivers:
                    receiver.send({"version": version, "sources": metadata})
                for receiver in receivers:
                    result = receiver.receive()
                    assert result["validated"], result
                    assert Path(result["production"]).is_relative_to(PACKAGE), result
            for worker in workers:
                worker.send({"stop": True})
            for worker in workers:
                worker.process.wait(timeout=15)
                assert worker.process.returncode == 0
        finally:
            for worker in workers:
                worker.close()


def run_replica_case(*, independent_slots=False, failure=False):
    """Keep A/B allocations alive, publishing only selected current replicas."""
    if os.getenv("MX_REQUIRE_CPU_NIXL") != "1" and os.getenv("MX_RUN_CPU_NIXL") != "1":
        pytest.skip("set MX_RUN_CPU_NIXL=1 to run real CPU/NIXL integration")
    env = dict(os.environ, PYTHONPATH=str(PACKAGE), UCX_TLS=os.getenv("UCX_TLS", "tcp,self"),
               UCX_NET_DEVICES="lo", CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1",
               MX_RESHARD_PUBLISH_DIGEST="0")
    config = dict(publishers=2, receivers=1, dtype="float32", replicas=True)
    workers = []
    timings = []
    with tempfile.TemporaryDirectory(prefix="mx-replica-test-") as directory:
        try:
            sources = {}
            for rank in range(2):
                for replica in ("A", "B"):
                    worker = Worker("source", dict(config, rank=rank, replica=replica), env, directory)
                    workers.append(worker)
                    sources[rank, replica] = worker
                    worker.send({"version": 0})
            metadata = {key: worker.receive() for key, worker in sources.items()}
            receiver = Worker("receiver", dict(config, rank=0), env, directory)
            workers.append(receiver)
            sequence = ("A", "A", "B", "B", "A", "B")
            second = ("B", "A", "A", "B", "B", "A") if independent_slots else sequence
            for version, replicas in enumerate(zip(sequence, second)):
                selected = []
                for rank, replica in enumerate(replicas):
                    # Unselected allocations retain older values: stale bindings fail exact checks.
                    worker = sources[rank, replica]
                    worker.send({"version": version + 1})
                    metadata[rank, replica] = worker.receive()
                    selected.append(metadata[rank, replica]["session"])
                receiver.send({"version": version + 1, "sources": list(metadata.values()),
                               "all_sources": list(metadata.values()), "selected_workers": selected,
                               "retry": failure and version == 2})
                result = receiver.receive()
                assert result["validated"]
                assert Path(result["production"]).is_relative_to(PACKAGE)
                timings.append(dict(cycle=version, replicas=replicas, **result["timing"]))
            output = os.getenv("MX_REPLICA_NIXL_TIMINGS")
            if output:
                with open(output, "a") as stream:
                    stream.write(json.dumps(dict(backend="real CPU NIXL", independent_slots=independent_slots,
                                                 failure=failure, cycles=timings)) + "\n")
            for worker in workers:
                worker.send({"stop": True})
            for worker in workers:
                worker.process.wait(timeout=15)
                assert worker.process.returncode == 0
        finally:
            for worker in workers:
                worker.close()
    return timings
