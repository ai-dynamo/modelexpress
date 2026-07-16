# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""External file-based coordinator for the NCCL M2N GPU test only."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from uuid import UUID


def _canonical_uuid4(value: str) -> str:
    parsed = UUID(value)
    if parsed.version != 4 or str(parsed) != value:
        raise ValueError("MX_M2N_ATTEMPT_ID must be a canonical UUIDv4")
    return value


def _atomic_immutable_decision(attempt_directory: Path, value: str) -> None:
    decision = attempt_directory / "decision"
    temporary = attempt_directory / f".decision.{os.getpid()}.tmp"
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, decision)
    finally:
        temporary.unlink(missing_ok=True)


def _abort_mx(
    client: MxM2nBootstrapClient,
    key: m2n_bootstrap_pb2.M2nBootstrapKey,
    reason: str,
) -> None:
    try:
        client.abort_bootstrap(
            key,
            requested_by="m2n-test-coordinator",
            reason=reason.encode("utf-8")[:1024].decode("utf-8", errors="ignore"),
            timeout_s=5.0,
        )
    except BaseException as error:
        print(f"warning: AbortBootstrap failed after abort decision: {error}", file=sys.stderr)


def run_coordinator(
    *,
    root: Path,
    attempt_id: str,
    world_size: int,
    timeout_s: float,
    client: MxM2nBootstrapClient,
    key: m2n_bootstrap_pb2.M2nBootstrapKey,
) -> int:
    if world_size <= 0:
        raise ValueError("WORLD_SIZE must be positive")
    if timeout_s <= 0:
        raise ValueError("MX_M2N_TIMEOUT_S must be positive")
    attempt_directory = root / _canonical_uuid4(attempt_id)
    attempt_directory.mkdir(parents=True, exist_ok=True)
    decision = attempt_directory / "decision"
    if decision.exists():
        raise RuntimeError(f"coordinator decision already exists: {decision}")

    expected = {f"ready-{rank}" for rank in range(world_size)}
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        failures = sorted(attempt_directory.glob("failed-*"))
        if failures:
            reason = "rank failure markers: " + ",".join(path.name for path in failures)
            try:
                _atomic_immutable_decision(attempt_directory, f"abort:{reason}")
            except FileExistsError:
                value = decision.read_text(encoding="utf-8")
                if not value.startswith("abort:"):
                    raise
            _abort_mx(client, key, reason)
            return 1

        ready = {path.name for path in attempt_directory.glob("ready-*")}
        if ready == expected:
            try:
                _atomic_immutable_decision(attempt_directory, "release")
                return 0
            except FileExistsError:
                value = decision.read_text(encoding="utf-8")
                if value.startswith("abort:"):
                    return 1
                raise
        time.sleep(0.05)

    reason = f"timed out waiting for {world_size} ready ranks"
    try:
        _atomic_immutable_decision(attempt_directory, f"abort:{reason}")
    except FileExistsError:
        value = decision.read_text(encoding="utf-8")
        if not value.startswith("abort:"):
            raise
    _abort_mx(client, key, reason)
    return 1


def main() -> int:
    attempt_id = os.environ["MX_M2N_ATTEMPT_ID"]
    root = Path(os.environ["MX_M2N_COORDINATOR_DIR"])
    (root / _canonical_uuid4(attempt_id)).mkdir(parents=True, exist_ok=True)

    from modelexpress import m2n_bootstrap_pb2
    from modelexpress.m2n_bootstrap import MxM2nBootstrapClient

    world_size = int(os.environ["WORLD_SIZE"])
    timeout_s = float(os.environ.get("MX_M2N_TIMEOUT_S", "120"))
    key = m2n_bootstrap_pb2.M2nBootstrapKey(
        job_id=os.environ.get("MX_M2N_JOB_ID", "nccl-m2n-e2e"),
        attempt_id=attempt_id,
        cohort_id=os.environ.get("MX_M2N_COHORT_ID", "reshard-e2e"),
    )
    client = MxM2nBootstrapClient()
    try:
        return run_coordinator(
            root=root,
            attempt_id=attempt_id,
            world_size=world_size,
            timeout_s=timeout_s,
            client=client,
            key=key,
        )
    finally:
        client.close()


if __name__ == "__main__":
    sys.exit(main())
