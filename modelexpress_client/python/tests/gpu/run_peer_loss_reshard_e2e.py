# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Two-GPU in-flight peer-loss validation for NCCL M2N fail-stop.

Run this driver directly, not through ``torchrun``. Both ranks return from
native M2N group submission with pending PP streams. The parent then kills the
source while reshard work remains queued and releases the destination into
MX's bounded completion polling. The survivor verifies quarantine and prompt
rejection of in-process recovery.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import signal
import socket
import time
import traceback
from datetime import timedelta
from multiprocessing.connection import wait
from typing import Any

EXPECTED_M2N_REF = "45c3f9b96663276c12437bdd9eb5bcf5a4b343a8"
SETUP_TIMEOUT_S = 180.0
POST_FAULT_TIMEOUT_S = 660.0
PROMPT_REJECTION_TIMEOUT_S = 2.0


SOURCE_DELAY_CYCLES = 30_000_000_000


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _prompt_failure(
    label: str,
    operation: Any,
    *,
    expected_type: type[BaseException],
    required_message: str,
) -> dict[str, Any]:
    started = time.monotonic()
    try:
        operation()
    except Exception as exc:
        elapsed = time.monotonic() - started
        if not isinstance(exc, expected_type):
            raise AssertionError(  # noqa: TRY004 - this reports a test failure.
                f"{label} raised unexpected {type(exc).__name__}: {exc}"
            ) from exc
        if elapsed > PROMPT_REJECTION_TIMEOUT_S:
            raise AssertionError(
                f"{label} rejection took {elapsed:.3f}s; expected at most "
                f"{PROMPT_REJECTION_TIMEOUT_S:.3f}s"
            ) from exc
        if required_message not in str(exc).lower():
            raise AssertionError(
                f"{label} raised an unrelated {type(exc).__name__}: {exc}"
            ) from exc
        return {
            "operation": label,
            "error": type(exc).__name__,
            "message": str(exc),
            "elapsed_s": elapsed,
        }
    raise AssertionError(f"{label} unexpectedly succeeded after fail-stop")


def _run_worker(rank: int, port: int, control: Any) -> None:
    import torch
    import torch.distributed as dist
    from modelexpress.refit.reshard.transport.nccl_m2n import (
        M2nCohortRestartRequired,
        M2nPPGroupBootstrap,
        NcclM2nExecutor,
        ReshardParam,
    )
    from modelexpress.refit.reshard.transport.nccl_m2n.runtime import (
        _PPGroupState,
        _RuntimeState,
    )
    from nccl import core as nccl

    if torch.cuda.device_count() < 2:
        raise RuntimeError("peer-loss test requires at least two visible GPUs")
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=SETUP_TIMEOUT_S),
    )

    bootstrap: list[bytes | None] = [
        bytes(nccl.get_unique_id().as_bytes) if rank == 0 else None
    ]
    dist.broadcast_object_list(bootstrap, src=0)
    if bootstrap[0] is None:
        raise RuntimeError("failed to broadcast NCCL unique ID")

    executor = NcclM2nExecutor.create(
        rank,
        [
            M2nPPGroupBootstrap(
                group_id="peer-loss",
                key=(0, 0),
                unique_id=bootstrap[0],
                source_size=1,
                destination_size=1,
                comm_rank=rank,
            )
        ],
        max_cta=8,
        comm_init_timeout_s=SETUP_TIMEOUT_S,
        transfer_timeout_s=5.0,
        finalize_timeout_s=5.0,
    )
    # This fault-injection runner intentionally inspects private fail-stop state;
    # construction and caller lifecycle still use the public data-plane API.
    runtime = executor._runtime
    (pp_group,) = runtime.pp_groups
    elements = 4 * 1024 * 1024
    tensor = torch.full(
        (elements,),
        17.0 if rank == 0 else 0.0,
        dtype=torch.float32,
        device=f"cuda:{rank}",
    )
    param = ReshardParam(
        name="peer_loss.weight",
        global_shape=(elements,),
        shard_dim=0,
        local_tensor=tensor,
        local_shard_index=0,
    )
    updates = {(0, 0): [param]}
    control.send(("GROUP_READY", rank))
    if not control.poll(SETUP_TIMEOUT_S):
        raise TimeoutError("parent did not start peer-loss execution")
    command = control.recv()
    if command != ("START",):
        raise RuntimeError(f"unexpected parent command: {command!r}")

    if rank == 0:
        # Keep source reshard work pending after native group submission. Stream
        # ordering places real M2N work behind this delay without blocking host.
        with torch.cuda.stream(pp_group.stream):
            torch.cuda._sleep(SOURCE_DELAY_CYCLES)

    gate_state: dict[str, bool | None] = {
        "gate_used": False,
        "stream_ready": None,
    }
    original_poll_completion = runtime._poll_pp_groups_completion

    def gate_after_native_group_end(
        pp_groups: Any,
        *,
        operation: str,
        deadline: float,
    ) -> None:
        if not gate_state["gate_used"]:
            gate_state["gate_used"] = True
            stream_ready = bool(pp_group.stream.query())
            gate_state["stream_ready"] = stream_ready
            control.send(("AFTER_GROUP_END", rank, stream_ready))
            if not control.poll(SETUP_TIMEOUT_S):
                raise TimeoutError("parent did not inject in-flight peer loss")
            command = control.recv()
            if command != ("PEER_KILLED",):
                raise RuntimeError(f"unexpected parent command: {command!r}")
            if rank == 0:
                raise AssertionError("source peer was expected to be killed")
        original_poll_completion(
            pp_groups,
            operation=operation,
            deadline=deadline,
        )

    runtime._poll_pp_groups_completion = gate_after_native_group_end
    commit_called = False
    original_copy = executor._copy_into_live

    def recording_copy(live_param: Any, staged: Any) -> None:
        nonlocal commit_called
        commit_called = True
        original_copy(live_param, staged)

    executor._copy_into_live = recording_copy
    started = time.monotonic()
    try:
        executor.stage(updates)
    except M2nCohortRestartRequired as exc:
        transfer_elapsed = time.monotonic() - started
        _require(
            isinstance(exc, M2nCohortRestartRequired),
            "peer-loss stage did not raise M2nCohortRestartRequired: "
            f"{type(exc).__name__}: {exc}",
        )
        _require(exc.operation == "stage", "typed failure operation was not stage")
        _require(
            exc.phase == "completion",
            f"typed failure phase was not completion: {exc.phase!r}",
        )
        _require(
            exc.group_ids == ("peer-loss",),
            f"typed failure group scope was incomplete: {exc.group_ids!r}",
        )
        _require(
            exc.pp_group_keys == ((0, 0),),
            f"typed failure PP-key scope was incomplete: {exc.pp_group_keys!r}",
        )
        _require(exc.__cause__ is not None, "typed failure lost its root cause")
        transfer_error = {
            "type": type(exc).__name__,
            "message": str(exc)[:500],
            "elapsed_s": transfer_elapsed,
            "operation": exc.operation,
            "phase": exc.phase,
            "group_ids": exc.group_ids,
            "pp_group_keys": exc.pp_group_keys,
            "cause_type": type(exc.__cause__).__name__,
            "cause_message": str(exc.__cause__)[:500],
        }
    else:
        raise AssertionError("peer-loss transfer unexpectedly succeeded")

    state = executor._states[(0, 0)]
    _require(gate_state["gate_used"], "post-group completion gate was not used")
    _require(
        gate_state["stream_ready"] is False,
        "destination PP stream was not pending at fault injection",
    )
    _require(runtime._state is _RuntimeState.POISONED, "runtime was not poisoned")
    _require(runtime._restart_required, "runtime did not require process restart")
    _require(runtime._handle_quarantined, "M2N handle was not quarantined")
    _require(runtime._handle is not None, "quarantined M2N handle was released")
    _require(
        pp_group.state is _PPGroupState.POISONED,
        "PP group was not poisoned",
    )
    _require(
        len(runtime._quarantined_batches) == 1,
        "submitted model batch was not retained exactly once",
    )
    _require(
        runtime._quarantined_batches[0].pp_group is pp_group,
        "retained batch refers to the wrong PP group",
    )
    _require(executor._poisoned, "executor was not poisoned")
    _require(bool(state.staged), "destination whole-version staging was released")
    _require(not commit_called, "live destination commit ran after peer loss")

    prompt_failures = [
        _prompt_failure(
            "second stage",
            lambda: executor.stage(updates),
            expected_type=M2nCohortRestartRequired,
            required_message="restart",
        ),
        _prompt_failure(
            "executor close",
            executor.close,
            expected_type=M2nCohortRestartRequired,
            required_message="restart",
        ),
    ]
    abort_deadline = time.monotonic() + PROMPT_REJECTION_TIMEOUT_S
    while not pp_group.abort_attempted and time.monotonic() < abort_deadline:
        time.sleep(0.001)
    _require(pp_group.abort_attempted, "communicator abort was not attempted")
    result = {
        "artifact_ref": EXPECTED_M2N_REF,
        "scenario": (
            "PP1->PP1 source killed after both native group submissions "
            "returned with pending PP streams; survivor quarantines resources "
            "and rejects in-process reuse"
        ),
        "stream_pending_at_fault": gate_state["stream_ready"] is False,
        "transfer_error": transfer_error,
        "prompt_failures": prompt_failures,
        "staging_retained": bool(state.staged),
        "commit_called": commit_called,
        "abort_attempted": pp_group.abort_attempted,
        "abort_completed": pp_group.aborted,
    }
    control.send(("PASS", rank, result))
    control.close()
    # Quarantined CUDA/M2N/NCCL state must be reclaimed by process exit only.
    os._exit(0)


def _worker(rank: int, port: int, control: Any) -> None:
    try:
        _run_worker(rank, port, control)
    except BaseException:  # noqa: BLE001 - child must report every fatal outcome.
        try:
            control.send(("ERROR", rank, traceback.format_exc()))
            control.close()
        finally:
            os._exit(1)


def _receive(
    controls: dict[int, Any],
    processes: dict[int, Any],
    *,
    deadline: float,
    ignore_exits: set[int] | None = None,
) -> tuple[Any, ...]:
    ignored = ignore_exits or set()
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("peer-loss test parent watchdog expired")
        ready = wait(tuple(controls.values()), timeout=min(0.25, remaining))
        for connection in ready:
            try:
                return connection.recv()
            except EOFError as exc:
                rank = next(
                    key for key, value in controls.items() if value is connection
                )
                raise RuntimeError(
                    f"worker {rank} closed its control pipe without a result"
                ) from exc
        for rank, process in processes.items():
            if rank in ignored:
                continue
            if process.exitcode is not None:
                raise RuntimeError(
                    f"worker {rank} exited early with code {process.exitcode}"
                )


def main() -> int:
    artifact_ref = os.environ.get("NCCL_M2N_TEST_REF")
    if artifact_ref != EXPECTED_M2N_REF:
        raise RuntimeError(
            "peer-loss test requires runner attestation for exact bounded M2N "
            f"artifact {EXPECTED_M2N_REF}; got {artifact_ref!r}"
        )

    context = mp.get_context("spawn")
    port = _free_port()
    processes: dict[int, Any] = {}
    controls: dict[int, Any] = {}
    try:
        for rank in range(2):
            parent_control, child_control = context.Pipe(duplex=True)
            process = context.Process(
                target=_worker,
                args=(rank, port, child_control),
                name=f"m2n-peer-loss-rank-{rank}",
            )
            process.start()
            child_control.close()
            controls[rank] = parent_control
            processes[rank] = process

        setup_deadline = time.monotonic() + SETUP_TIMEOUT_S
        ready_ranks: set[int] = set()
        while len(ready_ranks) != 2:
            message = _receive(
                controls,
                processes,
                deadline=setup_deadline,
            )
            if message[0] == "ERROR":
                raise RuntimeError(f"worker {message[1]} failed:\n{message[2]}")
            if message[0] != "GROUP_READY":
                raise RuntimeError(f"unexpected setup message: {message!r}")
            ready_ranks.add(int(message[1]))

        controls[0].send(("START",))
        controls[1].send(("START",))
        pending_streams: dict[int, bool] = {}
        while len(pending_streams) != 2:
            message = _receive(
                controls,
                processes,
                deadline=setup_deadline,
            )
            if message[0] == "ERROR":
                raise RuntimeError(f"worker {message[1]} failed:\n{message[2]}")
            if message[0] != "AFTER_GROUP_END":
                raise RuntimeError(f"unexpected pre-fault message: {message!r}")
            rank = int(message[1])
            stream_ready = bool(message[2])
            if stream_ready:
                raise RuntimeError(
                    f"worker {rank} PP stream completed before fault injection"
                )
            pending_streams[rank] = not stream_ready

        victim = processes[0]
        if not victim.is_alive():
            raise RuntimeError("source peer exited before the parent fault injection")
        os.kill(victim.pid, signal.SIGKILL)
        victim.join(timeout=30.0)
        if victim.is_alive() or victim.exitcode != -signal.SIGKILL:
            raise RuntimeError(
                f"source peer kill was not confirmed; exitcode={victim.exitcode}"
            )
        controls[0].close()
        del controls[0]
        controls[1].send(("PEER_KILLED",))

        message = _receive(
            controls,
            processes,
            deadline=time.monotonic() + POST_FAULT_TIMEOUT_S,
            ignore_exits={0},
        )
        if message[0] == "ERROR":
            raise RuntimeError(f"worker {message[1]} failed:\n{message[2]}")
        if message[0] != "PASS" or message[1] != 1:
            raise RuntimeError(f"unexpected survivor result: {message!r}")

        survivor = processes[1]
        survivor.join(timeout=10.0)
        if survivor.is_alive() or survivor.exitcode != 0:
            raise RuntimeError(
                f"survivor did not exit cleanly; exitcode={survivor.exitcode}"
            )
        print(json.dumps(message[2], indent=2, sort_keys=True), flush=True)
        return 0
    finally:
        for process in processes.values():
            if process.is_alive():
                os.kill(process.pid, signal.SIGKILL)
            process.join(timeout=10.0)
        for control in controls.values():
            control.close()


if __name__ == "__main__":
    raise SystemExit(main())
