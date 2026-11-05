#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read-only preflight for a ModelExpress RL weight refit worker.

Run it inside the trainer or generator container, with the same environment
the worker will run with. It checks the things that make a refit fail late
or hang instead of failing at startup:

* ``m2n``  - the NCCL M2N collective path: the libnccl this process actually
  maps, the ``nccl`` / ``nccl.m2n`` Python packages, ``NCCL_COMM_BLOCKING``,
  and every ``MX_NCCL_REFIT_*`` value against the client's own validation.
* ``nixl`` - the NIXL reshard path: NIXL availability through the
  ModelExpress client, ``VLLM_PLUGINS`` filtering, ``MX_P2P_METADATA`` and
  the RL load-strategy settings.

Both paths optionally probe TCP reachability of the ModelExpress server.
Nothing is written, no RPC is issued, and no environment value other than the
ones named above is read or printed. Output is one JSON document on stdout.
Exit code is 0 when no check failed, 1 otherwise.
"""

from __future__ import annotations

import argparse
import ctypes
import importlib
import importlib.util
import json
import math
import os
import socket
import sys
from dataclasses import asdict, dataclass

# The reshard entry points need this libnccl release (modelexpress_rl.collective.backend.MIN_NCCL).
MIN_NCCL = (2, 30, 7)
DEFAULT_CONNECT_TIMEOUT_S = 5.0

# Mirrors modelexpress_rl/collective/envs.py: kind decides the validation rule.
#   positive_int / positive_float: must parse and be > 0
#   bytes: must parse as int and be >= 0 (0 disables the bound)
#   flag: one of 0/1/true/false/yes/no/on/off (case-insensitive) or empty
M2N_ENV_RULES = {
    "MX_NCCL_REFIT_NUM_STREAMS": "positive_int",
    "MX_NCCL_REFIT_GROUP_TIMEOUT_S": "positive_float",
    "MX_NCCL_REFIT_POLL_INTERVAL_S": "positive_float",
    "MX_NCCL_REFIT_COMM_INIT_TIMEOUT_S": "positive_float",
    "MX_NCCL_REFIT_TRANSFER_TIMEOUT_S": "positive_float",
    "MX_NCCL_REFIT_ABORT_TIMEOUT_S": "positive_float",
    "MX_NCCL_REFIT_REGISTRATION_TTL_S": "positive_int",
    "MX_NCCL_REFIT_PIPELINE_DEPTH": "positive_int",
    "MX_NCCL_REFIT_BLOCKING_DATA_PLANE": "flag",
    "MX_NCCL_REFIT_GROUPED_SUBMISSION": "flag",
    "MX_NCCL_REFIT_RECEIVE_WINDOW_BYTES": "bytes",
    "MX_NCCL_REFIT_SEND_WINDOW_BYTES": "bytes",
    "MX_HEARTBEAT_INTERVAL_SECS": "positive_int",
}
TRUE_FLAGS = ("1", "true", "yes", "on")
FALSE_FLAGS = ("0", "false", "no", "off")


@dataclass
class Check:
    name: str
    status: str  # "pass", "warn", "fail", "skip"
    detail: str


def _validate(name: str, kind: str, raw: str) -> str | None:
    """Return an error string, or None when the value is acceptable."""
    if kind == "flag":
        if raw.strip().lower() in ("",) + TRUE_FLAGS + FALSE_FLAGS:
            return None
        return f"{name}={raw!r} is not a recognized boolean"
    try:
        value = float(raw) if kind == "positive_float" else int(raw)
    except ValueError:
        return f"{name}={raw!r} does not parse"
    if kind == "bytes":
        return None if value >= 0 else f"{name} must not be negative"
    if kind == "positive_float" and not math.isfinite(value):
        return f"{name} must be finite"
    return None if value > 0 else f"{name} must be positive"


def check_env_values(rules: dict[str, str]) -> list[Check]:
    checks = []
    for name, kind in rules.items():
        raw = os.environ.get(name)
        if raw is None:
            continue
        error = _validate(name, kind, raw)
        checks.append(
            Check(name, "fail" if error else "pass", error or f"{name}={raw}")
        )
    if not checks:
        checks.append(Check("env-values", "pass", "no overrides set; client defaults apply"))
    return checks


def loaded_nccl_version() -> tuple[int, int, int] | None:
    """Ask the libnccl this process maps, not the one torch was built against."""
    try:
        lib = ctypes.CDLL("libnccl.so.2")
        raw = ctypes.c_int()
        if lib.ncclGetVersion(ctypes.byref(raw)) != 0:
            return None
    except OSError:
        return None
    value = raw.value
    return (value // 10000, (value // 100) % 100, value % 100)


def _module_present(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def check_m2n() -> list[Check]:
    checks = []
    version = loaded_nccl_version()
    if version is None:
        checks.append(
            Check("libnccl", "fail", "libnccl.so.2 did not resolve or ncclGetVersion failed")
        )
    else:
        text = ".".join(map(str, version))
        ok = version >= MIN_NCCL
        checks.append(
            Check(
                "libnccl",
                "pass" if ok else "fail",
                f"mapped libnccl {text}; reshard needs >= {'.'.join(map(str, MIN_NCCL))}",
            )
        )
    for module, hint in (
        ("nccl", "install nccl4py"),
        ("nccl.m2n", "install nccl-extensions[cu12] or nccl-extensions[cu13] to match CUDA"),
        ("modelexpress_rl.collective", "install the modelexpress distribution"),
    ):
        present = _module_present(module)
        checks.append(
            Check(f"import:{module}", "pass" if present else "fail", "found" if present else hint)
        )
    blocking = os.environ.get("NCCL_COMM_BLOCKING")
    if blocking in (None, "", "0"):
        checks.append(Check("NCCL_COMM_BLOCKING", "pass", "unset or 0"))
    else:
        checks.append(
            Check(
                "NCCL_COMM_BLOCKING",
                "fail",
                "must be unset or 0: blocking init defeats MX_NCCL_REFIT_COMM_INIT_TIMEOUT_S",
            )
        )
    grouped = os.environ.get("MX_NCCL_REFIT_GROUPED_SUBMISSION", "").strip().lower()
    if grouped in TRUE_FLAGS:
        checks.append(
            Check(
                "grouped-submission",
                "warn",
                "grouped submission needs GIN (RDMA devices; NCCL_GIN_TYPE=2 for proxy GIN) "
                "and must match on every participant (it is part of the plan digest)",
            )
        )
    checks.extend(check_env_values(M2N_ENV_RULES))
    checks.append(
        Check(
            "process-bound",
            "warn",
            "MX_NCCL_REFIT_TRANSFER_TIMEOUT_S cannot interrupt a reshard blocked in native "
            "code; bound each worker process from outside (run_e2e.sh: PROC_TIMEOUT_S)",
        )
    )
    return checks


def check_nixl() -> list[Check]:
    checks = []
    if _module_present("modelexpress_rl"):
        checks.append(Check("import:modelexpress_rl", "pass", "found"))
        try:
            nixl_transfer = importlib.import_module("modelexpress.nixl_transfer")
            available = bool(nixl_transfer.is_nixl_available())
            checks.append(
                Check(
                    "nixl",
                    "pass" if available else "fail",
                    "available" if available else "NIXL not importable",
                )
            )
        except Exception as error:  # noqa: BLE001 - report, do not crash
            checks.append(Check("nixl", "fail", f"probe raised {type(error).__name__}"))
    else:
        checks.append(
            Check("import:modelexpress_rl", "fail", "install the modelexpress distribution")
        )
        checks.append(Check("nixl", "skip", "modelexpress_rl is not installed"))
    plugins = os.environ.get("VLLM_PLUGINS")
    if plugins is not None:
        names = {item.strip() for item in plugins.split(",") if item.strip()}
        ok = "modelexpress_rl" in names
        checks.append(
            Check(
                "VLLM_PLUGINS",
                "pass" if ok else "fail",
                "includes modelexpress_rl" if ok else "filtered list must include modelexpress_rl",
            )
        )
    if os.environ.get("MX_P2P_METADATA", "1") == "0":
        checks.append(
            Check(
                "MX_P2P_METADATA",
                "warn",
                "=0: active refit of a generator that publishes runtime tensors "
                "fails before mutation",
            )
        )
    strategy = os.environ.get("MX_LOAD_STRATEGY_CHAIN", "INFERENCE").strip().upper()
    desired = os.environ.get("MX_REFIT_DESIRED_VERSION_UID", "").strip()
    order = os.environ.get("MX_GENERATOR_SOURCE_ORDER", "").upper()
    detail = f"MX_LOAD_STRATEGY_CHAIN={strategy}; desired UID {'set' if desired else 'unset'}"
    status = "pass"
    if strategy == "RL" and desired and "TRAINER" in order:
        status = "fail"
        detail += (
            "; TRAINER in MX_GENERATOR_SOURCE_ORDER is rejected for desired-version cold start"
        )
    checks.append(Check("load-strategy", status, detail))
    return checks


def check_endpoint(endpoint: str, timeout_s: float) -> Check:
    host, sep, port = endpoint.rpartition(":")
    if not sep or not port.isdigit():
        return Check("endpoint", "fail", f"{endpoint!r} is not host:port")
    try:
        with socket.create_connection((host.strip("[]"), int(port)), timeout=timeout_s):
            pass
    except OSError as error:
        reason = error.strerror or error
        return Check("endpoint", "fail", f"TCP connect to {endpoint} failed: {reason}")
    return Check("endpoint", "pass", f"TCP connect to {endpoint} ok")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--path", choices=("m2n", "nixl"), required=True)
    parser.add_argument("--endpoint", help="ModelExpress server host:port to probe over TCP")
    parser.add_argument("--timeout", type=float, default=DEFAULT_CONNECT_TIMEOUT_S)
    args = parser.parse_args(argv)

    checks = check_m2n() if args.path == "m2n" else check_nixl()
    if args.endpoint:
        checks.append(check_endpoint(args.endpoint, args.timeout))
    else:
        checks.append(Check("endpoint", "skip", "pass --endpoint host:port to probe the server"))

    failed = [c.name for c in checks if c.status == "fail"]
    print(
        json.dumps(
            {
                "path": args.path,
                "ok": not failed,
                "failed": failed,
                "checks": [asdict(c) for c in checks],
            },
            indent=2,
        )
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
