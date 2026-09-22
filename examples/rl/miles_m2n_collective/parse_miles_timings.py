# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Extract complete rank-zero MILES weight-update timer triplets from logs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

_TIMER_PATTERN = re.compile(
    r"Timer "
    r"(?P<name>update_weights_implementation|finalize_and_resume_engines|update_weights) "
    r"end \(elapsed: (?P<elapsed>[0-9]+(?:\.[0-9]+)?)s\)"
)
_IMPLEMENTATION = "update_weights_implementation"
_FINALIZE = "finalize_and_resume_engines"
_UPDATE = "update_weights"


def parse_log(log_path: Path, *, expected_updates: int) -> dict[str, Any]:
    if expected_updates < 0:
        raise ValueError("expected_updates must be non-negative")

    updates: list[dict[str, Any]] = []
    pending: dict[str, float] = {}
    for line_number, line in enumerate(
        log_path.read_text(errors="replace").splitlines(), start=1
    ):
        if "actor_cell0_rank0" not in line:
            continue
        match = _TIMER_PATTERN.search(line)
        if match is None:
            continue
        name = match.group("name")
        elapsed = float(match.group("elapsed"))
        if name in pending:
            raise ValueError(
                f"duplicate Timer {name} before update completion at line {line_number}"
            )
        if name == _IMPLEMENTATION:
            if pending:
                observed = ", ".join(sorted(pending))
                raise ValueError(
                    "timer triplet is out of order at "
                    f"line {line_number}; observed {observed} before {name}"
                )
            pending[name] = elapsed
            continue
        if name == _FINALIZE:
            if set(pending) != {_IMPLEMENTATION}:
                raise ValueError(
                    "timer triplet is out of order at "
                    f"line {line_number}; {_FINALIZE} must follow {_IMPLEMENTATION}"
                )
            pending[name] = elapsed
            continue
        if name == _UPDATE:
            missing = {_IMPLEMENTATION, _FINALIZE}.difference(pending)
            if missing:
                missing_names = ", ".join(sorted(missing))
                raise ValueError(
                    "Timer update_weights ended without a complete timer triplet "
                    f"at line {line_number}; missing {missing_names}"
                )
            index = len(updates)
            updates.append(
                {
                    "classification": "cold" if index == 0 else "steady",
                    "finalize_and_resume_engines_s": pending[_FINALIZE],
                    "index": index,
                    "update_weights_implementation_s": pending[_IMPLEMENTATION],
                    "update_weights_s": elapsed,
                }
            )
            pending.clear()
            continue

    if pending:
        names = ", ".join(sorted(pending))
        raise ValueError(
            f"log ended without a complete timer triplet; unmatched timers: {names}"
        )
    if len(updates) != expected_updates:
        raise ValueError(
            f"expected {expected_updates} complete updates, observed {len(updates)}"
        )
    return {
        "expected_update_count": expected_updates,
        "observed_update_count": len(updates),
        "schema_version": 1,
        "updates": updates,
    }


def write_result(output_path: Path, payload: dict[str, Any]) -> None:
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse complete MILES rank-zero weight-update timings."
    )
    parser.add_argument("log", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--expected-updates", type=int, required=True)
    parser.add_argument("--mode", choices=("broadcast", "external"), required=True)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--block", type=int, required=True)
    parser.add_argument("--repetition", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    payload = parse_log(args.log, expected_updates=args.expected_updates)
    payload["benchmark"] = {
        "block": args.block,
        "mode": args.mode,
        "repetition": args.repetition,
        "run_label": args.run_label,
    }
    write_result(args.output, payload)


if __name__ == "__main__":
    main()
