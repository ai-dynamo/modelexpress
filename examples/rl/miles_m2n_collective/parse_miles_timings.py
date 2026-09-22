# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Extract complete rank-zero MILES weight-update timer triplets from logs."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

_TIMER_PATTERN = re.compile(
    r"\[(?P<timestamp>\d{4}-\d{2}-\d{2} "
    r"\d{2}:\d{2}:\d{2}\.\d+) actor_cell0_rank0\].*Timer "
    r"(?P<name>update_weights_implementation|finalize_and_resume_engines|update_weights) "
    r"(?P<phase>start|end)"
)
_IMPLEMENTATION = "update_weights_implementation"
_FINALIZE = "finalize_and_resume_engines"
_UPDATE = "update_weights"
_ORDER = (
    (_UPDATE, "start"),
    (_IMPLEMENTATION, "start"),
    (_IMPLEMENTATION, "end"),
    (_FINALIZE, "start"),
    (_FINALIZE, "end"),
    (_UPDATE, "end"),
)


def _elapsed_seconds(start: datetime, end: datetime) -> float:
    elapsed = (end - start).total_seconds()
    if elapsed < 0:
        raise ValueError("timer end precedes timer start")
    return elapsed


def parse_log(log_path: Path, *, expected_updates: int) -> dict[str, Any]:
    if expected_updates < 0:
        raise ValueError("expected_updates must be non-negative")

    updates: list[dict[str, Any]] = []
    pending: dict[tuple[str, str], datetime] = {}
    for line_number, line in enumerate(
        log_path.read_text(errors="replace").splitlines(), start=1
    ):
        if "actor_cell0_rank0" not in line:
            continue
        match = _TIMER_PATTERN.search(line)
        if match is None:
            continue
        event = (match.group("name"), match.group("phase"))
        expected_event = _ORDER[len(pending)]
        if event in pending:
            raise ValueError(
                f"duplicate Timer {event[0]} {event[1]} before update completion "
                f"at line {line_number}"
            )
        if event != expected_event:
            raise ValueError(
                "timer triplet is out of order at "
                f"line {line_number}; expected Timer {expected_event[0]} "
                f"{expected_event[1]}, observed Timer {event[0]} {event[1]}"
            )
        pending[event] = datetime.fromisoformat(match.group("timestamp"))
        if event == (_UPDATE, "end"):
            implementation_s = _elapsed_seconds(
                pending[(_IMPLEMENTATION, "start")],
                pending[(_IMPLEMENTATION, "end")],
            )
            finalize_s = _elapsed_seconds(
                pending[(_FINALIZE, "start")],
                pending[(_FINALIZE, "end")],
            )
            update_s = _elapsed_seconds(
                pending[(_UPDATE, "start")],
                pending[(_UPDATE, "end")],
            )
            ordered_timestamps = [pending[item] for item in _ORDER]
            if ordered_timestamps != sorted(ordered_timestamps):
                raise ValueError(
                    "timer timestamps are not monotonically nested for the "
                    f"update ending at line {line_number}"
                )
            index = len(updates)
            updates.append(
                {
                    "classification": "cold" if index == 0 else "steady",
                    "finalize_and_resume_engines_s": finalize_s,
                    "index": index,
                    "timing_source": "rank_zero_log_timestamps",
                    "update_weights_implementation_s": implementation_s,
                    "update_weights_s": update_s,
                }
            )
            pending.clear()

    if pending:
        names = ", ".join(f"{name} {phase}" for name, phase in pending)
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
        "schema_version": 2,
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
