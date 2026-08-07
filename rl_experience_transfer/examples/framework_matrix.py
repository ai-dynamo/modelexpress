# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run every declared producer-to-consumer native adapter pairing."""

from __future__ import annotations

import argparse
import json
import tempfile
from itertools import product
from pathlib import Path

if __package__:
    from .framework_roundtrip import FRAMEWORKS, run_framework_case
else:
    from framework_roundtrip import FRAMEWORKS, run_framework_case


def run_matrix(queue_root: Path) -> dict[str, object]:
    """Run the 4x4 conversion matrix and return a machine-readable summary."""

    cases = [
        run_framework_case(producer, consumer, queue_root / f"{producer}-to-{consumer}")
        for producer, consumer in product(FRAMEWORKS, repeat=2)
    ]
    return {
        "cases": cases,
        "converted": sum(case["expected_outcome"] == "converted" for case in cases),
        "expected_rejections": sum(
            case["expected_outcome"] == "rejected_as_unsafe" for case in cases
        ),
        "frameworks": list(FRAMEWORKS),
        "result": "PASSED" if all(case["result"] == "PASSED" for case in cases) else "FAILED",
        "total": len(cases),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-root", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    try:
        if arguments.queue_root is None:
            with tempfile.TemporaryDirectory(prefix="rlxfer-framework-matrix-") as temporary:
                result = run_matrix(Path(temporary))
        else:
            result = run_matrix(arguments.queue_root)
    except Exception as error:
        result = {"error": f"{type(error).__name__}: {error}", "result": "FAILED"}
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if result["result"] == "PASSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
