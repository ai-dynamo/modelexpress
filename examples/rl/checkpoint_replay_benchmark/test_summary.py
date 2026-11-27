# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only checks for paired aggregation; these are not GPU measurements."""

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).with_name("summarize_pause.py")


class SummaryTests(unittest.TestCase):
    def fixture(self, root, omit_mapped=False):
        for run in range(1, 4):
            rows = [
                {
                    "kind": "qualification",
                    "result": "PASS",
                    "capture_s": 10,
                    "reference_hashes": [{"rank": 0, "sha256": "fixture"}],
                }
            ]
            for arm, t, kind in [
                ("control", 0.02, "measurement"),
                ("ordinary", 1000, "warmup"),
                ("ordinary", 12, "measurement"),
                ("mapped", 9, "measurement"),
            ]:
                if omit_mapped and run == 3 and arm == "mapped":
                    continue
                rows.append(
                    {
                        "kind": kind,
                        "arm": arm,
                        "pair": 0,
                        "correctness": "PASS",
                        "stream_gaps_s": [t] * 4,
                        "requests": [{"tokens": 384, "start": 0, "end": t}] * 4,
                        "pause_rpc_s": t,
                        "scheduler_hold_s": t,
                        "install_rpc_s": t,
                        "max_stream_gap_s": t,
                        "events_during_hold": 0,
                    }
                )
            rows.append({"kind": "complete", "pairs": 1})
            (root / f"pause-run{run}.jsonl").write_text(
                "".join(json.dumps(x) + "\n" for x in rows)
            )

    def test_pairing_and_excluded_controls_warmups(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.fixture(root)
            subprocess.run(["python3", str(SCRIPT), d], check=True, capture_output=True)
            s = json.loads((root / "pause-summary.json").read_text())
            self.assertEqual(s["pairs"], 3)
            self.assertEqual(s["pause_rpc_s"]["ordinary_median_s"], 12)
            self.assertEqual(s["pause_rpc_s"]["paired_saving_median_s"], 3)
            self.assertEqual(s["pause_rpc_s"]["paired_reduction_median_pct"], 25)

    def test_rejects_incomplete_pairs(self):
        with tempfile.TemporaryDirectory() as d:
            self.fixture(Path(d), omit_mapped=True)
            result = subprocess.run(
                ["python3", str(SCRIPT), d], capture_output=True, check=False
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((Path(d) / "pause-summary.json").exists())


if __name__ == "__main__":
    unittest.main()
