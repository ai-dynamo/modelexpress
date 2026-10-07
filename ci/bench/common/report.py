# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Report saved results for the selected model, including failed/incomplete runs."""

import json
import re
import sys
from pathlib import Path

import scenario
import validation


def build_report(root):
    config = json.loads((root / "config.json").read_text())
    case = scenario.load(config)
    log = (
        (root / "bench-driver.log").read_text()
        if (root / "bench-driver.log").exists()
        else ""
    )
    records = {}
    for line in log.splitlines():
        if line.startswith("RESULT "):
            _, role, step, body = line.split(" ", 3)
            records[role, step] = json.loads(body)
    report = {
        "status": "FAILED",
        "validation_status": "FAILED",
        "measurement_status": "INVALID",
        "latency_summary": {},
        "config": config,
        "images": json.loads((root / "images.json").read_text()),
        "workers": {},
    }

    def result(role, step):
        record = records[role, step]
        assert record["http_status"] == 200 and record["response"]["ok"], record
        return record["response"]["result"]

    for role in config["roles"]:
        text = (
            (root / f"{role}-worker.log").read_text(errors="replace")
            if (root / f"{role}-worker.log").exists()
            else ""
        )
        times = [
            float(x)
            for x in re.findall(
                r"Model loading took .*? memory and ([\d.]+) seconds", text
            )
        ]
        worker = {
            "model_load_seconds": times,
            "refit": records.get((role, "refit")),
            "failures": [],
        }
        worker.update(case.worker_evidence(text, config))
        # Preserve streamed records even when an OOM prevented an RPC response.
        for line in text.splitlines():
            if "HOTLOAD_BENCHMARK " in line:
                try:
                    row = json.loads(line.split("HOTLOAD_BENCHMARK ", 1)[1])
                except ValueError:
                    continue
                if row.get("phase", "").endswith("-failed"):
                    worker["failures"].append(row)
        pod_file = root / f"{role}-pod.json"
        if pod_file.exists():
            worker["pod"] = json.loads(pod_file.read_text())
        report["workers"][role] = worker
    try:
        assert "BENCH_PASS" in log.splitlines(), (
            "Driver failed or did not finish; inspect per-rank failures and bench-driver.log"
        )
        publication = json.loads((root / "publication.json").read_text())
        report["publication"] = publication
        assert (
            publication["run"] == config["run"]
            and publication["model_revision"] == config["revision"]
        )
        baseline, updated = {}, {}
        for role in config["roles"]:
            baseline[role] = validation.hashes(result(role, "base-hashes"), config)
            updated[role] = validation.hashes(result(role, "updated-hashes"), config)
            case.validate_inventory(baseline[role], updated[role])
            validation.refit(result(role, "refit"), config, role)
            validation.inference(result(role, "post-refit-inference"))
            if config["expected_host_scales_per_rank"] is not None:
                for step in [
                    "baseline-host-scales",
                    "immediate-post-refit-host-scales",
                    "post-inference-host-scales",
                ]:
                    validation.scales(result(role, step), config)
            worker = report["workers"][role]
            pod = worker["pod"]
            assert pod["status"].get("containerStatuses")
            assert all(
                x["restartCount"] == 0 and "terminated" not in x["state"]
                for x in pod["status"]["containerStatuses"]
            )
        case.compare_workers(baseline, config)
        case.compare_workers(updated, config)
        case.validate_report(result, report, root)
        report["validation_status"] = "PASS"
        summary = {
            role: validation.latency(
                result(role, "refit"),
                config,
                report["workers"][role]["model_load_seconds"],
                records[role, "refit"]["seconds"],
            )
            for role in config["roles"]
        }
        report.update(
            status="PASS", measurement_status="VALID", latency_summary=summary
        )
    except (AssertionError, KeyError, ValueError, OSError, TypeError) as error:
        report["failure_reason"] = str(error) or type(error).__name__
    return report


if __name__ == "__main__":
    root = Path(sys.argv[1])
    report = build_report(root)
    (root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    for role, worker in report["workers"].items():
        print(
            role,
            "model distribution/load seconds:",
            worker["model_load_seconds"],
        )
        if worker["refit"]:
            for rank in worker["refit"]["response"].get("result", []):
                print(
                    "  rank",
                    rank.get("rank"),
                    "stage:",
                    rank.get("stage_seconds"),
                    "install:",
                    rank.get("install_seconds"),
                    "error:",
                    rank.get("error"),
                    "metrics:",
                    rank.get("metrics"),
                )
    print(
        "Validation:",
        report["validation_status"],
        "Measurements:",
        report["measurement_status"],
    )
    if report["latency_summary"]:
        print("LATENCY_SUMMARY", json.dumps(report["latency_summary"]))
    print(report["status"], report.get("failure_reason", ""))
    print(root / "report.json")
    sys.exit(0 if report["status"] == "PASS" else 1)
