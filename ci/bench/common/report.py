# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Report saved results for the selected model, including failed/incomplete runs."""

import json
import re
import sys
from pathlib import Path

import validation


def build_report(root):
    config = json.loads((root / "config.json").read_text())
    log = (
        (root / "bench-driver.log").read_text()
        if (root / "bench-driver.log").exists()
        else ""
    )
    records = {}
    record_errors = []
    for number, line in enumerate(log.splitlines(), 1):
        if line.startswith("RESULT "):
            try:
                _, role, step, body = line.split(" ", 3)
                records[role, step] = json.loads(body)
            except ValueError as error:
                record_errors.append({"line": number, "error": str(error)})
    report = {
        "status": "FAILED",
        "validation_status": "FAILED",
        "measurement_status": "INVALID",
        "latency_summary": {},
        "config": config,
        "images": json.loads((root / "images.json").read_text()),
        "workers": {},
        "record_errors": record_errors,
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
        times = []
        times_by_rank = {}
        for line in text.splitlines():
            match = re.search(
                r"Model loading took .*? memory and ([\d.]+) seconds", line
            )
            if match:
                seconds = float(match[1])
                times.append(seconds)
                rank_match = re.search(r"\bWorker_TP(\d+)\b", line)
                rank = int(rank_match[1]) if rank_match else (
                    0 if config["tp"] == 1 else -1
                )
                times_by_rank.setdefault(rank, []).append(seconds)
        wire = [
            line
            for line in text.splitlines()
            if config.get("peer_transfer_marker", "RDMA transfer complete:") in line
        ]
        worker = {
            "model_load_seconds": times,
            "model_load_seconds_by_rank": times_by_rank,
            "rdma_transfer_records": wire,
            "refit": records.get((role, "refit")),
            "failures": [],
        }
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
        assert not record_errors, "Malformed RESULT records; inspect record_errors"
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
            assert baseline[role] != updated[role], "No updated tensors"
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
            text = (root / f"{role}-worker.log").read_text()
            if role == "s3":
                assert "Streaming weights from s3://" in text
            else:
                assert worker["rdma_transfer_records"], "No RDMA completion evidence"
                assert not any(
                    x in text
                    for x in [
                        "Trying strategy: model_streamer",
                        "Streaming weights from s3://",
                        "Trying strategy: instant_tensor",
                    ]
                ), "Peer cold load fell back"
            pod = worker["pod"]
            assert pod["status"].get("containerStatuses")
            assert all(
                x["restartCount"] == 0 and "terminated" not in x["state"]
                for x in pod["status"]["containerStatuses"]
            )
        validation.checkpoint(result("s3", "verify-checkpoint"), config, publication)
        if "peer" in config["roles"]:
            assert (
                baseline["s3"] == baseline["peer"] and updated["s3"] == updated["peer"]
            )
            assert (
                report["workers"]["s3"]["pod"]["spec"]["nodeName"]
                != report["workers"]["peer"]["pod"]["spec"]["nodeName"]
            )
        report["validation_status"] = "PASS"
        summary = {
            role: validation.latency(
                result(role, "refit"),
                config,
                report["workers"][role]["model_load_seconds_by_rank"],
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
