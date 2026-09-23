"""Report saved results for the selected model, including failed/incomplete runs."""

import json
import re
import sys
from pathlib import Path

import model_checks
import validation


def build_report(root):
    config = json.loads((root / "config.json").read_text())
    log = (
        (root / "e2e-driver.log").read_text()
        if (root / "e2e-driver.log").exists()
        else ""
    )
    records = {}
    for line in log.splitlines():
        if line.startswith("RESULT "):
            _, role, step, body = line.split(" ", 3)
            records[role, step] = json.loads(body)
    report = {
        "status": "FAILED",
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
        wire = [
            line
            for line in text.splitlines()
            if config.get("peer_transfer_marker", "RDMA transfer complete:") in line
        ]
        worker = {
            "model_load_seconds": times,
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
        anonymous = []
        events = None
        monitor = root / f"{role}-monitor.log"
        if monitor.exists():
            for line in monitor.read_text().splitlines():
                if line.startswith("HOST_MEMORY "):
                    row = json.loads(line[len("HOST_MEMORY ") :])
                    stat = dict(
                        s.split() for s in row.get("memory.stat", "").splitlines()
                    )
                    if "anon" in stat:
                        anonymous.append(int(stat["anon"]) / 2**30)
                    events = row.get("memory.events")
        worker.update(
            peak_observed_anon_gib=max(anonymous) if anonymous else None,
            memory_events=events,
        )
        pod_file = root / f"{role}-pod.json"
        if pod_file.exists():
            worker["pod"] = json.loads(pod_file.read_text())
        report["workers"][role] = worker
    try:
        assert "E2E_PASS" in log.splitlines(), (
            "Driver failed or did not finish; inspect per-rank failures and e2e-driver.log"
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
            # The pinned runtimes log the model-load interval once on rank zero.
            assert len(worker["model_load_seconds"]) == 1, (
                "Expected one fresh model load"
            )
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
            if worker["memory_events"]:
                assert (
                    int(
                        dict(s.split() for s in worker["memory_events"].splitlines())[
                            "oom_kill"
                        ]
                    )
                    == 0
                )
        for row in validation.ranks(result("s3", "verify-checkpoint"), config).values():
            assert row["verified"] and row["sha256"] == row["expected_sha256"]
        if "peer" in config["roles"]:
            assert (
                baseline["s3"] == baseline["peer"] and updated["s3"] == updated["peer"]
            )
            assert (
                report["workers"]["s3"]["pod"]["spec"]["nodeName"]
                != report["workers"]["peer"]["pod"]["spec"]["nodeName"]
            )
        report.update(model_checks.validate(config, result))
        report["status"] = "PASS"
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
            "peak observed anonymous GiB:",
            worker["peak_observed_anon_gib"],
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
    print(report["status"], report.get("failure_reason", ""))
    print(root / "report.json")
    sys.exit(0 if report["status"] == "PASS" else 1)
