"""Report saved results for the selected model, including failed/incomplete runs."""

import json
import math
import re
import sys
from pathlib import Path

import model_checks
import validation


def _build_report(root):
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
            if (role, step) in records:
                raise ValueError(f"Duplicate driver record: {role} {step}")
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
            "sequential_refits": {
                step: records.get((role, prefix + "refit"))
                for step, prefix in [("d1", ""), ("d2", "d2-")]
            },
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
        trials = validation.trials(publication, config)
        baseline = {
            role: validation.hashes(result(role, "base-hashes"), config)
            for role in config["roles"]
        }
        if "peer" in baseline:
            assert baseline["s3"] == baseline["peer"], "Cold peer tensors differ"
        sessions = validation.sessions(config, result)
        for role in config["roles"]:
            worker = report["workers"][role]
            assert not worker["failures"], "Worker reported a failed operation"
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
            assert worker["memory_events"], "Missing memory monitor evidence"
            events = dict(line.split() for line in worker["memory_events"].splitlines())
            assert int(events["oom"]) == int(events["oom_kill"]) == 0, "Worker OOM"
        if "peer" in config["roles"]:
            assert (
                report["workers"]["s3"]["pod"]["spec"]["nodeName"]
                != report["workers"]["peer"]["pod"]["spec"]["nodeName"]
            )
        previous_layout = None
        report["sequential_refits"] = {}
        report["model_checks"] = {}
        for step, trial in enumerate(trials, 1):
            prefix = "" if step == 1 else "d2-"

            def step_result(role, name, prefix=prefix):
                return result(role, prefix + name)

            layout = validation.ranks(step_result("s3", "layout-before"), config)
            if previous_layout is not None:
                assert layout == previous_layout, "Checkpoint changed between updates"
            baseline = validation.update(
                config, trial, step_result, baseline, sessions, reuse=step == 2
            )
            previous_layout = validation.ranks(
                step_result("s3", "layout-after"), config
            )
            report["model_checks"][f"d{step}"] = model_checks.validate(
                config, step_result
            )
            step_records = {}
            for role in config["roles"]:
                record = records[role, prefix + "refit"]
                assert math.isfinite(record["seconds"]) and record["seconds"] > 0
                step_records[role] = record
            report["sequential_refits"][f"d{step}"] = step_records
        first = report["sequential_refits"]["d1"]["s3"]["seconds"]
        second = report["sequential_refits"]["d2"]["s3"]["seconds"]
        report["s3_savings"] = {
            "seconds": first - second,
            "percent": 100 * (first - second) / first,
        }
        report["status"] = "PASS"
    except (
        AssertionError,
        KeyError,
        ValueError,
        OSError,
        TypeError,
        IndexError,
    ) as error:
        report["failure_reason"] = str(error) or type(error).__name__
    return report


def build_report(root):
    try:
        return _build_report(root)
    except (
        AssertionError,
        KeyError,
        ValueError,
        OSError,
        TypeError,
        IndexError,
    ) as error:
        return {
            "status": "FAILED",
            "workers": {},
            "failure_reason": str(error) or type(error).__name__,
        }


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
