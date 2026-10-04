#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Download the previous scheduled nightly's compliance artifact for osrb.py to diff against.

A missing baseline (first nightly, expired artifacts) is recorded as unavailable.
Any API or download error fails the job instead of producing an all-added diff.
"""

import argparse
import http.client
import io
import json
import os
import sys
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

API = "https://api.github.com"


def fail(msg: str) -> None:
    print(f"::error::{msg}", file=sys.stderr)
    sys.exit(1)


def get(url: str, token: str, raw: bool = False):
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})
    # Signed artifact URLs authenticate with their query string; a forwarded bearer token gets HTTP 401.
    req.add_unredirected_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = resp.read()
    return data if raw else json.loads(data)


def unavailable(args: argparse.Namespace, checked: int) -> dict:
    where = f"of {args.workflow} on {args.branch} in the last {args.max_age_days} days"
    if checked:
        artifact = args.artifact.replace("{sha}", "<sha>")
        reason = (f"none of the {checked} earlier {args.event} runs {where} passed '{args.require_job}' "
                  f"and has artifact {artifact}")
        print(f"::warning::{reason}")
    else:
        reason = f"first nightly to diff: no earlier {args.event} run {where}"
    return {"available": False, "reason": reason, "runs_checked": checked}


def job_succeeded(repo: str, run_id: int, job_name: str, token: str) -> bool:
    page = 1
    while True:
        query = urllib.parse.urlencode({"filter": "all", "per_page": 100, "page": page})
        jobs = get(f"{API}/repos/{repo}/actions/runs/{run_id}/jobs?{query}", token)["jobs"]
        if any(j["name"] == job_name and j["conclusion"] == "success" for j in jobs):
            return True
        if len(jobs) < 100:
            return False
        page += 1


def find_baseline(args: argparse.Namespace, token: str) -> dict:
    repo = args.repo
    current = get(f"{API}/repos/{repo}/actions/runs/{args.run_id}", token)
    newest = current["created_at"]
    oldest = (datetime.fromisoformat(newest.replace("Z", "+00:00")) - timedelta(days=args.max_age_days))
    oldest = oldest.strftime("%Y-%m-%dT%H:%M:%SZ")
    checked = 0
    page = 1
    while True:
        query = urllib.parse.urlencode({"branch": args.branch, "event": args.event, "per_page": 100, "page": page})
        runs = get(f"{API}/repos/{repo}/actions/workflows/{args.workflow}/runs?{query}", token)["workflow_runs"]
        for run in runs:
            if str(run["id"]) == str(args.run_id) or run["created_at"] >= newest:
                continue
            if run["created_at"] < oldest:
                return unavailable(args, checked)
            checked += 1
            # Only evidence that reached release-automation counts as the previous submission.
            if not job_succeeded(repo, run["id"], args.require_job, token):
                continue
            name = args.artifact.format(sha=run["head_sha"])
            query = urllib.parse.urlencode({"name": name, "per_page": 100})
            artifacts = get(f"{API}/repos/{repo}/actions/runs/{run['id']}/artifacts?{query}", token)["artifacts"]
            artifacts = [a for a in artifacts if a["name"] == name and not a.get("expired")]
            if not artifacts:
                continue
            artifact = max(artifacts, key=lambda a: a["created_at"])
            data = get(artifact["archive_download_url"], token, raw=True)
            dest = Path(args.dest)
            dest.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                zf.extractall(dest)
            return {"available": True, "repo": repo, "dir": str(dest), "artifact": name,
                    "run_id": run["id"], "run_url": run["html_url"], "head_sha": run["head_sha"],
                    "created_at": run["created_at"]}
        if len(runs) < 100:
            return unavailable(args, checked)
        page += 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--run-id", required=True, help="the current run, excluded from the search")
    parser.add_argument("--artifact", required=True, help="artifact name with {sha} for the run's head SHA")
    parser.add_argument("--dest", required=True)
    parser.add_argument("--info", required=True, help="JSON file describing the baseline, read by osrb.py")
    parser.add_argument("--workflow", default="nightly-ci.yml")
    parser.add_argument("--branch", default="main")
    parser.add_argument("--event", default="schedule")
    parser.add_argument("--max-age-days", type=int, default=35)
    parser.add_argument("--require-job", required=True, help="job that must have succeeded in the baseline run")
    args = parser.parse_args()
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        fail("GH_TOKEN is not set")
    try:
        info = find_baseline(args, token)
    except (OSError, http.client.HTTPException, KeyError, ValueError, zipfile.BadZipFile) as exc:
        fail(f"baseline lookup for {args.artifact} failed: {exc}")
    Path(args.info).parent.mkdir(parents=True, exist_ok=True)
    Path(args.info).write_text(json.dumps(info, indent=2) + "\n")
    print(f"baseline for {args.artifact}: {json.dumps(info)}")


if __name__ == "__main__":
    main()
