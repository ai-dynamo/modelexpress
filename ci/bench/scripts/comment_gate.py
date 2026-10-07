# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Authorize comments and trusted mirror pushes before privileged CI starts."""

import argparse
import json
import os
import re
import shlex
import urllib.request
from pathlib import Path


def authorize_revision(repository, number, sha, get):
    pr = get(f"repos/{repository}/pulls/{number}")
    sha = sha or pr["head"]["sha"]
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError("Invalid PR head SHA")
    if pr["state"] != "open" or pr["head"]["sha"] != sha:
        raise ValueError("The run must name the current head of an open PR")
    mirror = get(f"repos/{repository}/git/ref/heads/pull-request/{number}")
    if mirror["object"]["sha"] != sha:
        raise ValueError("Wait for copy-pr-bot approval/mirroring of this SHA")
    return sha


def authorize(event, repository, get):
    if event.get("action") != "created" or not event.get("issue", {}).get(
        "pull_request"
    ):
        raise ValueError("Expected a newly created PR comment")
    catalog = json.loads(
        (Path(__file__).resolve().parents[1] / "profiles.json").read_text()
    )
    parser = argparse.ArgumentParser(
        prog="/bench", exit_on_error=False, add_help=False, allow_abbrev=False
    )
    parser.add_argument(
        "--model", choices=catalog["models"], default=catalog["default_model"]
    )
    parser.add_argument(
        "--scenario", choices=catalog["scenarios"], default=catalog["default_scenario"]
    )
    parser.add_argument("--sha")
    words = shlex.split(event["comment"]["body"].strip())
    if not words or words[0] != "/bench" or "\n" in event["comment"]["body"].strip():
        raise ValueError(
            "Use /bench [--model PROFILE] [--scenario SCENARIO] [--sha FULL_SHA]"
        )
    try:
        args, unknown = parser.parse_known_args(words[1:])
    except argparse.ArgumentError as error:
        raise ValueError(str(error)) from error
    if unknown or (args.sha and not re.fullmatch(r"[0-9a-f]{40}", args.sha)):
        raise ValueError(
            "Use /bench [--model PROFILE] [--scenario SCENARIO] [--sha FULL_SHA]"
        )
    number = event["issue"]["number"]
    login = event["comment"]["user"]["login"]
    permission = get(f"repos/{repository}/collaborators/{login}/permission")
    if permission["permission"] not in {"write", "maintain", "admin"}:
        raise ValueError("The commenter must have repository write permission")
    sha = authorize_revision(repository, number, args.sha, get)
    return {"sha": sha, "model": args.model, "scenario": args.scenario}


def authorize_workflow(environment, get):
    for key in [
        "KUBE_CONTEXT",
        "MX_BENCH_S3_ROLE_ARN",
        "MX_CI_S3_BUCKET",
        "MX_CI_S3_REGION",
    ]:
        if not environment.get(key):
            raise ValueError("Missing repository configuration: " + key)
    sha = environment.get("TEST_SHA", "")
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError("sha must be a full lowercase commit SHA")
    runtime = environment.get("RUNTIME_BASE", "")
    if not re.fullmatch(r"[a-zA-Z0-9./:_-]+@sha256:[0-9a-f]{64}", runtime):
        raise ValueError("runtime must be digest-pinned")
    profiles = Path(__file__).resolve().parents[1]
    catalog = json.loads((profiles / "profiles.json").read_text())
    model, scenario = environment["MODEL_PROFILE"], environment["SCENARIO"]
    if model not in catalog["models"]:
        for key in catalog["models"]:
            profile = json.loads(
                (profiles / "profiles" / key / "profile.json").read_text()
            )
            if model == profile["model"]:
                model = key
                break
    if model not in catalog["models"] or scenario not in catalog["scenarios"]:
        raise ValueError("Unknown model or scenario")
    if environment.get("GITHUB_EVENT_NAME") == "push":
        match = re.fullmatch(
            r"refs/heads/pull-request/([1-9][0-9]*)", environment.get("GITHUB_REF", "")
        )
        if (
            environment["GITHUB_REPOSITORY"] != "ai-dynamo/modelexpress"
            or not match
            or environment.get("GITHUB_SHA") != sha
        ):
            raise ValueError("Expected an exact copy-pr-bot mirror push")
        number = int(match[1])
        requested_number = int(environment["PR_NUMBER"])
        if requested_number not in (0, number):
            raise ValueError("pull_request does not match the mirrored branch")
        authorize_revision(environment["GITHUB_REPOSITORY"], number, sha, get)
        return {"sha": sha, "model": model, "scenario": scenario}
    number = int(environment["PR_NUMBER"])
    if number < 1:
        raise ValueError("pull_request must be a positive PR number")
    event = {
        "action": "created",
        "issue": {"number": number, "pull_request": {"number": number}},
        "comment": {
            "body": shlex.join(
                [
                    "/bench",
                    "--sha",
                    sha,
                    "--model",
                    model,
                    "--scenario",
                    scenario,
                ]
            ),
            "user": {"login": environment["GITHUB_ACTOR"]},
        },
    }
    return authorize(event, environment["GITHUB_REPOSITORY"], get)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow-call", action="store_true")
    args = parser.parse_args()

    def get(path):
        request = urllib.request.Request(
            f"https://api.github.com/{path}",
            headers={
                "Authorization": "Bearer " + os.environ["GH_TOKEN"],
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)

    if args.workflow_call:
        result = authorize_workflow(os.environ, get)
    else:
        event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
        result = authorize(event, os.environ["GITHUB_REPOSITORY"], get)
    with Path(os.environ["GITHUB_OUTPUT"]).open("a") as output:
        output.writelines(f"{key}={value}\n" for key, value in result.items())
    print(f"Authorized benchmark run: {result}")


if __name__ == "__main__":
    main()
