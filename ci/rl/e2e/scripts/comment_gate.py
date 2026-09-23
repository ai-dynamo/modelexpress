"""Authorize an exact, already mirrored PR revision before privileged CI starts."""

import argparse
import json
import os
import re
import shlex
import urllib.request
from pathlib import Path


def authorize(event, repository, get):
    if event.get("action") != "created" or not event.get("issue", {}).get(
        "pull_request"
    ):
        raise ValueError("Expected a newly created PR comment")
    catalog = json.loads(
        (Path(__file__).resolve().parents[1] / "profiles.json").read_text()
    )
    parser = argparse.ArgumentParser(
        prog="/e2e-test", exit_on_error=False, add_help=False, allow_abbrev=False
    )
    parser.add_argument(
        "--model", choices=catalog["models"], default=catalog["default_model"]
    )
    parser.add_argument("--sha")
    words = shlex.split(event["comment"]["body"].strip())
    if not words or words[0] != "/e2e-test" or "\n" in event["comment"]["body"].strip():
        raise ValueError("Use /e2e-test [--model PROFILE] [--sha FULL_SHA]")
    try:
        args, unknown = parser.parse_known_args(words[1:])
    except argparse.ArgumentError as error:
        raise ValueError(str(error)) from error
    if unknown or (args.sha and not re.fullmatch(r"[0-9a-f]{40}", args.sha)):
        raise ValueError("Use /e2e-test [--model PROFILE] [--sha FULL_SHA]")
    number = event["issue"]["number"]
    login = event["comment"]["user"]["login"]
    permission = get(f"repos/{repository}/collaborators/{login}/permission")
    if permission["permission"] not in {"write", "maintain", "admin"}:
        raise ValueError("The commenter must have repository write permission")
    pr = get(f"repos/{repository}/pulls/{number}")
    sha = args.sha or pr["head"]["sha"]
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError("Invalid PR head SHA")
    if pr["state"] != "open" or pr["head"]["sha"] != sha:
        raise ValueError("The command must name the current head of an open PR")
    mirror = get(f"repos/{repository}/git/ref/heads/pull-request/{number}")
    if mirror["object"]["sha"] != sha:
        raise ValueError(
            "Wait for copy-pr-bot approval/mirroring of this SHA, then comment again"
        )
    return {"sha": sha, "model": args.model}


def main():
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

    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
    result = authorize(event, os.environ["GITHUB_REPOSITORY"], get)
    with Path(os.environ["GITHUB_OUTPUT"]).open("a") as output:
        output.writelines(f"{key}={value}\n" for key, value in result.items())
    print(f"Authorized E2E test: {result}")


if __name__ == "__main__":
    main()
