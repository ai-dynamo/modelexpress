# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The comment boundary must never authorize an unapproved PR revision."""

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "comment_gate", Path(__file__).resolve().parents[1] / "scripts/comment_gate.py"
)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)
SHA = "a" * 40
REPO = "ai-dynamo/modelexpress"


def request(*, body=None, permission="write", head=SHA, mirror=SHA, state="open"):
    event = {
        "action": "created",
        "issue": {"number": 42, "pull_request": {"url": "https://example.test/pr/42"}},
        "comment": {
            "body": body or f"/bench --sha {SHA}",
            "user": {"login": "maintainer"},
        },
    }
    responses = {
        f"repos/{REPO}/collaborators/maintainer/permission": {"permission": permission},
        f"repos/{REPO}/pulls/42": {"head": {"sha": head}, "state": state},
        f"repos/{REPO}/git/ref/heads/pull-request/42": {"object": {"sha": mirror}},
    }
    return event, responses.__getitem__


def test_authorized_comment_returns_only_approved_immutable_sha():
    event, get = request()
    assert gate.authorize(event, REPO, get) == {"sha": SHA, "model": "nemotron"}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"permission": "read"},
        {"permission": "triage"},
        {"head": "b" * 40},
        {"mirror": "b" * 40},
        {"state": "closed"},
        {"body": "/bench --sha main"},
        {"body": "/bench --sha " + SHA + "\necho hacked"},
        {"body": "/bench --sha " + SHA[:7]},
        {"body": "/ok to test " + SHA},
        {"body": "/e2e-test"},
        {"body": "/bench-test"},
    ],
)
def test_invalid_or_unapproved_request_is_rejected(kwargs):
    event, get = request(**kwargs)
    with pytest.raises(ValueError):
        gate.authorize(event, REPO, get)


def test_missing_mirror_fails_closed():
    event, get = request()

    def missing(path):
        if "/git/ref/" in path:
            raise OSError("404")
        return get(path)

    with pytest.raises(OSError):
        gate.authorize(event, REPO, missing)


@pytest.mark.parametrize("change", ["issue", "edited"])
def test_only_new_pr_comments_are_accepted(change):
    event, get = request()
    if change == "issue":
        del event["issue"]["pull_request"]
    else:
        event["action"] = "edited"
    with pytest.raises(ValueError):
        gate.authorize(event, REPO, get)


@pytest.mark.parametrize(
    "command,model", [("/bench", "nemotron"), ("/bench --model kimi", "kimi")]
)
def test_command_resolves_current_approved_head_and_model(command, model):
    event, get = request(body=command)
    assert gate.authorize(event, REPO, get) == {"sha": SHA, "model": model}


@pytest.mark.parametrize(
    "command",
    [
        "/bench --model unknown",
        "/bench --model ../kimi",
        "/bench --paths both",
    ],
)
def test_unsupported_options_fail_closed(command):
    event, get = request(body=command)
    with pytest.raises(ValueError):
        gate.authorize(event, REPO, get)
