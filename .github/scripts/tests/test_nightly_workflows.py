# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import yaml

WORKFLOWS = Path(__file__).resolve().parents[2] / "workflows"


def workflow(name):
    return yaml.load((WORKFLOWS / name).read_text(), Loader=yaml.BaseLoader)


def execute(script, environment):
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "output"
        result = subprocess.run(
            ["bash", "-e", "-o", "pipefail"], input=script, text=True,
            capture_output=True,
            env={**os.environ, **environment, "GITHUB_OUTPUT": str(output)},
        )
        values = dict(line.split("=", 1) for line in output.read_text().splitlines()) if output.exists() else {}
        return result, values


class NightlyWorkflowTests(unittest.TestCase):
    def test_only_coverage_receives_oidc_permission(self):
        core = workflow("modelexpress-core-ci.yml")
        self.assertEqual(core.get("permissions"), {"contents": "read"})
        oidc_jobs = [
            name for name, job in core["jobs"].items()
            if job.get("permissions", core["permissions"]).get("id-token") == "write"
        ]
        self.assertEqual(oidc_jobs, ["code-coverage"])

    def test_slack_failure_payload_preserves_quotes_backslashes_and_newlines(self):
        steps = workflow("notify-slack.yml")["jobs"]["notify-slack"]["steps"]
        names = ['Test "quoted" path C:\\models\nanother line', "second lane"]
        jobs = {"jobs": [{"name": name, "status": "completed", "conclusion": "failure"} for name in names]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "jobs.json").write_text(json.dumps(jobs))
            script = 'curl() { cat "$JOB_RESPONSE"; }\n' + steps[0]["run"]
            script = script.replace("${{ github.repository }}", "example/repo").replace("${{ github.run_id }}", "123")
            result, values = execute(script, {
                "JOB_RESPONSE": str(root / "jobs.json"), "GITHUB_ENV": str(root / "env"),
                "GITHUB_TOKEN": "test-token", "IGNORED_JOBS": "", "FORCE_ALERT": "false",
            })
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(values["has_failures"], "true")
            failed = (root / "env").read_text().split("FAILED_JOBS<<EOF\n", 1)[1].rsplit("\nEOF\n", 1)[0]
        payload = steps[1]["with"]["payload"]
        payload = payload.replace("${{ toJSON(env.FAILED_JOBS) }}", json.dumps(failed))
        payload = payload.replace("${{ env.FAILED_JOBS }}", failed)
        parsed = yaml.safe_load(payload)
        self.assertEqual(parsed["blocks"][2]["text"]["text"], "\n".join(f":failed: {name}" for name in names))

    def test_preflight_rejects_forks_and_non_main_refs(self):
        script = workflow("nightly-ci.yml")["jobs"]["preflight"]["steps"][0]["run"]
        for repository, ref, expected in [
            ("ai-dynamo/modelexpress", "refs/heads/main", 0),
            ("contributor/modelexpress", "refs/heads/main", 1),
            ("ai-dynamo/modelexpress", "refs/heads/feature", 1),
            ("ai-dynamo/modelexpress", "refs/tags/main", 1),
        ]:
            with self.subTest(repository=repository, ref=ref):
                result, _ = execute(script, {"REPOSITORY": repository, "REF": ref})
                self.assertEqual(result.returncode, expected, result.stderr)

    def test_gpu_slots_support_main_runs_and_preserve_pr_assignment(self):
        inference = workflow("modelexpress-inference-gpu-ci.yml")["jobs"]["admission-slots"]["steps"][0]["run"]
        rl = workflow("modelexpress-rl-ci.yml")["jobs"]
        validate = next(step["run"] for step in rl["validate"]["steps"] if step.get("id") == "inputs")
        validate = "python3 - <<'PYTHON' >> \"$GITHUB_OUTPUT\"\n" + validate.split("python3 - <<'PYTHON' >> \"$GITHUB_OUTPUT\"\n", 1)[1]
        vime = rl["build-vime-delta-refit"]["steps"][0]["run"]
        for event, ref, seed in [
            ("schedule", "refs/heads/main", 1001),
            ("workflow_dispatch", "refs/heads/main", 1001),
            ("push", "refs/heads/pull-request/42", 42),
        ]:
            with self.subTest(event=event):
                environment = {
                    "GITHUB_EVENT_NAME": event, "GITHUB_REF": ref,
                    "GITHUB_RUN_ID": "1001", "PR_NUMBER": "0",
                    "MX_CI_GPU_CONCURRENCY": "8", "MX_CI_A100B_CONCURRENCY": "4",
                    "MX_CI_RL_GPU_CONCURRENCY": "4",
                }
                result, values = execute(inference, environment)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(json.loads(values["slots"]), [(seed + i) % 8 for i in range(16)])
                self.assertEqual(json.loads(values["a100b_slots"]), [(seed + i) % 4 for i in range(4)])
                result, values = execute(validate, environment)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(values["gpu_slot"], str(seed % 4))
                result, values = execute(vime, {**environment, "GPU_SLOT": values["gpu_slot"]})
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(values["gpu_slot"], str((seed + 1) % 4))

    def test_status_requires_every_enabled_suite(self):
        status = workflow("nightly-ci.yml")["jobs"]["nightly-status"]
        results = {job: {"result": "success"} for job in status["needs"]}
        results["manual-release-approval"]["result"] = "skipped"
        environment = {"RELEASE": "false", "RUN_TESTS": "true", "SKIP_GITLAB": "false"}
        script = status["steps"][0]["run"]
        result, _ = execute(script, {**environment, "RESULTS": json.dumps(results)})
        self.assertEqual(result.returncode, 0, result.stderr)
        for suite in ["core-tests", "inference-tests", "rl-tests"]:
            for conclusion in ["skipped", "failure", "cancelled"]:
                with self.subTest(suite=suite, conclusion=conclusion):
                    changed = {**results, suite: {"result": conclusion}}
                    result, _ = execute(script, {**environment, "RESULTS": json.dumps(changed)})
                    self.assertNotEqual(result.returncode, 0)
        for suite in ["core-tests", "inference-tests", "rl-tests"]:
            results[suite]["result"] = "skipped"
        result, _ = execute(script, {**environment, "RUN_TESTS": "false", "RESULTS": json.dumps(results)})
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
