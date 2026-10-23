# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

import yaml

WORKFLOWS = Path(__file__).resolve().parents[2] / "workflows"
ROOT = WORKFLOWS.parents[1]


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
        admission = workflow("modelexpress-inference-gpu-ci.yml")["jobs"]["admission-slots"]
        self.assertEqual(admission["steps"][0]["env"]["MX_CI_A100B_CONCURRENCY"],
                         "${{ vars.MX_CI_A100B_CONCURRENCY || '1' }}")
        inference = admission["steps"][0]["run"]
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
                self.assertEqual(json.loads(values["slots"]), [(seed + i) % 8 for i in range(20)])
                self.assertEqual(json.loads(values["a100b_slots"]), [(seed + i) % 4 for i in range(6)])
                result, values = execute(validate, environment)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(values["gpu_slot"], str(seed % 4))
                result, values = execute(vime, {**environment, "GPU_SLOT": values["gpu_slot"]})
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(values["gpu_slot"], str((seed + 1) % 4))

    def test_inference_tests_wait_only_for_their_worker_image(self):
        jobs = workflow("modelexpress-inference-ci.yml")["jobs"]
        for framework in ["vllm", "sglang", "sglang-mooncake", "dynamo-vllm", "dynamo-sglang"]:
            with self.subTest(framework=framework):
                gpu = jobs[f"gpu-tests-{framework}"]
                self.assertEqual(set(gpu["needs"]), {
                    f"build-client-{framework}", "build-server", "ensure-vcluster",
                })
                self.assertEqual(gpu["with"]["framework"], framework)
        for framework in ["vllm", "sglang"]:
            streamer = jobs[f"model-streamer-{framework}"]
            self.assertEqual(streamer["needs"], [f"build-client-{framework}"])
            self.assertEqual(len(streamer["strategy"]["matrix"]["include"]), 2)
        self.assertEqual(set(jobs["gpu-cleanup"]["needs"]), {
            f"gpu-tests-{framework}"
            for framework in ["vllm", "sglang", "sglang-mooncake", "dynamo-vllm", "dynamo-sglang", "large"]
        })

    def test_workflow_level_supersession_preserves_cleanup(self):
        e2e = workflow("modelexpress-e2e-ci.yml")
        self.assertEqual(e2e["concurrency"], {
            "group": "${{ github.workflow }}-${{ github.ref }}",
            "cancel-in-progress": "true",
        })

        inference = workflow("modelexpress-inference-ci.yml")
        rl = workflow("modelexpress-rl-ci.yml")
        for child in [inference, rl]:
            self.assertNotIn("concurrency_key", child["on"]["workflow_call"]["inputs"])
            for job in child["jobs"].values():
                self.assertNotIn("concurrency", job)

        self.assertIn("always()", inference["jobs"]["gpu-cleanup"]["if"])
        self.assertIn("always()", rl["jobs"]["cleanup"]["if"])
        self.assertEqual(rl["concurrency"], {
            "group": "${{ github.workflow }}-${{ github.event.issue.number || github.run_id }}",
            "cancel-in-progress": "${{ github.event_name == 'issue_comment' }}",
        })

        nightly = workflow("nightly-ci.yml")["jobs"]
        self.assertNotIn("concurrency_key", nightly["inference-tests"]["with"])
        self.assertNotIn("concurrency_key", nightly["rl-tests"]["with"])

    def test_large_gpu_tests_wait_for_all_small_tests(self):
        jobs = workflow("modelexpress-inference-ci.yml")["jobs"]
        large = jobs["gpu-tests-large"]
        small = {f"gpu-tests-{framework}" for framework in [
            "vllm", "sglang", "sglang-mooncake", "dynamo-vllm", "dynamo-sglang",
        ]}
        prerequisites = {
            "build-server", "ensure-vcluster", "build-client-vllm",
            "build-client-sglang", "build-client-dynamo-vllm", "build-client-dynamo-sglang",
        }
        self.assertEqual(set(large["needs"]), small | prerequisites)
        self.assertNotIn("if", large)
        self.assertEqual(large["with"]["framework"], "large")
        self.assertIn("gpu-tests-large", jobs["gpu-cleanup"]["needs"])
        self.assertIn("gpu-tests-large", jobs["status-check"]["needs"])
        gpu = workflow("modelexpress-inference-gpu-ci.yml")["jobs"]["test"]
        self.assertIn("matrix.large && 'modelexpress-gpu-a100a-large-workload'", gpu["concurrency"]["group"])
        self.assertEqual(gpu["concurrency"]["cancel-in-progress"], "false")

    def test_inference_matrix_has_only_real_cases(self):
        jobs = workflow("modelexpress-inference-gpu-ci.yml")["jobs"]
        self.assertEqual(set(jobs), {"admission-slots", "test"})
        test = jobs["test"]
        self.assertNotIn("if", test)
        include = test["strategy"]["matrix"]["include"]
        match = re.search(r"fromJSON\('((?:''|[^'])*)'\)", include)
        partitions = json.loads(match[1].replace("''", "'"))
        self.assertEqual({key: len(rows) for key, rows in partitions.items()}, {
            "vllm": 10, "sglang": 6, "sglang-mooncake": 1, "dynamo-vllm": 1, "dynamo-sglang": 1, "large": 7,
        })
        namespaces = set()
        for framework, cases in partitions.items():
            for case in cases:
                if framework == "large":
                    self.assertTrue(case["large"])
                else:
                    self.assertEqual(case["framework"], framework)
                    self.assertFalse(case.get("large", False))
                self.assertNotIn("${{", case["name"])
                self.assertNotIn("${{", case["namespace"])
                self.assertNotIn(case["namespace"], namespaces)
                namespaces.add(case["namespace"])
                self.assertIn(case["kind"], {"p2p", "fleet", "dynamo", "stale"})
                if case["kind"] == "dynamo":
                    suffix = "-39868265163-10"
                    grove_label = f"{case['namespace']}{suffix}-{case['dgd_name']}{suffix}"
                    self.assertLessEqual(len(grove_label), 63, grove_label)
        large = [case for cases in partitions.values() for case in cases if case.get("large")]
        self.assertEqual(sorted(case["fleet_size"] for case in large if case["kind"] == "fleet"), ["15", "5", "5"])
        sglang_mla = next(case for case in partitions["sglang"] if case.get("measure_vram"))
        self.assertEqual(sglang_mla["pytest_args"], "--expected-max-total-tokens 131072")
        self.assertEqual(
            {case["name"] for case in large if case["kind"] != "fleet"},
            {
                "P2P TP test (vllm TP=2 multi-node)",
                "P2P TP test (sglang TP=2 multi-node)",
                "P2P test with Dynamo (sglang disaggregated)",
                "P2P test with Dynamo (disaggregated)",
            },
        )
        # Select the pool's slot array before indexing: slot zero must not
        # fall through to the A100a array under expression truthiness rules.
        lane = test["env"]["MX_CI_GPU_LANE"]
        self.assertIn("a100b_slots) || fromJSON(needs.admission-slots.outputs.slots))[matrix.gpu_slot]", lane)

    def test_sglang_mla_uses_fixed_token_pool_and_strict_vram_parity(self):
        manifest = yaml.load(
            (ROOT / "ci/k8s/client/sgl/manifest-azure-tp2.yaml").read_text(),
            Loader=yaml.BaseLoader,
        )
        args = manifest["spec"]["template"]["spec"]["containers"][0]["args"]
        cap_index = args.index("--max-total-tokens")
        self.assertEqual(args[cap_index + 1], "131072")

        action = yaml.load(
            (ROOT / ".github/actions/run-mx-p2p-test/action.yml").read_text(),
            Loader=yaml.BaseLoader,
        )
        wait = next(
            step
            for step in action["runs"]["steps"]
            if step.get("name") == "Wait for target /health to return 200"
        )
        self.assertIn('\n    "0.5"\n', wait["run"])

    def test_inference_status_rejects_missing_or_failed_frameworks(self):
        status = workflow("modelexpress-inference-ci.yml")["jobs"]["status-check"]
        results = {job: {"result": "success"} for job in status["needs"]}
        results["gpu-cleanup"]["result"] = "skipped"
        script = status["steps"][0]["run"]
        def check(values):
            command = script.replace("${{ toJson(needs) }}", json.dumps(values))
            return execute(command, {})[0]
        self.assertEqual(check(results).returncode, 0)
        for job in status["needs"]:
            if job == "gpu-cleanup":
                continue
            for result in ["failure", "cancelled", "skipped"]:
                with self.subTest(job=job, result=result):
                    changed = {**results, job: {"result": result}}
                    self.assertNotEqual(check(changed).returncode, 0)
        self.assertNotEqual(check({**results, "gpu-cleanup": {"result": "failure"}}).returncode, 0)

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
