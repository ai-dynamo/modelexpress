# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib
import importlib.util
import io
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

import yaml

spec = importlib.util.spec_from_file_location(
    "check_streamer_inference", Path(__file__).resolve().parents[1] / "check_streamer_inference.py"
)
streamer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(streamer)


class StreamerInferenceTests(unittest.TestCase):
    def response(self, **changes):
        return {
            "choices": [{"text": " Paris", "finish_reason": "length"}],
            "usage": {"completion_tokens": 32},
            **changes,
        }

    def check(self, body):
        with patch.object(
            streamer.urllib.request, "urlopen",
            return_value=io.BytesIO(json.dumps(body).encode()),
        ) as request:
            with contextlib.redirect_stdout(io.StringIO()):
                streamer.check_inference("http://localhost:18888", "Qwen/Qwen3.5-0.8B")
        return request

    def test_request_is_deterministic_and_requires_a_full_decode(self):
        request = self.check(self.response())
        payload = json.loads(request.call_args.args[0].data)
        self.assertEqual(payload["temperature"], 0)
        self.assertEqual(payload["seed"], 0)
        self.assertTrue(payload["ignore_eos"])
        self.assertEqual(payload["max_tokens"], 32)
        self.assertEqual(request.call_args.kwargs["timeout"], 60)

    def test_empty_early_stopped_and_invalid_responses_still_fail(self):
        for changes in [
            {"choices": []},
            {"choices": [{"text": "", "finish_reason": "stop", "matched_stop": 248046}],
             "usage": {"completion_tokens": 1}},
            {"choices": [{"text": "  ", "finish_reason": "length"}]},
            {"choices": [{"text": " Paris", "finish_reason": "stop"}]},
            {"usage": {"completion_tokens": 1}},
            {"usage": {}},
        ]:
            with self.subTest(changes=changes), self.assertRaises(AssertionError):
                self.check(self.response(**changes))

    def test_http_and_malformed_json_errors_propagate(self):
        with patch.object(
            streamer.urllib.request, "urlopen",
            side_effect=urllib.error.HTTPError("url", 500, "failed", {}, None),
        ):
            with self.assertRaises(urllib.error.HTTPError):
                streamer.check_inference("http://localhost:18888", "model")
        with patch.object(streamer.urllib.request, "urlopen", return_value=io.BytesIO(b"not json")):
            with self.assertRaises(json.JSONDecodeError):
                streamer.check_inference("http://localhost:18888", "model")


class StreamerLaunchTests(unittest.TestCase):
    def test_load_wait_requires_the_selected_loading_path(self):
        action = Path(__file__).resolve().parents[2] / "actions/run-mx-streamer-test/action.yml"
        steps = yaml.safe_load(action.read_text())["runs"]["steps"]
        script = next(step["run"] for step in steps if step["name"] == "Wait for model load to complete")
        for fallback, logs, expected in [
            ("false", "Model streamer weight loading complete", 0),
            ("true", "Strategy model_streamer failed\nWeights loaded from disk", 0),
            ("true", "Strategy model_streamer failed", 1),
            ("true", "Weights loaded from disk", 1),
            ("false", "server crashed", 1),
        ]:
            with self.subTest(fallback=fallback, logs=logs), tempfile.TemporaryDirectory() as directory:
                log = Path(directory) / "server.log"
                log.write_text(logs)
                result = subprocess.run(
                    ["bash", "-e", "-o", "pipefail"], input=script,
                    text=True, capture_output=True, timeout=5,
                    env={**os.environ, "STREAMER_LOG": str(log), "STREAMER_PID": "2147483647",
                         "LOAD_TIMEOUT": "1", "TEST_MODEL_STREAMER_FALLBACK": fallback},
                )
                self.assertEqual(result.returncode, expected, result.stderr)
                if expected:
                    self.assertIn(logs, result.stdout)

    def test_launches_engines_in_job_container_and_cleans_up(self):
        action = Path(__file__).resolve().parents[2] / "actions/run-mx-streamer-test/action.yml"
        steps = yaml.safe_load(action.read_text())["runs"]["steps"]
        cleanup = next(step["run"] for step in steps if step["name"] == "Cleanup server")
        for framework in ["vllm", "sglang"]:
            launch = next(step["run"] for step in steps
                          if step.get("if") == f"inputs.framework == '{framework}'")
            for fallback in ["false", "true"]:
                with self.subTest(framework=framework, fallback=fallback), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    python = root / "python3"
                    python.write_text(
                        f"#!{sys.executable}\n"
                        "import json, os, pathlib, sys, time\n"
                        "path = pathlib.Path(os.environ['CAPTURE'])\n"
                        "path.with_suffix('.tmp').write_text(json.dumps({\n"
                        "    'args': sys.argv[1:], 'env': dict(os.environ)}))\n"
                        "path.with_suffix('.tmp').replace(path)\n"
                        "print('Model streamer weight loading complete', flush=True)\n"
                        "time.sleep(30)\n"
                    )
                    python.chmod(0o755)
                    environment = {
                        **os.environ, "PATH": f"{directory}:{os.environ['PATH']}",
                        "CAPTURE": str(root / "capture.json"), "GITHUB_ENV": str(root / "env"),
                        "STREAMER_LOG": str(root / "server.log"), "MODEL": "test/model",
                        "MODEL_S3_URI": "s3://bucket/models/test/model", "PORT": "18888",
                        "S3_REGION": "us-west-2", "STREAMER_CONCURRENCY": "16",
                        "TEST_MODEL_STREAMER_FALLBACK": fallback, "HF_TOKEN": "test-token",
                        "AWS_ROLE_ARN": "test-role", "AWS_WEB_IDENTITY_TOKEN_FILE": "/test/token",
                    }
                    result = subprocess.run(["bash", "-e", "-o", "pipefail"], input=launch,
                                            text=True, capture_output=True, env=environment, timeout=5)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    pid = dict(line.split("=", 1) for line in (root / "env").read_text().splitlines())["STREAMER_PID"]
                    try:
                        deadline = time.monotonic() + 5
                        while not (root / "capture.json").exists() and time.monotonic() < deadline:
                            time.sleep(0.01)
                        captured = json.loads((root / "capture.json").read_text())
                        args = captured["args"]
                        flag = "--model" if framework == "vllm" else "--model-path"
                        expected = "s3://bucket/models/test/model" if framework == "vllm" and fallback == "false" else "test/model"
                        self.assertEqual(args[args.index(flag) + 1], expected)
                        self.assertEqual(captured["env"]["MX_MODEL_URI"], environment["MODEL_S3_URI"])
                        self.assertEqual(captured["env"]["AWS_WEB_IDENTITY_TOKEN_FILE"], "/test/token")
                        self.assertEqual(captured["env"]["HF_TOKEN"], "test-token")
                        result = subprocess.run(["bash", "-e", "-o", "pipefail"], input=cleanup,
                                                text=True, capture_output=True,
                                                env={**environment, "STREAMER_PID": pid}, timeout=5)
                        self.assertEqual(result.returncode, 0, result.stderr)
                        self.assertIn("Model streamer weight loading complete", result.stdout)
                    finally:
                        try:
                            os.kill(int(pid), signal.SIGKILL)
                        except ProcessLookupError:
                            pass


if __name__ == "__main__":
    unittest.main()
