# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib
import importlib.util
import io
import json
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

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


if __name__ == "__main__":
    unittest.main()
