# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Require a bounded, deterministic decode after the streamer path check."""

import json
import os
import urllib.request

TOKENS = 32


def check_inference(url: str, model: str) -> dict:
    """Require one nonblank completion that reaches the requested token limit."""
    payload = {
        "model": model,
        "prompt": "The capital of France is",
        "max_tokens": TOKENS,
        "temperature": 0,
        "seed": 0,
        "ignore_eos": True,
    }
    request = urllib.request.Request(
        f"{url}/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        body = json.load(response)
    print("Response:", json.dumps(body))
    choices = body.get("choices", [])
    assert len(choices) == 1, f"Expected one completion: {body}"
    text = choices[0].get("text")
    assert isinstance(text, str) and text.strip(), f"No completion text: {body}"
    assert choices[0].get("finish_reason") == "length", f"Decode stopped early: {body}"
    assert body.get("usage", {}).get("completion_tokens") == TOKENS, (
        f"Expected {TOKENS} generated tokens: {body}"
    )
    print("Inference OK:", repr(text[:60]))
    return body


if __name__ == "__main__":
    check_inference(f"http://localhost:{int(os.environ['PORT'])}", os.environ["MODEL"])
