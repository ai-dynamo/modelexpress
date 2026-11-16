# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check requests using the pinned AIPerf dataset loader and endpoint formatter."""

import json
from types import SimpleNamespace

import pytest
from aiperf.accuracy.benchmarks.mmlu import GENERATION_SIZE, STOP_SEQUENCE
from aiperf.accuracy.models import BenchmarkProblem
from aiperf.common.models import ModelEndpointInfo, RequestInfo
from aiperf.dataset.loader.accuracy_dataset_loader import AccuracyDatasetLoader
from aiperf.endpoints.openai_chat import ChatEndpoint

import inference_parity


@pytest.mark.parametrize("model", ["zai-org/GLM-4.7-Flash", "Qwen/Qwen3.5-0.8B"])
@pytest.mark.parametrize("worker_id", [None, 42])
def test_mmlu_request_preserves_turn_boundary_stop(monkeypatch, tmp_path, model, worker_id):
    commands = []

    def capture(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(wait=lambda timeout: 1)

    monkeypatch.setattr(inference_parity.subprocess, "Popen", capture)
    with pytest.raises(AssertionError, match="AIPerf failed"):
        inference_parity.profile("http://fixture", model, tmp_path / "profile", worker_id=worker_id)
    command = commands[0]
    extra = json.loads(command[command.index("--extra-inputs") + 1])
    run = SimpleNamespace(random_seed=0, cfg=SimpleNamespace(
        get_default_dataset=lambda: SimpleNamespace(random_seed=0),
        accuracy=SimpleNamespace(system_prompt=inference_parity.SYSTEM_PROMPT),
    ))
    problem = BenchmarkProblem(
        prompt="Question: What is 2 + 2?\nA. 3\nB. 4\nC. 5\nD. 6\nAnswer:",
        ground_truth="B", task="elementary_mathematics",
        metadata={"stop_sequence": STOP_SEQUENCE, "generation_size": GENERATION_SIZE},
    )
    conversation = AccuracyDatasetLoader(run=run)._convert_to_conversations([problem])[0]
    endpoint = ModelEndpointInfo(
        models={"models": [{"name": model}], "model_selection_strategy": "round_robin"},
        endpoint={"streaming": False, "extra": list(extra.items())},
    )
    request = RequestInfo.model_construct(turns=conversation.turns, model_endpoint=endpoint)
    payload = ChatEndpoint(endpoint).format_payload(request)

    assert payload["stop"] == ["\n"]
    assert payload["max_completion_tokens"] == inference_parity.ANSWER_TOKENS
    assert payload["messages"] == conversation.turns[0].raw_messages
    assert payload["temperature"] == 0 and payload["seed"] == 0
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert "stop_regex" not in payload
    if worker_id is not None:
        assert payload["nvext"]["decode_worker_id"] == worker_id
        assert "backend_instance_id" not in payload["nvext"]
    else:
        assert "nvext" not in payload
