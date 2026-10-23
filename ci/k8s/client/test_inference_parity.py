# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import json
import io
import urllib.error
from contextlib import nullcontext

import pytest

import inference_parity
import test_dynamo_p2p


CASE = f"{inference_parity.SUBJECTS[0]}/session_000000"


@pytest.fixture
def records():
    return [
        {
            "metadata": {"conversation_id": f"session_{index:06d}"},
            "payload": {"messages": [{"role": "user", "content": f"fixture {index}"}]},
            "status": 200,
            "error": None,
            "responses": [{"text": json.dumps({
                "choices": [{"message": {"content": f"answer {index}"}, "finish_reason": "stop"}],
                "usage": {"completion_tokens": 3},
                "nvext": {"worker_id": {"decode_worker_id": 42}},
            })}],
        }
        for index in range(inference_parity.SAMPLES_PER_SUBJECT)
    ]


def write_records(directory, records):
    for subject in inference_parity.SUBJECTS:
        output = directory / subject
        output.mkdir(exist_ok=True)
        inputs = {"data": [
            {"session_id": f"session_{index:06d}",
             "payloads": [{"messages": [{"role": "user", "content": f"fixture {index}"}]}]}
            for index in range(inference_parity.SAMPLES_PER_SUBJECT)
        ]}
        grades = [
            {"conversation_id": f"session_{index:06d}", "task": subject, "expected": "A",
             "actual": "A", "unparsed": False, "passed": True, "model_output": f"answer {index}"}
            for index in range(inference_parity.SAMPLES_PER_SUBJECT)
        ]
        (output / "inputs.json").write_text(json.dumps(inputs))
        (output / "accuracy_export.jsonl").write_text("".join(json.dumps(grade) + "\n" for grade in grades))
        (output / "profile_export_raw.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records)
        )


def test_accepts_complete_export_in_any_order(tmp_path, records):
    write_records(tmp_path, records[::-1])
    responses = inference_parity.read_responses(tmp_path, worker_id=42)
    assert len(responses) == inference_parity.SAMPLE_COUNT
    inference_parity.compare(responses, copy.deepcopy(responses), tmp_path / "comparison.json")
    assert json.loads((tmp_path / "comparison.json").read_text())["matched"] == inference_parity.SAMPLE_COUNT


def test_wording_changes_do_not_fail_accuracy_comparison(tmp_path, records):
    write_records(tmp_path, records)
    source = inference_parity.read_responses(tmp_path)
    target = copy.deepcopy(source)
    target[CASE]["text"] = "Another wording with the same correct answer"
    inference_parity.compare(source, target, tmp_path / "comparison.json")


@pytest.mark.parametrize("lost", [0, 1, 6])
def test_small_paired_score_variation_is_reported_without_failing(tmp_path, records, lost):
    write_records(tmp_path, records)
    source = inference_parity.read_responses(tmp_path)
    target = copy.deepcopy(source)
    for case in list(target)[:lost]:
        target[case]["passed"] = False
    inference_parity.compare(source, target, tmp_path / "comparison.json")
    report = json.loads((tmp_path / "comparison.json").read_text())
    assert report["score"]["regressions"] == lost


def test_significant_paired_score_regression_fails(tmp_path, records):
    write_records(tmp_path, records)
    source = inference_parity.read_responses(tmp_path)
    target = copy.deepcopy(source)
    for case in list(target)[:7]:
        target[case]["passed"] = False
    with pytest.raises(AssertionError, match="accuracy regressed"):
        inference_parity.compare(source, target, tmp_path / "comparison.json")
    report = json.loads((tmp_path / "comparison.json").read_text())
    assert report["score"]["paired_p_value"] == 1 / 128


def test_score_improvements_and_offsetting_changes_are_allowed(tmp_path, records):
    write_records(tmp_path, records)
    source = inference_parity.read_responses(tmp_path)
    target = copy.deepcopy(source)
    cases = list(source)
    for case in cases[:8]:
        source[case].update(passed=False, actual="B")
    for case in cases[8:12]:
        target[case].update(passed=False, actual="B")
    inference_parity.compare(source, target, tmp_path / "comparison.json")
    score = json.loads((tmp_path / "comparison.json").read_text())["score"]
    assert (score["regressions"], score["improvements"]) == (4, 8)


def test_unparseable_answers_cannot_hide_behind_equal_scores(tmp_path, records):
    write_records(tmp_path, records)
    source = inference_parity.read_responses(tmp_path)
    for row in source.values():
        row.update(passed=False, actual="", unparsed=True)
    with pytest.raises(AssertionError, match="answer parsing failed"):
        inference_parity.compare(source, copy.deepcopy(source), tmp_path / "comparison.json")


def test_missing_target_samples_fail_before_score_comparison(tmp_path, records):
    write_records(tmp_path, records)
    source = inference_parity.read_responses(tmp_path)
    target = copy.deepcopy(source)
    del target[CASE]
    with pytest.raises(AssertionError, match="dataset differs"):
        inference_parity.compare(source, target, tmp_path / "comparison.json")


@pytest.mark.parametrize("failure", ["missing", "duplicate", "unexpected", "error", "http", "prompt"])
def test_rejects_incomplete_or_failed_requests(tmp_path, records, failure):
    if failure == "missing":
        records.pop()
    elif failure == "duplicate":
        records.append(records[0])
    elif failure == "unexpected":
        records[0]["metadata"]["conversation_id"] = "unexpected"
    elif failure == "error":
        records[0]["error"] = {"message": "connection reset"}
    elif failure == "http":
        records[0]["status"] = 500
    else:
        records[0]["payload"]["messages"] = [{"role": "user", "content": "changed"}]
    write_records(tmp_path, records)
    with pytest.raises(AssertionError):
        inference_parity.read_responses(tmp_path)


@pytest.mark.parametrize("failure", ["empty", "finish", "tokens", "routing"])
def test_rejects_invalid_completions_and_wrong_worker(tmp_path, records, failure):
    body = json.loads(records[0]["responses"][0]["text"])
    if failure == "empty":
        body["choices"][0]["message"]["content"] = "  "
    elif failure == "finish":
        body["choices"][0]["finish_reason"] = "error"
    elif failure == "tokens":
        body["usage"]["completion_tokens"] = 0
    else:
        body["nvext"]["worker_id"]["decode_worker_id"] = 43
    records[0]["responses"][0]["text"] = json.dumps(body)
    write_records(tmp_path, records)
    with pytest.raises(AssertionError):
        inference_parity.read_responses(tmp_path, worker_id=42)


@pytest.mark.parametrize("field,value", [("text", "corrupted weights"), ("finish_reason", "length"),
                                         ("completion_tokens", 9)])
def test_output_changes_preserve_diagnostic_without_failing(tmp_path, records, field, value):
    write_records(tmp_path, records)
    source = inference_parity.read_responses(tmp_path)
    target = copy.deepcopy(source)
    target[CASE][field] = value
    report = tmp_path / "comparison.json"
    inference_parity.compare(source, target, report)
    assert json.loads(report.read_text())["mismatches"][CASE]["target"][field] == value


@pytest.mark.parametrize("initial_ids", [{42}, {41, 42}])
def test_dynamo_profiles_only_the_new_worker_and_checks_routing(monkeypatch, tmp_path, initial_ids):
    monkeypatch.setattr(test_dynamo_p2p, "artifact_dir", lambda _: tmp_path)
    monkeypatch.setattr(test_dynamo_p2p, "port_forward", lambda *a, **k: nullcontext(18000))
    discovered = iter([initial_ids, initial_ids | {43}])
    monkeypatch.setattr(test_dynamo_p2p, "_discovered_worker_ids", lambda *a: next(discovered))
    monkeypatch.setattr(test_dynamo_p2p.time, "sleep", lambda _: None)
    (tmp_path / "initial-workers.json").write_text(json.dumps(sorted(initial_ids)))
    (tmp_path / "source-worker.json").write_text("42")
    monkeypatch.setattr(test_dynamo_p2p, "read_responses", lambda *a, **k: {"case": {"messages": [], "task": "fixture", "expected": "A", "actual": "A", "unparsed": False, "passed": True}})
    calls = []
    probes = []
    monkeypatch.setattr(test_dynamo_p2p, "_wait_for_worker_route", lambda port, model, worker_id: probes.append(worker_id))

    def profile(url, model, directory, *, worker_id):
        calls.append(worker_id)
        return {"case": {"messages": [], "task": "fixture", "expected": "A", "actual": "A", "unparsed": False, "passed": True}}

    monkeypatch.setattr(test_dynamo_p2p, "profile", profile)
    test_dynamo_p2p.test_frontend_inference("test", "model", "dgd", False)
    assert calls == [43]
    assert probes == [43]


def test_dynamo_waits_for_discovered_worker_to_be_routable(monkeypatch):
    calls = []

    def response(request):
        assert request.full_url.endswith("/v1/chat/completions")
        payload = json.loads(request.data)
        calls.append(payload)
        if len(calls) < 3:
            raise urllib.error.HTTPError(request.full_url, 400, "Bad Request", {}, io.BytesIO(
                b'{"message":"nvext.decode_worker_id=43 does not identify a known worker"}'
            ))
        return {"nvext": {"worker_id": {"decode_worker_id": 43}}}

    monkeypatch.setattr(test_dynamo_p2p, "completion_response", response)
    monkeypatch.setattr(test_dynamo_p2p.time, "sleep", lambda _: None)
    test_dynamo_p2p._wait_for_worker_route(18000, "model", 43)
    assert len(calls) == 3
    assert all(row["nvext"]["decode_worker_id"] == 43 for row in calls)


@pytest.mark.parametrize("actual", [42, None])
def test_dynamo_route_probe_rejects_wrong_or_missing_worker(monkeypatch, actual):
    monkeypatch.setattr(test_dynamo_p2p, "completion_response", lambda _: {
        "nvext": {"worker_id": {"decode_worker_id": actual}}
    })
    with pytest.raises(AssertionError):
        test_dynamo_p2p._wait_for_worker_route(18000, "model", 43)


def test_dynamo_route_probe_does_not_retry_other_http_errors(monkeypatch):
    def response(request):
        raise urllib.error.HTTPError(request.full_url, 400, "Bad Request", {}, io.BytesIO(b'{"message":"invalid model"}'))

    monkeypatch.setattr(test_dynamo_p2p, "completion_response", response)
    with pytest.raises(urllib.error.HTTPError):
        test_dynamo_p2p._wait_for_worker_route(18000, "model", 43)


def test_dynamo_route_probe_fails_if_advertised_worker_never_registers(monkeypatch):
    calls = []

    def response(request):
        calls.append(request)
        raise urllib.error.HTTPError(request.full_url, 400, "Bad Request", {}, io.BytesIO(
            b'{"message":"nvext.decode_worker_id=43 does not identify a known worker"}'
        ))

    monkeypatch.setattr(test_dynamo_p2p, "completion_response", response)
    monkeypatch.setattr(test_dynamo_p2p.time, "sleep", lambda _: None)
    with pytest.raises(AssertionError, match="never became routable"):
        test_dynamo_p2p._wait_for_worker_route(18000, "model", 43)
    assert len(calls) == 90


def test_dynamo_rejects_source_only_routing(monkeypatch, tmp_path):
    monkeypatch.setattr(test_dynamo_p2p, "artifact_dir", lambda _: tmp_path)
    monkeypatch.setattr(test_dynamo_p2p, "port_forward", lambda *a, **k: nullcontext(18000))
    monkeypatch.setattr(test_dynamo_p2p, "_discovered_worker_ids", lambda *a: {42})
    monkeypatch.setattr(test_dynamo_p2p.time, "sleep", lambda _: None)
    (tmp_path / "initial-workers.json").write_text("[42]")
    monkeypatch.setattr(test_dynamo_p2p, "read_responses", lambda *a, **k: {"case": {"messages": [], "task": "fixture", "expected": "A", "actual": "A", "unparsed": False, "passed": True}})
    (tmp_path / "source-worker.json").write_text("42")
    with pytest.raises(AssertionError, match="one target"):
        test_dynamo_p2p.test_frontend_inference("test", "model", "dgd", False)


@pytest.mark.parametrize("failure", ["missing", "duplicate", "output", "task"])
def test_rejects_incomplete_or_invalid_grading(tmp_path, records, failure):
    write_records(tmp_path, records)
    path = tmp_path / inference_parity.SUBJECTS[0] / "accuracy_export.jsonl"
    grades = list(map(json.loads, path.read_text().splitlines()))
    if failure == "missing":
        grades.pop()
    elif failure == "duplicate":
        grades.append(grades[0])
    elif failure == "output":
        grades[0]["model_output"] = "different response"
    else:
        grades[0]["task"] = "mmlu"
    path.write_text("".join(json.dumps(grade) + "\n" for grade in grades))
    with pytest.raises(AssertionError):
        inference_parity.read_responses(tmp_path)


@pytest.mark.parametrize("field", ["messages", "expected", "task"])
def test_rejects_dataset_or_grade_drift(tmp_path, records, field):
    write_records(tmp_path, records)
    source = inference_parity.read_responses(tmp_path)
    target = copy.deepcopy(source)
    target[CASE][field] = "changed"
    with pytest.raises(AssertionError, match="session_000000"):
        inference_parity.compare(source, target, tmp_path / "comparison.json")


def test_replay_preserves_variation_logprobs_and_request_errors(monkeypatch, tmp_path):
    import io
    import urllib.request

    responses = iter([
        {"choices": [{"text": "first", "logprobs": {"top_logprobs": [{"first": -0.1}]}}]},
        TimeoutError("probe timed out"),
        {"choices": [{"text": "second", "logprobs": {"top_logprobs": [{"second": -0.2}]}}]},
    ])
    payloads = []

    def request(req, timeout):
        payloads.append(json.loads(req.data))
        response = next(responses)
        if isinstance(response, Exception):
            raise response
        return io.BytesIO(json.dumps(response).encode())

    monkeypatch.setattr(urllib.request, "urlopen", request)
    original = {"messages": [{"role": "user", "content": "Question"}], "text": "first"}
    report = tmp_path / "replays.json"
    inference_parity.replay_mismatch("http://fixture", "model", original, report)
    result = json.loads(report.read_text())
    assert result["original"] == original
    assert len(result["replays"]) == 3
    assert result["replays"][1] == {"error": "probe timed out"}
    assert result["replays"][0]["response"]["choices"][0]["logprobs"]["top_logprobs"]
    assert result["replays"][2]["response"]["choices"][0]["text"] == "second"
    assert all(p == payloads[0] for p in payloads)
    assert payloads[0]["temperature"] == payloads[0]["seed"] == 0


def test_failed_hardware_probe_preserves_parity_failure(monkeypatch, tmp_path):
    import test_p2p_k8s

    monkeypatch.setattr(test_p2p_k8s, "artifact_dir", lambda _: tmp_path)
    monkeypatch.setattr(test_p2p_k8s, "_pod_name", lambda *a, **k: a[1])
    monkeypatch.setattr(test_p2p_k8s, "port_forward", lambda *a, **k: nullcontext(k["local_port"]))
    source = {
        str(i): {"messages": [], "task": "fixture", "expected": "A", "actual": "A", "unparsed": False,
                 "passed": True, "text": "A"}
        for i in range(8)
    }
    target = copy.deepcopy(source)
    target["0"]["text"] = "Different wording with the same correct answer"
    for i in range(1, 8):
        target[str(i)].update(passed=False, actual="B", text="B")
    outputs = iter([source, target])
    monkeypatch.setattr(test_p2p_k8s, "profile", lambda *a: next(outputs))
    replays = []
    monkeypatch.setattr(test_p2p_k8s, "replay_mismatch", lambda *a: replays.append(a[2]))

    def unavailable(*args, **kwargs):
        raise RuntimeError("pod no longer available")

    monkeypatch.setattr(test_p2p_k8s, "kubectl", unavailable)
    with pytest.raises(AssertionError, match="accuracy regressed"):
        test_p2p_k8s.test_source_target_inference_matches("test", "model", 8001, 8002, False, False)
    assert replays == [source["1"], target["1"]]
    assert json.loads((tmp_path / "comparison.json").read_text())["score"]["regressions"] == 7
    assert "pod no longer available" in (tmp_path / "source-diagnostic-error.txt").read_text()


def test_dynamo_discovery_filters_non_generation_endpoints(monkeypatch):
    worker_id = 7411001889333074
    monkeypatch.setattr(test_dynamo_p2p, "completion_response", lambda request: {
        "instances": [
            {"instance_id": worker_id, "endpoint": "generate"},
            {"instance_id": 99, "endpoint": "load_metrics"},
        ]
    })
    assert test_dynamo_p2p._discovered_worker_ids(8000) == {worker_id}


@pytest.mark.parametrize("tokens", [128, 8])
def test_long_decode_checks_sustained_generation_and_keeps_responses(monkeypatch, tmp_path, tokens):
    import io
    import urllib.request

    payloads = []

    def request(req, timeout):
        payloads.append(json.loads(req.data))
        return io.BytesIO(json.dumps({
            "usage": {"completion_tokens": tokens},
            "choices": [{"message": {"content": "A detailed explanation"}, "finish_reason": "length"}],
        }).encode())

    monkeypatch.setattr(urllib.request, "urlopen", request)
    report = tmp_path / "long-decode.json"
    if tokens == 128:
        inference_parity.long_decode("http://fixture", "model", report)
        assert len(payloads) == 2
    else:
        with pytest.raises(AssertionError, match="did not reach 128"):
            inference_parity.long_decode("http://fixture", "model", report)
    assert json.loads(report.read_text())[0]["response"]["usage"]["completion_tokens"] == tokens
    assert all(payload["ignore_eos"] and payload["max_tokens"] == 128 for payload in payloads)


def test_accuracy_pilot_records_both_sample_sizes_and_native_repeat(monkeypatch, tmp_path):
    import test_p2p_k8s

    monkeypatch.setattr(test_p2p_k8s, "artifact_dir", lambda _: tmp_path)
    monkeypatch.setattr(test_p2p_k8s, "_pod_name", lambda *a, **k: a[1])
    monkeypatch.setattr(test_p2p_k8s, "port_forward", lambda *a, **k: nullcontext(k["local_port"]))
    calls = []

    def profile(url, model, directory, *, samples_per_subject=16):
        calls.append((directory.name, samples_per_subject))
        return {"case": {"messages": [], "task": "fixture", "expected": "A", "actual": "A", "unparsed": False, "passed": True}}

    monkeypatch.setattr(test_p2p_k8s, "profile", profile)
    test_p2p_k8s.test_source_target_inference_matches("test", "model", 8001, 8002, True, False)
    assert calls == [("source", 16), ("target", 16), ("source-64", 8), ("source-repeat", 16)]
    assert json.loads((tmp_path / "native-repeat-comparison.json").read_text())["score"]["regressions"] == 0


def test_standalone_source_baseline_does_not_require_a_target(monkeypatch, tmp_path):
    import test_p2p_k8s

    monkeypatch.setattr(test_p2p_k8s, "artifact_dir", lambda _: tmp_path)
    pods = []
    monkeypatch.setattr(test_p2p_k8s, "_pod_name", lambda _, name, **kwargs: pods.append(name) or name)
    monkeypatch.setattr(test_p2p_k8s, "port_forward", lambda *a, **k: nullcontext(k["local_port"]))
    calls = []
    monkeypatch.setattr(test_p2p_k8s, "profile", lambda url, model, directory: calls.append(directory.name))
    test_p2p_k8s.test_source_target_inference_matches("test", "model", 8001, 8002, False, True)
    assert pods == ["mx-source"]
    assert calls == ["source"]


def test_standalone_reuses_the_captured_source_baseline(monkeypatch, tmp_path):
    import test_p2p_k8s

    (tmp_path / "source").mkdir()
    monkeypatch.setattr(test_p2p_k8s, "artifact_dir", lambda _: tmp_path)
    monkeypatch.setattr(test_p2p_k8s, "_pod_name", lambda _, name, **kwargs: name)
    monkeypatch.setattr(test_p2p_k8s, "port_forward", lambda *a, **k: nullcontext(k["local_port"]))
    records = {"case": {"messages": [], "task": "fixture", "expected": "A", "actual": "A", "unparsed": False, "passed": True}}
    monkeypatch.setattr(test_p2p_k8s, "read_responses", lambda path: records)
    calls = []
    monkeypatch.setattr(test_p2p_k8s, "profile", lambda url, model, directory: calls.append(directory.name) or records)
    test_p2p_k8s.test_source_target_inference_matches("test", "model", 8001, 8002, False, False)
    assert calls == ["target"]


@pytest.mark.parametrize("difference", [None, "bytes", "shape", "missing", "draft"])
def test_weight_integrity_is_independent_of_accuracy(tmp_path, difference):
    source = [{"weight": {"shape": [2, 3], "dtype": "bfloat16", "sha256": "1234"}}, {"draft": {"sha256": "5678"}}]
    target = copy.deepcopy(source)
    if difference == "bytes":
        target[0]["weight"]["sha256"] = "changed"
    elif difference == "shape":
        target[0]["weight"]["shape"] = [3, 2]
    elif difference == "missing":
        target = []
    elif difference == "draft":
        target[1]["draft"]["sha256"] = "changed"
    def logs(rows):
        return "\n".join("(EngineCore) [MX-CI-WEIGHTS] " + json.dumps(row) for row in rows)
    report = tmp_path / "weights.json"
    if difference is None:
        assert inference_parity.compare_weight_digests(logs(source), logs(target), report)
    else:
        with pytest.raises(AssertionError, match="tensor bytes differ"):
            inference_parity.compare_weight_digests(logs(source), logs(target), report)
    assert json.loads(report.read_text())["matched"] == (difference is None)


def test_uninstrumented_workers_are_not_claimed_as_byte_verified(tmp_path):
    report = tmp_path / "weights.json"
    assert not inference_parity.compare_weight_digests("normal engine logs", "normal engine logs", report)
    assert not report.exists()


def test_enabled_weight_audit_cannot_silently_skip(monkeypatch, tmp_path):
    from types import SimpleNamespace
    import test_p2p_k8s

    monkeypatch.setattr(test_p2p_k8s, "artifact_dir", lambda _: tmp_path)
    monkeypatch.setattr(test_p2p_k8s, "_pod_name", lambda *a, **k: a[1])
    info = {"items": [{"spec": {"containers": [{"env": [{"name": "MX_CI_WEIGHT_DIGESTS", "value": "1"}]}]}}]}
    monkeypatch.setattr(test_p2p_k8s, "kubectl", lambda *args, **kwargs: SimpleNamespace(stdout=json.dumps(info) if args[0] == "get" else "normal engine log"))
    with pytest.raises(AssertionError, match="Enabled CI weight audit"):
        test_p2p_k8s.test_finalized_tensor_bytes_match("test")


@pytest.mark.parametrize("model", ["model", "deepseek-ai/DeepSeek-V2-Lite"])
def test_invalid_baseline_stops_before_running_remaining_subjects(monkeypatch, tmp_path, records, model):
    import shutil
    from types import SimpleNamespace

    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    write_records(fixtures, records)
    for subject in inference_parity.SUBJECTS:
        path = fixtures / subject / "accuracy_export.jsonl"
        grades = [json.loads(line) for line in path.read_text().splitlines()]
        for grade in grades:
            grade.update(actual="An explanation without an answer", model_output="An explanation without an answer", passed=False, unparsed=True)
        path.write_text("".join(json.dumps(grade) + "\n" for grade in grades))
        path = fixtures / subject / "profile_export_raw.jsonl"
        raw = [json.loads(line) for line in path.read_text().splitlines()]
        for record in raw:
            body = json.loads(record["responses"][0]["text"])
            body["choices"][0]["message"]["content"] = "An explanation without an answer"
            record["responses"][0]["text"] = json.dumps(body)
        path.write_text("".join(json.dumps(record) + "\n" for record in raw))
    calls = []

    def aiperf(command, **kwargs):
        calls.append(command)
        subject = command[command.index("--accuracy-tasks") + 1]
        directory = command[command.index("--artifact-dir") + 1]
        shutil.copytree(fixtures / subject, directory, dirs_exist_ok=True)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(inference_parity.subprocess, "run", aiperf)
    with pytest.raises(AssertionError, match="unparseable MMLU"):
        inference_parity.profile("http://fixture", model, tmp_path / "profile")
    assert len(calls) == 1
    assert "--no-server-metrics" in calls[0] and "--no-gpu-telemetry" in calls[0]
    extra = json.loads(calls[0][calls[0].index("--extra-inputs") + 1])
    assert extra["max_completion_tokens"] == 8 and extra["chat_template_kwargs"] == {"enable_thinking": False}
    assert extra.get("stop") == (["User:"] if model == "deepseek-ai/DeepSeek-V2-Lite" else None)
    assert calls[0][calls[0].index("--accuracy-system-prompt") + 1] == inference_parity.SYSTEM_PROMPT
