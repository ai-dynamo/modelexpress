# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AIPerf source/target inference checks, using its pinned raw export schema."""

import hashlib
import json
import math
import subprocess
import time
from pathlib import Path


SUBJECTS = (
    "elementary_mathematics", "high_school_computer_science", "high_school_biology",
    "high_school_world_history", "logical_fallacies", "moral_disputes", "management", "global_facts",
)
SAMPLES_PER_SUBJECT = 16
SAMPLE_COUNT = len(SUBJECTS) * SAMPLES_PER_SUBJECT
ANSWER_TOKENS = 8
SYSTEM_PROMPT = "Answer the multiple-choice question with exactly one uppercase letter: A, B, C, or D. Do not explain or repeat the question."


def artifact_dir(namespace: str) -> Path:
    return Path("inference-artifacts") / namespace


def compare_weight_digests(source_log: str, target_log: str, report: Path) -> bool:
    digests = {
        side: [json.loads(line.split("[MX-CI-WEIGHTS] ", 1)[1]) for line in logs.splitlines() if "[MX-CI-WEIGHTS] " in line]
        for side, logs in (("source", source_log), ("target", target_log))
    }
    if not any(digests.values()):
        return False
    matched = bool(digests["source"] and digests["target"] and digests["source"] == digests["target"])
    report.write_text(json.dumps({"matched": matched, **digests}, indent=2) + "\n")
    assert matched, f"Finalized source/target tensor bytes differ or an audit is missing; see {report}"
    return True


def _read_subject(directory: Path, subject: str, count: int, worker_id: int | None) -> dict:
    inputs = json.loads((directory / "inputs.json").read_text())["data"][:count]
    prompts = {row["session_id"]: row["payloads"][0]["messages"] for row in inputs}
    assert len(prompts) == count, f"Incomplete MMLU inputs in {directory}"
    grades = {}
    for line in (directory / "accuracy_export.jsonl").read_text().splitlines():
        grade = json.loads(line)
        case = grade["conversation_id"]
        assert case in prompts and case not in grades, f"Unexpected or duplicate grade: {case}"
        assert grade["task"] == subject, f"{case}: unexpected accuracy task"
        assert grade["expected"] in "ABCD" and len(grade["expected"]) == 1, f"{case}: invalid ground truth"
        assert isinstance(grade["passed"], bool), f"{case}: invalid grade"
        grades[case] = grade
    assert grades.keys() == prompts.keys(), f"Missing MMLU grades in {directory}"
    records = {}
    for line in (directory / "profile_export_raw.jsonl").read_text().splitlines():
        record = json.loads(line)
        case = record["metadata"]["conversation_id"]
        assert case in prompts, f"Unexpected prompt {case!r} in {directory}"
        assert case not in records, f"Duplicate response for {case} in {directory}"
        assert record.get("error") is None, f"{case}: request failed: {record.get('error')}"
        assert record["status"] == 200, f"{case}: HTTP {record['status']}"
        assert record["payload"]["messages"] == prompts[case], f"{case}: prompt changed"
        assert len(record["responses"]) == 1, f"{case}: expected one non-streaming response"
        body = json.loads(record["responses"][0]["text"])
        assert len(body["choices"]) == 1, f"{case}: expected one completion"
        choice = body["choices"][0]
        text = choice["message"].get("content")
        assert isinstance(text, str) and text.strip(), f"{case}: empty completion"
        assert choice["finish_reason"] in {"stop", "length"}, f"{case}: unexpected finish: {choice}"
        assert body["usage"]["completion_tokens"] > 0, f"{case}: no generated tokens"
        assert grades[case]["model_output"] == text, f"{case}: grade/output mismatch"
        assert grades[case]["passed"] == (grades[case]["actual"] == grades[case]["expected"]), f"{case}: inconsistent grade"
        if worker_id is not None:
            actual = body["nvext"]["worker_id"]["decode_worker_id"]
            assert actual == worker_id, f"{case}: routed to worker {actual}, expected {worker_id}"
        records[case] = {
            "messages": prompts[case], "task": subject,
            "text": text,
            "finish_reason": choice["finish_reason"],
            "completion_tokens": body["usage"]["completion_tokens"],
            "expected": grades[case]["expected"],
            "passed": grades[case]["passed"],
            "actual": grades[case]["actual"], "unparsed": grades[case]["unparsed"],
        }
    assert records.keys() == prompts.keys(), f"Missing responses in {directory}: {prompts.keys() - records.keys()}"
    return records


def read_responses(directory: Path, *, worker_id: int | None = None, samples_per_subject: int = SAMPLES_PER_SUBJECT) -> dict:
    return {
        f"{subject}/{case}": row
        for subject in SUBJECTS
        for case, row in _read_subject(directory / subject, subject, samples_per_subject, worker_id).items()
    }


def profile(url: str, model: str, directory: Path, *, worker_id: int | None = None,
            samples_per_subject: int = SAMPLES_PER_SUBJECT) -> dict:
    directory.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    extra = {"temperature": 0, "seed": 0, "max_completion_tokens": ANSWER_TOKENS,
             "chat_template_kwargs": {"enable_thinking": False}}
    if model == "deepseek-ai/DeepSeek-V2-Lite":
        extra["stop"] = ["User:"]
    if worker_id is not None:
        extra["nvext"] = {
            "backend_instance_id": worker_id,
            "decode_worker_id": worker_id,
            "extra_fields": ["worker_id"],
        }
    base_command = [
        "aiperf", "profile", "--model", model, "--url", url,
        "--endpoint-type", "chat", "--accuracy-benchmark", "mmlu",
        "--accuracy-n-shots", "1", "--accuracy-no-enable-cot", "--dataset-sampling-strategy", "sequential",
        "--accuracy-system-prompt", SYSTEM_PROMPT,
        "--random-seed", "0", "--concurrency", "1", "--workers-max", "1",
        "--request-count", str(samples_per_subject),
        "--request-timeout-seconds", "300", "--extra-inputs", json.dumps(extra),
        "--tokenizer", "builtin", "--use-server-token-count",
        "--no-server-metrics", "--no-gpu-telemetry",
        "--export-level", "raw", "--ui", "none",
    ]
    timings = {}
    records = {}
    for subject in SUBJECTS:
        output = directory / subject
        output.mkdir()
        command = [*base_command, "--accuracy-tasks", subject, "--artifact-dir", str(output)]
        (output / "command.json").write_text(json.dumps(command, indent=2) + "\n")
        subject_started = time.monotonic()
        with (output / "aiperf.log").open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=300)
        timings[subject] = time.monotonic() - subject_started
        (directory / "timing.json").write_text(json.dumps({
            "samples": len(SUBJECTS) * samples_per_subject,
            "wall_seconds": time.monotonic() - started, "subjects": timings,
        }, indent=2) + "\n")
        assert result.returncode == 0, f"AIPerf failed; see {output / 'aiperf.log'}"
        records.update({
            f"{subject}/{case}": row
            for case, row in _read_subject(output, subject, samples_per_subject, worker_id).items()
        })
        invalid = sum(row["actual"] not in {"A", "B", "C", "D"} for row in records.values())
        assert invalid <= len(SUBJECTS) * samples_per_subject * 0.05, f"Too many unparseable MMLU answers after {subject}: {invalid}; see {output}"
    dataset = {
        "benchmark": "mmlu", "dataset": "lighteval/mmlu", "subjects": SUBJECTS, "split": "test",
        "samples_per_subject": samples_per_subject, "sampling": "first N per subject, sequential", "random_seed": 0,
        "examples": {
            case: hashlib.sha256(json.dumps([row["messages"], row["expected"]],
                                            ensure_ascii=False).encode()).hexdigest()
            for case, row in records.items()
        },
    }
    (directory / "dataset.json").write_text(json.dumps(dataset, indent=2) + "\n")
    return records


def compare(source: dict, target: dict, report: Path) -> None:
    mismatches = {
        case: {"source": source.get(case), "target": target.get(case)}
        for case in sorted(source.keys() | target.keys())
        if source.get(case) != target.get(case)
    }
    common = source.keys() & target.keys()
    dataset_mismatches = [
        case for case in sorted(source.keys() | target.keys())
        if case not in common or any(source[case][key] != target[case][key] for key in ("messages", "task", "expected"))
    ]
    lost = sum(source[case]["passed"] and not target[case]["passed"] for case in common)
    gained = sum(not source[case]["passed"] and target[case]["passed"] for case in common)
    discordant = lost + gained
    p_value = sum(math.comb(discordant, k) for k in range(lost, discordant + 1)) / 2**discordant
    score = {
        "source_correct": sum(row["passed"] for row in source.values()),
        "target_correct": sum(row["passed"] for row in target.values()),
        "samples": len(common), "regressions": lost, "improvements": gained,
        "paired_p_value": p_value, "significance": 0.01,
        "by_subject": {
            subject: {
                side: sum(row["passed"] for row in rows.values() if row["task"] == subject)
                for side, rows in (("source", source), ("target", target))
            }
            for subject in sorted({row["task"] for row in source.values()})
        },
    }
    parsing = {
        side: {
            "invalid_answers": sum(row["actual"] not in {"A", "B", "C", "D"} for row in rows.values()),
            "grader_fallbacks": sum(row["unparsed"] for row in rows.values()),
        }
        for side, rows in (("source", source), ("target", target))
    }
    summary = {"matched": len(common - mismatches.keys()), "mismatches": mismatches,
               "dataset_mismatches": dataset_mismatches, "score": score, "parsing": parsing}
    report.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    assert not dataset_mismatches, f"Source/target dataset differs for {dataset_mismatches}; see {report}"
    assert source and target, f"No accuracy samples; see {report}"
    assert all(values["invalid_answers"] <= len(common) * 0.05 for values in parsing.values()), f"MMLU answer parsing failed; see {report}"
    assert p_value >= 0.01, f"Source/target accuracy regressed ({score}); see {report}"


def long_decode(url: str, model: str, report: Path) -> None:
    """Exercise sustained decode independently of the short-answer score."""
    import urllib.request

    responses = []
    for prompt in (
        "Write a detailed story about an astronaut exploring an unfamiliar planet.",
        "Explain in detail how to multiply two three-digit numbers, with worked examples.",
    ):
        payload = {
            "model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 128, "ignore_eos": True, "temperature": 0, "seed": 0,
            "stream": False, "chat_template_kwargs": {"enable_thinking": False},
        }
        request = urllib.request.Request(
            f"{url}/v1/chat/completions", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            body = json.load(response)
        responses.append({"payload": payload, "response": body})
        report.write_text(json.dumps(responses, ensure_ascii=False, indent=2) + "\n")
        assert body["usage"]["completion_tokens"] == 128, f"Long decode did not reach 128 tokens; see {report}"
        assert body["choices"][0]["message"]["content"].strip(), f"Empty long decode; see {report}"
        assert body["choices"][0]["finish_reason"] == "length", f"Unexpected long-decode finish; see {report}"


def replay_mismatch(url: str, model: str, original: dict, report: Path) -> None:
    """Capture repeatability and token log-probabilities without changing parity."""
    import urllib.request

    payload = {
        "model": model, "messages": original["messages"], "max_tokens": ANSWER_TOKENS,
        "temperature": 0, "seed": 0, "stream": False,
        "chat_template_kwargs": {"enable_thinking": False}, "logprobs": True, "top_logprobs": 5,
    }
    diagnostic = {"payload": payload, "original": original, "replays": []}
    for _ in range(3):
        request = urllib.request.Request(
            f"{url}/v1/chat/completions", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                body = json.load(response)
            diagnostic["replays"].append({"response": body})
        except Exception as error:
            diagnostic["replays"].append({"error": str(error)})
        report.write_text(json.dumps(diagnostic, ensure_ascii=False, indent=2) + "\n")
