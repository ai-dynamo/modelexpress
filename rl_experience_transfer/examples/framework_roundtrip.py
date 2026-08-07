# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real adapter/transport smoke test; this is not a full framework trainer run."""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import numpy as np

from rlxfer.adapters import (
    ExperienceAdapter,
    IncompatibleExperienceError,
    MilesAdapter,
    SlimeAdapter,
    create_adapter,
)
from rlxfer.api import ExperienceConsumer, ExperienceProducer
from rlxfer.compatibility import CompatibilityRequirements
from rlxfer.model import ExperienceBatch, TensorPayload
from rlxfer.serialization import JsonExperienceSerializer
from rlxfer.transport import ReceiptState
from rlxfer.transports.filesystem import FileSystemTransport


def _prime_native() -> object:
    types = importlib.import_module("prime_rl.transport.types")
    sample = types.TrainingSample(
        prompt_ids=[1, 2],
        prompt_mask=[False, False],
        completion_ids=[3, 4],
        completion_mask=[True, True],
        completion_logprobs=[-0.1, -0.2],
        completion_temperatures=[1.0, 1.0],
        env_name="rlxfer-smoke",
        advantage=1.0,
        reward=1.0,
    )
    return types.TrainingBatch(examples=[sample], step=0, run_idx=0)


def _grouped_native(framework: str) -> object:
    types = importlib.import_module(f"{framework}.utils.types")
    outputs = importlib.import_module(f"{framework}.rollout.base_types")
    sample = types.Sample(
        index=0,
        group_index=0,
        prompt="Count to two.",
        tokens=[1, 2, 3, 4],
        response="one two",
        response_length=2,
        reward=1.0,
        loss_mask=[1, 1],
        weight_versions=["0"],
        rollout_log_probs=[-0.1, -0.2],
        status=types.Sample.Status.COMPLETED,
    )
    return outputs.RolloutFnTrainOutput(samples=[[sample]], metrics={"smoke": 1})


def _nemo_native() -> object:
    interfaces = importlib.import_module("nemo_rl.experience.interfaces")
    completion = interfaces.Completion(
        message_log=[
            {
                "role": "assistant",
                "content": "one two",
                "token_ids": [3, 4],
                "generation_logprobs": [-0.1, -0.2],
            }
        ],
        env_extras=None,
        truncated=False,
        reward=1.0,
    )
    return interfaces.PromptGroupRecord(
        prompt_idx=0,
        prompt=[{"role": "user", "content": "Count to two."}],
        extra_env_info=None,
        metadata={"smoke": True},
        completions=[completion],
        rollout_metrics={"tokens": 2},
    )


_BUILDERS: Mapping[str, Callable[[], object]] = MappingProxyType(
    {
        "miles": lambda: _grouped_native("miles"),
        "nemo_rl": _nemo_native,
        "prime_rl": _prime_native,
        "slime": lambda: _grouped_native("slime"),
    }
)
FRAMEWORKS = tuple(sorted(_BUILDERS))
SUPPORTED_CONVERSIONS = frozenset(
    {(framework, framework) for framework in FRAMEWORKS} | {("miles", "slime"), ("slime", "miles")}
)


def supports_conversion(producer_framework: str, consumer_framework: str) -> bool:
    """Return whether a lossless native conversion is part of the adapter contract."""

    return (producer_framework, consumer_framework) in SUPPORTED_CONVERSIONS


def _consumer_adapter(producer_framework: str, consumer_framework: str) -> ExperienceAdapter:
    adapter = create_adapter(consumer_framework)
    if producer_framework == consumer_framework:
        return adapter
    requirements = CompatibilityRequirements(
        consumer_framework=consumer_framework,
        consumer_framework_version=adapter.framework_version,
        algorithm="grpo",
        chat_template="rlxfer-test-template",
        model_id="rlxfer-test-model",
        padding="right",
        reward_definition="rlxfer-test-reward",
        sequence_format="prompt-response",
        tokenizer_id="rlxfer-test-tokenizer",
        truncation="right",
    )
    if consumer_framework == "miles":
        return MilesAdapter(requirements)
    if consumer_framework == "slime":
        return SlimeAdapter(requirements)
    return adapter


def _with_declared_semantics(batch: ExperienceBatch) -> ExperienceBatch:
    return replace(
        batch,
        metadata=replace(
            batch.metadata,
            algorithm="grpo",
            chat_template="rlxfer-test-template",
            model_id="rlxfer-test-model",
            padding="right",
            reward_definition="rlxfer-test-reward",
            sequence_format="prompt-response",
            tokenizer_id="rlxfer-test-tokenizer",
            truncation="right",
        ),
    )


def _training_signature(batch: ExperienceBatch) -> list[dict[str, object]]:
    signature = []
    for trajectory in batch.trajectories:
        signature.append(
            {
                "log_probs": _payload_values(trajectory.log_probs),
                "rewards": dict(trajectory.rewards),
                "terminal": trajectory.terminal,
                "tokens": _payload_values(trajectory.tokens),
                "truncated": trajectory.truncated,
            }
        )
    return signature


def _payload_values(payload: TensorPayload | None) -> object:
    return None if payload is None else np.asarray(payload.data).tolist()


def _tiny_update(batch: ExperienceBatch) -> dict[str, object]:
    torch = importlib.import_module("torch")
    payload = next(
        (
            trajectory.tokens
            for trajectory in batch.trajectories
            if trajectory.tokens is not None and trajectory.tokens.shape[-1] > 0
        ),
        None,
    )
    if payload is None:
        raise RuntimeError("the canonical batch has no tokens for the optimizer smoke step")
    tokens = torch.as_tensor(payload.data, device="cpu").reshape(-1).long().remainder(128)
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Embedding(128, 8), torch.nn.Linear(8, 1))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    prediction = model(tokens).reshape(-1)
    target = torch.ones_like(prediction)
    loss = torch.nn.functional.mse_loss(prediction, target)
    optimizer.zero_grad()
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters()]
    gradients_present = all(gradient is not None for gradient in gradients)
    gradients_finite = all(
        bool(torch.isfinite(gradient).all()) for gradient in gradients if gradient is not None
    )
    optimizer.step()
    parameter_changed = any(
        not torch.equal(old, parameter.detach())
        for old, parameter in zip(before, model.parameters(), strict=True)
    )
    return {
        "gradient_finite": gradients_finite,
        "gradient_present": gradients_present,
        "loss": float(loss.detach()),
        "model": "Embedding(128,8)+Linear(8,1)",
        "parameter_changed": parameter_changed,
        "torch_version": str(torch.__version__),
    }


def run_framework_case(
    producer_framework: str,
    consumer_framework: str,
    queue_dir: Path,
) -> dict[str, object]:
    """Exercise one declared native conversion or its required safe rejection."""

    started = time.perf_counter()
    expected_conversion = supports_conversion(producer_framework, consumer_framework)
    producer_adapter = create_adapter(producer_framework)
    consumer_adapter = _consumer_adapter(producer_framework, consumer_framework)
    native = _BUILDERS[producer_framework]()
    batch = _with_declared_semantics(producer_adapter.from_framework(native))
    serializer = JsonExperienceSerializer(checksum=True)
    transport = FileSystemTransport(queue_dir)
    try:
        receipt = ExperienceProducer(transport, serializer=serializer).publish(
            batch,
            idempotency_key=(f"{producer_framework}:{consumer_framework}:{batch.experience_id}"),
            timeout=5.0,
        )
        consumer = ExperienceConsumer(
            transport,
            adapter=consumer_adapter,
            serializer=serializer,
        )
        try:
            delivery = consumer.receive(timeout=5.0)
        except IncompatibleExperienceError as error:
            receipt_state = receipt.wait(timeout=5.0).state
            if expected_conversion:
                raise
            if receipt_state is not ReceiptState.REJECTED:
                raise RuntimeError(
                    f"unexpected rejection receipt state: {receipt_state.value}"
                ) from error
            return {
                "consumer_framework": consumer_framework,
                "coverage": "native adapter contract and safe-rejection integration",
                "elapsed_seconds": time.perf_counter() - started,
                "expected_outcome": "rejected_as_unsafe",
                "producer_framework": producer_framework,
                "python_version": platform.python_version(),
                "rejection": str(error),
                "result": "PASSED",
                "transport": "filesystem",
            }
        if delivery is None:
            raise RuntimeError("filesystem transport timed out")
        if not expected_conversion:
            delivery.reject("an undeclared cross-framework conversion unexpectedly validated")
            raise RuntimeError("unsafe cross-framework conversion unexpectedly succeeded")
        native_training = delivery.to_framework()
        training_batch = consumer_adapter.from_framework(native_training)
        if _training_signature(training_batch) != _training_signature(batch):
            delivery.reject("native reconstruction changed canonical training values")
            raise RuntimeError("native reconstruction changed canonical training values")
        update = _tiny_update(training_batch)
        delivery.ack()
        receipt_state = receipt.wait(timeout=5.0).state
    finally:
        transport.close()
    if receipt_state is not ReceiptState.ACKED:
        raise RuntimeError(f"unexpected delivery receipt state: {receipt_state.value}")
    token_count = sum(
        trajectory.tokens.shape[-1]
        for trajectory in training_batch.trajectories
        if isinstance(trajectory.tokens, TensorPayload)
    )
    return {
        "acknowledged": True,
        "canonical_schema": training_batch.metadata.schema_version,
        "consumer_adapter_version": consumer_adapter.adapter_version,
        "consumer_framework": consumer_framework,
        "consumer_framework_version": consumer_adapter.framework_version,
        "coverage": "native adapter integration; not a full framework trainer",
        "elapsed_seconds": time.perf_counter() - started,
        "expected_outcome": "converted",
        "native_training_type": type(native_training).__name__,
        "producer_adapter_version": producer_adapter.adapter_version,
        "producer_framework": producer_framework,
        "producer_framework_version": producer_adapter.framework_version,
        "python_version": platform.python_version(),
        "result": "PASSED",
        "tokens": token_count,
        "trajectories": len(training_batch.trajectories),
        "transport": "filesystem",
        **update,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("producer_framework", choices=FRAMEWORKS)
    parser.add_argument("--consumer-framework", choices=FRAMEWORKS)
    parser.add_argument("--queue-dir", type=Path)
    parser.add_argument("--output", type=Path, help="optional JSON result path")
    return parser


def main() -> int:
    args = _parser().parse_args()
    consumer_framework = args.consumer_framework or args.producer_framework
    try:
        if args.queue_dir is None:
            with tempfile.TemporaryDirectory(prefix="rlxfer-framework-") as temporary:
                result = run_framework_case(
                    args.producer_framework,
                    consumer_framework,
                    Path(temporary),
                )
        else:
            result = run_framework_case(
                args.producer_framework,
                consumer_framework,
                args.queue_dir,
            )
    except ImportError as error:
        result = {
            "consumer_framework": consumer_framework,
            "coverage": "native adapter integration; not a full framework trainer",
            "error": f"{type(error).__name__}: {error}",
            "producer_framework": args.producer_framework,
            "result": "BLOCKED",
        }
    except Exception as error:
        result = {
            "consumer_framework": consumer_framework,
            "coverage": "native adapter integration; not a full framework trainer",
            "error": f"{type(error).__name__}: {error}",
            "producer_framework": args.producer_framework,
            "result": "FAILED",
        }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if result["result"] == "PASSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
