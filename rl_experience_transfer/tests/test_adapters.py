# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from enum import Enum
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from rlxfer.adapters import (
    AdapterRegistry,
    MilesAdapter,
    MissingDependencyError,
    NemoRLAdapter,
    PrimeRLAdapter,
    SlimeAdapter,
    create_adapter,
)
from rlxfer.adapters.compat import verify_framework_version
from rlxfer.compatibility import CompatibilityRequirements
from rlxfer.errors import CompatibilityError
from rlxfer.model import TensorPayload


@pytest.mark.unit
def test_adapter_registry_is_instance_scoped() -> None:
    assert isinstance(create_adapter("nemo_rl"), NemoRLAdapter)
    registry = AdapterRegistry()
    registry.register("custom", SlimeAdapter)
    adapter = create_adapter("custom", registry)
    assert isinstance(adapter, SlimeAdapter)
    assert adapter.adapter_version == "0.1.0"
    with pytest.raises(ValueError, match="unknown adapter"):
        create_adapter("custom")

    with pytest.raises(CompatibilityError, match="outside adapter"):
        verify_framework_version("slime", "9.0.0")


def _install_module(monkeypatch: pytest.MonkeyPatch, name: str, **values: object) -> None:
    module = ModuleType(name)
    for key, value in values.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)


@dataclass
class _EncodedTensor:
    dtype: str
    shape: list[int]
    data: bytes


@dataclass
class _RoutedExperts:
    data: bytes
    shape: list[int]
    dtype: str


@dataclass
class _TrainingSample:
    prompt_ids: list[int]
    prompt_mask: list[bool]
    completion_ids: list[int]
    completion_mask: list[bool]
    completion_logprobs: list[float]
    completion_temperatures: list[float]
    env_name: str
    teacher_logprobs: list[float] | None = None
    advantage: float | None = None
    reward: float | None = None
    mm_kwargs: dict[str, _EncodedTensor] | None = None
    routed_experts: _RoutedExperts | None = None
    mm_token_type_ids: list[int] | None = None
    training_mode: str = "rl"


@dataclass
class _TrainingBatch:
    examples: list[_TrainingSample]
    step: int
    run_idx: int | None = None


def _install_prime(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_module(monkeypatch, "prime_rl", __version__="2873bf2")
    _install_module(monkeypatch, "prime_rl.transport")
    _install_module(
        monkeypatch,
        "prime_rl.transport.types",
        EncodedTensor=_EncodedTensor,
        RoutedExperts=_RoutedExperts,
        TrainingSample=_TrainingSample,
        TrainingBatch=_TrainingBatch,
    )


@pytest.mark.unit
def test_prime_training_batch_round_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_prime(monkeypatch)
    sample = _TrainingSample(
        prompt_ids=[1, 2],
        prompt_mask=[False, False],
        completion_ids=[3, 4],
        completion_mask=[True, True],
        completion_logprobs=[-0.1, -0.2],
        completion_temperatures=[0.8, 0.8],
        env_name="math",
        advantage=1.25,
        reward=1.0,
        mm_kwargs={"pixels": _EncodedTensor("uint8", [2], b"\x01\x02")},
        routed_experts=_RoutedExperts(b"\x03", [1, 1, 1], "uint8"),
    )
    native = _TrainingBatch([sample], step=7, run_idx=2)

    batch = PrimeRLAdapter().from_framework(native)

    assert batch.metadata.producer_framework == "prime_rl"
    assert batch.trajectories[0].tokens is not None
    assert np.asarray(batch.trajectories[0].tokens.data).tolist() == [3, 4]
    assert batch.trajectories[0].advantages is not None
    assert batch.trajectories[0].advantages.shape == (2,)
    restored = PrimeRLAdapter().to_framework(batch)
    assert isinstance(restored, _TrainingBatch)
    assert restored.step == 7
    assert restored.examples[0].completion_ids == [3, 4]
    assert isinstance(restored.examples[0].routed_experts, _RoutedExperts)
    assert restored.examples[0].routed_experts.data == b"\x03"
    assert restored.examples[0].mm_kwargs is not None
    assert isinstance(restored.examples[0].mm_kwargs["pixels"], _EncodedTensor)
    assert restored.examples[0].mm_kwargs["pixels"].data == b"\x01\x02"


@pytest.mark.unit
def test_prime_rejects_misaligned_completion(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_prime(monkeypatch)
    sample = _TrainingSample(
        prompt_ids=[1],
        prompt_mask=[False],
        completion_ids=[2, 3],
        completion_mask=[True],
        completion_logprobs=[-0.1, -0.2],
        completion_temperatures=[1.0, 1.0],
        env_name="test",
    )
    with pytest.raises(ValueError, match="completion_mask length"):
        PrimeRLAdapter().from_framework(_TrainingBatch([sample], step=0))


class _Status(Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    TRUNCATED = "truncated"


@dataclass
class _GroupedSample:
    index: int | None = None
    group_index: int | None = None
    prompt: str = ""
    tokens: list[int] = field(default_factory=list)
    response: str = ""
    response_length: int = 0
    reward: float | None = None
    loss_mask: list[int] | None = None
    weight_versions: list[str] = field(default_factory=list)
    rollout_log_probs: list[float] | None = None
    teacher_log_probs: list[float] | None = None
    status: _Status = _Status.PENDING
    metadata: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> _GroupedSample:
        state = dict(value)
        state["status"] = _Status(state["status"])
        accepted = cls.__dataclass_fields__
        return cls(**{key: item for key, item in state.items() if key in accepted})


@dataclass
class _RolloutOutput:
    samples: list[list[_GroupedSample]]
    metrics: dict[str, Any] | None = None


def _install_grouped(monkeypatch: pytest.MonkeyPatch, framework: str, version: str) -> None:
    _install_module(monkeypatch, framework, __version__=version)
    _install_module(monkeypatch, f"{framework}.utils")
    _install_module(monkeypatch, f"{framework}.utils.types", Sample=_GroupedSample)
    _install_module(monkeypatch, f"{framework}.rollout")
    _install_module(
        monkeypatch,
        f"{framework}.rollout.base_types",
        RolloutFnTrainOutput=_RolloutOutput,
    )


@pytest.mark.unit
def test_slime_round_trip_retains_groups_and_miles_rejects_unknown_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_grouped(monkeypatch, "slime", "0.3.1")
    _install_grouped(monkeypatch, "miles", "0.2.1")
    first = _GroupedSample(
        index=10,
        group_index=2,
        prompt="2+2?",
        tokens=[10, 11, 12],
        response="4",
        response_length=1,
        reward=1.0,
        loss_mask=[1],
        weight_versions=["3"],
        rollout_log_probs=[-0.2],
        status=_Status.COMPLETED,
        metadata={"source": "real-shape-contract"},
    )
    second = _GroupedSample(
        index=11,
        group_index=3,
        tokens=[20, 21],
        response_length=1,
        loss_mask=[1],
        rollout_log_probs=[-0.3],
        status=_Status.TRUNCATED,
    )
    native = _RolloutOutput([[first], [second]], metrics={"rollout": 2})

    batch = SlimeAdapter().from_framework(native)

    assert len(batch.trajectories) == 2
    assert isinstance(batch.trajectories[0].tokens, TensorPayload)
    assert np.asarray(batch.trajectories[0].tokens.data).tolist() == [12]
    restored = SlimeAdapter().to_framework(batch)
    assert isinstance(restored, _RolloutOutput)
    assert [len(group) for group in restored.samples] == [1, 1]
    assert restored.samples[0][0].status is _Status.COMPLETED
    assert restored.metrics == {"rollout": 2}

    with pytest.raises(ValueError, match="cross-framework slime -> miles conversion is unsafe"):
        MilesAdapter().to_framework(batch)

    semantics = {
        "algorithm": "grpo",
        "tokenizer_id": "tokenizer-v1",
        "model_id": "model-v1",
        "reward_definition": "math-v1",
        "sequence_format": "prompt-response",
        "padding": "right",
        "chat_template": "chat-v1",
        "truncation": "right",
    }
    for name, value in semantics.items():
        setattr(batch.metadata, name, value)
    requirements = CompatibilityRequirements(
        consumer_framework="miles",
        consumer_framework_version="0.2.1",
        algorithm="grpo",
        tokenizer_id="tokenizer-v1",
        model_id="model-v1",
        reward_definition="math-v1",
        sequence_format="prompt-response",
        padding="right",
        chat_template="chat-v1",
        truncation="right",
    )
    cross_framework = MilesAdapter(requirements).to_framework(batch)
    assert isinstance(cross_framework, _RolloutOutput)


@pytest.mark.unit
def test_grouped_adapter_rejects_multidimensional_sample_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_grouped(monkeypatch, "slime", "0.3.1")
    sample = _GroupedSample(tokens=np.ones((2, 2), dtype=np.int64), response_length=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="tokens must be one-dimensional"):
        SlimeAdapter().from_framework([[sample]])


class _BatchedDataDict(dict[str, Any]):
    pass


@dataclass
class _Completion:
    message_log: list[dict[str, Any]]
    env_extras: dict[str, Any] | None
    truncated: bool
    reward: float


@dataclass
class _PromptGroupRecord:
    prompt_idx: int
    prompt: list[dict[str, Any]]
    extra_env_info: dict[str, Any] | None
    metadata: dict[str, Any]
    completions: list[_Completion]
    rollout_metrics: dict[str, Any]


def _install_nemo(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_module(monkeypatch, "nemo_rl", __version__="0.7.0")
    _install_module(monkeypatch, "nemo_rl.distributed")
    _install_module(
        monkeypatch,
        "nemo_rl.distributed.batched_data_dict",
        BatchedDataDict=_BatchedDataDict,
    )
    _install_module(monkeypatch, "nemo_rl.experience")
    _install_module(
        monkeypatch,
        "nemo_rl.experience.interfaces",
        Completion=_Completion,
        PromptGroupRecord=_PromptGroupRecord,
    )


@pytest.mark.unit
def test_nemo_batched_data_dict_mapping_round_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_nemo(monkeypatch)
    native = _BatchedDataDict(
        input_ids=np.asarray([[1, 2, 3], [4, 5, 0]], dtype=np.int64),
        input_lengths=np.asarray([3, 2], dtype=np.int32),
        generation_logprobs=np.asarray([[-0.1, -0.2, -0.3], [-0.4, -0.5, 0.0]], dtype=np.float32),
        token_mask=np.asarray([[1, 1, 1], [1, 1, 0]], dtype=np.bool_),
    )

    batch = NemoRLAdapter().from_framework((native, {"latency": 0.1}))

    assert len(batch.trajectories) == 2
    assert batch.tensors["input_ids"].dtype == "int64"
    restored = NemoRLAdapter().to_framework(batch)
    assert isinstance(restored, _BatchedDataDict)
    np.testing.assert_array_equal(restored["input_ids"], native["input_ids"])
    np.testing.assert_array_equal(restored["token_mask"], native["token_mask"])


@pytest.mark.unit
@pytest.mark.parametrize(
    ("input_ids", "input_lengths", "message"),
    [
        (np.asarray([[1, 2]]), np.asarray([-1]), "must be non-negative"),
        (np.asarray([[1, 2]]), np.asarray([3]), "exceeds padded row width"),
        (np.asarray([[1, 2]]), np.asarray([1.5]), "must be an integer"),
        (np.asarray([[1, 2], [3, 4]]), np.asarray([2]), "must equal input_ids rows"),
        (np.asarray([1, 2]), np.asarray([1, 1]), "one-dimensional rows"),
    ],
)
def test_nemo_rejects_invalid_batch_lengths(
    monkeypatch: pytest.MonkeyPatch,
    input_ids: np.ndarray[Any, Any],
    input_lengths: np.ndarray[Any, Any],
    message: str,
) -> None:
    _install_nemo(monkeypatch)
    with pytest.raises((TypeError, ValueError), match=message):
        NemoRLAdapter().from_framework(
            _BatchedDataDict(input_ids=input_ids, input_lengths=input_lengths)
        )


@pytest.mark.unit
def test_nemo_prompt_group_record_round_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_nemo(monkeypatch)
    completion = _Completion(
        message_log=[
            {
                "role": "assistant",
                "content": "four",
                "token_ids": [4, 5],
                "generation_logprobs": [-0.1, -0.2],
            }
        ],
        env_extras={"turns": 1},
        truncated=False,
        reward=1.0,
    )
    native = _PromptGroupRecord(
        prompt_idx=8,
        prompt=[{"role": "user", "content": "2+2?"}],
        extra_env_info=None,
        metadata={"task": "math"},
        completions=[completion],
        rollout_metrics={"tokens": 2},
    )

    batch = NemoRLAdapter().from_framework(native)

    assert batch.trajectories[0].response == "four"
    restored = NemoRLAdapter().to_framework(batch)
    assert isinstance(restored, _PromptGroupRecord)
    assert isinstance(restored.completions[0], _Completion)
    assert restored.completions[0].message_log[0]["token_ids"] == [4, 5]


@pytest.mark.unit
def test_missing_optional_dependency_is_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = NemoRLAdapter()
    batch = adapter.from_framework(
        {
            "input_ids": np.asarray([[1]], dtype=np.int64),
            "input_lengths": np.asarray([1], dtype=np.int32),
        }
    )

    def missing_import(name: str) -> ModuleType:
        raise ModuleNotFoundError(name)

    monkeypatch.setattr("rlxfer.adapters.base.import_module", missing_import)
    with pytest.raises(MissingDependencyError, match=r"rl-experience-transfer\[nemo-rl\]"):
        adapter.to_framework(batch)
