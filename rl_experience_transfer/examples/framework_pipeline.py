# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU framework-pipeline checks using pinned upstream code and test-time wiring."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import inspect
import json
import os
import sys
import tempfile
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

from rlxfer.adapters import MilesAdapter, NemoRLAdapter, PrimeRLAdapter, SlimeAdapter
from rlxfer.api import ExperienceConsumer, ExperienceProducer
from rlxfer.transport import ReceiptState
from rlxfer.transports.filesystem import FileSystemTransport

FRAMEWORKS = ("nemo_rl", "prime_rl", "slime", "miles")


def _source_roots() -> dict[str, Path]:
    roots = {
        framework: Path(os.environ[f"RLXFER_{framework.upper()}_SOURCE"]).resolve()
        for framework in FRAMEWORKS
    }
    missing = [str(root) for root in roots.values() if not root.is_dir()]
    if missing:
        raise FileNotFoundError(f"framework source directories do not exist: {missing}")
    return roots


def _source_package(name: str, path: Path) -> ModuleType:
    package = ModuleType(name)
    package.__path__ = [str(path)]
    sys.modules[name] = package
    return package


def install_framework_source_shims() -> None:
    """Bypass eager GPU launch imports while retaining pinned source modules."""

    roots = _source_roots()

    _source_package("prime_rl.transport", roots["prime_rl"] / "prime_rl" / "transport")

    miles_data_source = ModuleType("miles.rollout.data_source")
    miles_data_source.DataSource = object  # type: ignore[attr-defined]
    sys.modules[miles_data_source.__name__] = miles_data_source

    nemo_root = roots["nemo_rl"] / "nemo_rl"
    nemo_logger = ModuleType("nemo_rl.utils.logger")
    nemo_logger.Logger = object  # type: ignore[attr-defined]
    sys.modules[nemo_logger.__name__] = nemo_logger

    nemo_policy = _source_package("nemo_rl.models.policy", nemo_root / "models" / "policy")
    nemo_policy.TokenizerConfig = dict  # type: ignore[attr-defined]
    _source_package("nemo_rl.models.generation", nemo_root / "models" / "generation")
    _source_package(
        "nemo_rl.algorithms.async_utils",
        nemo_root / "algorithms" / "async_utils",
    )
    nemo_replay = ModuleType("nemo_rl.algorithms.async_utils.replay_buffer")
    nemo_replay.TQReplayBuffer = object  # type: ignore[attr-defined]
    sys.modules[nemo_replay.__name__] = nemo_replay
    _source_package("nemo_rl.algorithms.loss", nemo_root / "algorithms" / "loss")


class _TransferBridge:
    def __init__(self, queue: Path, adapter: Any) -> None:
        self._producer = ExperienceProducer(FileSystemTransport(queue), adapter=adapter)
        self._consumer = ExperienceConsumer(FileSystemTransport(queue), adapter=adapter)
        self._receipt: Any = None
        self._delivery: Any = None

    def __enter__(self) -> _TransferBridge:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback
        if self._delivery is not None and not self._delivery.settled:
            self._delivery.reject("framework pipeline did not finish")
        self._producer.close()
        self._consumer.close()

    def publish(self, native: object) -> None:
        if self._receipt is not None:
            raise RuntimeError("the pipeline bridge accepts one native batch")
        self._receipt = self._producer.publish(native, timeout=5.0)

    def receive(self) -> object:
        self._delivery = self._consumer.receive(timeout=5.0)
        if self._delivery is None:
            raise TimeoutError("framework pipeline transfer timed out")
        return self._delivery.to_framework()

    def settle(self) -> dict[str, object]:
        if self._delivery is None or self._receipt is None:
            raise RuntimeError("cannot settle an incomplete framework transfer")
        self._delivery.ack()
        result = self._receipt.wait(timeout=5.0)
        if result.state is not ReceiptState.ACKED:
            raise RuntimeError(f"unexpected receipt state {result.state.value!r}")
        return {
            "acknowledged": True,
            "canonical_schema": self._delivery.batch.metadata.schema_version,
            "experience_id": self._delivery.experience_id,
            "trajectories": len(self._delivery.batch.trajectories),
        }


def _source_file(framework: str, component: Any) -> str:
    source = inspect.getsourcefile(inspect.unwrap(component))
    if source is None:
        raise RuntimeError(f"cannot locate source for {component!r}")
    root = _source_roots()[framework]
    path = Path(source).resolve()
    try:
        return str(path.relative_to(root))
    except ValueError as error:
        message = f"{component!r} loaded from {path}, outside pinned source {root}"
        raise RuntimeError(message) from error


def _optimizer_update(torch: Any, model: Any, loss: Any) -> dict[str, object]:
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    gradient_finite = bool(gradients) and all(
        bool(torch.isfinite(gradient).all()) for gradient in gradients
    )
    optimizer.step()
    parameter_changed = any(
        not torch.equal(previous, parameter.detach())
        for previous, parameter in zip(before, model.parameters(), strict=True)
    )
    return {
        "gradient_finite": gradient_finite,
        "gradient_present": bool(gradients),
        "loss": float(loss.detach()),
        "parameter_changed": parameter_changed,
    }


def _prime_pipeline(queue: Path) -> dict[str, object]:
    torch: Any = importlib.import_module("torch")

    base: Any = importlib.import_module("prime_rl.transport.base")
    types: Any = importlib.import_module("prime_rl.transport.types")
    batch_module: Any = importlib.import_module("prime_rl.trainer.batch")
    loss_module: Any = importlib.import_module("prime_rl.trainer.rl.loss")
    configs: Any = importlib.import_module("prime_rl.configs.trainer")

    with _TransferBridge(queue, PrimeRLAdapter()) as bridge:

        class Sender(base.TrainingBatchSender):  # type: ignore[misc]
            async def send(self, batch: object) -> None:
                bridge.publish(batch)

        class Receiver(base.TrainingBatchReceiver):  # type: ignore[misc]
            def can_receive(self) -> bool:
                return True

            def receive(self) -> list[object]:
                return [bridge.receive()]

        native = types.TrainingBatch(
            examples=[
                types.TrainingSample(
                    prompt_ids=[1, 2],
                    prompt_mask=[False, False],
                    completion_ids=[3, 4],
                    completion_mask=[True, True],
                    completion_logprobs=[-0.1, -0.2],
                    completion_temperatures=[1.0, 1.0],
                    env_name="rlxfer-ci",
                    advantage=1.0,
                    reward=1.0,
                )
            ],
            step=3,
            run_idx=0,
        )
        sender = Sender(queue)
        receiver = Receiver()
        asyncio.run(sender.send(native))
        restored = receiver.receive()[0]
        if not isinstance(restored, types.TrainingBatch):
            raise TypeError(f"expected TrainingBatch, received {type(restored).__name__}")

        micro = batch_module.prepare_sample(restored.examples[0], seq_len=8)
        input_ids = torch.tensor([micro.input_ids], dtype=torch.long)
        inference_logprobs = torch.tensor([micro.inference_logprobs], dtype=torch.float32)
        advantages = torch.tensor([micro.advantages], dtype=torch.float32)
        loss_mask = torch.tensor([micro.loss_mask], dtype=torch.bool)
        temperatures = torch.tensor([micro.temperatures], dtype=torch.float32)
        torch.manual_seed(11)
        model = torch.nn.Sequential(torch.nn.Embedding(16, 8), torch.nn.Linear(8, 16))
        logits = model(input_ids) / temperatures.unsqueeze(-1)
        labels = loss_module.shift_tensor_left(input_ids)
        trainer_logprobs = loss_module.selective_log_softmax(logits, labels)
        trainer_logprobs = loss_module.shift_tensor_right(trainer_logprobs, pad_value=0.0)
        loss, _ = loss_module.compute_loss(
            trainer_logprobs=[trainer_logprobs[0]],
            inference_logprobs=[inference_logprobs[0]],
            teacher_logprobs=None,
            advantages=[advantages[0]],
            loss_mask=[loss_mask[0]],
            loss_fns=loss_module.setup_loss_fns(configs.DefaultLossConfig()),
            loss_scale=max(int(loss_mask.sum()), 1),
            training_mode=restored.examples[0].training_mode,
        )
        update = _optimizer_update(torch, model, loss)
        settlement = bridge.settle()

    return {
        "framework": "prime_rl",
        "rollout_component": "prime_rl.transport.base.TrainingBatchSender.send",
        "rollout_source": _source_file("prime_rl", base.TrainingBatchSender.send),
        "trainer_component": "prime_rl.trainer.rl.loss.compute_loss",
        "trainer_source": _source_file("prime_rl", loss_module.compute_loss),
        "native_training_type": type(restored).__name__,
        "external_fixtures": "synthetic TrainingBatch; CPU tiny policy",
        "result": "PASSED",
        **settlement,
        **update,
    }


def _grouped_pipeline(framework: str, queue: Path) -> dict[str, object]:
    torch: Any = importlib.import_module("torch")

    outputs: Any = importlib.import_module(f"{framework}.rollout.base_types")
    types: Any = importlib.import_module(f"{framework}.utils.types")
    adapter: Any
    if framework == "slime":
        loss_module = importlib.import_module("slime.utils.ppo_utils")
        adapter = SlimeAdapter()
    else:
        loss_module = importlib.import_module("miles.backends.training_utils.loss_hub.math_utils")
        adapter = MilesAdapter()

    class DataSource:
        def get_samples(self, rollout_id: int) -> list[list[object]]:
            sample = types.Sample(
                index=rollout_id,
                group_index=0,
                rollout_id=rollout_id,
                prompt="Count to two.",
                tokens=[1, 2, 3, 4],
                response="one two",
                response_length=2,
                reward=1.0,
                loss_mask=[1, 1],
                weight_versions=[str(rollout_id)],
                rollout_log_probs=[-0.1, -0.2],
                status=types.Sample.Status.COMPLETED,
            )
            return [[sample]]

    def rollout_plugin(
        args: object,
        rollout_id: int,
        data_source: DataSource,
        evaluation: bool = False,
    ) -> object:
        del args
        if evaluation:
            raise ValueError("the training pipeline fixture does not run evaluation")
        return outputs.RolloutFnTrainOutput(
            samples=data_source.get_samples(rollout_id),
            metrics={"fixture": "cpu"},
        )

    native = outputs.call_rollout_fn(
        rollout_plugin,
        None,
        5,
        DataSource(),
        evaluation=False,
    )
    with _TransferBridge(queue, adapter) as bridge:
        bridge.publish(native)
        restored = bridge.receive()
        if not isinstance(restored, outputs.RolloutFnTrainOutput):
            raise TypeError(f"expected RolloutFnTrainOutput, received {type(restored).__name__}")
        sample = restored.samples[0][0]
        response_tokens = torch.tensor(sample.tokens[-sample.response_length :], dtype=torch.long)
        old_logprobs = torch.tensor(sample.rollout_log_probs, dtype=torch.float32)
        advantages = torch.full_like(old_logprobs, float(sample.reward))
        torch.manual_seed(13)
        model = torch.nn.Sequential(torch.nn.Embedding(16, 8), torch.nn.Linear(8, 16))
        logits = model(response_tokens)
        current_logprobs = (
            logits.log_softmax(-1).gather(-1, response_tokens.unsqueeze(-1)).squeeze(-1)
        )
        losses, _ = loss_module.compute_policy_loss(
            old_logprobs - current_logprobs,
            advantages,
            0.2,
            0.2,
        )
        update = _optimizer_update(torch, model, losses.mean())
        settlement = bridge.settle()

    return {
        "framework": framework,
        "rollout_component": f"{framework}.rollout.base_types.call_rollout_fn",
        "rollout_source": _source_file(framework, outputs.call_rollout_fn),
        "trainer_component": f"{loss_module.__name__}.compute_policy_loss",
        "trainer_source": _source_file(framework, loss_module.compute_policy_loss),
        "native_training_type": type(restored).__name__,
        "external_fixtures": "test-time rollout plugin; CPU tiny policy",
        "result": "PASSED",
        **settlement,
        **update,
    }


def _nemo_pipeline(queue: Path) -> dict[str, object]:
    torch: Any = importlib.import_module("torch")

    manager_module: Any = importlib.import_module("nemo_rl.experience.rollout_manager")
    interfaces: Any = importlib.import_module("nemo_rl.environments.interfaces")
    batched: Any = importlib.import_module("nemo_rl.distributed.batched_data_dict")
    loss_module: Any = importlib.import_module("nemo_rl.algorithms.loss.loss_functions")

    class Tokenizer:
        def decode(self, tokens: object, *, skip_special_tokens: bool) -> str:
            del tokens, skip_special_tokens
            return "one two"

        def __call__(self, text: str, **kwargs: object) -> SimpleNamespace:
            del text, kwargs
            return SimpleNamespace(input_ids=torch.tensor([[5]], dtype=torch.long))

    class Generation:
        calls = 0

        async def generate_async(self, data: Any) -> Any:
            self.calls += 1
            prompt_ids = data["input_ids"]
            generated = torch.tensor([[3, 4]], dtype=torch.long)
            output_ids = torch.cat([prompt_ids, generated], dim=1)
            logprobs = torch.zeros_like(output_ids, dtype=torch.float32)
            logprobs[:, -2:] = torch.tensor([[-0.1, -0.2]])
            yield (
                0,
                batched.BatchedDataDict(
                    {
                        "output_ids": output_ids,
                        "generation_lengths": torch.tensor([2]),
                        "unpadded_sequence_lengths": torch.tensor([output_ids.shape[1]]),
                        "logprobs": logprobs,
                        "truncated": torch.tensor([False]),
                    }
                ),
            )

    def calculate_rewards(data: Any, task_to_env: object) -> object:
        del task_to_env
        return interfaces.EnvironmentReturn(
            observations=[{"role": "user", "content": "done"}],
            metadata=data["extra_env_info"],
            next_stop_strings=[None],
            rewards=torch.tensor([1.0]),
            terminateds=torch.tensor([True]),
            answers=[None],
        )

    with _TransferBridge(queue, NemoRLAdapter()) as bridge:

        class TransferBuffer:
            def __init__(self) -> None:
                self.group_id: str | None = None

            def reserve(self, *, weight_version: int, target_step: int | None) -> str:
                self.group_id = f"nemo-{weight_version}-{target_step}"
                return self.group_id

            async def commit(
                self,
                group_id: str,
                record: object,
                start_weight_version: int,
                end_weight_version: int,
            ) -> None:
                if group_id != self.group_id or start_weight_version != end_weight_version:
                    raise RuntimeError("NeMo rollout version or reservation changed")
                bridge.publish(record)

            async def remove_group(self, group_id: str) -> int:
                if group_id != self.group_id:
                    raise ValueError(f"unknown group {group_id!r}")
                self.group_id = None
                return 1

        generation = Generation()
        manager = manager_module.RolloutManager(
            tokenizer=Tokenizer(),
            task_to_env={"cpu": object()},
            num_generations_per_prompt=2,
            max_seq_len=16,
            max_rollout_turns=1,
            policy_generation=generation,
            tq_buffer=TransferBuffer(),
        )
        manager.set_weight_version(4)
        original_rewards = manager_module.calculate_rewards
        manager_module.calculate_rewards = calculate_rewards
        try:
            asyncio.run(
                manager.generate_and_push(
                    {
                        "idx": 7,
                        "message_log": [
                            {
                                "role": "user",
                                "content": "Count to two.",
                                "token_ids": torch.tensor([1, 2]),
                            }
                        ],
                        "extra_env_info": {},
                        "task_name": "cpu",
                        "stop_strings": None,
                    },
                    target_step=5,
                )
            )
        finally:
            manager_module.calculate_rewards = original_rewards

        restored = bridge.receive()
        record_type = importlib.import_module("nemo_rl.experience.interfaces").PromptGroupRecord
        if not isinstance(restored, record_type):
            raise TypeError(f"expected PromptGroupRecord, received {type(restored).__name__}")

        response_tokens: list[list[int]] = []
        response_logprobs: list[list[float]] = []
        for completion in restored.completions:
            assistant = next(
                message
                for message in completion.message_log
                if message.get("generation_logprobs") is not None
            )
            response_tokens.append(torch.as_tensor(assistant["token_ids"]).tolist())
            response_logprobs.append(
                torch.as_tensor(assistant["generation_logprobs"]).float().tolist()
            )

        tokens = torch.tensor([[0, *values] for values in response_tokens], dtype=torch.long)
        old = torch.tensor([[0.0, *values] for values in response_logprobs])
        mask = torch.tensor([[0.0, 1.0, 1.0]] * len(response_tokens))
        advantages = torch.tensor([[0.0, sign, sign] for sign in (1.0, -1.0)], dtype=torch.float32)
        training_data = batched.BatchedDataDict(
            {
                "input_ids": tokens,
                "advantages": advantages,
                "prev_logprobs": old,
                "generation_logprobs": old,
                "token_mask": mask,
                "sample_mask": torch.ones(len(response_tokens)),
            }
        )
        torch.manual_seed(17)
        model = torch.nn.Sequential(torch.nn.Embedding(16, 8), torch.nn.Linear(8, 16))
        logits = model(tokens[:, 1:])
        current = logits.log_softmax(-1).gather(-1, tokens[:, 1:].unsqueeze(-1)).squeeze(-1)
        loss_fn = loss_module.ClippedPGLossFn(
            loss_module.ClippedPGLossConfig(reference_policy_kl_penalty=0.0)
        )
        loss, _ = loss_fn(
            current,
            training_data,
            global_valid_seqs=torch.tensor(float(len(response_tokens))),
            global_valid_toks=mask[:, 1:].sum(),
        )
        update = _optimizer_update(torch, model, loss)
        settlement = bridge.settle()

    return {
        "framework": "nemo_rl",
        "rollout_component": "nemo_rl.experience.rollout_manager.RolloutManager.generate_and_push",
        "rollout_source": _source_file("nemo_rl", manager_module.RolloutManager.generate_and_push),
        "trainer_component": "nemo_rl.algorithms.loss.loss_functions.ClippedPGLossFn",
        "trainer_source": _source_file("nemo_rl", loss_module.ClippedPGLossFn.__call__),
        "native_rollout_type": type(restored).__name__,
        "native_training_type": type(training_data).__name__,
        "external_fixtures": "deterministic generation/environment; CPU tiny policy",
        "generation_calls": generation.calls,
        "result": "PASSED",
        **settlement,
        **update,
    }


def run_framework_pipeline(framework: str, queue: Path) -> dict[str, object]:
    """Run one pinned framework rollout-to-loss component pipeline."""

    if framework not in FRAMEWORKS:
        raise ValueError(f"unknown framework {framework!r}")
    if framework == "nemo_rl":
        return _nemo_pipeline(queue)
    if framework == "prime_rl":
        return _prime_pipeline(queue)
    return _grouped_pipeline(framework, queue)


def _markdown(results: list[dict[str, object]]) -> str:
    lines = [
        "## RL Experience framework pipeline components",
        "",
        "| Framework | Real rollout boundary | Real training component | "
        "CPU-only stand-ins | Result |",
        "|---|---|---|---|---|",
    ]
    for result in results:
        lines.append(
            "| {framework} | `{rollout_component}` | `{trainer_component}` | "
            "{external_fixtures} | {result} |".format(**result)
        )
    lines.extend(
        [
            "",
            "Each row imports the named boundary from the pinned upstream checkout, transfers "
            "native experience through `ExperienceProducer` and `ExperienceConsumer`, runs a "
            "finite backward/optimizer step, and acknowledges only after the update.",
            "GPU model servers and distributed trainer engines remain separate accelerator gates.",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("framework", choices=(*FRAMEWORKS, "all"))
    parser.add_argument("--format", choices=("json", "markdown"), default="json")
    args = parser.parse_args()
    install_framework_source_shims()
    selected = FRAMEWORKS if args.framework == "all" else (args.framework,)
    with tempfile.TemporaryDirectory(prefix="rlxfer-framework-pipeline-") as temporary:
        root = Path(temporary)
        results = [run_framework_pipeline(name, root / name) for name in selected]
    print(_markdown(results) if args.format == "markdown" else json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
