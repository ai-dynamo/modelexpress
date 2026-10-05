"""Run a real PrimeRL worker and refit a small tied Llama through public MX APIs."""

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path


def verify_worker(worker, source_path):
    import torch
    from safetensors.torch import load_file

    model = worker.model_runner.get_model()
    source = load_file(source_path)
    expected = {}
    for name, value in source.items():
        if ".self_attn.q_proj." in name or ".mlp.gate_proj." in name:
            if ".self_attn.q_proj." in name:
                destination = name.replace("q_proj", "qkv_proj")
                values = [
                    source[name.replace("q_proj", part)]
                    for part in ("q_proj", "k_proj", "v_proj")
                ]
            else:
                destination = name.replace("gate_proj", "gate_up_proj")
                values = [value, source[name.replace("gate_proj", "up_proj")]]
            expected[destination] = torch.cat(values)
        elif any(part in name for part in (".k_proj.", ".v_proj.", ".up_proj.")):
            continue
        else:
            expected[name] = value
    actual = dict(model.named_parameters())
    assert set(actual) == set(expected), (
        set(actual) - set(expected),
        set(expected) - set(actual),
    )
    for name, parameter in actual.items():
        assert torch.equal(parameter.cpu(), expected[name]), name
    assert model.lm_head.weight is model.model.embed_tokens.weight
    digest = hashlib.sha256()
    for name in sorted(actual):
        digest.update(name.encode())
        digest.update(
            actual[name].detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        )
    return {
        "parameters": len(actual),
        "exact_parameters": True,
        "tied_embedding_identity": True,
        "sha256": digest.hexdigest(),
        "model_class": f"{type(model).__module__}.{type(model).__name__}",
    }


def generate(llm):
    from vllm import SamplingParams

    output = llm.generate(
        [{"prompt_token_ids": [1, 7, 19, 23]}],
        SamplingParams(temperature=0, max_tokens=8, ignore_eos=True, logprobs=1),
        use_tqdm=False,
    )[0].outputs[0]
    assert len(output.token_ids) == 8
    assert output.logprobs is not None
    assert all(
        math.isfinite(item.logprob)
        for token in output.logprobs
        for item in token.values()
    )
    return {"token_ids": list(output.token_ids), "finite_logprobs": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--server-url", required=True)
    args = parser.parse_args()
    # Every complete owner fits, while the model requires repeated arena reuse.
    os.environ["MX_REFIT_STAGING_BYTES"] = str(192 * 1024)
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    import torch
    import torch.distributed as dist
    import vllm
    from modelexpress_rl import (
        FSDPTrainerContext,
        ModelExpressControlClient,
        ModelExpressTrainerClient,
        ModelExpressTrainerConfig,
        TrainerStagingMode,
        WeightPayloadFormat,
        WeightVersionState,
    )
    from safetensors.torch import save_file
    from transformers import LlamaConfig, LlamaForCausalLM
    from vllm import LLM

    assert vllm.__version__ == "0.30.0", vllm.__version__
    assert torch.cuda.device_count() == 1
    args.work_dir.mkdir(parents=True, exist_ok=False)
    checkpoint = args.work_dir / "tiny-tied-llama"
    torch.manual_seed(1729)
    config = LlamaConfig(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        tie_word_embeddings=True,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    reference = LlamaForCausalLM(config).to(torch.bfloat16)
    reference.save_pretrained(checkpoint)
    tensors = {
        name: parameter.detach() for name, parameter in reference.named_parameters()
    }
    expected_path = args.work_dir / "expected.safetensors"
    save_file(tensors, expected_path)
    result = {
        "schema": "mx-generic-direct-tied-worker-v1",
        "passed": False,
        "torch": torch.__version__,
        "vllm": vllm.__version__,
        "updates": [],
        "scope": "One H100, real PrimeRL vLLM worker, local tied Llama and public MX apply_weight_streaming via NIXL; COPY_TO_HOST trainer. No distributed or full GLM claim.",
        "receive_arena_cap_bytes": 192 * 1024,
        "trainer_staging_mode": "COPY_TO_HOST",
        "generator_install_mode": "DIRECT",
    }
    trainer = control = llm = None
    generator_initialized = False
    try:
        llm = LLM(
            model=str(checkpoint),
            skip_tokenizer_init=True,
            dtype="bfloat16",
            tensor_parallel_size=1,
            max_model_len=64,
            max_num_seqs=1,
            enforce_eager=True,
            enable_prefix_caching=False,
            kv_cache_memory_bytes=64 * 1024**2,
            seed=1729,
            worker_extension_cls="tied_worker.TiedProbeWorker",
        )
        result["initial_generation"] = generate(llm)
        result["initial_verification"] = llm.collective_rpc(
            "verify_probe_weights", args=(str(expected_path),)
        )
        server_host, server_port = args.server_url.rsplit(":", 1)
        llm.collective_rpc("init_broadcaster", args=(server_host, int(server_port)))
        generator_initialized = True
        dist.init_process_group(
            "gloo",
            init_method=f"file://{args.work_dir / 'trainer-rendezvous'}",
            rank=0,
            world_size=1,
        )
        trainer = ModelExpressTrainerClient.initialize(
            ModelExpressTrainerConfig(
                engine_context=FSDPTrainerContext(),
                device_id=0,
                model_name=str(checkpoint),
                server_url=args.server_url,
                staging_mode=TrainerStagingMode.COPY_TO_HOST,
                payload_format=WeightPayloadFormat.FULL_TENSOR,
            )
        )
        slot = trainer.bind_tensors(tensors)
        control = ModelExpressControlClient.connect(server_url=args.server_url)
        previous = result["initial_verification"][0]["sha256"]
        for step in range(3):
            with torch.no_grad():
                for parameter in reference.parameters():
                    parameter.add_(0.00390625 * (step + 1))
            save_file(tensors, expected_path)
            version = control.create_weight_version(
                model_name=str(checkpoint),
                idempotency_key=f"gate-b:{step}",
                uid=f"gate-b:{step}",
                payload_format=WeightPayloadFormat.FULL_TENSOR,
                expected_source_slots=[slot],
            )
            trainer.publish_version(version=version.ref)
            deadline = time.monotonic() + 60
            while (
                control.get_weight_version(version.version_id).state
                is not WeightVersionState.READY
            ):
                if time.monotonic() >= deadline:
                    raise TimeoutError("Published version did not become READY")
                time.sleep(0.05)
            llm.collective_rpc("begin_probe_measurement")
            llm.collective_rpc(
                "update_weights_from_path", kwargs={"version_uid": version.version_id}
            )
            verification = llm.collective_rpc(
                "verify_probe_weights", args=(str(expected_path),)
            )
            assert verification[0]["sha256"] != previous
            previous = verification[0]["sha256"]
            result["updates"].append(
                {
                    "step": step,
                    "verification": verification,
                    "generation": generate(llm),
                    "memory": llm.collective_rpc("end_probe_measurement"),
                }
            )
            control.delete_weight_version(version.version_id)
            trainer.release_version(version=version.ref)
        result["passed"] = True
    finally:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
        if generator_initialized:
            llm.collective_rpc("close_probe_generator")
        if trainer is not None:
            trainer.close()
        if control is not None:
            control.close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
