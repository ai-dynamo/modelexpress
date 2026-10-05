"""CUDA reuse, engine reload and graph-address checks for generic installation.

The composite contains native vLLM MLA and MoE implementations. It is a small
installer probe, not a full Qwen/DeepSeek forward or a distributed benchmark.
"""

import argparse
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import vllm
from modelexpress.refit.timing import RefitTimingRecorder, use_refit_timing
from modelexpress_rl.inference.engines.vllm.installer import _VllmInstaller
from modelexpress_rl.inference.plan import PreparedStreamingTensors
from torch import nn
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.attention.attention import set_default_quant_scales
from vllm.model_executor.layers.attention.mla_attention import MLAAttention
from vllm.model_executor.layers.layernorm import LayerNorm
from vllm.model_executor.model_loader.reload.layerwise import (
    record_metadata_for_reloading,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader


class FixtureMLA(MLAAttention):
    def __init__(self):
        nn.Module.__init__(self)
        self.num_heads, self.qk_nope_head_dim = 2, 192
        self.v_head_dim, self.kv_lora_rank = 256, 512
        self.dcp_q_replicate = False
        self.is_aiter_triton_fp4_bmm_enabled = False
        self.is_aiter_triton_fp8_bmm_enabled = False
        self.is_amx_bmm_enabled = False
        self.quant_config = None
        self.layer_name = "generic_reload_probe"
        self.impl = SimpleNamespace(process_weights_after_loading=lambda dtype: None)
        self.kv_b_proj = nn.Linear(512, 896, bias=False, dtype=torch.bfloat16)
        self.kv_b_proj.quant_method = None
        set_default_quant_scales(self, register_buffer=True)


def native_moe():
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEConfig,
        FusedMoEParallelConfig,
        RoutingMethodType,
    )
    from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts
    from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEKernel
    from vllm.model_executor.layers.fused_moe.prepare_finalize.no_dp_ep import (
        MoEPrepareAndFinalizeNoDPEPModular,
    )
    from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
        UnquantizedFusedMoEMethod,
    )

    parallel = FusedMoEParallelConfig(
        tp_size=1,
        pcp_size=1,
        dp_size=1,
        ep_size=1,
        tp_rank=0,
        pcp_rank=0,
        dp_rank=0,
        ep_rank=0,
        sp_size=1,
        use_ep=False,
        all2all_backend="allgather_reducescatter",
        enable_eplb=False,
    )
    config = FusedMoEConfig(
        num_experts=8,
        experts_per_token=2,
        hidden_dim=128,
        intermediate_size=256,
        num_local_experts=8,
        num_logical_experts=8,
        activation=MoEActivation.SILU,
        device=torch.device("cuda", 0),
        routing_method=RoutingMethodType.DeepSeekV3,
        moe_parallel_config=parallel,
        in_dtype=torch.bfloat16,
        router_logits_dtype=torch.float32,
        moe_backend="triton",
        max_num_tokens=32,
    )
    method = UnquantizedFusedMoEMethod(config)
    layer = nn.Module()
    layer.moe_config = config
    method.create_weights(layer, 8, 128, 256, torch.bfloat16)
    layer.quant_method = method
    # Native constructors establish the same modular kernel used for subsequent
    # post-load updates. This avoids pretending the fixture is a RoutedExperts
    # model or inventing routing-table internals for initial kernel creation.
    quant = method.get_fused_moe_quant_config(layer)
    method.moe_quant_config = quant
    method.moe_kernel = FusedMoEKernel(
        MoEPrepareAndFinalizeNoDPEPModular(), TritonExperts(config, quant)
    )
    return layer


class Composite(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(128, 128, bias=False, dtype=torch.bfloat16)
        self.attention = FixtureMLA()
        self.indexer = nn.Module()
        self.indexer.k_norm = LayerNorm(128, eps=1e-6)
        self.attention.indexer = self.indexer
        self.output_norm = LayerNorm(128, eps=1e-6)
        self.output_norm.bias = self.indexer.k_norm.bias
        self.experts = native_moe()

    def load_weights(self, weights):
        for name, value in weights:
            parameter = self.get_parameter(name)
            getattr(parameter, "weight_loader", default_weight_loader)(parameter, value)


def geometry(model):
    return {
        name: (id(value), value.data_ptr(), tuple(value.shape), tuple(value.stride()))
        for name, value in model.named_parameters(remove_duplicate=False)
    }


def initialize(config):
    with torch.device("cuda"), set_current_vllm_config(config):
        model = Composite()
        for index, value in enumerate(model.parameters()):
            value.fill_((index + 1) / 64)
        manifest = [
            (name, p.dtype, tuple(p.shape)) for name, p in model.named_parameters()
        ]
        record_metadata_for_reloading(model)
    with set_current_vllm_config(config):
        model.experts.quant_method.process_weights_after_loading(model.experts)
        model.attention.process_weights_after_loading(torch.bfloat16)
    assert model.attention._k_scale_cpu.device.type == "cpu"
    assert model.attention._v_scale_cpu.device.type == "cpu"
    return model, manifest


def consumer(model, x):
    # Consumers capture real destination pointers, including derived MLA tensors.
    # This is intentionally not a claim about a complete attention/MoE forward.
    return (
        torch.bmm(x, model.attention.W_UV),
        model.dense.weight.float().sum(),
        model.experts.w13_weight.float().sum(),
        model.experts.w2_weight.float().sum(),
    )


@torch.inference_mode()
def qualify(packed):
    config = VllmConfig()
    model, manifest = initialize(config)
    reference, reference_manifest = initialize(config)
    assert manifest == reference_manifest
    loader = _VllmInstaller(
        model=model,
        vllm_config=config,
        model_config=SimpleNamespace(dtype=torch.bfloat16),
        device=torch.device("cuda"),
    )
    addresses = geometry(model)
    x = torch.full((2, 3, 512), 0.125, device="cuda", dtype=torch.bfloat16)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            consumer(model, x)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_values = consumer(model, x)
    records = []
    previous = None
    for step in range(3):
        _, layout = loader.capture(manifest)
        values = {
            name: (
                (
                    (torch.arange(torch.Size(shape).numel()).reshape(shape) % 101 - 50)
                    / 128
                )
                + (step + 1) / 16
            ).to(dtype)
            for name, dtype, shape in manifest
        }
        assert set(layout) == set(values), (set(layout), set(values))
        groups = {}
        for name, value in values.items():
            groups.setdefault(name.rsplit(".", 1)[0], {})[name] = value
        batches = [values] if packed else list(groups.values())
        arena_bytes = max(
            sum(v.numel() * v.element_size() for v in b.values()) for b in batches
        )
        arena = torch.empty(arena_bytes, device="cuda", dtype=torch.uint8)
        exhausted = []

        def refill():
            for batch in batches:
                arena.fill_(0xA5)
                offset = 0
                staged = {}
                for name, value in batch.items():
                    length = value.numel() * value.element_size()
                    view = (
                        arena[offset : offset + length]
                        .view(value.dtype)
                        .reshape(value.shape)
                    )
                    view.copy_(value)
                    staged[name] = view
                    offset += length
                yield staged
            # A callback retaining an arena view observes poison after completion.
            arena.fill_(0xA5)
            exhausted.append(True)

        prepared = PreparedStreamingTensors(
            batches=refill,
            parameter_names=frozenset(values),
            transfer_metrics={},
        )
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        recorder = RefitTimingRecorder(
            backend="generic_probe", version=f"packed={packed}:{step}", rank=0
        )
        with use_refit_timing(recorder):
            loader.install_streaming(prepared)
        recorder.finish()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        assert exhausted == [True]
        for name, value in values.items():
            assert torch.equal(model.get_parameter(name).cpu(), value), name
            reference.get_parameter(name).copy_(value)
        # The oracle uses the actual engine implementation on independent storage.
        reference.experts.quant_method.process_weights_after_loading(reference.experts)
        reference.attention.process_weights_after_loading(torch.bfloat16)
        for name in ("W_UV", "W_UK_T"):
            assert torch.equal(
                getattr(model.attention, name), getattr(reference.attention, name)
            ), name
        for name in ("_k_scale_cpu", "_v_scale_cpu"):
            actual, expected_host = (
                getattr(model.attention, name),
                getattr(reference.attention, name),
            )
            assert actual.device.type == "cpu" and expected_host.device.type == "cpu", (
                name
            )
            assert torch.equal(actual, expected_host), name
        expected = consumer(reference, x)
        graph.replay()
        torch.cuda.synchronize()
        assert all(
            torch.equal(actual, want)
            for actual, want in zip(graph_values, expected, strict=True)
        )
        assert previous is None or not torch.equal(previous, expected[0])
        previous = expected[0].clone()
        assert geometry(model) == addresses
        assert model.attention.indexer is model.indexer
        assert model.output_norm.bias is model.indexer.k_norm.bias
        records.append(
            {
                "step": step,
                "batches": len(batches),
                "arena_bytes": arena_bytes,
                "exact_parameters": True,
                "exact_native_mla_state": True,
                "native_moe_post_load": True,
                "graph_consumers_exact": True,
                "arena_poisoned_after_reuse": True,
                "stable_destinations_and_aliases": True,
                "cuda_allocated_before_install_bytes": baseline,
                "cuda_peak_allocated_bytes": peak,
                "cuda_peak_incremental_install_bytes": peak - baseline,
                "metrics": prepared.transfer_metrics,
                "native_stage_timing": recorder.as_dict(),
            }
        )
    return {"packed": packed, "records": records}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert vllm.__version__ == "0.30.0", vllm.__version__
    assert torch.cuda.device_count() == 1
    result = {
        "schema": "mx-generic-streaming-cuda-v1",
        "passed": False,
        "vllm": vllm.__version__,
        "torch": torch.__version__,
        "arms": [],
        "scope": "Native vLLM reload and post-load implementations on a composite CUDA fixture. Independent receive arena reused and poisoned, tied/shared owners, captured graph consumers, peak allocation. Not full MLA/MoE forward or transport qualification.",
        "sources": {
            str(Path(inspect.getfile(cls))): hashlib.sha256(
                Path(inspect.getfile(cls)).read_bytes()
            ).hexdigest()
            for cls in (_VllmInstaller, MLAAttention)
        },
    }
    try:
        for packed in (False, True):
            result["arms"].append(qualify(packed))
        result["passed"] = True
    except BaseException as error:
        result["error"] = repr(error)
        raise
    finally:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
