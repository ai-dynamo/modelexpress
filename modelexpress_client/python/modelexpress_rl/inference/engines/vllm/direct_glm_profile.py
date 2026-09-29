# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned GLM-5 eager BF16 campaign profile; not a general vLLM allowlist.

Source hashes pin inspected vLLM 0.30.0 and Torch 2.13.0 runtime files.
The registry selects the CUDA model below. The adapter additionally validates
live configuration, helpers and destinations. Full-model admission and refit
qualification are still required before relying on this source profile.
"""

VLLM_VERSION = "0.30.0"
MODEL_CLASS = "vllm.models.deepseek_v32.nvidia.model.DeepseekV32ForCausalLM"

MODULE_COUNTS = {
    "vllm.models.deepseek_v32.nvidia.model.DeepseekV32ForCausalLM": 1,
    "vllm.models.deepseek_v32.nvidia.model.DeepseekV32Model": 1,
    "vllm.model_executor.layers.vocab_parallel_embedding.VocabParallelEmbedding": 1,
    "torch.nn.modules.container.ModuleList": 1,
    "vllm.models.deepseek_v32.nvidia.model.DeepseekV32DecoderLayer": 78,
    "vllm.models.deepseek_v32.attention.DeepseekV32Attention": 78,
    "vllm.model_executor.models.deepseek_v2.DeepSeekV2FusedQkvAProjLinear": 78,
    "vllm.model_executor.layers.layernorm.RMSNorm": 313,
    "vllm.model_executor.layers.linear.ColumnParallelLinear": 156,
    "vllm.model_executor.layers.linear.RowParallelLinear": 156,
    "vllm.model_executor.layers.rotary_embedding.base.RotaryEmbedding": 1,
    "vllm.model_executor.layers.rotary_embedding.common.ApplyRotaryEmb": 1,
    "vllm.models.deepseek_v32.attention.DeepseekV32Indexer": 78,
    "vllm.model_executor.layers.linear.ReplicatedLinear": 78,
    "vllm.model_executor.layers.linear.MergedColumnParallelLinear": 156,
    "vllm.model_executor.layers.layernorm.LayerNorm": 78,
    "vllm.model_executor.models.deepseek_v2.DeepseekV32IndexerCache": 78,
    "vllm.model_executor.layers.sparse_attn_indexer.SparseAttnIndexer": 78,
    "vllm.model_executor.layers.attention.mla_attention._DecodeConcatQuantFP8": 78,
    "vllm.model_executor.layers.quantization.input_quant_fp8.QuantFP8": 78,
    "vllm.model_executor.models.deepseek_v2.DeepseekV2MLP": 78,
    "vllm.model_executor.layers.activation.SiluAndMul": 78,
    "vllm.model_executor.models.deepseek_v2.DeepseekV2MoE": 75,
    "vllm.model_executor.layers.fused_moe.router.gate_linear.GateLinear": 75,
    "vllm.model_executor.layers.fused_moe.runner.moe_runner.MoERunner": 75,
    "vllm.model_executor.layers.fused_moe.routed_experts.RoutedExperts": 75,
    "vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method.UnquantizedFusedMoEMethod": 75,
    "vllm.model_executor.layers.fused_moe.runner.shared_experts.SharedExperts": 75,
    "vllm.model_executor.layers.vocab_parallel_embedding.ParallelLMHead": 1,
    "vllm.model_executor.layers.logits_processor.LogitsProcessor": 1,
}

SOURCE_HASHES = {
    "torch/nn/modules/container.py": "73417ef19501a3e0fef8c6377ab660aac8c3e9b5034f3e652e6c6c91afd68bb8",
    "torch/nn/modules/module.py": "341153e4099444f923f5b0766e8e41b45fbe909b87f4737cb28c828f297b7baa",
    "vllm/model_executor/custom_op.py": "967b74c14b8a8e69af8828300414fbd3a6954ec573a253ca0f67aa02b0ed473f",
    "vllm/model_executor/layers/activation.py": "3cbefe5f03c7f5223327ce1f18d3f5a1782964a07d64d7994a1e9e64c0e345d8",
    "vllm/model_executor/layers/attention/attention.py": "bc897e452f3aea603e353ca42a365d3be8836e9485d78cbf8cc4677baca2fb42",
    "vllm/model_executor/layers/attention/mla_attention.py": "4ec101d7c882910bd09ec677e2c42df5d9fe33f99417172136adb312761cb42e",
    "vllm/model_executor/layers/attention/sparse_mla_attention.py": "ae0aa4fee580a5fcb22f912078d2df142dd7e576c790383211e5c06e484c0e27",
    "vllm/model_executor/layers/fused_allreduce_gemma_rms_norm.py": "aec049372738a38fd644305c51e3a0068b008f3ca16da9dfca57191c434ab03f",
    "vllm/model_executor/layers/fused_embed_norm.py": "261915444d720e70d7d5fb0293850d3a2520e33f3252ed88c6907cf61d79a199",
    "vllm/model_executor/layers/fused_moe/activation.py": "78dec0e85e0c040304515f64ccf8f2c6a7a2e5b9f1124a2138a97d9a02585541",
    "vllm/model_executor/layers/fused_moe/config.py": "a3a87cebcc7ac85d73f471dcfafc2d899c6fc08f264777b74a0e919adac9e03f",
    "vllm/model_executor/layers/fused_moe/expert_map_manager.py": "308c00f63fcb6f44518ec3a56d8f902c7bba75928c2e3aea30fbe1f3f473973b",
    "vllm/model_executor/layers/fused_moe/experts/triton_moe.py": "806fbccfe30f5fbd578acf773add99bdf022b12d617958bd6281aef6de369701",
    "vllm/model_executor/layers/fused_moe/modular_kernel.py": "8dc4e0dbbf35b3aac8add63ca6f2a7bbfdadaabbd60826e24598b22c7d8e5c22",
    "vllm/model_executor/layers/fused_moe/oracle/unquantized.py": "c8b8203bd0b73bb7f586f703607394d663f14c05711416ad42d45af564382223",
    "vllm/model_executor/layers/fused_moe/prepare_finalize/no_dp_ep.py": "adbbf529cd22059c95b5380952943a70cf11e05e321dcd707f296bebde61a6e7",
    "vllm/model_executor/layers/fused_moe/routed_experts.py": "68c48e5df0ea4a7cc1f04436b7b9153eb53802ecd9bb55d38db8738508daac6d",
    "vllm/model_executor/layers/fused_moe/router/gate_linear.py": "faa43362c2e7c568498a923c682ca18cae75e87f48d00882770821cb19c21198",
    "vllm/model_executor/layers/fused_moe/router/grouped_topk_router.py": "048339e1799cdd42c9dae686a6cb891046577fb391b500f0d12eed73c8f61e52",
    "vllm/model_executor/layers/fused_moe/runner/moe_runner.py": "05893188402fe37229bf72d75088d66193c417d5d502daafd3a55e1bfedaa60c",
    "vllm/model_executor/layers/fused_moe/runner/shared_experts.py": "d89e4c79c46f6fe61262c51f3facf102e71ce1e4eb3db67f72b6bd8470937be9",
    "vllm/model_executor/layers/fused_moe/unquantized_fused_moe_method.py": "182a5ec1874ece2091a24c5303f6ab9ea64a9a61adf325b863927f0cdd83e5d3",
    "vllm/model_executor/layers/layernorm.py": "4126258bf85aa3af54c78cfbf2a0f32000491e42be37661f2d1ac9f0beea00f3",
    "vllm/model_executor/layers/linear.py": "094fdf956c35bcfcb44b924a4ff60bb1768285963e4e1ae27f3022dd7e1e852d",
    "vllm/model_executor/layers/logits_processor.py": "6b0603d67b0c756253c2fdc882a3896d2e873a16e9aa2ef877aabca8d36bdb5f",
    "vllm/model_executor/layers/mla.py": "594f71cb59ca7e9d7e26d76914272ce1f5b7ca280107756ca9dd03b1403cf1a2",
    "vllm/model_executor/layers/quantization/input_quant_fp8.py": "b16ee19f3f75affbaad6feb2099583932f8e4d5c51616b84d8aa5316a72b6ef2",
    "vllm/model_executor/layers/quantization/utils/quant_utils.py": "caf0868998aab778a517ef6efbcce95a6a9dd1987f0c8b417a490eecd6057526",
    "vllm/model_executor/layers/rotary_embedding/base.py": "c81804498f08f072356a26b41967f211475df2694dabf5da9f488c074d1fdef5",
    "vllm/model_executor/layers/rotary_embedding/common.py": "a56a50bc6235a4ca4c5241259a81de30dc65670467addc4fd7c52e1099481904",
    "vllm/model_executor/layers/sparse_attn_indexer.py": "4622ce35e084314b7b0c16e78c170e760282ef8c59d141da7973bb5c92516a3e",
    "vllm/model_executor/layers/utils.py": "88dee156d427709feb8a560f61eba8e0ebfd1dc10885978ccdce3a952c6afb10",
    "vllm/model_executor/layers/vocab_parallel_embedding.py": "d42574ac79bd7d7847548784d2272c7c40cbd93fe6cdadb15e8c2db79d2074d7",
    "vllm/model_executor/models/deepseek_v2.py": "c658788dc880c5e57400f3bc59537386f24605e157d3cad5b6d5bc8f31bf2328",
    "vllm/model_executor/models/registry.py": "a08a98aaae52ced32226aa647f58d682ac600b9572651a9b377b97846bc99212",
    "vllm/model_executor/parameter.py": "ff6054fbd19ec932c548d562c6f4cc4506383b71cbe411fcfd554c8b1d87b510",
    "vllm/models/common/ops/fused_allreduce_rms_norm.py": "036e07b212f3154fb58a26da97c18015315cc736f89097eac1b733456d4c3e4c",
    "vllm/models/common/ops/sequence_parallel.py": "699fde98360fc8d604055e2987d93108ee489e87cc5563949d2a3dcd838eb8bd",
    "vllm/models/deepseek_v32/__init__.py": "be2bdc7f98691848500c532bb25784badab16adeb16313e5c66800910f49b981",
    "vllm/models/deepseek_v32/attention.py": "3da3b2611a0785805f34e85d52cd2e5655259aa3d7fd001a22b1958f957d5908",
    "vllm/models/deepseek_v32/common/kernels.py": "3d52c2718c58fa0d320efef4757fa5c50b0695caf673632dea341ee246af8a04",
    "vllm/models/deepseek_v32/nvidia/glm52_low_latency_gemm.py": "912e0d144520205c2606e62c5e1db64091a4c75b273bac8dccc6b54eaa8a262a",
    "vllm/models/deepseek_v32/nvidia/model.py": "e53c068dc43981c20bfd83384764b57234a7e3c5e5235b988cdecebc1b5788e4",
    "vllm/models/deepseek_v32/nvidia/ops/fused_q_cutedsl.py": "548992905e056e4021e052609664f422c753f24a3eaaac46963957442c2ba808",
    "vllm/v1/attention/backend.py": "01ac364372afa75fdb22486aa3a9c7cf9aebb742e3471dfb9e0f9d9d36b734bc",
    "vllm/v1/attention/backends/mla/flashattn_mla_sparse.py": "67dd3ead2e8b9d8cda0825ecc93779cb7d670628bc741c759a7d3cd5d474030f",
    "vllm/v1/attention/backends/mla/index_group.py": "c24e028c3c160a9bb92e043da5b5b34c805e84745b7c4b8da6fb8702a0ddd641",
    "vllm/v1/attention/backends/mla/prefill/flash_attn.py": "5e648c82c82e5f81fe11458eb82b9f90b50fc06df2ade3bf111ee6b7a2ee8481",
    "vllm/v1/worker/workspace.py": "d21c08167d1d0d1c0cddaa80453a53a0b4dec5d81b5a537c94a21324344976b3",
}
