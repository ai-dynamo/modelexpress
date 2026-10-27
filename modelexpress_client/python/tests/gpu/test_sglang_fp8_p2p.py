# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real SGLang block-FP8 cold loads; no checkpoint or shared service required."""

import json
from types import SimpleNamespace
from uuid import uuid4

import pytest
import torch

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
        reason="requires SM100+ for FlashInfer MXFP8",
    ),
]


@pytest.fixture(params=["flashinfer_cutedsl", "flashinfer_cutlass"])
def sglang_adapter(tmp_path, request):
    pytest.importorskip("sglang")
    from modelexpress.engines.sglang.adapter import SglangAdapter
    from sglang.srt.layers.quantization.fp8_utils import initialize_fp8_gemm_config
    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

    (tmp_path / "config.json").write_text(
        json.dumps(
            dict(
                architectures=["LlamaForCausalLM"],
                model_type="llama",
                hidden_size=256,
                intermediate_size=512,
                num_hidden_layers=1,
                num_attention_heads=4,
                num_key_value_heads=4,
                vocab_size=256,
                max_position_embeddings=2048,
            )
        )
    )
    set_global_server_args_for_scheduler(
        ServerArgs(
            model_path=str(tmp_path),
            skip_tokenizer_init=True,
            load_format="dummy",
            dtype="bfloat16",
            fp8_gemm_runner_backend=request.param,
        )
    )
    initialize_fp8_gemm_config()
    return SglangAdapter(
        SimpleNamespace(tp_rank=0),
        SimpleNamespace(is_draft_model=False),
        SimpleNamespace(device="cuda:0", gpu_id=0),
    )


def _allocate():
    from modelexpress.load_strategy.context import LoadResult
    from sglang.srt.layers.linear import ReplicatedLinear
    from sglang.srt.layers.quantization.fp8 import Fp8Config
    from sglang.srt.layers.quantization.kv_cache import BaseKVCacheMethod

    config = Fp8Config(
        is_checkpoint_fp8_serialized=True,
        activation_scheme="dynamic",
        weight_block_size=[32, 32],
        scale_fmt="ue8m0",
        use_mxfp8=False,
    )
    with torch.device("cuda:0"):
        model = torch.nn.Module()
        # Mixed layer decisions exercise per-layer rather than global dispatch.
        for name in ("eligible", "ineligible"):
            layer = ReplicatedLinear(
                256, 256, bias=False, params_dtype=torch.bfloat16, quant_config=config
            )
            assert layer.quant_method.block_fp8_as_mxfp8
            model.add_module(name, layer)
        model.attn = torch.nn.Module()
        model.attn.quant_method = BaseKVCacheMethod(config)
        model.attn.quant_method.create_weights(model.attn)
    return LoadResult(value=model, model=model)


def _descriptors(tensors):
    from modelexpress.types import TensorDescriptor

    return [
        TensorDescriptor(
            name=n,
            addr=t.data_ptr(),
            size=t.numel() * t.element_size(),
            device_id=0,
            dtype=str(t.dtype),
        )
        for n, t in tensors.items()
    ]


@torch.no_grad()
def _donor(adapter):
    result = _allocate()
    for name, scale in (("eligible", 1.0), ("ineligible", 1.5)):
        layer = getattr(result.model, name)
        # Exactly representable FP8 values, not uninitialized dummy weights.
        values = (torch.arange(256 * 256, device="cuda").reshape(256, 256) % 9 - 4).float() / 4
        layer.weight.copy_(values)
        layer.weight_scale_inv.fill_(scale)
    result.model.attn.k_scale.fill_(2.0)
    result.model.attn.v_scale.fill_(4.0)
    return adapter.after_weight_iter_load(result)


@torch.no_grad()
def _check(adapter, donor, receiver):
    source, target = adapter.discover_tensors(donor), adapter.discover_tensors(receiver)
    assert source.keys() == target.keys()
    for name, tensor in source.items():
        assert torch.equal(
            tensor.contiguous().reshape(-1).view(torch.uint8),
            target[name].contiguous().reshape(-1).view(torch.uint8),
        ), name
    x = torch.ones(128, 256, device="cuda", dtype=torch.bfloat16)
    for name, ready in (("eligible", True), ("ineligible", False)):
        a, b = getattr(donor.model, name), getattr(receiver.model, name)
        assert a.block_fp8_mxfp8_ready == b.block_fp8_mxfp8_ready == ready
        torch.testing.assert_close(
            a.quant_method.apply(a, x), b.quant_method.apply(b, x), rtol=0, atol=0
        )
    assert receiver.model.attn.k_scale_float == donor.model.attn.k_scale_float == 2.0
    assert receiver.model.attn.v_scale_float == donor.model.attn.v_scale_float == 4.0


@torch.no_grad()
def test_local_copy_uses_manifest_without_priming(sglang_adapter):
    adapter = sglang_adapter
    donor, receiver = _donor(adapter), _allocate()
    source = adapter.discover_tensors(donor)
    adapter.prepare_rdma_target(receiver)
    adapter.before_rdma_receive(receiver)
    assert not receiver.model.eligible.block_fp8_mxfp8_ready
    assert not hasattr(receiver.model.eligible, "weight_scale_inv_swizzled")
    scales = receiver.model.eligible.weight_scale_inv.clone()
    adapter.prepare_rdma_target_from_manifest(receiver, _descriptors(source))
    assert torch.equal(scales, receiver.model.eligible.weight_scale_inv)
    target = adapter.discover_tensors(receiver)
    assert source.keys() == target.keys()
    addresses = {n: t.data_ptr() for n, t in target.items()}
    for name in source:
        target[name].copy_(source[name])
    torch.cuda.synchronize()
    adapter.after_rdma_receive(receiver)
    assert addresses == {n: t.data_ptr() for n, t in adapter.discover_tensors(receiver).items()}
    _check(adapter, donor, receiver)


@pytest.mark.parametrize("inconsistent_manifest", [False, True])
@torch.no_grad()
def test_real_nixl_cold_receive(sglang_adapter, inconsistent_manifest):
    from modelexpress import p2p_pb2
    from modelexpress.adapter import StrategyFailed
    from modelexpress.load_strategy.context import LoadContext
    from modelexpress.load_strategy.rdma_strategy import RdmaStrategy
    from modelexpress.nixl_transfer import NixlTransferManager, is_nixl_available

    if not is_nixl_available():
        pytest.skip("NIXL is not installed")
    adapter = sglang_adapter
    donor, receiver = _donor(adapter), _allocate()
    source_manager = NixlTransferManager(f"fp8-source-{uuid4()}", 0)
    target_manager = NixlTransferManager(f"fp8-target-{uuid4()}", 0)
    try:
        source_manager.initialize()
        target_manager.initialize()
        source_manager.register_tensors(adapter.discover_tensors(donor))
        descriptors = source_manager.tensor_descriptors
        if inconsistent_manifest:
            descriptors = [t for t in descriptors if t.name != "eligible.weight_scale_inv_swizzled"]
        worker = p2p_pb2.WorkerMetadata(
            tensors=[p2p_pb2.TensorDescriptor(**vars(t)) for t in descriptors],
            nixl_metadata=source_manager.nixl_metadata,
            accelerator="cuda",
        )
        ctx = LoadContext(
            model_config=adapter.model_config,
            load_config=adapter.load_config,
            target_device=torch.device("cuda:0"),
            global_rank=0,
            worker_rank=0,
            local_rank=0,
            device_id=0,
            identity=p2p_pb2.SourceIdentity(model_name="synthetic-fp8"),
            mx_client=SimpleNamespace(REQUIRES_P2P_METADATA=True),
            worker_id="target",
            adapter=adapter,
            accelerator_backend=adapter.accelerator_backend,
            nixl_manager=target_manager,
        )
        if inconsistent_manifest:
            with pytest.raises(StrategyFailed, match="disagree") as error:
                RdmaStrategy()._load_as_target(receiver, ctx, worker, "source", "worker")
            assert error.value.mutated
            assert not receiver.model.eligible.block_fp8_mxfp8_ready
        else:
            result = RdmaStrategy()._load_as_target(receiver, ctx, worker, "source", "worker")
            _check(adapter, donor, result)
    finally:
        target_manager.shutdown()
        source_manager.shutdown()
