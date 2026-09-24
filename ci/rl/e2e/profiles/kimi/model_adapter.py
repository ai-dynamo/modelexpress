"""Kimi MLA projection mutation and derived-weight verification."""


def extra_tensor_names(index):
    names = sorted(
        n
        for n in index["weight_map"]
        if n.endswith(".layers.0.self_attn.kv_b_proj.weight")
    )
    assert len(names) == 1, "Expected one first-layer MLA projection"
    return names


def mutate_extra(tensors):
    assert len(tensors) == 1
    for tensor in tensors.values():
        tensor.add_(0.0009765625)


def verify_derived(worker, phase):
    import hashlib

    import torch
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        get_and_maybe_dequant_weights,
    )

    assert phase in ("baseline", "updated")
    rows = []
    for name, layer in worker.model_runner.get_model().named_modules():
        if not (hasattr(layer, "W_UV") and hasattr(layer, "W_UK_T")):
            continue
        weight = get_and_maybe_dequant_weights(
            layer.kv_b_proj,
            out_dtype=worker.vllm_config.model_config.dtype,
        )
        weight = weight.T.reshape(
            layer.kv_lora_rank,
            layer.num_heads,
            layer.qk_nope_head_dim + layer.v_head_dim,
        )
        uk, uv = weight.split([layer.qk_nope_head_dim, layer.v_head_dim], dim=-1)
        for key, expected in {
            "W_UV": uv.transpose(0, 1),
            "W_UK_T": uk.permute(1, 2, 0),
        }.items():
            actual = getattr(layer, key)
            torch.testing.assert_close(actual, expected)
            digest = hashlib.sha256(
                actual.detach().cpu().contiguous().view(torch.uint8).numpy()
            ).hexdigest()
            rows.append(
                {
                    "name": name + "." + key,
                    "sha256": digest,
                    "pointer": actual.data_ptr(),
                }
            )
    assert rows, "No MLA derived tensors checked"
    current = {r["name"]: r for r in rows}
    changed = []
    if phase == "baseline":
        worker._derived_baseline = current
    else:
        baseline = worker._derived_baseline
        assert current.keys() == baseline.keys()
        assert all(r["pointer"] == baseline[n]["pointer"] for n, r in current.items())
        changed = [
            n for n, r in current.items() if r["sha256"] != baseline[n]["sha256"]
        ]
        targeted = {n for n in current if ".layers.0.self_attn." in n}
        assert len(targeted) == 2 and set(changed) == targeted, (
            "MLA projection delta did not refresh both targeted weights"
        )
    return {"verified": True, "tensors": rows, "changed": changed}
