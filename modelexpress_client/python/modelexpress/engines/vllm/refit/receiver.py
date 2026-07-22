# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""vLLM receiver for the no-gather slice-resharding weight refit.

Implements the two engine hooks of
:class:`~modelexpress.refit.reshard.receiver.ReshardReceiver`
for vLLM:

  * :meth:`_capture` - build a fresh UNQUANTIZED meta twin of the model and drive
    its ``load_weights`` with the record-only geometry capture, so we record where
    each source lands in the bf16 load-time layout (the live params are post-quant
    for a quantized model, so we can't capture on them).
  * :meth:`_install` - install the RDMA'd receive buffers into the live params via
    vLLM's layerwise reload: ``initialize_layerwise_reload`` reverts the live
    params to bf16 load-time skeletons + snapshots the CUDA-graph-bound kernel
    tensors, we ``setattr`` the receive buffers as each layer's load-time params,
    run per-layer PWAL (quantize/derive for fp8, no-op for bf16), then restore the
    graph-bound tensors + any bare-attr derived tensors (Marlin ``workspace``,
    MLA ``W_UV``/``W_UK_T``).

Everything else (discover, plan, transport, buffers, router dtype-cast) is the
engine-agnostic base.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch.nn import Module

from modelexpress.refit.reshard.geometry import capture_geometry
from modelexpress.refit.reshard.receiver import ReshardReceiver
from modelexpress.refit.reshard.types import CaptureResult

logger = logging.getLogger("modelexpress.engines.vllm.refit.receiver")


class VllmReshardReceiver(ReshardReceiver):
    """Slice-resharding weight receiver for a vLLM model."""

    def __init__(
        self, *, model: Module, vllm_config: Any, model_config: Any, **base_kwargs: Any
    ) -> None:
        self._model = model
        self._vllm_config = vllm_config
        self._model_config = model_config
        super().__init__(device=base_kwargs.pop("device"), **base_kwargs)

    @property
    def _is_quantized(self) -> bool:
        """True when the live model was quantized (fp8, etc.) - its live params are
        then post-PWAL (Marlin-packed / kernel layout), not the bf16 load-time
        layout, so capture runs on a meta twin and quantized layers re-quantize
        via PWAL on install."""
        return getattr(self._vllm_config, "quant_config", None) is not None

    # ---------------------------------------------------------------- capture
    def _build_meta_twin(self) -> Module:
        """Build a fresh twin of the model on ``meta`` to capture the bf16
        load-time slice geometry.

        Stripped of quantization: vLLM's fp8 ``create_weights`` makes params
        directly as ``float8_e4m3fn``, so a same-config twin would have fp8 params
        and every source would be a dtype mismatch (no bf16 layout to slice into).
        An unquantized twin has bf16 params with the same structural fusion =
        the load-time layout; the live fp8 model is re-quantized separately on
        install. ``meta`` -> zero storage, discarded after capture."""
        import copy as _copy

        from vllm.model_executor.model_loader.utils import initialize_model
        from vllm.utils.torch_utils import set_default_torch_dtype

        # Strip quantization so the twin builds Unquantized (bf16) params.
        twin_config = _copy.copy(self._vllm_config)
        twin_mc = _copy.copy(self._vllm_config.model_config)
        twin_mc.quantization = None
        twin_config.model_config = twin_mc
        twin_config.quant_config = None
        # Attention.__init__ rejects a duplicate layer prefix in
        # compilation_config.static_forward_context - the live model already
        # populated it, so the twin needs its OWN empty registry.
        twin_cc = _copy.copy(self._vllm_config.compilation_config)
        twin_cc.static_forward_context = {}
        twin_config.compilation_config = twin_cc

        # Without set_default_torch_dtype the layers create params at torch's
        # default (fp32), not the model's bf16 (vLLM's loader wraps init the same).
        with set_default_torch_dtype(self._model_config.dtype), torch.device("meta"):
            twin = initialize_model(twin_config)
        logger.info(
            "[reshard] built unquantized meta-twin for bf16 load-time geometry capture"
        )
        return twin

    def _capture(self, manifest: list) -> "tuple[CaptureResult, dict]":
        from vllm.model_executor.model_loader.weight_utils import default_weight_loader

        # Always capture on a fresh meta twin (uniform for bf16 + quantized): its
        # load_weights is pre-PWAL (bf16 load-time), which for a quantized model is
        # the layout we slice into, and for bf16 is identical to the live params.
        # Pass default_weight_loader so params WITHOUT a custom weight_loader
        # (norms) are stamped and their copies attributed rather than dropped.
        twin = self._build_meta_twin()
        capture = capture_geometry(
            twin, manifest, default_weight_loader=default_weight_loader
        )
        logger.info(
            "[reshard] captured %d copies, %d unsupported (quantized=%s)",
            len(capture.copies),
            len(capture.unsupported),
            self._is_quantized,
        )
        param_layout = {
            n: (tuple(p.shape), p.dtype) for n, p in twin.named_parameters()
        }
        return capture, param_layout

    # ---------------------------------------------------------------- install
    def _install(self, recv_buffers: dict) -> None:
        self._process_and_commit(recv_buffers)
        _update_mla_absorbed_weights(self._model)

    @torch.no_grad()
    def _process_and_commit(self, recv_buffers: dict) -> None:
        """Install the RDMA'd receive buffers into the live params, per layer.

        ``initialize_layerwise_reload`` reverts the live params to bf16 load-time
        skeletons and snapshots the CUDA-graph-bound kernel tensors; we ``setattr``
        each layer's receive buffers as its load-time params, run PWAL (quantize +
        derive scales for fp8, no-op for bf16), then copy the result back into the
        saved kernel tensors so the compiled graph stays intact. Grouped by layer
        since PWAL is a per-layer op."""
        import torch.nn as nn
        from vllm.config import set_current_vllm_config
        from vllm.model_executor.layers.quantization.base_config import (
            QuantizeMethodBase,
        )
        from vllm.model_executor.model_loader.reload.layerwise import (
            LAYERWISE_INFO,
            _copy_and_restore_kernel_tensors,
            finalize_layerwise_reload,
            initialize_layerwise_reload,
        )

        model = self._model
        # Bridge captured param names -> live layer modules via the live model's
        # own module hierarchy (authoritative; PWAL is a per-layer op).
        recv = set(recv_buffers)
        groups: dict = {}  # live module -> [(full_name, leaf)]
        for mod_name, module in model.named_modules():
            for leaf, _ in module.named_parameters(recurse=False):
                full = f"{mod_name}.{leaf}" if mod_name else leaf
                if full in recv:
                    groups.setdefault(module, []).append((full, leaf))

        # Snapshot BARE-ATTRIBUTE tensors (plain module.__dict__ tensors, NOT
        # params/buffers) before ilr. These are graph-bound scratch/derived that
        # PWAL re-creates at new addresses - Marlin's ``layer.workspace`` (vLLM
        # leaves it unregistered) and MLA ``W_UV``/``W_UK_T``. ilr saves only
        # params+buffers, so without this the CUDA graph replays reading the freed
        # boot address -> hang. We hold the boot tensors and re-attach after PWAL.
        bare_snapshot: dict = {}
        for module in model.modules():
            attrs = {
                n: t for n, t in module.__dict__.items() if isinstance(t, torch.Tensor)
            }
            if attrs:
                bare_snapshot[module] = attrs

        with torch.device(self._device), set_current_vllm_config(self._vllm_config):
            initialize_layerwise_reload(model)
            for layer, params in groups.items():
                info = LAYERWISE_INFO.get(layer)
                for full, leaf in params:
                    setattr(
                        layer,
                        leaf,
                        nn.Parameter(recv_buffers[full], requires_grad=False),
                    )
                quant_method = getattr(layer, "quant_method", None)
                if isinstance(quant_method, QuantizeMethodBase):
                    if hasattr(layer, "_already_called_process_weights_after_loading"):
                        delattr(layer, "_already_called_process_weights_after_loading")
                    quant_method.process_weights_after_loading(layer)
                if info is not None and info.kernel_tensors is not None:
                    _copy_and_restore_kernel_tensors(layer, info)
                if info is not None:
                    info.reset()
            finalize_layerwise_reload(model, self._model_config)

            # Restore boot bare-attr tensors into their ORIGINAL graph-bound
            # storage. For scratch (workspace) re-attaching suffices; for a derived
            # tensor with meaningful content (MLA absorbed matrices) copy the fresh
            # content into the boot storage first, then re-attach - so the graph
            # reads the original address with the updated values.
            for module, attrs in bare_snapshot.items():
                for n, boot_t in attrs.items():
                    cur = module.__dict__.get(n)
                    if isinstance(cur, torch.Tensor) and cur is not boot_t:
                        if cur.shape == boot_t.shape and cur.dtype == boot_t.dtype:
                            boot_t.data.copy_(cur)
                        else:
                            logger.error(
                                "[reshard] bare-attr %s.%s changed shape/dtype across refit "
                                "(cur %s/%s vs boot %s/%s); skipping copy, re-attaching STALE boot tensor",
                                type(module).__name__,
                                n,
                                tuple(cur.shape),
                                cur.dtype,
                                tuple(boot_t.shape),
                                boot_t.dtype,
                            )
                    setattr(module, n, boot_t)

        # Sanity: any param/buffer left on meta after the restore is a CUDA-graph
        # hang cause (replay reads unbacked memory).
        meta = [n for n, p in model.named_parameters() if p.device.type == "meta"]
        if meta:
            logger.error(
                "[reshard] POST-COMMIT META PARAMS (graph-hang risk): %d %s",
                len(meta),
                meta[:10],
            )


def _update_mla_absorbed_weights(model: Module) -> None:
    """Recompute MLA absorbed KV weights (``W_UV``/``W_UK_T``) in place after the
    ``kv_b_proj`` update. No-op for non-MLA models. In-place ``copy_`` (not a
    reassign) keeps the CUDA-graph-bound address, paired with the bare-attr
    re-attach in :meth:`_process_and_commit`.

    TODO(generalize derived tensors): MLA's absorbed matrices are one instance of
    a "derived tensor" - a graph-bound value COMPUTED from params and cached as a
    bare module attribute, not a param/buffer. ``_process_and_commit`` already
    preserves such tensors' graph addresses generically (bare-attr snapshot +
    content re-copy), but the RECOMPUTE of their content is model-specific and
    hardcoded here by attribute name. Replace this with a general mechanism - e.g.
    have the layer expose a ``recompute_derived()`` hook, or drive the recompute
    off vLLM's own ``process_weights_after_loading`` so any model's derived
    tensors (not just MLA) are refreshed without a bespoke function."""
    for _name, module in model.named_modules():
        if not (hasattr(module, "W_UV") or hasattr(module, "W_UK_T")) or not hasattr(
            module, "kv_b_proj"
        ):
            continue
        out_dtype = (
            module.W_UV.dtype if hasattr(module, "W_UV") else module.W_UK_T.dtype
        )
        kv_b_proj_weight = module.kv_b_proj.weight.view(
            module.num_heads, module.qk_nope_head_dim + module.v_head_dim, -1
        )
        w_uk, w_uv = kv_b_proj_weight.split(
            [module.qk_nope_head_dim, module.v_head_dim], dim=1
        )
        if hasattr(module, "W_UV"):
            module.W_UV.copy_(w_uv.transpose(0, 1).to(out_dtype))
        if hasattr(module, "W_UK_T"):
            module.W_UK_T.copy_(w_uk.permute(1, 2, 0).to(out_dtype))
