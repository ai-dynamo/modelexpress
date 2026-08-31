# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM load-layout capture and graph-safe weight installation.

Capture records where each published source lands in vLLM's load-time layout,
tracing the live model with its params reverted to bf16 load-time skeletons via
layerwise reload. Installation uses vLLM's layerwise reload and post-load
processing to update the live model while preserving storage already referenced
by compiled CUDA graphs.
"""

from __future__ import annotations

import copy
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from modelexpress.refit.reshard.geometry import (
    capture_weights,
    convert_source_weights,
)
from modelexpress.refit.reshard.types import IncompleteRefit
from modelexpress_rl.inference.plan import (
    EngineCapabilities,
    EngineInstaller,
    PreparedArtifact,
    PreparedCheckpointArtifact,
    PreparedEngineTensors,
)
from modelexpress_rl.inference.receiver import PreparedCheckpoint

if TYPE_CHECKING:
    from torch.nn import Module
    from vllm.config import ModelConfig, VllmConfig

    from modelexpress.refit.reshard.types import CaptureResult

logger = logging.getLogger("modelexpress_rl.inference.engines.vllm.installer")


class _VllmInstaller(EngineInstaller):
    """Capture vLLM's load layout and install verified staged tensors."""

    def __init__(
        self,
        *,
        model: Module,
        vllm_config: VllmConfig,
        model_config: ModelConfig,
        device: torch.device,
        convert_native_to_hf: Callable[[dict], dict] | None = None,
    ) -> None:
        self._model = model
        self._vllm_config = vllm_config
        self._model_config = model_config
        self._device = device
        self._convert_native_to_hf = convert_native_to_hf
        self._parameter_layout: dict[
            str, tuple[tuple[int, ...], torch.dtype]
        ] | None = None

    @property
    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            artifact_types=frozenset(
                {
                    PreparedEngineTensors,
                    PreparedCheckpointArtifact,
                }
            )
        )

    def install(self, prepared: PreparedArtifact) -> dict[str, float]:
        started = time.perf_counter()
        if isinstance(prepared, PreparedEngineTensors):
            self.install_tensors(prepared.staged.tensors)
        elif isinstance(prepared, PreparedCheckpointArtifact):
            checkpoint = prepared.checkpoint
            if not isinstance(checkpoint, PreparedCheckpoint):
                raise TypeError("checkpoint preparation has an invalid value")
            self.install_checkpoint(checkpoint.path)
        else:
            raise TypeError(
                f"unsupported prepared artifact {type(prepared).__name__}"
            )
        return {"perf/mx_receive_install_time": time.perf_counter() - started}

    @property
    def _is_quantized(self) -> bool:
        """Whether the live model uses a post-load quantized kernel layout."""
        return getattr(self._vllm_config, "quant_config", None) is not None

    def _build_meta_twin(self) -> Module:
        """Build an unquantized, storage-free copy of the load-time model.

        Quantized live parameters are already packed for their kernels, so they
        cannot describe where bf16 trainer weights land during normal loading.
        The unquantized meta twin has the same structural fusion and load-time
        parameter layout without allocating tensor storage.
        """
        try:
            from vllm.model_executor.model_loader.utils import initialize_model
            from vllm.utils.torch_utils import set_default_torch_dtype
        except (ImportError, AttributeError) as error:
            raise RuntimeError(
                "ModelExpress refit requires vLLM's layerwise reload APIs"
            ) from error

        # Strip quantization so capture observes pre-PWAL load-time parameters,
        # not fp8/Marlin kernel storage.
        twin_config = copy.copy(self._vllm_config)
        twin_model_config = copy.copy(self._vllm_config.model_config)
        twin_model_config.quantization = None
        twin_config.model_config = twin_model_config
        twin_config.quant_config = None
        # The live model already populated static_forward_context. The twin needs
        # a separate empty registry or Attention initialization rejects duplicate
        # layer prefixes.
        twin_compilation_config = copy.copy(self._vllm_config.compilation_config)
        twin_compilation_config.static_forward_context = {}
        twin_config.compilation_config = twin_compilation_config

        # Match vLLM's normal loader initialization; otherwise torch's fp32
        # default would produce the wrong destination dtype during capture.
        with set_default_torch_dtype(self._model_config.dtype), torch.device("meta"):
            return initialize_model(twin_config)

    @staticmethod
    def _layout_of(
        model: Module,
    ) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
        """Describe one model's named parameter shapes and dtypes."""
        return {
            name: (tuple(parameter.shape), parameter.dtype)
            for name, parameter in model.named_parameters()
        }

    def capture(
        self, manifest: list[tuple[str, torch.dtype, tuple[int, ...]]]
    ) -> tuple[
        CaptureResult,
        dict[str, tuple[tuple[int, ...], torch.dtype]],
    ]:
        """Record how published tensors map into vLLM's load-time parameters.

        Captures on the LIVE model with its params reverted to bf16 load-time
        skeletons via layerwise reload; graph-bound kernel tensors are restored
        afterward without finalizing (finalizing would commit the empty skeletons
        and corrupt the live params).
        """
        try:
            from vllm.config import set_current_vllm_config
            from vllm.model_executor.model_loader.reload.layerwise import (
                LAYERWISE_INFO,
                _get_original_loader,
                _place_kernel_tensors,
                initialize_layerwise_reload,
            )
            from vllm.model_executor.model_loader.weight_utils import (
                default_weight_loader,
            )
        except (ImportError, AttributeError) as error:
            raise RuntimeError(
                "ModelExpress refit requires vLLM's layerwise reload APIs"
            ) from error

        model = self._model
        with torch.device(self._device), set_current_vllm_config(self._vllm_config):
            initialize_layerwise_reload(model)
            try:
                # Trace the ORIGINAL loaders, not the reload shims they were wrapped in.
                for _, param in model.named_parameters():
                    param.weight_loader = _get_original_loader(param)
                # The explicit default loader stamps params without a custom
                # weight_loader (norms) so their copies are attributed, not dropped.
                capture = capture_weights(
                    model,
                    convert_source_weights(self._convert_native_to_hf, manifest),
                    default_weight_loader=default_weight_loader,
                )
                param_layout = {
                    name: (tuple(p.shape), p.dtype)
                    for name, p in model.named_parameters()
                }
            finally:
                for layer in model.modules():
                    info = LAYERWISE_INFO.get(layer)
                    if info is not None:
                        if info.kernel_tensors is not None:
                            _place_kernel_tensors(layer, info)
                        info.reset()
        logger.info(
            "captured %d copies and %d unsupported sources (quantized=%s)",
            len(capture.copies),
            len(capture.unsupported),
            self._is_quantized,
        )
        self._parameter_layout = param_layout
        return capture, param_layout

    def parameter_layout(self) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
        """Return the canonical load-time layout used by peer staging buffers."""
        if self._parameter_layout is None:
            self._parameter_layout = self._layout_of(self._build_meta_twin())
        return self._parameter_layout

    def install_tensors(self, tensors: dict[str, torch.Tensor]) -> None:
        """Install verified load-layout tensors without changing graph addresses."""
        self._process_and_commit(tensors)
        _update_mla_absorbed_weights(self._model, quantized=self._is_quantized)
        torch.cuda.synchronize(self._device)

    def install_checkpoint(self, path: str | Path) -> None:
        """Reload a prepared safetensors checkpoint into the live model."""
        try:
            from vllm.model_executor.model_loader.default_loader import (
                DefaultModelLoader,
            )
        except (ImportError, AttributeError) as error:
            raise RuntimeError(
                "ModelExpress refit requires vLLM's default model loader"
            ) from error

        load_config = copy.copy(self._vllm_config.load_config)
        try:
            load_config.load_format = "safetensors"
        except AttributeError:
            object.__setattr__(load_config, "load_format", "safetensors")
        model_config = copy.copy(self._model_config)
        model_config.model = str(path)
        model_config.revision = None
        loader = DefaultModelLoader(load_config)

        self._reload(lambda: loader.load_weights(self._model, model_config))
        _update_mla_absorbed_weights(self._model, quantized=self._is_quantized)
        torch.cuda.synchronize(self._device)

    @torch.no_grad()
    def _process_and_commit(self, tensors: dict[str, torch.Tensor]) -> None:
        """Run vLLM's per-layer post-load processing into graph-bound storage.

        ``initialize_layerwise_reload`` restores load-time parameter skeletons
        and snapshots kernel tensors. Each verified staging tensor is attached to
        its layer, PWAL derives the runtime representation, and vLLM copies the
        result back into the original kernel storage used by CUDA graphs.
        """
        from torch import nn

        try:
            from vllm.model_executor.layers.quantization.base_config import (
                QuantizeMethodBase,
            )
            from vllm.model_executor.model_loader.reload.layerwise import (
                LAYERWISE_INFO,
                _copy_and_restore_kernel_tensors,
            )
        except (ImportError, AttributeError) as error:
            raise RuntimeError(
                "ModelExpress refit requires vLLM's layerwise reload APIs"
            ) from error

        def load() -> None:
            # Quantized models expose kernel-packed parameters before layerwise
            # reload and load-time parameters after it. Resolve the captured
            # names only after vLLM has restored that load-time hierarchy.
            groups: dict[Module, list[tuple[str, str]]] = {}
            matched: set[str] = set()
            for module_name, module in self._model.named_modules():
                for leaf, _parameter in module.named_parameters(recurse=False):
                    full_name = f"{module_name}.{leaf}" if module_name else leaf
                    if full_name in tensors:
                        groups.setdefault(module, []).append((full_name, leaf))
                        matched.add(full_name)
            unmatched = sorted(set(tensors) - matched)
            if unmatched:
                raise IncompleteRefit(
                    "vLLM layerwise reload did not expose every staged parameter; "
                    f"unmatched={unmatched[:10]}"
                )

            for layer, parameters in groups.items():
                info = LAYERWISE_INFO.get(layer)
                for full_name, leaf in parameters:
                    setattr(
                        layer,
                        leaf,
                        nn.Parameter(tensors[full_name], requires_grad=False),
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

        self._reload(load)

    @torch.no_grad()
    def _reload(self, load: Callable[[], None]) -> None:
        """Run one weight loader inside vLLM's graph-safe reload window."""
        try:
            from vllm.config import set_current_vllm_config
            from vllm.model_executor.model_loader.reload.layerwise import (
                finalize_layerwise_reload,
                initialize_layerwise_reload,
            )
        except (ImportError, AttributeError) as error:
            raise RuntimeError(
                "ModelExpress refit requires vLLM's layerwise reload APIs"
            ) from error

        # vLLM also keeps graph-bound tensors as plain object attributes rather
        # than registered parameters or buffers. Layerwise reload does not save
        # these. Snapshot their original storage so Marlin workspaces and MLA
        # derived tensors are not replaced with addresses absent from the graph.
        bare_tensors = {
            module: {
                name: value
                for name, value in module.__dict__.items()
                if isinstance(value, torch.Tensor)
            }
            for module in self._model.modules()
        }
        bare_tensors = {
            module: values for module, values in bare_tensors.items() if values
        }

        with torch.device(self._device), set_current_vllm_config(self._vllm_config):
            initialize_layerwise_reload(self._model)
            load()
            finalize_layerwise_reload(self._model, self._model_config)

            # PWAL may recreate a bare attribute. Copy meaningful derived content
            # into the original graph-bound tensor, then reattach that tensor.
            # Scratch tensors such as workspaces need only be reattached.
            for module, attributes in bare_tensors.items():
                for name, graph_tensor in attributes.items():
                    current = module.__dict__.get(name)
                    if (
                        isinstance(current, torch.Tensor)
                        and current is not graph_tensor
                    ):
                        if (
                            current.shape == graph_tensor.shape
                            and current.dtype == graph_tensor.dtype
                        ):
                            graph_tensor.data.copy_(current)
                        else:
                            logger.error(
                                "%s.%s changed shape or dtype during refit; "
                                "restoring its previous graph-bound tensor",
                                type(module).__name__,
                                name,
                            )
                    setattr(module, name, graph_tensor)

        # A parameter left on meta has no backing storage. CUDA-graph replay would
        # read an invalid address, so reject the update and let the framework
        # restart the engine.
        meta_parameters = [
            name
            for name, parameter in self._model.named_parameters()
            if parameter.device.type == "meta"
        ]
        if meta_parameters:
            raise IncompleteRefit(
                "vLLM refit left parameters on the meta device; "
                f"count={len(meta_parameters)}, names={meta_parameters[:10]}"
            )


def _update_mla_absorbed_weights(model: Module, *, quantized: bool) -> None:
    """Refresh MLA tensors derived from ``kv_b_proj`` in graph-bound storage.

    ``W_UV`` and ``W_UK_T`` are cached bare attributes rather than parameters or
    buffers. Updating them in place preserves the addresses captured by CUDA
    graphs.

    TODO: Replace this MLA-specific recomputation with an engine-owned derived
    tensor hook when vLLM exposes one. Address preservation is generic above;
    recomputing the value is still model-specific here.
    """
    for _name, module in model.named_modules():
        if not (hasattr(module, "W_UV") or hasattr(module, "W_UK_T")) or not hasattr(
            module, "kv_b_proj"
        ):
            continue
        if quantized:
            raise IncompleteRefit(
                "MLA derived-weight refresh from a quantized kv_b_proj is unsupported"
            )
        output_dtype = (
            module.W_UV.dtype if hasattr(module, "W_UV") else module.W_UK_T.dtype
        )
        kv_b_proj_weight = module.kv_b_proj.weight.view(
            module.num_heads,
            module.qk_nope_head_dim + module.v_head_dim,
            -1,
        )
        w_uk, w_uv = kv_b_proj_weight.split(
            [module.qk_nope_head_dim, module.v_head_dim], dim=1
        )
        if hasattr(module, "W_UV"):
            module.W_UV.copy_(w_uv.transpose(0, 1).to(output_dtype))
        if hasattr(module, "W_UK_T"):
            module.W_UK_T.copy_(w_uk.permute(1, 2, 0).to(output_dtype))


__all__: list[str] = []
