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
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from modelexpress.accelerators import accelerator_backend_for
from modelexpress.engines.vllm.host_quantization import (
    refresh_host_quantization_state,
)
from modelexpress.refit.reshard.geometry import (
    capture_weights,
    convert_source_weights,
)
from modelexpress.refit.reshard.types import IncompleteRefit
from modelexpress.refit.timing import refit_span

from modelexpress_rl.inference.plan import (
    EngineCapabilities,
    EngineInstaller,
    PreparedArtifact,
    PreparedCheckpointArtifact,
    PreparedEngineTensors,
    PreparedRuntimeTensors,
)
from modelexpress_rl.inference.receiver import PreparedCheckpoint

if TYPE_CHECKING:
    from modelexpress.refit.reshard.types import CaptureResult
    from torch.nn import Module
    from vllm.config import ModelConfig, VllmConfig

logger = logging.getLogger("modelexpress_rl.inference.engines.vllm.installer")


@contextmanager
def _preserve_generated_vision_buffers(model: Module, layerwise_info):
    """Exclude Kimi's generated temporal embedding from checkpoint materialization."""
    saved = []
    try:
        for layer in model.modules():
            if (
                type(layer).__module__ != "vllm.model_executor.models.kimi_k25_vit"
                or type(layer).__name__ != "Learnable2DInterpPosEmbDivided_fixed"
            ):
                continue
            from vllm.model_executor.models.kimi_k25_vit import (
                Learnable2DInterpPosEmbDivided_fixed,
            )

            if type(layer) is not Learnable2DInterpPosEmbDivided_fixed:
                continue
            buffer = layer._buffers.get("time_weight")
            info = layerwise_info.get(layer)
            if (
                buffer is None
                or "time_weight" not in layer._non_persistent_buffers_set
                or info is None
            ):
                raise IncompleteRefit("Kimi generated temporal buffer contract changed")
            metadata = info.restore_metadata
            saved.append((layer, buffer, info, metadata))
            # Native reload otherwise allocates this buffer with empty_strided
            # and copies unwritten bytes back when the spatial weight loads.
            info.restore_metadata = (
                metadata[0],
                {name: value for name, value in metadata[1].items() if name != "time_weight"},
            )
            delattr(layer, "time_weight")
            object.__setattr__(layer, "time_weight", buffer)
        yield
    finally:
        for layer, buffer, info, metadata in reversed(saved):
            info.restore_metadata = metadata
            delattr(layer, "time_weight")
            layer.register_buffer("time_weight", buffer, persistent=False)


def _reserve_runtime_buffer_slots(model: Module, layerwise_info) -> None:
    """Keep late-created kernel buffers registered while PWAL runs again."""
    for layer in model.modules():
        info = layerwise_info.get(layer)
        if info is None or info.kernel_tensors is None:
            continue
        _, buffers = info.kernel_tensors
        for name, buffer in buffers.items():
            if name in layer._buffers:
                continue
            if hasattr(layer, name):
                raise IncompleteRefit(
                    f"{type(layer).__name__}.{name} conflicts with a runtime buffer"
                )
            layer.register_buffer(name, buffer)


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
        runtime_tensors: dict[str, torch.Tensor] | None = None,
        checkpoint_tensor_mapping: dict[str, str] | None = None,
    ) -> None:
        self._model = model
        self._vllm_config = vllm_config
        self._model_config = model_config
        self._device = device
        self._convert_native_to_hf = convert_native_to_hf
        self._runtime_tensors = runtime_tensors
        self._checkpoint_fenced = False
        self._checkpoint_tensor_mapping = (
            dict(checkpoint_tensor_mapping) if checkpoint_tensor_mapping is not None else None
        )

    @property
    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            artifact_types=frozenset(
                {
                    PreparedEngineTensors,
                    PreparedRuntimeTensors,
                    PreparedCheckpointArtifact,
                }
            )
        )

    def install(self, prepared: PreparedArtifact) -> dict[str, float]:
        if self._checkpoint_fenced:
            raise IncompleteRefit("checkpoint transaction failed; worker recovery required")
        started = time.perf_counter()
        metrics = prepared.metrics
        if isinstance(prepared, PreparedEngineTensors):
            self.install_tensors(prepared.staged.tensors)
        elif isinstance(prepared, PreparedRuntimeTensors):
            self.install_runtime_tensors(prepared.staged.tensors)
        elif isinstance(prepared, PreparedCheckpointArtifact):
            checkpoint = prepared.checkpoint
            if not isinstance(checkpoint, PreparedCheckpoint):
                raise TypeError("checkpoint preparation has an invalid value")
            self.install_checkpoint(checkpoint.path)
        else:
            raise TypeError(
                f"unsupported prepared artifact {type(prepared).__name__}"
            )
        metrics["perf/mx_receive_install_time"] = time.perf_counter() - started
        return metrics

    @property
    def _is_quantized(self) -> bool:
        """Whether the live model uses a post-load quantized kernel layout."""
        return getattr(self._vllm_config, "quant_config", None) is not None

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
        return capture, param_layout

    def install_tensors(self, tensors: dict[str, torch.Tensor]) -> None:
        """Install verified load-layout tensors without changing graph addresses."""
        self._process_and_commit(tensors)
        # Synchronization is paid once per install, separately from per-layer work.
        with refit_span("post_install"):
            torch.cuda.synchronize(self._device)

    def install_runtime_tensors(self, tensors: dict[str, torch.Tensor]) -> None:
        """Finish a direct peer transfer into existing graph-bound storage."""
        if self._runtime_tensors is None:
            raise RuntimeError("vLLM runtime tensor installation is unavailable")
        destinations = self._runtime_tensors
        local_only = sorted(set(destinations) - set(tensors))
        source_only = sorted(set(tensors) - set(destinations))
        if local_only or source_only:
            raise IncompleteRefit(
                "vLLM runtime tensor set differs from the staged peer: "
                f"{len(local_only)} local-only, {len(source_only)} source-only"
            )
        for name, source in tensors.items():
            if destinations[name] is not source:
                raise IncompleteRefit(
                    "vLLM runtime P2P must write directly into live storage"
                )

        if getattr(self._model_config, "enforce_eager", False):
            with refit_span("post_install"):
                refresh_host_quantization_state(
                    self._model,
                    self._vllm_config,
                    accelerator_backend_for(self._device),
                    allow_warm=True,
                )

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
        # Same synchronize as install_tensors, so a checkpoint refit
        # reports the stage too rather than charging it to the caller's total.
        with refit_span("post_install"):
            torch.cuda.synchronize(self._device)

    def install_checkpoint_collectively(
        self,
        prepared: PreparedCheckpoint,
        *,
        serving_version: str,
        world_size: int,
        all_gather: Callable[[object], tuple[object, ...]],
        installation_context: Callable,
        activate: Callable[[], None],
        commit_version: Callable[[], None],
        fence: Callable[[], None],
    ) -> str:
        """Experimental partial refit; all ranks must enter a drained safe point.

        Supply the canonical method's context with activate=False, followed by
        its explicit activate callback. The caller owns the serving-version
        commit and a fence that prevents inference/publication after ANY failure.
        The default warm-refit session does not opt into this API.
        """
        from vllm.config import set_current_vllm_config

        from ...checkpoint_transaction import refit_checkpoint_collectively
        from .partial_checkpoint import prepare_partial_checkpoint

        if self._checkpoint_fenced:
            raise IncompleteRefit("checkpoint transaction failed; worker recovery required")

        def fail_closed() -> None:
            self._checkpoint_fenced = True
            fence()

        def gather_with_mapping(value):
            peers = all_gather((self._checkpoint_tensor_mapping, value))
            if any(
                not isinstance(peer, tuple) or len(peer) != 2
                or peer[0] != self._checkpoint_tensor_mapping
                for peer in peers
            ):
                raise IncompleteRefit("checkpoint tensor mappings differ across ranks")
            return tuple(peer[1] for peer in peers)

        with torch.device(self._device), set_current_vllm_config(self._vllm_config):
            return refit_checkpoint_collectively(
                prepared,
                serving_version=serving_version,
                world_size=world_size,
                all_gather=gather_with_mapping,
                installation_context=installation_context,
                prepare_partial=lambda checkpoint: prepare_partial_checkpoint(
                    self._model, self._vllm_config, checkpoint,
                    serving_version=serving_version,
                    tensor_mapping=self._checkpoint_tensor_mapping,
                ),
                install_full=lambda: self.install_checkpoint(prepared.path),
                synchronize=lambda: torch.cuda.synchronize(self._device),
                activate=activate,
                commit_version=commit_version,
                fence=fail_closed,
            )

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

            # Per-layer spans, accumulated across every layer into two stages.
            # This loop is the whole of what a framework sees as "install", and
            # the two things it does have unrelated costs: post-load processing
            # is compute that scales with quantization scheme, while the copy
            # back is bandwidth into storage the CUDA graphs already point at.
            # Charged together they cannot be acted on.
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
                    with refit_span("transformation"):
                        quant_method.process_weights_after_loading(layer)
                if info is not None and info.kernel_tensors is not None:
                    with refit_span("installation"):
                        _copy_and_restore_kernel_tensors(layer, info)
                if info is not None:
                    info.reset()

        self._reload(load)

    @torch.no_grad()
    def _reload(self, load: Callable[[], None]) -> None:
        """Run one weight loader inside vLLM's graph-safe reload window."""
        if self._checkpoint_fenced:
            raise IncompleteRefit("checkpoint transaction failed; worker recovery required")
        try:
            from vllm.config import set_current_vllm_config
            from vllm.model_executor.model_loader.reload.layerwise import (
                LAYERWISE_INFO,
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

        with (
            torch.device(self._device),
            set_current_vllm_config(self._vllm_config),
            _preserve_generated_vision_buffers(self._model, LAYERWISE_INFO),
        ):
            initialize_layerwise_reload(self._model)
            _reserve_runtime_buffer_slots(self._model, LAYERWISE_INFO)
            load()
            finalize_layerwise_reload(self._model, self._model_config)

            # PWAL may recreate a bare attribute. Copy meaningful derived content
            # into the original graph-bound tensor, then reattach that tensor.
            # Scratch tensors such as workspaces need only be reattached.
            for module, attributes in bare_tensors.items():
                for name, graph_tensor in attributes.items():
                    current = module.__dict__.get(name)
                    if name in ("W_UV", "W_UK_T") and current is graph_tensor:
                        raise IncompleteRefit(
                            f"{type(module).__name__}.{name} was not refreshed by "
                            "vLLM post-load processing during reload"
                        )
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


__all__: list[str] = []
