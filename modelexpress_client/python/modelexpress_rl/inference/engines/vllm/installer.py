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
from modelexpress.refit.timing import refit_span

from modelexpress_rl.inference.plan import (
    EngineCapabilities,
    EngineInstaller,
    PreparedArtifact,
    PreparedCheckpointArtifact,
    PreparedEngineTensors,
    PreparedStreamingTensors,
    PreparedRuntimeTensors,
)
from modelexpress_rl.inference.receiver import PreparedCheckpoint

if TYPE_CHECKING:
    from modelexpress.refit.reshard.types import CaptureResult
    from torch.nn import Module
    from vllm.config import ModelConfig, VllmConfig

logger = logging.getLogger("modelexpress_rl.inference.engines.vllm.installer")


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
    ) -> None:
        self._model = model
        self._vllm_config = vllm_config
        self._model_config = model_config
        self._device = device
        self._convert_native_to_hf = convert_native_to_hf
        self._runtime_tensors = runtime_tensors

    @property
    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            artifact_types=frozenset(
                {
                    PreparedEngineTensors,
                    PreparedRuntimeTensors,
                    PreparedCheckpointArtifact,
                }
                | ({PreparedStreamingTensors} if not self._is_quantized else set())
            )
        )

    def install(self, prepared: PreparedArtifact) -> dict[str, float]:
        started = time.perf_counter()
        metrics = prepared.metrics
        if isinstance(prepared, PreparedEngineTensors):
            self.install_tensors(prepared.staged.tensors)
        elif isinstance(prepared, PreparedStreamingTensors):
            self.install_streaming(prepared)
            # install_streaming records into the artifact's own metrics dict;
            # re-read it so those entries travel with the install timing.
            metrics = prepared.metrics
            metrics["streaming_apply_s"] = time.perf_counter() - started
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

    @staticmethod
    def _parameter_aliases(model: Module) -> list[list[tuple[Module, str]]]:
        groups: dict[int, list[tuple[Module, str]]] = {}
        for module in model.modules():
            for name, parameter in module._parameters.items():
                if parameter is not None:
                    groups.setdefault(id(parameter), []).append((module, name))
        return [group for group in groups.values() if len(group) > 1]

    @staticmethod
    def _restore_parameter_aliases(aliases: list[list[tuple[Module, str]]]) -> None:
        # vLLM restores metadata separately for each module. Reconnect shared
        # parameters so a tied loader still covers one canonical destination.
        for group in aliases:
            first_module, first_name = group[0]
            parameter = getattr(first_module, first_name)
            for module, name in group[1:]:
                other = getattr(module, name)
                if other.shape != parameter.shape or other.dtype != parameter.dtype:
                    raise IncompleteRefit(
                        "tied parameters have incompatible load-time layouts"
                    )
                setattr(module, name, parameter)

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
        aliases = self._parameter_aliases(model)
        with torch.device(self._device), set_current_vllm_config(self._vllm_config):
            initialize_layerwise_reload(model)
            try:
                self._restore_parameter_aliases(aliases)
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
        # Derived-weight fixups plus the synchronize that makes the whole
        # install observable. Separate from the per-layer stages because it is
        # paid once and does not scale with the number of layers, so folding it
        # in would make those look worse than they are on small models.
        with refit_span("post_install"):
            _update_mla_absorbed_weights(self._model, quantized=self._is_quantized)
            torch.cuda.synchronize(self._device)

    @torch.no_grad()
    def install_streaming(self, prepared: PreparedStreamingTensors) -> None:
        """Commit complete modules into existing storage before arena reuse."""
        if self._is_quantized:
            raise IncompleteRefit(
                "bounded streaming currently requires an unquantized engine"
            )

        metrics = prepared.transfer_metrics
        load_s = 0.0
        commit_s = 0.0
        scan_s = 0.0

        def retains_arena(module: Module, arena_storage: set[int]) -> bool:
            values = [
                *module.parameters(recurse=False),
                *module.buffers(recurse=False),
            ]
            values.extend(
                v for v in module.__dict__.values() if isinstance(v, torch.Tensor)
            )
            return any(
                v.device.type != "meta"
                and v.untyped_storage().data_ptr() in arena_storage
                for v in values
            )

        def load():
            nonlocal load_s, commit_s, scan_s
            load_started = time.perf_counter()
            expected = set(dict(self._model.named_parameters()))
            if expected != prepared.parameter_names:
                raise IncompleteRefit(
                    "streaming parameter coverage differs from the live load layout"
                )
            # Name resolution and owning-module membership are properties of the
            # load layout, which `expected` has just pinned, so one walk serves
            # every batch and removes two of the three the install used to make.
            # The arena-retention scan is not such a property and still walks
            # the live tree per batch; see the scan below.
            owner_of: dict[str, str] = {}
            owned_by: dict[str, set[str]] = {}
            resolved: dict[str, tuple[Module, str]] = {}
            for module_name, module in self._model.named_modules():
                owned = set()
                for leaf, _ in module.named_parameters(recurse=False):
                    full_name = f"{module_name}.{leaf}" if module_name else leaf
                    owned.add(full_name)
                    resolved[full_name] = (module, leaf)
                owned_by[module_name] = owned
                for name in owned:
                    owner_of[name] = module_name
            installed = set()
            arena_storages: set[int] = set()
            batches = prepared.batches()
            try:
                for tensors in batches:
                    names = set(tensors)
                    if not names or names - expected or names & installed:
                        raise IncompleteRefit(
                            "invalid or repeated streaming parameter batch"
                        )
                    touched = {owner_of[name] for name in names}
                    for module_name in touched:
                        if not owned_by[module_name] <= names:
                            raise IncompleteRefit(
                                "streaming batch splits an owning module"
                            )
                    commit_started = time.perf_counter()
                    self._process_and_commit(tensors, reload=False, resolved=resolved)
                    arena_storage = {
                        tensor.untyped_storage().data_ptr()
                        for tensor in tensors.values()
                    }
                    # A loader may stash an arena view anywhere, including on a
                    # module this batch did not touch and on one it creates, so
                    # only a live whole-model walk can clear the arena for
                    # refill. Deferring any part of it to a post-install sweep
                    # is not equivalent: by then the arena has been overwritten,
                    # a consumer may have already committed the changed value,
                    # and a reference that was read and deleted leaves nothing
                    # to find.
                    scan_started = time.perf_counter()
                    for module in self._model.modules():
                        if retains_arena(module, arena_storage):
                            raise IncompleteRefit(
                                "engine retained bounded staging storage; restart required"
                            )
                    scan_s += time.perf_counter() - scan_started
                    installed.update(names)
                    torch.cuda.synchronize(self._device)
                    commit_s += time.perf_counter() - commit_started
            finally:
                batches.close()
            if installed != expected:
                raise IncompleteRefit(
                    "streaming transfer ended before every parameter was installed"
                )
            # Every batch already cleared its own arena against the live tree.
            # This repeats the check over every arena the install used, walking
            # the tree as it now stands rather than as it was cached, so a
            # module added during the final batch is still covered.
            scan_started = time.perf_counter()
            for module in self._model.modules():
                if retains_arena(module, arena_storages):
                    raise IncompleteRefit(
                        "engine retained bounded staging storage; restart required"
                    )
            scan_s += time.perf_counter() - scan_started
            load_s = time.perf_counter() - load_started

        reload_started = time.perf_counter()
        self._reload(load)
        metrics["reload_s"] = time.perf_counter() - reload_started - load_s
        metrics["install_commit_s"] = commit_s
        metrics["retention_scan_s"] = scan_s
        derived_started = time.perf_counter()
        _update_mla_absorbed_weights(self._model, quantized=False)
        torch.cuda.synchronize(self._device)
        metrics["derived_refresh_s"] = time.perf_counter() - derived_started
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
        # Same fixups and synchronize as install_tensors, so a checkpoint refit
        # reports the stage too rather than charging it to the caller's total.
        with refit_span("post_install"):
            _update_mla_absorbed_weights(self._model, quantized=self._is_quantized)
            torch.cuda.synchronize(self._device)

    @torch.no_grad()
    def _process_and_commit(
        self,
        tensors: dict[str, torch.Tensor],
        *,
        reload: bool = True,
        resolved: dict[str, tuple[Module, str]] | None = None,
    ) -> None:
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
            # A caller already inside that window may pass the resolution in
            # (streaming does, once per install) to avoid walking the model
            # for every batch.
            groups: dict[Module, list[tuple[str, str]]] = {}
            matched: set[str] = set()
            if resolved is not None:
                for full_name in tensors:
                    entry = resolved.get(full_name)
                    if entry is None:
                        continue
                    module, leaf = entry
                    groups.setdefault(module, []).append((full_name, leaf))
                    matched.add(full_name)
            else:
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
                if not reload and (info is None or info.kernel_tensors is None):
                    for full_name, leaf in parameters:
                        target = getattr(layer, leaf)
                        source = tensors[full_name]
                        if (
                            target.device.type == "meta"
                            or target.shape != source.shape
                            or target.dtype != source.dtype
                        ):
                            raise IncompleteRefit(
                                "unmanaged streaming parameter has no compatible live storage"
                            )
                        target.copy_(source)
                    continue
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

        if reload:
            self._reload(load)
        else:
            load()

    @torch.no_grad()
    def _reload(self, load: Callable[[], None]) -> None:
        """Run one weight loader inside vLLM's graph-safe reload window."""
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
        aliases = self._parameter_aliases(self._model)

        with torch.device(self._device), set_current_vllm_config(self._vllm_config):
            initialize_layerwise_reload(self._model)
            self._restore_parameter_aliases(aliases)
            _reserve_runtime_buffer_slots(self._model, LAYERWISE_INFO)
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
