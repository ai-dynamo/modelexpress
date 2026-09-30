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
import hashlib
import logging
import time
from collections.abc import Callable
from functools import cached_property
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
    PreparedDirectGroupTensors,
    PreparedEngineTensors,
    PreparedRuntimeTensors,
    PreparedStreamingTensors,
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
        self._capture_cache = None

    @cached_property
    def _original_loader(self) -> Callable:
        try:
            from vllm.model_executor.model_loader.reload.layerwise import (
                _get_original_loader,
            )
        except (ImportError, AttributeError) as error:
            raise RuntimeError(
                "ModelExpress refit requires vLLM's layerwise reload APIs"
            ) from error
        return _get_original_loader

    def _capture_key(self, manifest):
        original_loader = self._original_loader

        def function_identity(function):
            return (
                id(getattr(function, "__func__", function)),
                id(function.__self__) if hasattr(function, "__self__") else None,
            )

        parameters = tuple(
            (
                name,
                id(parameter),
                parameter.data_ptr(),
                tuple(parameter.shape),
                tuple(parameter.stride()),
                parameter.dtype,
                parameter.device,
                function_identity(original_loader(parameter)),
            )
            for name, parameter in self._model.named_parameters(remove_duplicate=False)
        )
        modules = tuple(
            (name, id(module), function_identity(getattr(module, "load_weights", None)))
            for name, module in self._model.named_modules()
        )
        routing_buffers = tuple(
            (
                name,
                id(buffer),
                buffer.data_ptr(),
                tuple(buffer.shape),
                hashlib.sha256(
                    buffer.detach().cpu().contiguous().numpy().tobytes()
                ).digest()
                if buffer.is_inference()
                else buffer._version,
            )
            for name, buffer in self._model.named_buffers()
            if not buffer.is_floating_point() and not buffer.is_complex()
        )
        return (
            tuple(manifest),
            parameters,
            modules,
            routing_buffers,
            id(self._convert_native_to_hf),
        )

    @property
    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            artifact_types=frozenset(
                {
                    PreparedEngineTensors,
                    PreparedDirectGroupTensors,
                    PreparedRuntimeTensors,
                    PreparedCheckpointArtifact,
                }
                | ({PreparedStreamingTensors} if not self._is_quantized else set())
            )
        )

    def prepare_streaming_artifact(
        self,
        *,
        version,
        source,
        batch_names,
        parameter_layout,
        staging_device,
        staging_buffers,
    ) -> PreparedArtifact:
        from modelexpress import envs

        from .direct_copy import prepare_direct_copy

        if staging_device != "cuda" or staging_buffers != 1:
            if envs.MX_REFIT_GLM_DIRECT:
                raise ValueError("GLM DIRECT requires one CUDA receive arena")
            return source
        if envs.MX_REFIT_GLM_DIRECT:
            from .direct_glm import prepare_glm_direct

            return prepare_glm_direct(
                self._model,
                self._vllm_config,
                version_id=version.version_id,
                source=source,
                batch_names=batch_names,
                parameter_layout=parameter_layout,
            )
        if self._is_quantized:
            return source
        selected = prepare_direct_copy(
            self._model,
            version_id=version.version_id,
            source=source,
            batch_names=batch_names,
        )
        if isinstance(selected, PreparedDirectGroupTensors):
            expected_layout = {
                item.name: (item.geometry[1], item.geometry[3])
                for item in selected.plan.destinations
            }
            if parameter_layout != expected_layout:
                return source
        return selected

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
        elif isinstance(prepared, PreparedDirectGroupTensors):
            from .direct_copy import install_direct_copy
            from .direct_glm import _GlmDirectPlan, install_glm_direct

            if type(prepared.plan) is _GlmDirectPlan:
                install_glm_direct(prepared, model=self._model)
            else:
                install_direct_copy(prepared, model=self._model)
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
            raise TypeError(f"unsupported prepared artifact {type(prepared).__name__}")
        if not isinstance(prepared, (PreparedStreamingTensors, PreparedDirectGroupTensors)):
            metrics["perf/mx_receive_install_time"] = time.perf_counter() - started
        return metrics

    @property
    def _is_quantized(self) -> bool:
        """Whether the live model uses a post-load quantized kernel layout."""
        return getattr(self._vllm_config, "quant_config", None) is not None

    @staticmethod
    def _parameter_aliases(model: Module) -> list[list[tuple[str, Module, str]]]:
        groups: dict[int, list[tuple[str, Module, str]]] = {}
        for path, module in model.named_modules(remove_duplicate=False):
            for name, parameter in module._parameters.items():
                if parameter is not None:
                    groups.setdefault(id(parameter), []).append((path, module, name))
        return [group for group in groups.values() if len(group) > 1]

    def _validate_alias_owners(
        self, aliases: list[list[tuple[str, Module, str]]]
    ) -> None:
        for group in aliases:
            for path, module, _ in group:
                try:
                    current = self._model.get_submodule(path)
                except AttributeError as error:
                    raise IncompleteRefit(
                        f"parameter alias owner {path!r} disappeared during refit"
                    ) from error
                if current is not module:
                    raise IncompleteRefit(
                        f"parameter alias owner {path!r} was replaced during refit"
                    )

    def _restore_parameter_aliases(
        self, aliases: list[list[tuple[str, Module, str]]]
    ) -> None:
        # vLLM restores metadata separately for each module. Reconnect shared
        # parameters so a tied loader still covers one canonical destination.
        self._validate_alias_owners(aliases)
        for group in aliases:
            _, first_module, first_name = group[0]
            parameter = getattr(first_module, first_name)
            for _, module, name in group[1:]:
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
        if not self._is_quantized and self._capture_cache is not None:
            key, result = self._capture_cache
            current_key = self._capture_key(manifest)
            if key == current_key:
                logger.info(
                    "reusing cached vLLM load layout (%d copies)", len(result[0].copies)
                )
                return copy.deepcopy(result)
        self._capture_cache = None
        try:
            from vllm.config import set_current_vllm_config
            from vllm.model_executor.model_loader.reload.layerwise import (
                LAYERWISE_INFO,
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

        original_loader = self._original_loader
        model = self._model
        aliases = self._parameter_aliases(model)
        with torch.device(self._device), set_current_vllm_config(self._vllm_config):
            initialize_layerwise_reload(model)
            try:
                self._restore_parameter_aliases(aliases)
                # Trace the ORIGINAL loaders, not the reload shims they were wrapped in.
                for _, param in model.named_parameters():
                    param.weight_loader = original_loader(param)
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
        if (
            not self._is_quantized
            and not capture.unsupported
            and not capture.unattributed
        ):
            self._capture_cache = (
                self._capture_key(manifest),
                copy.deepcopy((capture, param_layout)),
            )
        return capture, param_layout

    def install_tensors(self, tensors: dict[str, torch.Tensor]) -> None:
        """Install verified load-layout tensors without changing graph addresses."""
        self._process_and_commit(tensors)
        # Synchronization is paid once per install, separately from per-layer work.
        with refit_span("post_install"):
            torch.cuda.synchronize(self._device)

    @torch.no_grad()
    def install_streaming(self, prepared: PreparedStreamingTensors) -> None:
        """Commit complete modules into existing storage before arena reuse.

        Nothing about the module tree may be cached across batches. A post-load
        hook can replace a module, so a pinned module object goes stale, and it
        can add a Parameter to a module a later batch owns, so a batch that was
        complete when the layout was captured no longer is. Every resolution,
        completeness and retention check therefore walks the live tree.
        """
        if self._is_quantized:
            raise IncompleteRefit(
                "bounded streaming currently requires an unquantized engine"
            )

        metrics = prepared.transfer_metrics
        load_s = 0.0
        commit_s = 0.0
        setup_s = 0.0
        batch_scan_s = 0.0
        final_scan_s = 0.0
        batch_scans = 0
        final_scans = 0

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
            nonlocal load_s, commit_s
            nonlocal setup_s, batch_scan_s, final_scan_s, batch_scans, final_scans
            load_started = time.perf_counter()
            expected = set(dict(self._model.named_parameters()))
            if expected != prepared.parameter_names:
                raise IncompleteRefit(
                    "streaming parameter coverage differs from the live load layout"
                )
            # Resolution and the complete-owner check both happen inside
            # _process_and_commit, sharing the one live walk it already makes.
            installed = set()
            installed_parameters: dict[str, torch.Tensor] = {}
            arena_storages: set[int] = set()
            batches = prepared.batches()
            try:
                for tensors in batches:
                    names = set(tensors)
                    if not names or names - expected or names & installed:
                        raise IncompleteRefit(
                            "invalid or repeated streaming parameter batch"
                        )
                    commit_started = time.perf_counter()
                    self._process_and_commit(
                        tensors, reload=False, installed_parameters=installed_parameters
                    )
                    try:
                        installed_parameters.update(
                            (name, self._model.get_parameter(name)) for name in names
                        )
                    except AttributeError as error:
                        raise IncompleteRefit(
                            "installed canonical parameter disappeared"
                        ) from error
                    setup_started = time.perf_counter()
                    arena_storage = {
                        tensor.untyped_storage().data_ptr()
                        for tensor in tensors.values()
                    }
                    arena_storages |= arena_storage
                    setup_s += time.perf_counter() - setup_started
                    # Reject retained arena views before refill; a later loader
                    # could read overwritten data before the final scan.
                    scan_started = time.perf_counter()
                    for module in self._model.modules():
                        if retains_arena(module, arena_storage):
                            raise IncompleteRefit(
                                "engine retained bounded staging storage; restart required"
                            )
                    batch_scan_s += time.perf_counter() - scan_started
                    batch_scans += 1
                    installed.update(names)
                    torch.cuda.synchronize(self._device)
                    commit_s += time.perf_counter() - commit_started
            except BaseException:
                # close() reaches the transfer as GeneratorExit, which it cannot
                # tell apart from a clean abandonment. Report this install error;
                # the transfer has already logged its own cleanup failure.
                try:
                    batches.close()
                except Exception:
                    pass
                raise
            batches.close()
            if installed != expected:
                raise IncompleteRefit(
                    "streaming transfer ended before every parameter was installed"
                )
            # Repeat over every arena the install used, so a module the final
            # batch created is still covered.
            scan_started = time.perf_counter()
            for module in self._model.modules():
                if retains_arena(module, arena_storages):
                    raise IncompleteRefit(
                        "engine retained bounded staging storage; restart required"
                    )
            final_scan_s += time.perf_counter() - scan_started
            final_scans += 1
            load_s = time.perf_counter() - load_started

        reload_started = time.perf_counter()
        self._reload(load)
        metrics["reload_s"] = time.perf_counter() - reload_started - load_s
        metrics["install_commit_s"] = commit_s
        # retention_batch_scan_s is inside install_commit_s; retention_final_scan_s
        # is in neither it nor reload_s, so it has to be added, not inferred.
        metrics["retention_arena_setup_s"] = setup_s
        metrics["retention_batch_scan_s"] = batch_scan_s
        metrics["retention_final_scan_s"] = final_scan_s
        metrics["retention_batch_scans"] = batch_scans
        metrics["retention_final_scans"] = final_scans
        metrics["retention_scan_s"] = batch_scan_s + final_scan_s
        # MLA derived weights are refreshed by vLLM's post-load processing inside
        # the reload above, so only the once-per-install synchronize remains.
        sync_started = time.perf_counter()
        with refit_span("post_install"):
            torch.cuda.synchronize(self._device)
        metrics["post_install_sync_s"] = time.perf_counter() - sync_started

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

    @torch.no_grad()
    def _process_and_commit(
        self,
        tensors: dict[str, torch.Tensor],
        *,
        reload: bool = True,
        installed_parameters: dict[str, torch.Tensor] | None = None,
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
            # names only after vLLM has restored that load-time hierarchy, and
            # resolve them on every call: a streaming install runs this once per
            # batch, and a hook may have replaced a module since the last one.
            groups: dict[Module, list[tuple[str, str]]] = {}
            group_paths: dict[Module, str] = {}
            matched: set[str] = set()
            canonical: dict[int, str] = {}
            alias_groups: dict[int, list[tuple[str, Module, str]]] = {}
            for module_name, module in self._model.named_modules(remove_duplicate=False):
                duplicate_module = module in group_paths
                group_paths.setdefault(module, module_name)
                owned = set()
                missing = set()
                for leaf, parameter in module._parameters.items():
                    if parameter is None:
                        continue
                    full_name = f"{module_name}.{leaf}" if module_name else leaf
                    alias_groups.setdefault(id(parameter), []).append(
                        (module_name, module, leaf)
                    )
                    # Keep every alias path, but process each owning module once.
                    if duplicate_module:
                        continue
                    canonical_name = canonical.setdefault(id(parameter), full_name)
                    owned.add(full_name)
                    if canonical_name not in tensors:
                        previously_installed_alias = (
                            canonical_name != full_name
                            and installed_parameters is not None
                            and installed_parameters.get(canonical_name) is parameter
                        )
                        if not previously_installed_alias:
                            missing.add(canonical_name)
                    if full_name in tensors:
                        groups.setdefault(module, []).append((full_name, leaf))
                        matched.add(full_name)
                # Streaming installs an owning module at a time, so reject a
                # batch covering only part of one before any hook runs. Asked
                # of the live tree, since an earlier hook may have added a
                # Parameter here since the layout was captured.
                if (
                    not reload
                    and owned & tensors.keys()
                    and missing
                ):
                    raise IncompleteRefit(
                        f"streaming batch splits an owning module {module_name!r}; "
                        f"missing canonical parameters={sorted(missing)}"
                    )
            unmatched = sorted(set(tensors) - matched)
            if unmatched:
                raise IncompleteRefit(
                    "vLLM layerwise reload did not expose every staged parameter; "
                    f"unmatched={unmatched[:10]}"
                )
            aliases = [group for group in alias_groups.values() if len(group) > 1]

            # Per-layer spans, accumulated across every layer into two stages.
            # This loop is the whole of what a framework sees as "install", and
            # the two things it does have unrelated costs: post-load processing
            # is compute that scales with quantization scheme, while the copy
            # back is bandwidth into storage the CUDA graphs already point at.
            # Charged together they cannot be acted on.
            for layer, parameters in groups.items():
                # A packed batch resolves several owning modules before any of
                # their hooks run, and an earlier hook may replace a later
                # owner. Committing into the detached module would leave the
                # live one without the published bytes.
                if not reload and self._model.get_submodule(group_paths[layer]) is not layer:
                    raise IncompleteRefit(
                        f"a post-load hook replaced module {group_paths[layer]!r} "
                        "before its parameters were committed"
                    )
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
                self._restore_parameter_aliases(aliases)

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
            self._validate_alias_owners(aliases)

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
