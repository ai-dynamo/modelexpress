# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compiled layer replay for staged and streaming vLLM weight updates.

ModelExpress stages independent source slices and replays captured writes in order.
One reload plan materializes each owner before its first write, invokes native
post-load processing after its last dependency, and restores graph-bound storage.
Native finalization handles deferred attention after every layer has arrived.
"""

from __future__ import annotations

import copy
import hashlib
import logging
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping
from contextlib import closing
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING
from weakref import ref

import torch
from modelexpress import envs
from modelexpress.accelerators import accelerator_backend_for
from modelexpress.engines.vllm.host_quantization import (
    refresh_host_quantization_state,
)
from modelexpress_rl.refit.reshard.geometry import (
    capture_weights,
    convert_source_weights,
    overlapping_destination_groups,
)
from modelexpress_rl.refit.reshard.types import IncompleteRefit
from modelexpress_rl.refit.timing import refit_span

from modelexpress_rl import envs as rl_envs
from modelexpress_rl.inference.engines.vllm.surgical import (
    changed_since,
    module_groups,
)
from modelexpress_rl.inference.engines.vllm.stream_windows import (
    layer_windows,
    windowed_weights,
)

from modelexpress_rl.inference.engines.vllm.install_plan import (
    EngineInstallPlan,
    SourceInstallPlan,
    _EngineRuntimeBindings,
    _InputOwner,
    _InputSlot,
    _InstallExecution,
    _ReplaySchedule,
)
from modelexpress_rl.inference.plan import (
    EngineCapabilities,
    EngineInstaller,
    PreparedArtifact,
    PreparedCheckpointArtifact,
    PreparedEngineTensors,
    PreparedRuntimeTensors,
    PreparedStreamingTensors,
    SourceReplayBatch,
)
from modelexpress_rl.inference.receiver import PreparedCheckpoint
from modelexpress_rl.utils import index_checkpoint_tensors

if TYPE_CHECKING:
    from modelexpress_rl.inference.streaming_checkpoint import StreamedCheckpoint
    from modelexpress_rl.refit.reshard.types import CaptureResult
    from torch.nn import Module
    from vllm.config import ModelConfig, VllmConfig

logger = logging.getLogger("modelexpress_rl.inference.engines.vllm.installer")
_LAYERWISE_LOGGER = "vllm.model_executor.model_loader.reload.layerwise"


class _UnloadedLayerFilter(logging.Filter):
    """Count vLLM's per-layer warning for layers a delta leaves untouched."""

    def __init__(self) -> None:
        super().__init__()
        self.count = 0

    def filter(self, record: logging.LogRecord) -> bool:
        if record.msg == "%s: Failed to load weights":
            self.count += 1
            return False
        return True


_MODULE_GETATTRIBUTE = torch.nn.Module.__getattribute__
_MODULE_GETATTR = torch.nn.Module.__getattr__
_MODULE_GET_SUBMODULE = torch.nn.Module.get_submodule
_MODULE_SETATTR = torch.nn.Module.__setattr__
_MODULE_REGISTER_PARAMETER = torch.nn.Module.register_parameter


def _is_deferred_attention_layer(layer):
    from vllm.model_executor.layers import attention

    predicate = getattr(attention, "is_deferred_attention_layer", None)
    if predicate is not None:
        return predicate(layer)
    return isinstance(layer, (attention.Attention, attention.MLAAttention))


def _tensor_geometry(tensor):
    return (
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
        tensor.device,
        tensor.data_ptr(),
    )


def _clone_native_input(tensor):
    span = (
        1
        + sum(
            (size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride())
        )
        if tensor.numel()
        else 0
    )
    storage = tensor.data.as_strided((span,), (1,)).clone()
    isolated = torch.Tensor._make_subclass(
        type(tensor),
        storage.as_strided(tensor.shape, tensor.stride()),
        tensor.requires_grad,
    )
    isolated.__dict__.update(tensor.__dict__)
    return isolated, storage.numel() * storage.element_size()


def _runtime_tensor_layout(module):
    return tuple(
        (registry, name, _tensor_geometry(tensor))
        for registry in ("_parameters", "_buffers")
        for name, tensor in getattr(module, registry).items()
        if tensor is not None
    )


def _validate_runtime_tensor_layout(path, module, layout):
    for registry, name, geometry in layout:
        tensor = getattr(module, registry).get(name)
        if (
            not isinstance(tensor, torch.Tensor)
            or getattr(module, name, None) is not tensor
            or _tensor_geometry(tensor) != geometry
        ):
            raise IncompleteRefit(
                f"runtime tensor {path}.{name} disappeared or changed storage/geometry"
            )


@dataclass(frozen=True)
class _AliasEdge:
    path: str
    parent: Module
    name: str
    child: Module


@dataclass(frozen=True)
class _ParameterAliases:
    root: Module
    owners: tuple[tuple[str, Module], ...]
    edges: tuple[_AliasEdge, ...] | None
    groups: tuple[tuple[tuple[Module, str], ...], ...]
    ties: tuple[tuple[tuple[Module, str], ...], ...]
    writers: tuple[tuple[Module, type], ...]
    attributes: tuple[tuple[type, str], ...]

    @cached_property
    def by_owner(self):
        """Share each complete tie group with every layer that can change it."""
        modules = {"": self.root}
        if self.edges is None:
            return None
        for edge in self.edges:
            modules[edge.path] = edge.child
        owners = iter(self.owners)
        groups_by_owner = {}
        for group in self.groups:
            named = tuple((next(owners)[0], module, leaf) for module, leaf in group)
            for module in dict.fromkeys(module for module, _ in group):
                groups_by_owner.setdefault(module, []).append(named)
        return {
            module: _compile_parameter_aliases(self.root, groups, modules)
            for module, groups in groups_by_owner.items()
        }


def _compile_parameter_aliases(model, groups, modules) -> _ParameterAliases:
    groups = tuple(groups)
    owners, edges, full_groups, ties, writers, attributes = [], {}, [], [], {}, {}
    visited_paths = set()
    complete_paths = modules.get("") is model
    for group in groups:
        if len(group) < 2:
            continue
        slots = {}
        full_groups.append(tuple((module, leaf) for _, module, leaf in group))
        for path, module, leaf in group:
            owners.append((path, module))
            if modules.get(path) is not module:
                complete_paths = False
            slots.setdefault((id(module), leaf), (module, leaf))
            writers.setdefault(id(module), (module, type(module)))
            attributes.setdefault((type(module), leaf), None)
            pending = []
            if (
                type(path) is not str
                or path.startswith(".")
                or path.endswith(".")
                or ".." in path
            ):
                parent_path = ""
                for name in path.split(".") if path else ():
                    child_path = f"{parent_path}.{name}" if parent_path else name
                    pending.append((parent_path, name, child_path))
                    parent_path = child_path
                pending.reverse()
            else:
                child_path = path
                while child_path and child_path not in visited_paths:
                    parent_path, _, name = child_path.rpartition(".")
                    pending.append((parent_path, name, child_path))
                    child_path = parent_path
            for parent_path, name, child_path in reversed(pending):
                if parent_path not in modules or child_path not in modules:
                    complete_paths = False
                    break
                parent, child = modules[parent_path], modules[child_path]
                key = (id(parent), name)
                edge = edges.get(key)
                if edge is None:
                    edges[key] = _AliasEdge(child_path, parent, name, child)
                elif edge.child is not child:
                    complete_paths = False
                visited_paths.add(child_path)
        if len(slots) > 1:
            ties.append(tuple(slots.values()))
    return _ParameterAliases(
        model,
        tuple(owners),
        tuple(edges.values()) if complete_paths else None,
        tuple(full_groups),
        tuple(ties),
        tuple(writers.values()),
        tuple(attributes),
    )


def _has_descriptor(cls, name):
    for base in cls.__mro__:
        if name in base.__dict__:
            return hasattr(type(base.__dict__[name]), "__get__")
    return False


def _standard_lookup_containers(attributes):
    try:
        return (
            type(attributes.get("_parameters")) is dict
            and type(attributes.get("_buffers")) is dict
            and type(attributes.get("_modules")) is dict
        )
    except StopIteration as error:
        # Preserve the failure classification used by preparation retries.
        raise RuntimeError("generator raised StopIteration") from error


def _standard_module_access(module, name, checked_classes, checked_attributes):
    cls = type(module)
    if cls not in checked_classes:
        if (
            cls.__getattribute__ is not _MODULE_GETATTRIBUTE
            or getattr(cls, "__getattr__", None) is not _MODULE_GETATTR
        ):
            return False
        checked_classes.add(cls)
    key = (cls, name)
    if key not in checked_attributes:
        if _has_descriptor(cls, name):
            return False
        checked_attributes.add(key)
    # Custom lookup containers can also make repeated attribute reads dynamic.
    attributes = object.__getattribute__(module, "__dict__")
    return _standard_lookup_containers(attributes)


def _standard_parameter_writes(aliases: _ParameterAliases) -> bool:
    from torch.nn.modules.module import _global_parameter_registration_hooks

    if _global_parameter_registration_hooks:
        return False
    checked = set()
    for module, cls in aliases.writers:
        attributes = object.__getattribute__(module, "__dict__")
        if (
            type(module) is not cls
            or "register_parameter" in attributes
            or not _standard_lookup_containers(attributes)
        ):
            return False
        if cls not in checked:
            if (
                cls.__setattr__ is not _MODULE_SETATTR
                or cls.register_parameter is not _MODULE_REGISTER_PARAMETER
                or cls.__getattribute__ is not _MODULE_GETATTRIBUTE
                or getattr(cls, "__getattr__", None) is not _MODULE_GETATTR
            ):
                return False
            checked.add(cls)
    if any(_has_descriptor(cls, name) for cls, name in aliases.attributes):
        return False
    # Recheck ordinary registries without repeating custom truth callbacks.
    hooks = torch.nn.modules.module._global_parameter_registration_hooks
    if type(hooks) is dict or type(hooks) is OrderedDict:
        return not hooks
    return hooks is _global_parameter_registration_hooks


def _check_install_dispatch(*, bounded: bool) -> None:
    from torch.overrides import _get_current_function_mode_stack
    from torch.utils._device import DeviceContext
    from torch.utils._python_dispatch import _get_current_dispatch_mode

    if bounded and (
        any(
            type(mode) is not DeviceContext
            for mode in _get_current_function_mode_stack()
        )
        or _get_current_dispatch_mode() is not None
    ):
        raise IncompleteRefit(
            "weight installation does not support active tensor dispatch modes"
        )


def _receive_destination(target: torch.Tensor, *, bounded: bool) -> torch.Tensor:
    _check_install_dispatch(bounded=bounded)
    destination = target.data
    if type(destination) is not torch.Tensor:
        raise IncompleteRefit(
            "weight installation requires ordinary source and destination tensor views"
        )
    if (
        destination.is_meta
        or destination.data_ptr() != target.data_ptr()
        or destination.shape != target.shape
        or destination.stride() != target.stride()
        or destination.dtype != target.dtype
        or destination.device != target.device
        or destination.storage_offset() != target.storage_offset()
        or destination.untyped_storage().data_ptr()
        != target.untyped_storage().data_ptr()
    ):
        raise IncompleteRefit("load-time parameter does not expose its own storage")
    return destination


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
    """Capture vLLM's load layout and install received weights."""

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
        # Version whose checkpoint bytes the engine holds; None when unknown.
        self._live_version: str | None = None
        self._capture_cache = None
        self._debug_validate_generator_layout = (
            rl_envs.MX_REFIT_DEBUG_VALIDATE_GENERATOR_LAYOUT
        )
        self.engine_install_plan = None
        self.source_install_plan = None
        self._plan_generation = 0
        self._in_reload = False
        self._load_buffer_names = frozenset()
        self._persistent_names = frozenset()
        self._tied_input_peak_bytes = 0

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

    def capture_generator_fingerprint(self):
        if not self._debug_validate_generator_layout:
            return ()

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
        buffers = tuple(
            (
                name,
                id(buffer),
                buffer.data_ptr(),
                tuple(buffer.shape),
                tuple(buffer.stride()),
                buffer.dtype,
                buffer.device,
                function_identity(original_loader(buffer)),
                None
                if buffer.is_floating_point() or buffer.is_complex()
                else (
                    hashlib.sha256(
                        buffer.detach().cpu().contiguous().numpy().tobytes()
                    ).digest()
                    if buffer.is_inference()
                    else buffer._version
                ),
            )
            for name, buffer in self._model.named_buffers(remove_duplicate=False)
        )
        return (
            parameters,
            modules,
            buffers,
            id(self._convert_native_to_hf),
        )

    @property
    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            artifact_types=frozenset(
                {
                    PreparedEngineTensors,
                    PreparedRuntimeTensors,
                    PreparedCheckpointArtifact,
                    PreparedStreamingTensors,
                }
            ),
            streamed_checkpoints=(
                getattr(
                    getattr(self._vllm_config, "parallel_config", None),
                    "tensor_parallel_size",
                    1,
                )
                == 1
                or envs.MX_MS_DISTRIBUTED
            ),
        )

    def install(self, prepared: PreparedArtifact) -> dict[str, float]:
        started = time.perf_counter()
        self._tied_input_peak_bytes = 0
        metrics = prepared.metrics
        live_version, self._live_version = self._live_version, None
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
            if checkpoint.streaming is not None:
                self.install_streamed_checkpoint(checkpoint.streaming)
                metrics.update(checkpoint.streaming.cache_metrics)
            else:
                changed = (
                    changed_since(
                        checkpoint.delta_changes,
                        live_version=live_version,
                        target_version=checkpoint.target_version,
                    )
                    if rl_envs.MX_REFIT_DELTA_SURGICAL
                    else None
                )
                if changed is None:
                    self.install_checkpoint(checkpoint.path)
                else:
                    metrics.update(self.install_changed_tensors(checkpoint.path, changed))
            self._live_version = checkpoint.target_version
        else:
            raise TypeError(f"unsupported prepared artifact {type(prepared).__name__}")
        if not isinstance(prepared, PreparedStreamingTensors):
            metrics["perf/mx_receive_install_time"] = time.perf_counter() - started
        metrics["installation_tied_input_peak_bytes"] = self._tied_input_peak_bytes
        return metrics

    @property
    def _is_quantized(self) -> bool:
        """Whether the live model uses a post-load quantized kernel layout."""
        return getattr(self._vllm_config, "quant_config", None) is not None

    @staticmethod
    def _parameter_aliases(model: Module) -> _ParameterAliases:
        groups: dict[int, list[tuple[str, Module, str]]] = {}
        modules = {}
        for path, module in model.named_modules(remove_duplicate=False):
            modules[path] = module
            for name, parameter in module._parameters.items():
                if parameter is not None:
                    groups.setdefault(id(parameter), []).append((path, module, name))
        return _compile_parameter_aliases(model, groups.values(), modules)

    def _validate_alias_owners(self, aliases: _ParameterAliases) -> None:
        if self._model is not aliases.root:
            raise IncompleteRefit("parameter alias root was replaced during refit")
        standard = (
            aliases.edges is not None
            and type(self._model).get_submodule is _MODULE_GET_SUBMODULE
            and "get_submodule" not in object.__getattribute__(self._model, "__dict__")
        )
        checked_classes, checked_attributes = set(), set()
        if standard:
            for edge in aliases.edges:
                if not _standard_module_access(
                    edge.parent, edge.name, checked_classes, checked_attributes
                ):
                    standard = False
                    break
                try:
                    current = getattr(edge.parent, edge.name)
                except AttributeError as error:
                    raise IncompleteRefit(
                        f"parameter alias owner {edge.path!r} disappeared during refit"
                    ) from error
                if current is not edge.child:
                    raise IncompleteRefit(
                        f"parameter alias owner {edge.path!r} was replaced during refit"
                    )
        if standard:
            return
        for path, module in aliases.owners:
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

    def _load_time_parameter_aliases(
        self, aliases: _ParameterAliases, layerwise_info: Mapping
    ) -> _ParameterAliases:
        """Exclude runtime-only slots using vLLM's recorded load-time metadata."""
        self._validate_alias_owners(aliases)
        owners = iter(aliases.owners)
        groups = []
        changed = False
        for group in aliases.groups:
            load_group = []
            for module, name in group:
                path, _ = next(owners)
                info = layerwise_info.get(module)
                metadata = getattr(info, "restore_metadata", None)
                if metadata is not None and name not in metadata[0]:
                    changed = True
                    continue
                load_group.append((path, module, name))
            groups.append(load_group)
        if not changed:
            return aliases
        modules = dict(self._model.named_modules(remove_duplicate=False))
        return _compile_parameter_aliases(self._model, groups, modules)

    def _restore_parameter_aliases(self, aliases: _ParameterAliases) -> None:
        # vLLM restores metadata separately for each module. Reconnect shared
        # parameters so a tied loader still covers one canonical destination.
        self._validate_alias_owners(aliases)
        standard = _standard_parameter_writes(aliases)
        for group in aliases.ties if standard else aliases.groups:
            first_module, first_name = group[0]
            parameter = getattr(first_module, first_name)
            for module, name in group[1:]:
                other = getattr(module, name)
                if other.shape != parameter.shape or other.dtype != parameter.dtype:
                    raise IncompleteRefit(
                        "tied parameters have incompatible load-time layouts"
                    )
                if not standard or other is not parameter:
                    setattr(module, name, parameter)

    def capture(
        self, manifest: list[tuple[str, torch.dtype, tuple[int, ...]]]
    ) -> tuple[
        CaptureResult,
        dict[str, tuple[tuple[int, ...], torch.dtype]],
    ]:
        """Record how published tensors map into vLLM's load-time parameters.

        Captures on the LIVE model with its params reverted to engine load-time
        skeletons via layerwise reload; graph-bound kernel tensors are restored
        afterward without finalizing (finalizing would commit the empty skeletons
        and corrupt the live params).
        """

        gen_key = self.capture_generator_fingerprint()

        if self._capture_cache is not None:
            cached_gen_key, cached_manifest, result = self._capture_cache
            if cached_gen_key == gen_key and cached_manifest == tuple(manifest):
                logger.info(
                    "reusing cached vLLM load layout (%d copies)", len(result[0].copies)
                )
                return copy.deepcopy(result)
        self._capture_cache = None
        self.engine_install_plan = None
        self.source_install_plan = None
        self.__dict__.pop("_reload_plan", None)
        self._load_buffer_names = frozenset()
        self._persistent_names = frozenset()
        self._plan_generation += 1
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
        aliases = self._reload_plan.load_aliases
        with torch.device(self._device), set_current_vllm_config(self._vllm_config):
            initialize_layerwise_reload(model)
            try:
                self._restore_parameter_aliases(aliases)
                self._ensure_engine_install_plan()
                # Trace the ORIGINAL loaders, not the reload shims they were wrapped in.
                for _, param in (*model.named_parameters(), *model.named_buffers()):
                    param.weight_loader = original_loader(param)
                # The explicit default loader stamps params without a custom
                # weight_loader (norms) so their copies are attributed, not dropped.
                capture = capture_weights(
                    model,
                    convert_source_weights(
                        self._convert_native_to_hf, manifest, device=self._device
                    ),
                    default_weight_loader=default_weight_loader,
                    source_replay=True,
                )
                capture.generation = self._plan_generation
                param_layout = {
                    name: (tuple(p.shape), p.dtype)
                    for name, p in model.named_parameters()
                }
                copied = {item.param_name for item in capture.copies}
                self._persistent_names = frozenset(
                    name
                    for name, value in (
                        *model.named_parameters(),
                        *model.named_buffers(),
                    )
                    if name in copied and not value.is_meta
                )
                self._load_buffer_names = frozenset(
                    name for name, _ in model.named_buffers() if name in copied
                )
                param_layout.update(
                    (name, (tuple(buffer.shape), buffer.dtype))
                    for name, buffer in model.named_buffers()
                    if name in self._load_buffer_names
                )
            finally:
                for layer in model.modules():
                    info = LAYERWISE_INFO.get(layer)
                    if info is not None:
                        if info.kernel_tensors is not None:
                            _place_kernel_tensors(layer, info)
                        info.reset()
                if hasattr(model, "_original_do_torchao_reload"):
                    model._do_torchao_reload = model._original_do_torchao_reload
        logger.info(
            "captured %d copies and %d unsupported sources (quantized=%s)",
            len(capture.copies),
            len(capture.unsupported),
            self._is_quantized,
        )
        gen_key = self.capture_generator_fingerprint()
        capture.generator_fingerprint = gen_key
        if not capture.unsupported and not capture.unattributed:
            engine = self.engine_install_plan
            targets = engine.bind_targets(self._load_buffer_names)
            last_writes = {
                item.param_name: index for index, item in enumerate(capture.copies)
            }
            if not targets.by_name.keys() <= last_writes.keys():
                return capture, param_layout
            schedule = _ReplaySchedule.compile(
                targets,
                engine.runtime.overlap_groups,
                last_writes,
                len(capture.copies),
            )
            self.source_install_plan = SourceInstallPlan(
                targets,
                schedule,
            )
            self._capture_cache = (
                gen_key,
                tuple(manifest),
                copy.deepcopy((capture, param_layout)),
            )
        return capture, param_layout

    def install_tensors(self, tensors: SourceReplayBatch) -> None:
        """Replay verified source writes without changing graph addresses."""
        if not isinstance(tensors, SourceReplayBatch):
            raise IncompleteRefit("vLLM installation requires ordered source replay")

        def load(aliases: _ParameterAliases) -> None:
            plan = self._new_install_execution()
            self._apply_source_batch(plan, tensors)
            self._finish_source_replay(plan)

        self._reload(load)
        # Synchronization is paid once per install, separately from per-layer work.
        with refit_span("post_install"):
            torch.cuda.synchronize(self._device)

    def _drain_streaming(self) -> None:
        torch.cuda.synchronize(self._device)

    @torch.no_grad()
    def install_streaming(self, prepared: PreparedStreamingTensors) -> None:
        """Replay source writes into engine storage before arena reuse."""
        ownership = prepared.ownership
        if ownership.iterator is not None or ownership.release_blocked:
            raise IncompleteRefit(
                "streaming transaction still retains transfer resources"
            )

        metrics = prepared.transfer_metrics
        load_s = commit_s = 0.0
        live_storages = {
            tensor.untyped_storage().data_ptr()
            for tensor in (*self._model.parameters(), *self._model.buffers())
            if tensor.device.type != "meta"
        }
        primary_error = None

        def drain() -> None:
            try:
                self._drain_streaming()
            except BaseException:
                ownership.drain_failed = True
                raise

        def load(initial_aliases: _ParameterAliases) -> None:
            nonlocal load_s, commit_s
            load_started = time.perf_counter()
            plan = self._new_install_execution()
            plan.bounded = True
            expected = set(plan.targets.by_name)
            if expected != prepared.parameter_names:
                raise IncompleteRefit(
                    "streaming parameter coverage differs from the live load layout"
                )
            try:
                ownership.iterator = iter(prepared.batches())
            except BaseException:
                ownership.source_failed = True
                raise
            sentinel = object()
            while True:
                try:
                    tensors = next(ownership.iterator, sentinel)
                except BaseException:
                    # A CUDA fence does not establish that failed RDMA reads drained.
                    ownership.source_failed = True
                    raise
                if tensors is sentinel:
                    break
                if not isinstance(tensors, SourceReplayBatch):
                    raise IncompleteRefit(
                        "vLLM installation requires ordered source replay"
                    )
                commit_started = time.perf_counter()
                received_storages = {
                    tensor.untyped_storage().data_ptr() for tensor in tensors.values()
                }
                if (received_storages & live_storages) - {0}:
                    raise IncompleteRefit(
                        "received streaming tensor aliases live storage"
                    )
                self._apply_source_batch(plan, tensors)
                drain()
                commit_s += time.perf_counter() - commit_started
                del tensors
            if plan.replay_metadata is None or plan.installed != expected:
                raise IncompleteRefit(
                    "streaming transfer ended before every parameter was installed"
                )
            self._finish_source_replay(plan)
            load_s = time.perf_counter() - load_started

        reload_started = time.perf_counter()
        try:
            self._reload(load)
            reload_s = time.perf_counter() - reload_started - load_s
        except BaseException as error:
            primary_error = error
            raise
        finally:
            if not ownership.drain_failed:
                try:
                    sync_started = time.perf_counter()
                    with refit_span("post_install"):
                        drain()
                    metrics["post_install_sync_s"] = time.perf_counter() - sync_started
                    if ownership.iterator is not None and not ownership.source_failed:
                        try:
                            close = getattr(ownership.iterator, "close", None)
                            if close is not None:
                                close()
                        except BaseException:
                            ownership.close_failed = True
                            raise
                        ownership.iterator = None
                except BaseException as cleanup_error:
                    if primary_error is None:
                        raise
                    raise primary_error from cleanup_error

        metrics["reload_s"] = reload_s
        metrics["install_commit_s"] = commit_s

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

    def install_streamed_checkpoint(self, checkpoint: StreamedCheckpoint) -> None:
        """Stream once through graph-safe reload and tee to one writer per node."""
        from vllm.distributed import get_world_group
        from vllm.model_executor.model_loader.runai_streamer_loader import (
            RunaiModelStreamerLoader,
        )
        from vllm.model_executor.model_loader.weight_utils import (
            runai_safetensors_weights_iterator,
        )

        tp_size = getattr(
            getattr(self._vllm_config, "parallel_config", None),
            "tensor_parallel_size",
            1,
        )
        load_config = copy.copy(self._vllm_config.load_config)
        extra = dict(getattr(load_config, "model_loader_extra_config", None) or {})
        extra["distributed"] = tp_size > 1 and envs.MX_MS_DISTRIBUTED
        object.__setattr__(load_config, "model_loader_extra_config", extra)
        loader = RunaiModelStreamerLoader(load_config)
        window_layers = rl_envs.MX_REFIT_STREAM_WINDOW_LAYERS
        if window_layers:
            weights = windowed_weights(
                list(checkpoint.shard_uris),
                layer_windows(checkpoint.tensor_metadata, window_layers),
                is_distributed=loader._is_distributed,
            )
        else:
            weights = runai_safetensors_weights_iterator(
                list(checkpoint.shard_uris),
                load_config.use_tqdm_on_load,
                is_distributed=loader._is_distributed,
            )
        cache = get_world_group().local_rank == 0
        with (
            closing(weights),
            closing(checkpoint.validate(weights, cache=cache)) as verified,
        ):
            self._reload(lambda _aliases: self._model.load_weights(verified))
        if not checkpoint._complete:
            raise RuntimeError("full checkpoint stream was not completely consumed")
        with refit_span("post_install"):
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

        self._reload(lambda _aliases: loader.load_weights(self._model, model_config))
        # Same synchronize as install_tensors, so a checkpoint refit
        # reports the stage too rather than charging it to the caller's total.
        with refit_span("post_install"):
            torch.cuda.synchronize(self._device)

    def install_changed_tensors(
        self, path: str | Path, changed: frozenset[str]
    ) -> dict[str, float]:
        """Reload only the modules whose checkpoint tensors a delta changed.

        Untouched layers receive no weights, so vLLM's finalize places their
        existing kernel tensors back. If a fed module is still incomplete, the
        rest of the checkpoint is streamed through the same reload, which is
        then exactly a full reload.
        """
        from safetensors import safe_open
        from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
        from vllm.model_executor.model_loader.reload.layerwise import LAYERWISE_INFO

        started = time.perf_counter()
        _, locations, _ = index_checkpoint_tensors(path)
        packed: dict[str, list[str]] = {}
        for module in self._model.modules():
            packed.update(getattr(module, "packed_modules_mapping", None) or {})
        names = module_groups(changed, locations, packed)
        by_file: dict[Path, list[str]] = {}
        for name in sorted(
            names, key=lambda n: (str(locations[n][0]), locations[n][1])
        ):
            by_file.setdefault(locations[name][0], []).append(name)
        fallback = False

        def subset():
            for file, file_names in by_file.items():
                with safe_open(str(file), framework="pt") as tensors:
                    for name in file_names:
                        yield name, tensors.get_tensor(name)

        def load(_aliases: _ParameterAliases) -> None:
            nonlocal fallback
            self._model.load_weights(subset())
            incomplete = [
                type(layer).__name__
                for layer, info in LAYERWISE_INFO.items()
                if info.can_load()
                and 0 < info.load_numel < info.load_numel_total
                and not _is_deferred_attention_layer(layer)
            ]
            if not incomplete:
                return
            fallback = True
            logger.warning(
                "Surgical checkpoint install left %d modules incomplete (%s); "
                "loading the rest of the checkpoint",
                len(incomplete),
                sorted(set(incomplete))[:5],
            )
            load_config = copy.copy(self._vllm_config.load_config)
            object.__setattr__(load_config, "load_format", "safetensors")
            model_config = copy.copy(self._model_config)
            model_config.model = str(path)
            model_config.revision = None
            rest = DefaultModelLoader(load_config).get_all_weights(
                model_config, self._model
            )
            self._model.load_weights(
                (name, tensor) for name, tensor in rest if name not in names
            )

        unloaded = _UnloadedLayerFilter()
        layerwise_logger = logging.getLogger(_LAYERWISE_LOGGER)
        layerwise_logger.addFilter(unloaded)
        try:
            self._reload(load)
        finally:
            layerwise_logger.removeFilter(unloaded)
        with refit_span("post_install"):
            torch.cuda.synchronize(self._device)
        logger.info(
            "Surgical checkpoint install changed=%d loaded=%d untouched_layers=%d "
            "fallback=%s seconds=%.3f",
            len(changed),
            len(names),
            unloaded.count,
            fallback,
            time.perf_counter() - started,
        )
        return {
            "perf/mx_receive_surgical_tensors": float(len(names)),
            "perf/mx_receive_surgical_fallback": float(fallback),
        }

    def _ensure_engine_install_plan(self):
        if self.engine_install_plan is not None:
            return self.engine_install_plan
        inputs = []
        for path, layer in self._model.named_modules():
            slots = []
            for is_buffer, values in (
                (False, layer._parameters),
                (True, layer._buffers),
            ):
                for leaf, tensor in values.items():
                    if tensor is None:
                        continue
                    name = f"{path}.{leaf}" if path else leaf
                    slots.append(_InputSlot(name, leaf, id(tensor), is_buffer))
            inputs.append(_InputOwner(path, ref(layer), tuple(slots)))
        self.engine_install_plan = EngineInstallPlan(
            self._reload_plan,
            tuple(inputs),
        )
        return self.engine_install_plan

    def _new_install_execution(self):
        engine = self._ensure_engine_install_plan()
        source = self.source_install_plan
        targets = (
            source.targets
            if source is not None
            else engine.bind_targets(self._load_buffer_names)
        )
        return _InstallExecution(
            engine.runtime.load_aliases,
            targets,
            pending={
                name: ref(getattr(entry.layer, leaf))
                for entry in targets.owners
                for name, leaf, _ in entry.parameters
            },
            schedule=source.schedule if source is not None else None,
        )

    def _validate_source_owner(self, plan) -> None:
        layer = plan.layer
        if self._model.get_submodule(plan.path) is not layer:
            raise IncompleteRefit(
                f"a post-load hook replaced module {plan.path!r} "
                "before its parameters were committed"
            )
        current = {
            name for name, value in layer._parameters.items() if value is not None
        }
        current.update(
            name
            for name, value in layer._buffers.items()
            if value is not None
            and (f"{plan.path}.{name}" if plan.path else name)
            in self._load_buffer_names
        )
        required = {leaf for _name, leaf, _canonical in plan.parameters}
        missing = sorted(current - required)
        if required - current:
            raise IncompleteRefit("installed canonical parameter disappeared")
        if missing:
            raise IncompleteRefit(
                f"source replay owner {plan.path!r} gained unplanned parameters={missing}"
            )

    @torch.no_grad()
    def _apply_source_batch(
        self, plan: _InstallExecution, batch: SourceReplayBatch
    ) -> None:
        if batch.generation is not None and batch.generation != self._plan_generation:
            raise IncompleteRefit(
                "captured replay plan was invalidated before installation"
            )
        if plan.replay_metadata is not None and (
            batch.parameter_names is not plan.replay_metadata[0]
            or batch.last_writes is not plan.replay_metadata[1]
        ):
            raise IncompleteRefit(
                "source replay batch belongs to a different complete plan"
            )
        if plan.replay_metadata is None:
            if batch.parameter_names != plan.targets.by_name.keys():
                raise IncompleteRefit(
                    "source replay coverage differs from the live load layout"
                )
            if plan.schedule is None:
                plan.schedule = _ReplaySchedule.compile(
                    plan.targets,
                    self._runtime_replay_groups,
                    batch.last_writes,
                    batch.total_copies,
                )
            if batch.last_writes != plan.schedule.last_writes:
                raise IncompleteRefit("source replay completion schedule differs")
            plan.replay_metadata = (batch.parameter_names, batch.last_writes)
        if batch.total_copies != plan.schedule.total_copies or not batch.writes:
            raise IncompleteRefit("incompatible or empty source replay batch")
        for index, recorded, source in batch.writes:
            if index != plan.replay_next or index >= plan.schedule.total_copies:
                raise IncompleteRefit("source replay is not in recorded order")
            owner = plan.targets.by_name[recorded.param_name]
            if owner not in plan.replay_started:
                for pending_owner in plan.schedule.groups[owner]:
                    if pending_owner not in plan.replay_started:
                        self._begin_source_owner(pending_owner, plan)
            self._copy_source_write(owner, plan, recorded, source)
            plan.replay_next += 1
            for ready in plan.schedule.ready.get(index, ()):
                self._finish_source_group(ready, plan)

    def _isolate_native_tied_inputs(self, owners, plan):
        from vllm.model_executor.layers.quantization.base_config import (
            QuantizeMethodBase,
        )

        native = {
            owner
            for owner in owners
            if not plan.replay_started[owner][0]
            and isinstance(
                getattr(owner.layer, "quant_method", None), QuantizeMethodBase
            )
        }
        if not native:
            return
        slots = {}
        for owner in owners:
            for _, leaf, canonical in owner.parameters:
                slots.setdefault(canonical, {}).setdefault(owner, []).append(leaf)
        allocated = 0
        for consumers in slots.values():
            if len(consumers) < 2:
                continue
            # The last native consumer can use the original only when every
            # other consumer has its own input, including non-native restores.
            last = next(reversed(consumers)) if consumers.keys() <= native else None
            for owner, leaves in consumers.items():
                if owner not in native or owner is last:
                    continue
                original = getattr(owner.layer, leaves[0])
                isolated, nbytes = _clone_native_input(original)
                allocated += nbytes
                for leaf in leaves:
                    setattr(owner.layer, leaf, isolated)
        self._tied_input_peak_bytes = max(self._tied_input_peak_bytes, allocated)

    def _finish_source_group(self, owners, plan):
        self._isolate_native_tied_inputs(owners, plan)
        for owner in owners:
            self._process_owner(owner, plan.replay_started[owner][0])
        for owner in owners:
            self._restore_owner(owner, plan.replay_started[owner][0])
        for owner in owners:
            aliases = (
                plan.aliases
                if plan.aliases.by_owner is None
                else plan.aliases.by_owner.get(owner.layer)
            )
            if aliases is not None:
                self._restore_parameter_aliases(aliases)
            plan.replay_started.pop(owner)
            plan.installed.update(canonical for _, _, canonical in owner.parameters)

    def _copy_source_write(self, owner, plan, recorded, source):
        from modelexpress_rl.inference.nixl_staged_transfer import _replay_ops

        if self._model.get_submodule(owner.path) is not owner.layer:
            raise IncompleteRefit(f"replay owner {owner.path!r} was replaced")
        leaf = recorded.param_name.rpartition(".")[2]
        target = getattr(owner.layer, leaf)
        if target is not plan.replay_started[owner][1][leaf]:
            raise IncompleteRefit(
                f"pending parameter slot {recorded.param_name!r} was replaced"
            )
        destination = _receive_destination(target, bounded=plan.bounded)
        if target.dtype != recorded.dest_dtype:
            raise IncompleteRefit("recorded destination has no compatible live storage")
        destination = destination.as_strided(
            recorded.dest_shape, recorded.dest_stride,
            target.storage_offset() + recorded.dest_offset,
        )
        value = _replay_ops(source, recorded.op_chain)
        if tuple(value.shape) != tuple(recorded.dest_shape):
            raise IncompleteRefit("recorded source shape differs from its destination")
        if type(value) is not torch.Tensor:
            raise IncompleteRefit("weight installation requires an ordinary source tensor")
        torch.Tensor.copy_(destination, value)

    def _begin_source_owner(self, owner, plan):
        from vllm.model_executor.model_loader.reload.layerwise import LAYERWISE_INFO
        from vllm.model_executor.model_loader.reload.meta import materialize_layer

        self._validate_source_owner(owner)
        layer = owner.layer
        for name, leaf, _ in owner.parameters:
            if getattr(layer, leaf) is not plan.pending[name]():
                raise IncompleteRefit(
                    f"pending parameter slot {name!r} was replaced before its copy"
                )
        info = LAYERWISE_INFO.get(layer)
        if getattr(info, "loaded_weights", ()):
            raise IncompleteRefit("engine reload has pending checkpoint-loader calls")
        persistent = all(
            canonical in self._persistent_names and not getattr(layer, leaf).is_meta
            for _, leaf, canonical in owner.parameters
        )
        if info is not None and not persistent:
            materialize_layer(layer, info)
            aliases = (
                plan.aliases
                if plan.aliases.by_owner is None
                else plan.aliases.by_owner.get(layer)
            )
            if aliases is not None:
                self._restore_parameter_aliases(aliases)
                for tied_owner in plan.schedule.groups[owner]:
                    for name, leaf, _ in tied_owner.parameters:
                        if any(
                            tied_owner.layer is module and leaf == slot
                            for group in aliases.ties
                            for module, slot in group
                        ):
                            plan.pending[name] = ref(getattr(tied_owner.layer, leaf))
        # Reconnect compatible overlapping load-time views to the graph's common
        # backing. Their recorded copies then have the same alias semantics as
        # native loading; later full-owner restoration cannot undo another write.
        for name, leaf, _ in owner.parameters:
            original = self._runtime_replay_tensors.get(name)
            if original is not None:
                target = getattr(layer, leaf)
                if (
                    target.shape != original.shape
                    or target.dtype != original.dtype
                    or target.stride() != original.stride()
                ):
                    raise IncompleteRefit(
                        f"overlapping runtime destination {name!r} has a different load-time layout"
                    )
                target.data = original.data
        targets = {leaf: getattr(layer, leaf) for _, leaf, _ in owner.parameters}
        for _, leaf, canonical in owner.parameters:
            if canonical not in plan.replay_initialized:
                if canonical not in self._persistent_names:
                    targets[leaf].data.zero_()
                plan.replay_initialized.add(canonical)
        plan.replay_started[owner] = (persistent, targets)

    def _process_owner(self, owner, persistent):
        if persistent:
            return
        from vllm.model_executor.layers.quantization.base_config import (
            QuantizeMethodBase,
        )

        layer = owner.layer
        if hasattr(layer, "_already_called_process_weights_after_loading"):
            delattr(layer, "_already_called_process_weights_after_loading")
        quant_method = getattr(layer, "quant_method", None)
        if isinstance(quant_method, QuantizeMethodBase):
            quant_method.process_weights_after_loading(layer)
            update_tp = getattr(layer, "update_param_tp_status", None)
            if update_tp is not None:
                update_tp()

    @staticmethod
    def _restore_owner(owner, persistent):
        if persistent:
            return
        from vllm.model_executor.model_loader.reload.layerwise import (
            LAYERWISE_INFO,
            _copy_and_restore_kernel_tensors,
        )

        layer = owner.layer
        info = LAYERWISE_INFO.get(layer)
        if info is not None and info.kernel_tensors is not None:
            _copy_and_restore_kernel_tensors(layer, info)
        if info is not None and not _is_deferred_attention_layer(layer):
            info.reset()

    @staticmethod
    def _finish_source_replay(plan):
        if plan.replay_next != plan.schedule.total_copies or plan.replay_started:
            raise IncompleteRefit(
                "source replay ended before every recorded write was installed"
            )

    @cached_property
    def _reload_plan(self):
        from vllm.model_executor.model_loader.reload.layerwise import LAYERWISE_INFO

        # Deferred post-load hooks can replace bare tensors and registered buffers.
        # Preserve both for graph consumers and ModelExpress tensor registrations.
        owners = tuple(
            (path, ref(module))
            for path, module in self._model.named_modules(remove_duplicate=False)
        )
        runtime_layouts = {}
        for path, reference in owners:
            module = reference()
            if module is not None and id(module) not in runtime_layouts:
                runtime_layouts[id(module)] = (
                    path,
                    reference,
                    _runtime_tensor_layout(module),
                )
        del module
        bare_tensors = [
            (
                path,
                module,
                {
                    name: (
                        value,
                        _tensor_geometry(value),
                        name not in module._non_persistent_buffers_set
                        if name in module._buffers
                        else None,
                    )
                    for name, value in {**module.__dict__, **module._buffers}.items()
                    if isinstance(value, torch.Tensor)
                    and value.device.type == self._device.type
                    and (
                        self._device.index is None
                        or value.device.index == self._device.index
                    )
                },
            )
            for path, module in ((path, reference()) for path, reference in owners)
            if module is not None
        ]
        bare_tensors = [entry for entry in bare_tensors if entry[2]]
        runtime_tensors = dict(
            (
                *self._model.named_parameters(remove_duplicate=False),
                *self._model.named_buffers(remove_duplicate=False),
            )
        )
        replay_groups = overlapping_destination_groups(runtime_tensors.items())
        replay_tensors = {
            name: runtime_tensors[name]
            for group in replay_groups
            if len({id(runtime_tensors[name]) for name in group}) > 1
            for name in group
        }
        aliases = self._parameter_aliases(self._model)
        load_aliases = self._load_time_parameter_aliases(aliases, LAYERWISE_INFO)

        return _EngineRuntimeBindings(
            owners,
            runtime_layouts,
            bare_tensors,
            replay_groups,
            replay_tensors,
            aliases,
            load_aliases,
        )

    @torch.no_grad()
    def _reload(self, load: Callable[[_ParameterAliases], None]) -> None:
        """Run one weight loader inside vLLM's graph-safe reload window."""
        try:
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
            from vllm.model_executor.model_loader.reload.meta import materialize_layer
        except (ImportError, AttributeError) as error:
            raise RuntimeError(
                f"ModelExpress refit requires vLLM's layerwise reload APIs: {error}"
            ) from error

        if not all(
            callable(api)
            for api in (
                QuantizeMethodBase,
                _copy_and_restore_kernel_tensors,
                materialize_layer,
                initialize_layerwise_reload,
                finalize_layerwise_reload,
            )
        ):
            raise RuntimeError(
                "ModelExpress refit requires vLLM's layerwise reload APIs"
            )

        runtime = (
            self.engine_install_plan.runtime
            if self.engine_install_plan is not None
            else self._reload_plan
        )
        owners = runtime.modules
        runtime_layouts = runtime.layouts
        bare_tensors = runtime.bare_tensors
        self._runtime_replay_groups = runtime.overlap_groups
        self._runtime_replay_tensors = runtime.overlap_tensors
        aliases = runtime.aliases
        load_aliases = runtime.load_aliases

        # Native materialization enters each layer's recorded restore device.
        # PWAL also creates host tensors, so it must keep the ambient device.
        with set_current_vllm_config(self._vllm_config):
            initialize_layerwise_reload(self._model)
            self._restore_parameter_aliases(load_aliases)
            _reserve_runtime_buffer_slots(self._model, LAYERWISE_INFO)
            self._in_reload = True
            try:
                load(load_aliases)
            finally:
                self._in_reload = False
                self._runtime_replay_tensors = {}
                self._runtime_replay_groups = ()
            finalize_layerwise_reload(self._model, self._model_config)
            for path, reference in owners:
                try:
                    module = self._model.get_submodule(path)
                except AttributeError as error:
                    raise IncompleteRefit(
                        f"reload owner {path!r} disappeared during refit"
                    ) from error
                if module is not reference():
                    raise IncompleteRefit(
                        f"reload owner {path!r} was replaced during refit"
                    )
            self._restore_parameter_aliases(aliases)
            self._validate_alias_owners(aliases)

            # Preserve storage referenced by existing graph consumers.
            for path, module, attributes in bare_tensors:
                if self._model.get_submodule(path) is not module:
                    raise IncompleteRefit(
                        f"graph-bound tensor owner {path!r} was replaced"
                    )
                for name, (graph_tensor, original_geometry, persistent) in attributes.items():
                    current = getattr(module, name, None)
                    if (
                        not isinstance(current, torch.Tensor)
                        or _tensor_geometry(graph_tensor) != original_geometry
                        or current.shape != graph_tensor.shape
                        or current.dtype != graph_tensor.dtype
                        or current.device != graph_tensor.device
                    ):
                        raise IncompleteRefit(
                            f"graph-bound tensor {path}.{name} disappeared or changed geometry"
                        )
                    if current is not graph_tensor:
                        graph_tensor.copy_(current)
                    setattr(module, name, graph_tensor)
                    if persistent is not None:
                        if persistent:
                            module._non_persistent_buffers_set.discard(name)
                        else:
                            module._non_persistent_buffers_set.add(name)
            for path, reference, layout in runtime_layouts.values():
                _validate_runtime_tensor_layout(path, reference(), layout)

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
