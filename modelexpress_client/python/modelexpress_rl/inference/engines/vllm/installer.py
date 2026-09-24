# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM load-layout capture and graph-safe weight installation.

Capture records where each published source lands in vLLM's load-time layout,
tracing the live model with its params reverted to engine load-time skeletons via
layerwise reload. Installation uses vLLM's layerwise reload and post-load
processing to update the live model while preserving storage already referenced
by compiled CUDA graphs.
"""

from __future__ import annotations

import copy
import hashlib
import logging
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from functools import cached_property
from inspect import getattr_static
from contextlib import closing
from pathlib import Path
from types import GetSetDescriptorType
from typing import TYPE_CHECKING

import torch
from modelexpress import envs
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

from modelexpress_rl.inference.engines.vllm._capture_snapshot import _CaptureSnapshot
from modelexpress_rl.inference.plan import (
    EngineCapabilities,
    EngineInstaller,
    PreparedArtifact,
    PreparedCheckpointArtifact,
    PreparedEngineTensors,
    PreparedRuntimeTensors,
    PreparedStreamingTensors,
)
from modelexpress_rl.inference.receiver import PreparedCheckpoint

if TYPE_CHECKING:
    from modelexpress_rl.inference.streaming_checkpoint import StreamedCheckpoint
    from modelexpress.refit.reshard.types import CaptureResult
    from torch.nn import Module
    from vllm.config import ModelConfig, VllmConfig

logger = logging.getLogger("modelexpress_rl.inference.engines.vllm.installer")
_MODULE_GETATTRIBUTE = torch.nn.Module.__getattribute__
_MODULE_GETATTR = torch.nn.Module.__getattr__
_MODULE_GET_SUBMODULE = torch.nn.Module.get_submodule
_MODULE_SETATTR = torch.nn.Module.__setattr__
_MODULE_REGISTER_PARAMETER = torch.nn.Module.register_parameter
_MODULE_GETATTR_CODE = _MODULE_GETATTR.__code__
_MODULE_GET_SUBMODULE_CODE = _MODULE_GET_SUBMODULE.__code__


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
    structure: tuple | None = None
    root_type: type | None = None
    _owner_plans: dict = field(default_factory=dict, compare=False, repr=False)

    @cached_property
    def by_owner(self):
        """Share each complete tie group with every layer that can change it."""
        if self.edges is None or self.structure is None:
            return None
        modules = {"": self.root}
        for edge in self.edges:
            modules[edge.path] = edge.child
        owners = iter(self.owners)
        named_groups, groups_by_owner = [], {}
        for group in self.groups:
            named = tuple((next(owners)[0], module, leaf) for module, leaf in group)
            index = len(named_groups)
            named_groups.append(named)
            for module in dict.fromkeys(module for module, _ in group):
                groups_by_owner.setdefault(module, []).append(index)
        structures = (
            # Custom owner hashing can change classes between dictionary writes.
            _alias_structure(named_groups)
            if _ordinary_owner_hashes(self.writers)
            else None
        )
        result = {}
        for module, indices in groups_by_owner.items():
            groups = [named_groups[index] for index in indices]
            structure = (
                tuple(structures[index] for index in indices)
                if structures is not None
                else _alias_structure(groups)
            )
            plan = self._owner_plans.get(structure) if structure is not None else None
            if plan is None:
                plan = _compile_parameter_aliases(self.root, groups, modules)
                if structure is not None:
                    self._owner_plans[structure] = plan
            result[module] = plan
        return result


def _ordinary_owner_hashes(writers):
    checked = set()
    for module, _captured_class in writers:
        cls = type(module)
        if type(cls) is not type:
            return False
        if cls not in checked:
            if (
                getattr_static(cls, "__hash__") is not object.__hash__
                or getattr_static(cls, "__eq__") is not object.__eq__
            ):
                return False
            checked.add(cls)
    return True


def _alias_structure(groups):
    result = []
    inheritance = {}
    for group in groups:
        if len(group) < 2:
            continue
        entries = []
        for path, module, leaf in group:
            if (
                type(path) is not str
                or type(leaf) is not str
                or type(type(module)) is not type
            ):
                return None
            cls = type(module)
            bases = cls.__mro__
            cached = inheritance.get(cls)
            if cached is None or cached[0] is not bases:
                cached = (bases, tuple(map(id, bases)))
                inheritance[cls] = cached
            entries.append((path, id(module), leaf, cached[1]))
        result.append(tuple(entries))
    return tuple(result)


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
        _alias_structure(groups),
        type(model),
    )


def _select_parameter_aliases(model, groups, modules, initial_aliases):
    groups = tuple(groups)
    structure = _alias_structure(groups)
    same_owners = (
        initial_aliases is not None
        and initial_aliases.root is model
        and initial_aliases.root_type is type(model)
        and initial_aliases.structure is not None
        and initial_aliases.edges is not None
        and modules.get("") is model
        and all(
            modules.get(edge.path) is edge.child
            and modules.get(edge.path.rpartition(".")[0]) is edge.parent
            for edge in initial_aliases.edges
        )
    )
    if same_owners and structure == initial_aliases.structure:
        return initial_aliases
    # Only reuse the initial reload's owners. A later plan must die with its
    # batch so it cannot extend the lifetime of a module replaced by a hook.
    aliases = _compile_parameter_aliases(model, groups, modules)
    if (
        same_owners
        and structure is not None
        and sorted(structure) == sorted(initial_aliases.structure)
    ):
        # Reload can reorder parameter groups as it replaces meta tensors.
        # Keep fresh callback order, but share views of the initial owners.
        # These views cannot retain a module introduced by a later callback.
        aliases = replace(aliases, _owner_plans=initial_aliases._owner_plans)
    return aliases


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


def _dictionary_descriptor(cls):
    for base in cls.__mro__:
        if "__dict__" in base.__dict__:
            return base.__dict__["__dict__"]
    return None


_TENSOR_DICTIONARY = _dictionary_descriptor(torch.nn.Parameter)


def _ordinary_module_dictionary(cls):
    # A mixin before nn.Module can supply the ordinary instance dictionary.
    descriptor = _dictionary_descriptor(cls)
    return (
        type(descriptor) is GetSetDescriptorType
        and descriptor.__name__ == "__dict__"
        and descriptor.__objclass__ in cls.__mro__
    )


def _ordinary_alias_lookups(aliases: _ParameterAliases) -> bool:
    if (
        aliases.edges is None
        or _MODULE_GETATTR.__code__ is not _MODULE_GETATTR_CODE
        or _MODULE_GET_SUBMODULE.__code__ is not _MODULE_GET_SUBMODULE_CODE
        or type(aliases.root).get_submodule is not _MODULE_GET_SUBMODULE
        or not _ordinary_module_dictionary(type(aliases.root))
        or "get_submodule" in object.__getattribute__(aliases.root, "__dict__")
    ):
        return False
    checked_classes, checked_attributes = set(), set()
    for edge in aliases.edges:
        cls = type(edge.parent)
        if not _ordinary_module_dictionary(cls):
            return False
        attrs = object.__getattribute__(edge.parent, "__dict__")
        if type(attrs) is not dict or not _standard_module_access(
            edge.parent, edge.name, checked_classes, checked_attributes
        ):
            return False
    return True


_MATERIALIZE_MODULE_METHODS = tuple(
    (name, getattr(torch.nn.Module, name), getattr(torch.nn.Module, name).__code__)
    for name in ("__getattr__", "__setattr__", "register_parameter", "register_buffer")
)
_MATERIALIZE_TENSOR_ATTRIBUTES = tuple(
    (name, getattr(torch.nn.Parameter, name))
    for name in (
        "__getattribute__",
        "__torch_function__",
        "__torch_dispatch__",
        "is_meta",
        "size",
        "stride",
        "dtype",
    )
)
_TENSOR_FUNCTION = torch.Tensor.__torch_function__.__func__
_TENSOR_FUNCTION_CODE = _TENSOR_FUNCTION.__code__


def _native_parameter_dispatch():
    try:
        from vllm.model_executor.parameter import BasevLLMParameter
    except ImportError:
        return None
    # vLLM's stock wrapper only forwards tensor operations to Parameter.
    # Preserve that dispatch; later overrides or inheritance changes use full checks.
    function = getattr(BasevLLMParameter.__torch_function__, "__func__", None)
    if function is None or not hasattr(function, "__code__"):
        return None
    return (
        BasevLLMParameter,
        function,
        function.__code__,
        BasevLLMParameter.__mro__,
    )


def _standard_tensor_function(tensor_type, expected) -> bool:
    current = getattr(tensor_type, "__torch_function__", None)
    if tensor_type is torch.Tensor:
        return (
            getattr(current, "__func__", None) is _TENSOR_FUNCTION
            and _TENSOR_FUNCTION.__code__ is _TENSOR_FUNCTION_CODE
        )
    if expected is None or type(tensor_type) is not type(torch.nn.Parameter):
        return False
    base, function, code, bases = expected
    if (
        getattr(current, "__func__", None) is not function
        or getattr(current, "__self__", None) is not tensor_type
        or function.__code__ is not code
    ):
        return False
    mro = tensor_type.__mro__
    for index, cls in enumerate(mro):
        if cls is base:
            return len(mro[index:]) == len(bases) and all(
                actual is original for actual, original in zip(mro[index:], bases)
            )
    return False


def _materialization_is_local(layer, info, native_dispatch=None) -> bool:
    """Admit layer-local vLLM allocation only without user dispatch or hooks."""
    from torch.nn.modules import module as module_api
    from torch.overrides import _get_current_function_mode_stack
    from torch.utils._device import DeviceContext
    from torch.utils._python_dispatch import _get_current_dispatch_mode

    hooks = (
        module_api._global_parameter_registration_hooks,
        module_api._global_buffer_registration_hooks,
    )
    if (
        type(getattr(info, "restore_device", None)) is not torch.device
        or any(type(registry) not in (dict, OrderedDict) for registry in hooks)
        or any(hooks)
        or _get_current_dispatch_mode() is not None
        or any(
            type(mode) is not DeviceContext
            for mode in _get_current_function_mode_stack()
        )
    ):
        return False
    cls = type(layer)
    if (
        type(cls) is not type
        or cls.__getattribute__ is not _MODULE_GETATTRIBUTE
        or not _ordinary_module_dictionary(cls)
        or any(
            _has_descriptor(cls, name)
            for name in ("_parameters", "_buffers", "_modules")
        )
    ):
        return False
    attrs = object.__getattribute__(layer, "__dict__")
    if type(attrs) is not dict or not _standard_lookup_containers(attrs):
        return False
    if any(
        name in attrs or getattr(cls, name) is not method or method.__code__ is not code
        for name, method, code in _MATERIALIZE_MODULE_METHODS
    ):
        return False
    for name, tensor in (*attrs["_parameters"].items(), *attrs["_buffers"].items()):
        if type(name) is not str or _has_descriptor(cls, name):
            return False
        if tensor is None:
            continue
        tensor_type = type(tensor)
        if (
            type(tensor_type).__getattribute__ is not type.__getattribute__
            or _dictionary_descriptor(tensor_type) is not _TENSOR_DICTIONARY
        ):
            return False
        tensor_attrs = object.__getattribute__(tensor, "__dict__")
        if type(tensor_attrs) is not dict:
            return False
        for key, value in _MATERIALIZE_TENSOR_ATTRIBUTES:
            if key in tensor_attrs:
                return False
            if getattr(tensor_type, key, None) is value:
                continue
            if key != "__torch_function__" or not _standard_tensor_function(
                tensor_type, native_dispatch
            ):
                return False
    return True


def _copy_received(target: torch.Tensor, source: torch.Tensor) -> None:
    """Keep receive tensors out of parameter-specific copy dispatch."""
    from torch.overrides import _get_current_function_mode_stack
    from torch.utils._device import DeviceContext
    from torch.utils._python_dispatch import _get_current_dispatch_mode

    if (
        any(
            type(mode) is not DeviceContext
            for mode in _get_current_function_mode_stack()
        )
        or _get_current_dispatch_mode() is not None
    ):
        raise IncompleteRefit(
            "weight installation does not support active tensor dispatch modes"
        )
    destination = target.data
    if type(destination) is not torch.Tensor or type(source) is not torch.Tensor:
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
    torch.Tensor.copy_(destination, source)


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
        self._capture_cache = None
        self._native_parameter_dispatch = _native_parameter_dispatch()

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
                    PreparedRuntimeTensors,
                    PreparedCheckpointArtifact,
                }
                | ({PreparedStreamingTensors} if not self._is_quantized else set())
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
            if checkpoint.streaming is not None:
                self.install_streamed_checkpoint(checkpoint.streaming)
                metrics.update(checkpoint.streaming.cache_metrics)
            else:
                self.install_checkpoint(checkpoint.path)
        else:
            raise TypeError(f"unsupported prepared artifact {type(prepared).__name__}")
        if not isinstance(prepared, PreparedStreamingTensors):
            metrics["perf/mx_receive_install_time"] = time.perf_counter() - started
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
        if not self._is_quantized and self._capture_cache is not None:
            key, result = self._capture_cache
            current_key = self._capture_key(manifest)
            if key == current_key:
                copies = (
                    result.indices
                    if type(result) is _CaptureSnapshot
                    else result[0].copies
                )
                logger.info("reusing cached vLLM load layout (%d copies)", len(copies))
                if type(result) is _CaptureSnapshot:
                    return result.clone()
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
        aliases = self._load_time_parameter_aliases(
            self._parameter_aliases(model), LAYERWISE_INFO
        )
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
                _CaptureSnapshot.create(copy.deepcopy((capture, param_layout))),
            )
        return capture, param_layout

    def install_tensors(self, tensors: dict[str, torch.Tensor]) -> None:
        """Install verified load-layout tensors without changing graph addresses."""
        self._process_and_commit(tensors)
        # Synchronization is paid once per install, separately from per-layer work.
        with refit_span("post_install"):
            torch.cuda.synchronize(self._device)

    def _drain_streaming(self) -> None:
        torch.cuda.synchronize(self._device)

    @torch.no_grad()
    def install_streaming(self, prepared: PreparedStreamingTensors) -> None:
        """Copy complete groups into engine-owned storage before arena reuse."""
        if self._is_quantized:
            raise IncompleteRefit(
                "bounded streaming currently requires an unquantized engine"
            )
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
            expected = set(dict(self._model.named_parameters()))
            if expected != prepared.parameter_names:
                raise IncompleteRefit(
                    "streaming parameter coverage differs from the live load layout"
                )
            installed = set()
            installed_parameters: dict[str, torch.Tensor] = {}
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
                names = set(tensors)
                if not names or names - expected or names & installed:
                    raise IncompleteRefit(
                        "invalid or repeated streaming parameter batch"
                    )
                commit_started = time.perf_counter()
                received_storages = {
                    tensor.untyped_storage().data_ptr() for tensor in tensors.values()
                }
                if (received_storages & live_storages) - {0}:
                    raise IncompleteRefit(
                        "received streaming tensor aliases live storage"
                    )
                self._process_and_commit(
                    tensors,
                    reload=False,
                    installed_parameters=installed_parameters,
                    initial_aliases=initial_aliases,
                )
                try:
                    installed_parameters.update(
                        (name, self._model.get_parameter(name)) for name in names
                    )
                except AttributeError as error:
                    raise IncompleteRefit(
                        "installed canonical parameter disappeared"
                    ) from error
                installed.update(names)
                drain()
                commit_s += time.perf_counter() - commit_started
                del tensors
            if installed != expected:
                raise IncompleteRefit(
                    "streaming transfer ended before every parameter was installed"
                )
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
        from vllm.model_executor.model_loader.weight_utils import (
            runai_safetensors_weights_iterator,
        )

        tp_size = getattr(
            getattr(self._vllm_config, "parallel_config", None),
            "tensor_parallel_size",
            1,
        )
        weights = runai_safetensors_weights_iterator(
            list(checkpoint.shard_uris),
            self._vllm_config.load_config.use_tqdm_on_load,
            is_distributed=tp_size > 1 and envs.MX_MS_DISTRIBUTED,
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

    @torch.no_grad()
    def _process_and_commit(
        self,
        tensors: dict[str, torch.Tensor],
        *,
        reload: bool = True,
        installed_parameters: dict[str, torch.Tensor] | None = None,
        initial_aliases: _ParameterAliases | None = None,
    ) -> None:
        """Run vLLM's per-layer post-load processing into graph-bound storage.

        vLLM owns the materialized destinations and post-load processing. Receive
        arena tensors are copied into those destinations and never attached to
        a module or passed to a post-load callback.
        """
        try:
            from vllm.model_executor.layers.attention import is_deferred_attention_layer
            from vllm.model_executor.layers.quantization.base_config import (
                QuantizeMethodBase,
            )
            from vllm.model_executor.model_loader.reload.layerwise import (
                LAYERWISE_INFO,
                _copy_and_restore_kernel_tensors,
                _layerwise_process,
            )
            from vllm.model_executor.model_loader.reload.meta import materialize_layer
        except (ImportError, AttributeError) as error:
            raise RuntimeError(
                "ModelExpress refit requires vLLM's layerwise reload APIs"
            ) from error

        def load(_reload_aliases: _ParameterAliases | None = None) -> None:
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
            module_paths = {}
            for module_name, module in self._model.named_modules(
                remove_duplicate=False
            ):
                module_paths[module_name] = module
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
                # Each included owner needs complete canonical coverage. Check
                # the live tree before hooks run: earlier batches may have added
                # parameters since the layout was captured.
                if owned & tensors.keys() and missing:
                    kind = "staged" if reload else "streaming"
                    raise IncompleteRefit(
                        f"{kind} batch splits an owning module {module_name!r}; "
                        f"missing canonical parameters={sorted(missing)}"
                    )
            unmatched = sorted(set(tensors) - matched)
            if unmatched:
                raise IncompleteRefit(
                    "vLLM layerwise reload did not expose every staged parameter; "
                    f"unmatched={unmatched[:10]}"
                )
            aliases = _select_parameter_aliases(
                self._model, alias_groups.values(), module_paths, initial_aliases
            )

            self._validate_alias_owners(aliases)
            owner_aliases = (
                aliases.by_owner
                if _ordinary_alias_lookups(aliases)
                and _standard_parameter_writes(aliases)
                else None
            )
            # Subspans measure host calls; the batch fence completes GPU work.
            for layer, parameters in groups.items():
                local_aliases = (
                    aliases if owner_aliases is None else owner_aliases.get(layer)
                )

                # A packed batch resolves several owning modules before any of
                # their hooks run, and an earlier hook may replace a later
                # owner. Committing into the detached module would leave the
                # live one without the published bytes.
                if (
                    not reload
                    and self._model.get_submodule(group_paths[layer]) is not layer
                ):
                    raise IncompleteRefit(
                        f"a post-load hook replaced module {group_paths[layer]!r} "
                        "before its parameters were committed"
                    )
                info = LAYERWISE_INFO.get(layer)
                managed = info is not None and info.kernel_tensors is not None
                if info is not None:
                    local_materialization = (
                        owner_aliases is not None
                        and _materialization_is_local(
                            layer, info, self._native_parameter_dispatch
                        )
                    )
                    with refit_span(
                        "installation",
                        duration_key="materialization_s",
                        accumulate_metadata=True,
                    ):
                        materialize_layer(layer, info)
                    if not local_materialization:
                        self._restore_parameter_aliases(aliases)
                    elif local_aliases is not None:
                        self._restore_parameter_aliases(local_aliases)
                with refit_span(
                    "installation",
                    duration_key="receive_copy_s",
                    accumulate_metadata=True,
                ):
                    for full_name, leaf in parameters:
                        target = getattr(layer, leaf)
                        source = tensors[full_name]
                        if (
                            target.device.type == "meta"
                            or target.shape != source.shape
                            or target.dtype != source.dtype
                        ):
                            raise IncompleteRefit(
                                "streaming parameter has no compatible live storage"
                            )
                        _copy_received(target, source)
                if not reload and not managed:
                    continue
                deferred = is_deferred_attention_layer(layer)
                if managed and not deferred:
                    if getattr(info, "loaded_weights", ()):
                        raise IncompleteRefit(
                            "engine reload has pending checkpoint-loader calls"
                        )
                    with refit_span(
                        "installation",
                        duration_key="post_load_processing_s",
                        accumulate_metadata=True,
                    ):
                        _layerwise_process(layer, info)
                else:
                    quant_method = getattr(layer, "quant_method", None)
                    if isinstance(quant_method, QuantizeMethodBase):
                        if hasattr(
                            layer, "_already_called_process_weights_after_loading"
                        ):
                            delattr(
                                layer, "_already_called_process_weights_after_loading"
                            )
                        with refit_span("transformation"):
                            quant_method.process_weights_after_loading(layer)
                            update_tp = getattr(layer, "update_param_tp_status", None)
                            if update_tp is not None:
                                update_tp()
                    if managed:
                        with refit_span("installation"):
                            _copy_and_restore_kernel_tensors(layer, info)
                    if info is not None and not deferred:
                        info.reset()
                self._restore_parameter_aliases(aliases)

        if reload:
            self._reload(load)
        else:
            load()

    @torch.no_grad()
    def _reload(self, load: Callable[[_ParameterAliases], None]) -> None:
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

        # Layerwise reload does not save bare device tensors referenced by
        # captured graphs. Host bookkeeping follows the native reload lifecycle.
        def geometry(tensor):
            return (
                tuple(tensor.shape),
                tuple(tensor.stride()),
                tensor.dtype,
                tensor.device,
                tensor.data_ptr(),
            )

        bare_tensors = [
            (
                path,
                module,
                {
                    name: (value, geometry(value))
                    for name, value in module.__dict__.items()
                    if isinstance(value, torch.Tensor)
                    and value.device.type == self._device.type
                    and (
                        self._device.index is None
                        or value.device.index == self._device.index
                    )
                },
            )
            for path, module in self._model.named_modules()
        ]
        bare_tensors = [entry for entry in bare_tensors if entry[2]]
        aliases = self._parameter_aliases(self._model)
        load_aliases = self._load_time_parameter_aliases(aliases, LAYERWISE_INFO)

        # Native materialization enters each layer's recorded restore device.
        # PWAL also creates host tensors, so it must keep the ambient device.
        with set_current_vllm_config(self._vllm_config):
            initialize_layerwise_reload(self._model)
            self._restore_parameter_aliases(load_aliases)
            _reserve_runtime_buffer_slots(self._model, LAYERWISE_INFO)
            load(load_aliases)
            finalize_layerwise_reload(self._model, self._model_config)
            self._validate_alias_owners(aliases)

            # Preserve storage referenced by existing graph consumers.
            for path, module, attributes in bare_tensors:
                if self._model.get_submodule(path) is not module:
                    raise IncompleteRefit(
                        f"graph-bound tensor owner {path!r} was replaced"
                    )
                for name, (graph_tensor, original_geometry) in attributes.items():
                    current = module.__dict__.get(name)
                    if (
                        not isinstance(current, torch.Tensor)
                        or geometry(graph_tensor) != original_geometry
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
