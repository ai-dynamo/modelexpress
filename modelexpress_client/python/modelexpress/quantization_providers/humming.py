# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Humming quantization RDMA manifest provider."""

from __future__ import annotations

import importlib
import logging
from contextlib import contextmanager

import torch
import torch.nn as nn

from ..types import ManifestMismatchError
from .base import (
    MANIFEST_TENSOR_OVERRIDES_ATTR,
    NO_STRUCTURAL_REPLACE_POLICY,
    REJECT_IF_MISMATCH_POLICY,
    ManifestTensorDecision,
    QuantizationManifestProvider,
    tensor_data,
)

logger = logging.getLogger("modelexpress.quantization_providers.humming")

HUMMING_RUNTIME_TENSORS_ATTR = "_mx_rdma_tensors"
HUMMING_RUNTIME_ROLE = "humming_packed_weight"
HUMMING_DENSE_RUNTIME_ROLE = "humming_dense_weight"


class HummingManifestProvider(QuantizationManifestProvider):
    name = "humming"

    def enabled(self, quantization: str) -> bool:
        return str(quantization or "").lower() == self.name

    def capture_during_load(self, *, enabled: bool = True):
        return capture_runtime_tensors(enabled=enabled)

    def capture_from_model(self, model: nn.Module) -> int:
        return capture_runtime_tensors_from_model(model)

    def resolve_manifest_tensor(
        self,
        module: nn.Module,
        leaf: str,
        tensor: torch.Tensor,
        *,
        quantization: str = "",
    ) -> ManifestTensorDecision | None:
        del quantization
        quant_method = _quant_method_name(module)
        manifest_overrides = getattr(module, MANIFEST_TENSOR_OVERRIDES_ATTR, None)
        if isinstance(manifest_overrides, dict):
            override = manifest_overrides.get(leaf)
            if isinstance(override, tuple) and len(override) == 2:
                return ManifestTensorDecision(
                    tensor=tensor,
                    runtime_role=str(override[0] or leaf),
                    replace_policy=str(override[1] or "structural_replace"),
                )

        is_humming = (
            _is_humming_quant_method(quant_method)
            or isinstance(getattr(module, HUMMING_RUNTIME_TENSORS_ATTR, None), dict)
            or isinstance(getattr(module, "humming_metas", None), dict)
        )
        if leaf != "weight" or not is_humming:
            return None

        selected = tensor
        packed = _humming_runtime_tensor(module, leaf)
        if packed is not None and _is_humming_packed_weight(packed):
            selected = packed
        if _is_humming_packed_weight(selected):
            return ManifestTensorDecision(
                tensor=selected,
                runtime_role=HUMMING_RUNTIME_ROLE,
                replace_policy=NO_STRUCTURAL_REPLACE_POLICY,
            )

        logger.warning(
            "Humming weight %s.%s has no captured packed runtime tensor; "
            "publishing dense tensor with reject_if_mismatch policy "
            "(shape=%s dtype=%s)",
            type(module).__name__, leaf, list(tensor.shape), tensor.dtype,
        )
        return ManifestTensorDecision(
            tensor=selected,
            runtime_role=HUMMING_DENSE_RUNTIME_ROLE,
            replace_policy=REJECT_IF_MISMATCH_POLICY,
        )

    def skip_manifest_tensor(self, name: str, leaf: str, tensor_type: str) -> bool:
        del name, tensor_type
        return leaf == "locks"

    def align_target_module_from_source(
        self,
        module: nn.Module,
        leaf: str,
        desc,
    ) -> None:
        if leaf != "weight":
            return

        source_quant_method = str(getattr(desc, "quant_method", "") or "")
        if not source_quant_method.endswith(".UnquantizedLinearMethod"):
            return

        local_quant_method = getattr(module, "quant_method", None)
        local_quant_method_name = (
            f"{type(local_quant_method).__module__}."
            f"{type(local_quant_method).__qualname__}"
            if local_quant_method is not None else ""
        )
        if "humming" not in local_quant_method_name.lower():
            return

        try:
            from vllm.model_executor.layers.linear import UnquantizedLinearMethod
        except Exception as e:
            raise ManifestMismatchError(
                f"Cannot align RDMA target module {type(module).__name__}: "
                "source manifest requires UnquantizedLinearMethod, but vLLM "
                f"UnquantizedLinearMethod could not be imported: {e}"
            ) from e

        module.quant_method = UnquantizedLinearMethod()
        logger.debug(
            "Source-manifest target prepare: changed %s quant_method from %s "
            "to UnquantizedLinearMethod for source tensor %r",
            type(module).__name__,
            local_quant_method_name,
            getattr(desc, "name", ""),
        )

    def after_target_tensor_rebuilt(
        self,
        module: nn.Module,
        leaf: str,
        desc,
    ) -> None:
        del desc
        runtime_tensors = getattr(module, HUMMING_RUNTIME_TENSORS_ATTR, None)
        if isinstance(runtime_tensors, dict):
            runtime_tensors.pop(leaf, None)


def _quant_method_name(module: nn.Module) -> str:
    quant_method = getattr(module, "quant_method", None)
    if quant_method is None:
        return ""
    cls = type(quant_method)
    return f"{cls.__module__}.{cls.__qualname__}"


def _is_humming_quant_method(quant_method: str) -> bool:
    return "humming" in quant_method.lower()


def _is_humming_packed_weight(tensor: torch.Tensor) -> bool:
    if not isinstance(tensor, torch.Tensor):
        return False
    if tensor.numel() == 0:
        return False
    return tensor.dtype in (torch.int32, torch.int16, torch.int8, torch.uint8)


def _humming_runtime_tensor(module: nn.Module, leaf: str) -> torch.Tensor | None:
    runtime_tensors = getattr(module, HUMMING_RUNTIME_TENSORS_ATTR, None)
    if not isinstance(runtime_tensors, dict):
        return None
    tensor = tensor_data(runtime_tensors.get(leaf))
    if tensor is not None:
        return tensor
    if leaf == "weight" and len(runtime_tensors) == 1:
        return tensor_data(next(iter(runtime_tensors.values())))
    return None


def _record_runtime_tensor(
    layer: object,
    name: object,
    *,
    require_packed: bool = False,
) -> bool:
    if not isinstance(layer, nn.Module) or not name:
        return False
    runtime = getattr(layer, str(name), None)
    tensor = tensor_data(runtime)
    if tensor is None:
        return False
    if require_packed and not _is_humming_packed_weight(tensor):
        return False
    runtime_tensors = getattr(layer, HUMMING_RUNTIME_TENSORS_ATTR, None)
    if not isinstance(runtime_tensors, dict):
        runtime_tensors = {}
        setattr(layer, HUMMING_RUNTIME_TENSORS_ATTR, runtime_tensors)
    runtime_tensors[str(name)] = tensor
    tensor.mx_runtime_role = HUMMING_RUNTIME_ROLE
    tensor.mx_replace_policy = NO_STRUCTURAL_REPLACE_POLICY
    return True


def _record_runtime_tensor_value(
    layer: object,
    name: object,
    tensor: object,
    *,
    require_packed: bool = False,
) -> bool:
    if not isinstance(layer, nn.Module) or not name:
        return False
    tensor_data_value = tensor_data(tensor)
    if tensor_data_value is None:
        return False
    if require_packed and not _is_humming_packed_weight(tensor_data_value):
        return False
    runtime_tensors = getattr(layer, HUMMING_RUNTIME_TENSORS_ATTR, None)
    if not isinstance(runtime_tensors, dict):
        runtime_tensors = {}
        setattr(layer, HUMMING_RUNTIME_TENSORS_ATTR, runtime_tensors)
    runtime_tensors[str(name)] = tensor_data_value
    tensor_data_value.mx_runtime_role = HUMMING_RUNTIME_ROLE
    tensor_data_value.mx_replace_policy = NO_STRUCTURAL_REPLACE_POLICY
    return True


def _record_meta_runtime_tensors(layer: object) -> int:
    if not isinstance(layer, nn.Module):
        return 0
    metas = getattr(layer, "humming_metas", None)
    if not isinstance(metas, dict):
        return 0

    recorded = 0
    for sublayer_name, meta in metas.items():
        weight_name = getattr(meta, "weight_name", None)
        if not weight_name:
            continue
        if not _record_runtime_tensor(layer, weight_name, require_packed=True):
            logger.debug(
                "Humming module %s meta %r points to %r, but no packed "
                "runtime tensor is attached there",
                type(layer).__name__, sublayer_name, weight_name,
            )
            continue
        recorded += 1

    if recorded == 1:
        runtime_tensors = getattr(layer, HUMMING_RUNTIME_TENSORS_ATTR, None)
        if isinstance(runtime_tensors, dict) and "weight" not in runtime_tensors:
            for tensor in runtime_tensors.values():
                runtime_tensors["weight"] = tensor
                break
    return recorded


def _record_postload_runtime_tensors(layer: object) -> int:
    if not isinstance(layer, nn.Module):
        return 0
    recorded = _record_meta_runtime_tensors(layer)
    if recorded:
        return recorded
    recorded = 0
    if _record_runtime_tensor(layer, "weight", require_packed=True):
        recorded += 1
    elif _record_unique_packed_weight(layer):
        recorded += 1
    return recorded


def _iter_direct_module_tensors(module: nn.Module) -> list[tuple[str, torch.Tensor]]:
    tensors: list[tuple[str, torch.Tensor]] = []
    for collection_name, collection in (
        ("parameters", getattr(module, "_parameters", {})),
        ("buffers", getattr(module, "_buffers", {})),
    ):
        for name, value in collection.items():
            tensor = tensor_data(value)
            if tensor is not None:
                tensors.append((f"{collection_name}.{name}", tensor))
    for name, value in vars(module).items():
        if name.startswith("_"):
            continue
        tensor = tensor_data(value)
        if tensor is not None:
            tensors.append((f"attr.{name}", tensor))
    return tensors


def _is_packed_weight_candidate_name(name: str) -> bool:
    leaf = name.rsplit(".", 1)[-1]
    if leaf in {"locks", "weight_scale", "zero_point", "global_scale"}:
        return False
    if leaf.endswith("_scale") or leaf.endswith("_zero_point"):
        return False
    return leaf == "weight" or leaf.endswith("_weight")


def _record_unique_packed_weight(module: nn.Module) -> bool:
    current_weight = tensor_data(getattr(module, "weight", None))
    current_ptr = current_weight.data_ptr() if current_weight is not None else 0
    candidates: list[tuple[str, torch.Tensor]] = []
    seen_ptrs: set[int] = set()
    for name, tensor in _iter_direct_module_tensors(module):
        if not _is_packed_weight_candidate_name(name):
            continue
        if not _is_humming_packed_weight(tensor):
            continue
        if tensor.data_ptr() == current_ptr:
            continue
        if tensor.data_ptr() in seen_ptrs:
            continue
        seen_ptrs.add(tensor.data_ptr())
        candidates.append((name, tensor))

    if len(candidates) != 1:
        if candidates:
            logger.debug(
                "Humming module %s has ambiguous packed weight candidates: %s",
                type(module).__name__,
                ", ".join(f"{name}:{list(t.shape)}:{t.dtype}" for name, t in candidates),
            )
        return False

    name, tensor = candidates[0]
    runtime_tensors = getattr(module, HUMMING_RUNTIME_TENSORS_ATTR, None)
    if not isinstance(runtime_tensors, dict):
        runtime_tensors = {}
        setattr(module, HUMMING_RUNTIME_TENSORS_ATTR, runtime_tensors)
    runtime_tensors["weight"] = tensor
    tensor.mx_runtime_role = HUMMING_RUNTIME_ROLE
    tensor.mx_replace_policy = NO_STRUCTURAL_REPLACE_POLICY
    logger.debug(
        "Captured Humming packed runtime weight from %s.%s "
        "(shape=%s dtype=%s)",
        type(module).__name__, name, list(tensor.shape), tensor.dtype,
    )
    return True


def capture_runtime_tensors_from_model(model: nn.Module) -> int:
    recorded = 0
    for module in model.modules():
        if _record_postload_runtime_tensors(module):
            recorded += 1
    if recorded:
        logger.info("Captured %d Humming runtime packed weights from model", recorded)
    return recorded


@contextmanager
def capture_runtime_tensors(enabled: bool = True):
    if not enabled:
        yield
        return

    restore_callbacks = []

    def restore_attr(cls: type, name: str, raw: object) -> None:
        if raw is not None:
            setattr(cls, name, raw)
        else:
            delattr(cls, name)

    def call_layer_name_tensor(
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> tuple[object, object, object]:
        layer = args[0] if len(args) >= 1 else kwargs.get("layer")
        name = args[1] if len(args) >= 2 else kwargs.get("name")
        tensor = args[2] if len(args) >= 3 else kwargs.get("tensor")
        return layer, name, tensor

    try:
        humming_layer = importlib.import_module("humming.layer")
        HummingLayerMethod = getattr(humming_layer, "HummingLayerMethod", None)
        raw = (
            HummingLayerMethod.__dict__.get("may_set_param")
            if HummingLayerMethod is not None else None
        )
        original = (
            getattr(HummingLayerMethod, "may_set_param", None)
            if HummingLayerMethod is not None else None
        )
        if HummingLayerMethod is not None and original is not None:
            if isinstance(raw, classmethod):
                def wrapped_may_set_param(cls, *args, _original=original, **kwargs):
                    result = _original(*args, **kwargs)
                    layer, name, tensor = call_layer_name_tensor(args, kwargs)
                    _record_runtime_tensor_value(layer, name, tensor)
                    return result

                HummingLayerMethod.may_set_param = classmethod(wrapped_may_set_param)
            else:
                descriptor = staticmethod if isinstance(raw, staticmethod) else None

                def wrapped_may_set_param(*args, _original=original, **kwargs):
                    result = _original(*args, **kwargs)
                    layer, name, tensor = call_layer_name_tensor(args, kwargs)
                    _record_runtime_tensor_value(layer, name, tensor)
                    return result

                HummingLayerMethod.may_set_param = (
                    descriptor(wrapped_may_set_param)
                    if descriptor is not None else wrapped_may_set_param
                )
            restore_callbacks.append((
                restore_attr, HummingLayerMethod, "may_set_param", raw,
            ))
    except Exception as e:
        logger.debug("Humming may_set_param capture disabled: %s", e)

    try:
        vllm_humming = importlib.import_module(
            "vllm.model_executor.layers.quantization.humming"
        )
        for class_name in ("HummingLinearMethod", "HummingMoEMethod"):
            cls = getattr(vllm_humming, class_name, None)
            if cls is None:
                continue
            raw = cls.__dict__.get("process_weights_after_loading")
            original = getattr(cls, "process_weights_after_loading", None)
            if original is None:
                continue
            if isinstance(raw, classmethod):
                def wrapped_postload(
                    class_, layer, *args,
                    _original=original, _class_name=class_name, **kwargs,
                ):
                    result = _original(layer, *args, **kwargs)
                    _record_postload_runtime_tensors(layer)
                    return result

                setattr(cls, "process_weights_after_loading", classmethod(wrapped_postload))
            elif isinstance(raw, staticmethod):
                def wrapped_postload(
                    layer, *args,
                    _original=original, _class_name=class_name, **kwargs,
                ):
                    result = _original(layer, *args, **kwargs)
                    _record_postload_runtime_tensors(layer)
                    return result

                setattr(cls, "process_weights_after_loading", staticmethod(wrapped_postload))
            else:
                def wrapped_postload(
                    self, layer, *args,
                    _original=original, _class_name=class_name, **kwargs,
                ):
                    result = _original(self, layer, *args, **kwargs)
                    _record_postload_runtime_tensors(layer)
                    return result

                setattr(cls, "process_weights_after_loading", wrapped_postload)
            restore_callbacks.append((restore_attr, cls, "process_weights_after_loading", raw))
            logger.debug("Installed vLLM Humming post-load capture for %s", class_name)
    except Exception as e:
        logger.warning("vLLM Humming post-load capture disabled: %s", e)

    try:
        yield
    finally:
        for callback, cls, name, raw in reversed(restore_callbacks):
            callback(cls, name, raw)
