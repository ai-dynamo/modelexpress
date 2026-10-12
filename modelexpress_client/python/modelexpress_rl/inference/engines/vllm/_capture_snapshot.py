# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private immutable cache metadata; callers still receive mutable records."""

import copy
import copyreg
from dataclasses import dataclass
from types import FunctionType, GetSetDescriptorType

import torch
from modelexpress_rl.refit.reshard.types import CaptureResult, RecordedCopy

_RECORD_FIELDS = (
    "src_name",
    "op_chain",
    "param_name",
    "dest_offset",
    "dest_shape",
    "dest_stride",
    "dest_dtype",
    "owner_offset",
    "owner_stride",
)
_CAPTURE_FIELDS = (
    "copies",
    "unsupported",
    "unattributed",
    "unsupported_reasons",
    "source_replay",
    "generation",
    "owner_workspaces",
    "owner_dependencies",
    "destination_layouts",
    "generator_fingerprint",
)
_CLASS_DEFAULTS = {
    CaptureResult: {
        "unattributed": 0,
        "generator_fingerprint": None,
        "source_replay": False,
        "generation": None,
        "owner_dependencies": (),
    },
    RecordedCopy: {"owner_offset": None, "owner_stride": None},
}
_CAPTURE_CLASS = CaptureResult
_RECORD_CLASS = RecordedCopy
_COPY_METHODS = (
    "__new__",
    "__getattribute__",
    "__getattr__",
    "__setattr__",
    "__delattr__",
    "__getstate__",
    "__setstate__",
    "__reduce__",
    "__reduce_ex__",
    "__deepcopy__",
    "__del__",
    "__slots__",
)
_DEEPCOPY = copy.deepcopy
_BUILTIN_COPY = (
    (list, copy._deepcopy_list),
    (tuple, copy._deepcopy_tuple),
    (dict, copy._deepcopy_dict),
    *(
        (cls, copy._deepcopy_atomic)
        for cls in (str, int, float, bool, bytes, type(None))
    ),
)
_COPY_FUNCTIONS = (_DEEPCOPY, *(handler for _, handler in _BUILTIN_COPY))
_COPY_BINDINGS = tuple(
    (function, function.__code__, function.__defaults__, function.__kwdefaults__)
    for function in _COPY_FUNCTIONS
    if type(function) is FunctionType
)


def _ordinary_copy_classes() -> bool:
    if (
        CaptureResult is not _CAPTURE_CLASS
        or RecordedCopy is not _RECORD_CLASS
        or copy.deepcopy is not _DEEPCOPY
        or type(copy._deepcopy_dispatch) is not dict
    ):
        return False
    if len(_COPY_BINDINGS) != len(_COPY_FUNCTIONS) or any(
        function.__code__ is not code
        or function.__defaults__ is not defaults
        or function.__kwdefaults__ is not kwdefaults
        for function, code, defaults, kwdefaults in _COPY_BINDINGS
    ):
        return False
    dispatch = copy._deepcopy_dispatch
    if any(type(key) is not type for key in dispatch):
        return False
    if any(dispatch.get(cls) is not handler for cls, handler in _BUILTIN_COPY):
        return False
    if (
        type(copyreg.dispatch_table) is not dict
        or copy.dispatch_table is not copyreg.dispatch_table
        or any(type(key) is not type for key in copyreg.dispatch_table)
    ):
        return False
    if any(key is torch.dtype or key is slice for key in dispatch) or any(
        key is torch.dtype or key is slice for key in copyreg.dispatch_table
    ):
        return False
    for cls, fields in (
        (CaptureResult, _CAPTURE_FIELDS),
        (RecordedCopy, _RECORD_FIELDS),
    ):
        if type(cls) is not type or cls.__bases__ != (object,):
            return False
        namespace = vars(cls)
        if any(name in namespace for name in _COPY_METHODS):
            return False
        descriptor = namespace.get("__dict__")
        if (
            type(descriptor) is not GetSetDescriptorType
            or descriptor.__objclass__ is not cls
        ):
            return False
        if any(
            name in namespace
            and not (
                name in _CLASS_DEFAULTS[cls]
                and namespace[name] is _CLASS_DEFAULTS[cls][name]
            )
            for name in fields
        ):
            return False
        if any(key is cls for key in dispatch) or any(
            key is cls for key in copyreg.dispatch_table
        ):
            return False
    return True


def _immutable(value) -> bool:
    if type(value) in (str, int, float, bool, bytes, type(None), torch.dtype):
        return True
    if type(value) is slice:
        return all(_immutable(item) for item in (value.start, value.stop, value.step))
    return type(value) is tuple and all(_immutable(item) for item in value)


def _needs_copy(value) -> bool:
    return type(value) is slice or (
        type(value) is tuple and any(_needs_copy(item) for item in value)
    )


def _clone_value(value, memo):
    if type(value) is not tuple and type(value) is not slice:
        return value
    identity = id(value)
    if identity in memo:
        return memo[identity]
    if type(value) is slice:
        result = slice(
            *(
                _clone_value(item, memo)
                for item in (value.start, value.stop, value.step)
            )
        )
    else:
        items = tuple(_clone_value(item, memo) for item in value)
        result = value if all(a is b for a, b in zip(value, items)) else items
    memo[identity] = result
    return result


@dataclass(frozen=True)
class _CaptureSnapshot:
    fallback: tuple
    records: tuple
    indices: tuple[int, ...]
    layout: tuple
    metadata: tuple
    owner_workspaces: tuple
    destination_layouts: tuple

    @classmethod
    def create(cls, result: tuple):
        if (
            not _ordinary_copy_classes()
            or type(result) is not tuple
            or len(result) != 2
        ):
            return result
        capture, layout = result
        if type(capture) is not CaptureResult or type(layout) is not dict:
            return result
        state = vars(capture)
        if tuple(state) != _CAPTURE_FIELDS:
            return result
        if (
            type(capture.copies) is not list
            or not capture.copies
            or type(capture.unsupported) is not list
            or capture.unsupported
            or type(capture.unattributed) is not int
            or capture.unattributed
            or type(capture.unsupported_reasons) is not dict
            or capture.unsupported_reasons
            or layout is capture.unsupported_reasons
        ):
            return result
        metadata = (capture.source_replay, capture.generation, capture.owner_dependencies)
        mappings = (
            layout,
            capture.unsupported_reasons,
            capture.owner_workspaces,
            capture.destination_layouts,
        )
        if (
            not all(_immutable(value) for value in metadata)
            or any(type(mapping) is not dict for mapping in mappings)
            or len({id(mapping) for mapping in mappings}) != len(mappings)
            or not all(
                type(name) is str and _immutable(value)
                for mapping in mappings
                for name, value in mapping.items()
            )
        ):
            return result
        records, indices, seen = [], [], {}
        for record in capture.copies:
            if (
                type(record) is not RecordedCopy
                or tuple(vars(record)) != _RECORD_FIELDS
            ):
                return result
            values = tuple(vars(record).values())
            if not all(_immutable(value) for value in values):
                return result
            identity = id(record)
            if identity not in seen:
                seen[identity] = len(records)
                records.append(
                    (
                        values,
                        tuple(
                            i for i, value in enumerate(values) if _needs_copy(value)
                        ),
                    )
                )
            indices.append(seen[identity])
        if not _ordinary_copy_classes():
            return result
        return cls(
            result,
            tuple(records),
            tuple(indices),
            tuple(layout.items()),
            metadata,
            tuple(capture.owner_workspaces.items()),
            tuple(capture.destination_layouts.items()),
        )

    def clone(self):
        if not _ordinary_copy_classes():
            return copy.deepcopy(self.fallback)
        records = []
        memo = {}
        for values, positions in self.records:
            if positions:
                values = list(values)
                for index in positions:
                    values[index] = _clone_value(values[index], memo)
            record = object.__new__(RecordedCopy)
            vars(record).update(zip(_RECORD_FIELDS, values))
            records.append(record)
        capture = object.__new__(CaptureResult)
        vars(capture).update(
            copies=[records[index] for index in self.indices],
            unsupported=[],
            unattributed=0,
            unsupported_reasons={},
            source_replay=self.metadata[0],
            generation=self.metadata[1],
            owner_dependencies=_clone_value(self.metadata[2], memo),
            owner_workspaces={
                name: _clone_value(value, memo) for name, value in self.owner_workspaces
            },
            destination_layouts={
                name: _clone_value(value, memo)
                for name, value in self.destination_layouts
            },
            generator_fingerprint=self.fallback[0].generator_fingerprint,
        )
        result = (
            capture,
            {
                name: _clone_value(value, memo) if _needs_copy(value) else value
                for name, value in self.layout
            },
        )
        if not _ordinary_copy_classes():
            return copy.deepcopy(self.fallback)
        return result
