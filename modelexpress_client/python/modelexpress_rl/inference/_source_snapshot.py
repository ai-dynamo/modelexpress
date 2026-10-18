# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Immutable host metadata owned by the manifest-byte cache."""

from types import GetSetDescriptorType, MappingProxyType
from typing import NamedTuple

import torch
from modelexpress.refit.reshard.slice_plan import Shard
from modelexpress.refit.reshard.transfer_plan import SourceInfo


class _ShardSnapshot(NamedTuple):
    shard_offset: tuple
    shape: tuple
    session: str
    addr: int
    elsize: int
    digest: str | None


class _TensorSnapshot(NamedTuple):
    global_shape: tuple
    dtype: torch.dtype
    elsize: int
    shards: tuple


_SNAPSHOT_FIELDS = tuple(
    (
        cls,
        cls.__bases__,
        vars(cls),
        tuple((name, vars(cls)[name]) for name in cls._fields),
    )
    for cls in (_ShardSnapshot, _TensorSnapshot)
)
_TUPLE_GETATTRIBUTE = tuple.__getattribute__
_SOURCE_FIELDS = ("global_shape", "dtype", "elsize", "shards")
_SHARD_FIELDS = ("shard_offset", "shape", "session", "addr", "elsize", "digest")


def _snapshot_classes_unchanged() -> bool:
    if (
        _ShardSnapshot is not _SNAPSHOT_FIELDS[0][0]
        or _TensorSnapshot is not _SNAPSHOT_FIELDS[1][0]
    ):
        return False
    # Reuse live namespace views and avoid allocating iterators between class
    # checks: a cyclic-GC callback could otherwise change an earlier class.
    index = 0
    while index < len(_SNAPSHOT_FIELDS):
        row = _SNAPSHOT_FIELDS[index]
        cls = row[0]
        if type(cls) is not type or cls.__bases__ is not row[1]:
            return False
        namespace = row[2]
        if "__getattr__" in namespace:
            return False
        if (
            "__getattribute__" in namespace
            and namespace["__getattribute__"] is not _TUPLE_GETATTRIBUTE
        ):
            return False
        fields = row[3]
        field_index = 0
        while field_index < len(fields):
            field = fields[field_index]
            name = field[0]
            if name not in namespace or namespace[name] is not field[1]:
                return False
            field_index += 1
        index += 1
    return True


def _ordinary_record_class(cls, fields) -> bool:
    if type(cls) is not type or cls.__bases__ != (object,):
        return False
    namespace = vars(cls)
    if (
        "__getattr__" in namespace
        or cls.__getattribute__ is not object.__getattribute__
    ):
        return False
    descriptor = namespace.get("__dict__")
    if (
        type(descriptor) is not GetSetDescriptorType
        or descriptor.__objclass__ is not cls
    ):
        return False
    return all(
        name not in namespace
        or (cls is Shard and name == "digest" and namespace[name] is None)
        for name in fields
    )


def _record_state(value, cls, fields):
    if type(value) is not cls:
        return None
    state = vars(value)
    if type(state) is not dict:
        return None
    state = dict(state)
    if any(type(name) is not str for name in state):
        return None
    return state if tuple(state) == fields else None


def _shape(value) -> bool:
    return type(value) is tuple and all(type(size) is int for size in value)


def _freeze_sources(source_map) -> MappingProxyType | None:
    """Own only source rows; retain the resolver's outer metadata schema."""
    if type(source_map) is not dict:
        return None
    source_map = dict(source_map)
    if not (
        _ordinary_record_class(SourceInfo, _SOURCE_FIELDS)
        and _ordinary_record_class(Shard, _SHARD_FIELDS)
        and _snapshot_classes_unchanged()
    ):
        return None
    sources = {}
    for name, source in source_map.items():
        state = _record_state(source, SourceInfo, _SOURCE_FIELDS)
        if type(name) is not str or state is None:
            return None
        shape, dtype, elsize, shards = (state[field] for field in _SOURCE_FIELDS)
        if not (
            _shape(shape)
            and type(dtype) is torch.dtype
            and type(elsize) is int
            and type(shards) is list
        ):
            return None
        frozen_shards = []
        for shard in tuple(shards):
            state = _record_state(shard, Shard, _SHARD_FIELDS)
            if state is None:
                return None
            offset, shard_shape, session, address, shard_elsize, digest = (
                state[field] for field in _SHARD_FIELDS
            )
            if not (
                _shape(offset)
                and _shape(shard_shape)
                and type(session) is str
                and type(address) is int
                and type(shard_elsize) is int
                and (digest is None or type(digest) is str)
            ):
                return None
            frozen_shards.append(
                tuple.__new__(
                    _ShardSnapshot,
                    (offset, shard_shape, session, address, shard_elsize, digest),
                )
            )
        sources[name] = tuple.__new__(
            _TensorSnapshot, (shape, dtype, elsize, tuple(frozen_shards))
        )
    if not (
        _ordinary_record_class(SourceInfo, _SOURCE_FIELDS)
        and _ordinary_record_class(Shard, _SHARD_FIELDS)
        and _snapshot_classes_unchanged()
    ):
        return None
    # No caller retains the backing maps. Tuple rows contain only immutable
    # scalars and tuples, so a byte-identical manifest cannot acquire new fields.
    return MappingProxyType(sources)
