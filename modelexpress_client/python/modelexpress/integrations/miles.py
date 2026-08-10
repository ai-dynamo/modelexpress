# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Trainer-side Miles to ModelExpress publisher adapter.

The integration deliberately uses only the attribute-level Miles protocol.  In
particular, importing ModelExpress never imports Miles.

Every publication is stamped with the training step from the Miles request.
Receivers wait for that exact stamp before reading the stable registered
addresses, preventing a partially propagated update from installing mixed-step
bytes.
"""

from __future__ import annotations

import atexit
import hashlib
import inspect
import ipaddress
import logging
import threading
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from modelexpress import envs
from modelexpress.metadata.client_factory import create_metadata_client
from modelexpress.nixl_transfer import NixlTransferManager
from modelexpress.refit.reshard import (
    MegatronAliasInput,
    MxReshardRendezvous,
    build_hf_aliases,
    publish_registered_shard_table,
)

logger = logging.getLogger("modelexpress.integrations.miles")


def _as_int(value: object, label: str) -> int:
    if value is None:
        raise ValueError(f"Miles ModelExpress registration requires {label}")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Miles ModelExpress registration has invalid {label}={value!r}"
        ) from exc


def _require_text(value: object, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"Miles ModelExpress registration requires non-empty {label}")
    return text


def _routable_worker_host() -> str:
    host = _require_text(
        envs.MX_WORKER_HOST,
        "MX_WORKER_HOST (a receiver-routable trainer hostname or IP)",
    )
    lowered = host.lower().rstrip(".")
    if lowered in {"localhost", "localhost.localdomain"}:
        raise ValueError("MX_WORKER_HOST must be receiver-routable, not localhost")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return host
    if address.version == 6:
        raise ValueError(
            "MX_WORKER_HOST IPv6 literals are unsupported by the current "
            "ModelExpress host:port rendezvous contract; use a routable DNS name"
        )
    if address.is_loopback or address.is_unspecified:
        raise ValueError(f"MX_WORKER_HOST must be receiver-routable, got {host!r}")
    return host


def _tensor_signature(tensors: Mapping[str, Any]) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            name,
            int(tensor.data_ptr()),
            tuple(int(dim) for dim in tensor.shape),
            str(tensor.dtype),
        )
        for name, tensor in sorted(tensors.items())
    )


def _published_signature(
    published: Sequence[Any],
) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            item.name,
            str(item.dtype),
            int(item.elsize),
            tuple(int(dim) for dim in item.full_shape),
            tuple(
                (
                    shard.agent_name,
                    int(shard.device_id),
                    int(shard.addr),
                    tuple(int(value) for value in shard.shard_offset),
                    tuple(int(dim) for dim in shard.shape),
                )
                for shard in item.shards
            ),
        )
        for item in published
    )


def _request_tensors(request: Any) -> dict[str, Any]:
    tensors: dict[str, Any] = {}
    for spec in request.tensors:
        name = _require_text(spec.native_name, "tensor native_name")
        if name in tensors:
            raise ValueError(f"duplicate Miles native tensor name {name!r}")
        tensor = spec.tensor
        if not tensor.is_contiguous():
            raise ValueError(f"{name}: native Megatron storage must be contiguous")
        tensors[name] = tensor
    if not tensors:
        raise ValueError("Miles ModelExpress publish request contains no tensors")
    return tensors


def _alias_by_name(spec: Any) -> dict[str, Any]:
    aliases: dict[str, Any] = {}
    for alias in spec.aliases:
        name = _require_text(alias.hf_name, f"{spec.native_name} alias hf_name")
        if name in aliases:
            raise ValueError(f"{spec.native_name}: duplicate HF alias {name!r}")
        aliases[name] = alias
    if tuple(aliases) != tuple(spec.hf_names):
        raise ValueError(
            f"{spec.native_name}: hf_names {tuple(spec.hf_names)!r} do not match "
            f"alias names {tuple(aliases)!r}"
        )
    return aliases


def _validate_alias_output(
    spec: Any, aliases: Mapping[str, Any], published: Sequence[Any]
) -> None:
    actual = {item.name: item for item in published}
    if set(actual) != set(aliases) or len(actual) != len(published):
        raise ValueError(
            f"{spec.native_name}: generated aliases {sorted(actual)!r} do not "
            f"match Miles aliases {sorted(aliases)!r}"
        )
    for name, expected in aliases.items():
        item = actual[name]
        expected_shape = tuple(int(dim) for dim in expected.global_shape)
        if tuple(item.full_shape) != expected_shape:
            raise ValueError(
                f"{spec.native_name}/{name}: generated global shape "
                f"{tuple(item.full_shape)} != Miles {expected_shape}"
            )
        axis = expected.shard_axis
        shard_range = expected.local_shard_range
        if (axis is None) != (shard_range is None):
            raise ValueError(
                f"{spec.native_name}/{name}: incomplete Miles shard geometry"
            )
        if axis is None:
            if len(item.shards) != 1 or tuple(item.shards[0].shape) != expected_shape:
                raise ValueError(
                    f"{spec.native_name}/{name}: invalid replicated alias geometry"
                )
            continue
        axis = int(axis)
        lo, hi = (int(value) for value in shard_range)
        intervals = []
        for shard in item.shards:
            offset = tuple(int(value) for value in shard.shard_offset)
            shape = tuple(int(value) for value in shard.shape)
            if len(shape) != len(expected_shape) or any(
                offset[index] != 0 or shape[index] != expected_shape[index]
                for index in range(len(shape))
                if index != axis
            ):
                raise ValueError(
                    f"{spec.native_name}/{name}: generated shard is not axis-{axis} aligned"
                )
            intervals.append((offset[axis], offset[axis] + shape[axis]))
        intervals.sort()
        cursor = lo
        for start, end in intervals:
            if start != cursor or end <= start:
                raise ValueError(
                    f"{spec.native_name}/{name}: generated shards do not cover Miles range"
                )
            cursor = end
        if cursor != hi:
            raise ValueError(
                f"{spec.native_name}/{name}: generated range {(lo, cursor)} != Miles {(lo, hi)}"
            )


def _generic_alias_input(spec: Any, alias: Any) -> MegatronAliasInput | None:
    tensor = spec.tensor
    full_shape = tuple(int(dim) for dim in alias.global_shape)
    shard_axis = None if alias.shard_axis is None else int(alias.shard_axis)
    shard_range = (
        None
        if alias.local_shard_range is None
        else tuple(int(value) for value in alias.local_shard_range)
    )
    if (shard_axis is None) != (shard_range is None):
        raise ValueError(f"{spec.native_name}: incomplete generic alias shard geometry")

    metadata = dict(spec.conversion_metadata)
    if metadata.get("layout") == "padded_vocab":
        if shard_axis != 0 or shard_range is None:
            raise ValueError(
                f"{spec.native_name}: padded vocabulary must be sharded on axis 0"
            )
        native_range = spec.local_shard_range
        if native_range is None:
            if tuple(tensor.shape) != full_shape:
                tensor = tensor.narrow(0, 0, full_shape[0])
        else:
            native_lo, native_hi = (int(value) for value in native_range)
            lo, hi = shard_range
            if lo != native_lo or not native_lo <= hi <= native_hi:
                raise ValueError(
                    f"{spec.native_name}: invalid padded vocabulary canonical range"
                )
            valid_rows = hi - lo
            if valid_rows == 0:
                return None
            tensor = tensor.narrow(0, 0, valid_rows)

    if shard_axis is not None:
        lo, hi = shard_range
        if not 0 <= shard_axis < tensor.ndim or hi - lo != int(
            tensor.shape[shard_axis]
        ):
            raise ValueError(
                f"{spec.native_name}: alias shard range does not match local storage"
            )
        placement = "SHARD"
    else:
        if tuple(int(dim) for dim in tensor.shape) != full_shape:
            raise ValueError(
                f"{spec.native_name}: replicated alias shape does not match storage"
            )
        placement = "REPLICATE"

    if metadata.get("layout") == "padded_vocab":
        role = "vocab"
    elif placement == "REPLICATE":
        role = "replicated"
    elif shard_axis == 1:
        role = "row"
    else:
        role = str(spec.role)
    return MegatronAliasInput(
        name=str(spec.native_name),
        tensor=tensor,
        role=role,
        hf_names=(str(alias.hf_name),),
        global_shape=full_shape,
        placement_kind=placement,
        shard_axis=shard_axis,
        local_shard_range=shard_range,
    )


def _alias_input(spec: Any) -> tuple[MegatronAliasInput | None, Mapping[str, Any]]:
    aliases = _alias_by_name(spec)
    role = str(spec.role)
    placement = "SHARD" if spec.shard_axis is not None else "REPLICATE"
    if role == "qkv":
        metadata = dict(spec.conversion_metadata)
        required = ("head_dim", "local_query_groups", "query_heads_per_group")
        missing = [name for name in required if name not in metadata]
        if missing:
            raise ValueError(
                f"{spec.native_name}: QKV conversion metadata missing {missing}"
            )
        item = MegatronAliasInput(
            name=str(spec.native_name),
            tensor=spec.tensor,
            role="qkv_column",
            hf_names=tuple(str(name) for name in spec.hf_names),
            global_shape=tuple(int(dim) for dim in spec.global_shape),
            placement_kind=placement,
            shard_axis=None if spec.shard_axis is None else int(spec.shard_axis),
            local_shard_range=(
                None
                if spec.local_shard_range is None
                else tuple(int(value) for value in spec.local_shard_range)
            ),
            extras={
                "head_dim": str(metadata["head_dim"]),
                "num_heads_local": str(
                    int(metadata["local_query_groups"])
                    * int(metadata["query_heads_per_group"])
                ),
                "num_kv_heads_local": str(metadata["local_query_groups"]),
            },
        )
        return item, aliases
    if role == "gate_up":
        item = MegatronAliasInput(
            name=str(spec.native_name),
            tensor=spec.tensor,
            role="gated_mlp_column",
            hf_names=tuple(str(name) for name in spec.hf_names),
            global_shape=tuple(int(dim) for dim in spec.global_shape),
            placement_kind=placement,
            shard_axis=None if spec.shard_axis is None else int(spec.shard_axis),
            local_shard_range=(
                None
                if spec.local_shard_range is None
                else tuple(int(value) for value in spec.local_shard_range)
            ),
            extras={"gated_mlp_order": "gate_then_up"},
        )
        return item, aliases
    if len(aliases) != 1:
        raise ValueError(
            f"{spec.native_name}: generic aliasing requires exactly one HF name"
        )
    return _generic_alias_input(spec, next(iter(aliases.values()))), aliases


def _build_published(request: Any, agent_name: str) -> list[Any]:
    inputs: list[MegatronAliasInput] = []
    expected_aliases = []
    all_names: set[str] = set()
    for spec in request.tensors:
        item, aliases = _alias_input(spec)
        duplicate = all_names.intersection(aliases)
        if duplicate:
            raise ValueError(
                f"duplicate HF aliases across Miles tensors: {sorted(duplicate)!r}"
            )
        all_names.update(aliases)
        if item is not None:
            inputs.append(item)
            expected_aliases.append((spec, aliases))
    published = build_hf_aliases(inputs, agent_name=agent_name)
    cursor = 0
    for spec, aliases in expected_aliases:
        count = len(aliases)
        _validate_alias_output(spec, aliases, published[cursor : cursor + count])
        cursor += count
    return published


class MilesModelExpressPublisher:
    """Long-lived publisher implementing Miles' duck-typed trainer protocol."""

    def __init__(
        self,
        *,
        manager_factory: Callable[..., Any] = NixlTransferManager,
        client_factory: Callable[..., Any] = create_metadata_client,
        rendezvous_factory: Callable[..., Any] = MxReshardRendezvous,
        publish_fn: Callable[..., str] = publish_registered_shard_table,
        device_id_factory: Callable[[], int] | None = None,
    ) -> None:
        self._manager_factory = manager_factory
        self._client_factory = client_factory
        self._rendezvous_factory = rendezvous_factory
        self._publish_fn = publish_fn
        self._device_id_factory = device_id_factory
        self._lock = threading.RLock()
        self._configured = False
        self._closed = False
        self._atexit_registered = False
        self._manager: Any = None
        self._client: Any = None
        self._rendezvous: Any = None
        self._registration_signature: tuple[object, ...] | None = None
        self._native_signature: tuple[tuple[object, ...], ...] | None = None
        self._published_signature: tuple[tuple[object, ...], ...] | None = None
        self._metadata_endpoint = ""

    def configure(self, registration: Any) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("Miles ModelExpress publisher is closed")
            model_name = _require_text(
                getattr(registration, "model_name", None),
                "registration.model_name; update Miles to pass its configured model name",
            )
            worker_id = _require_text(registration.worker_id, "registration.worker_id")
            cohort_id = _require_text(registration.cohort_id, "registration.cohort_id")
            geometry = dict(registration.source_geometry)
            global_rank = _as_int(
                geometry.get("global_rank"),
                "source_geometry['global_rank']; update Miles to pass the trainer global rank",
            )
            logical_groups = tuple(str(value) for value in registration.logical_groups)
            signature = (
                model_name,
                worker_id,
                cohort_id,
                tuple(sorted(geometry.items())),
                logical_groups,
            )
            if self._configured:
                if signature == self._registration_signature:
                    return
                logger.info(
                    "Miles rollout cohort changed; rebuilding the ModelExpress "
                    "publisher session"
                )
                self._teardown_locked()

            host = _routable_worker_host()
            if self._device_id_factory is None:
                import torch

                device_id = int(torch.cuda.current_device())
            else:
                device_id = int(self._device_id_factory())
            listen_port = int(envs.MX_METADATA_PORT) + device_id
            if not 1 <= listen_port <= 65535:
                raise ValueError(f"invalid NIXL listen port {listen_port}")
            agent_token = hashlib.sha256(
                f"{worker_id}\0{cohort_id}".encode()
            ).hexdigest()[:12]
            manager = self._manager_factory(
                agent_name=f"mx-miles-{global_rank}-{agent_token}",
                device_id=device_id,
                listen_port=listen_port,
            )
            client = None
            rendezvous = None
            try:
                manager.initialize()
                client = self._client_factory(worker_rank=global_rank)
                rendezvous = self._rendezvous_factory(
                    client,
                    role="trainer",
                    rank=global_rank,
                    model_name=model_name,
                    worker_id=worker_id,
                )
            except Exception:
                if rendezvous is not None:
                    rendezvous.close()
                if client is not None:
                    client.close()
                manager.shutdown()
                raise
            self._manager = manager
            self._client = client
            self._rendezvous = rendezvous
            self._metadata_endpoint = f"{host}:{listen_port}"
            self._registration_signature = signature
            self._configured = True
            if not self._atexit_registered:
                atexit.register(self.close)
                self._atexit_registered = True

    def publish_and_execute(self, request: Any) -> None:
        with self._lock:
            if not self._configured or self._closed:
                raise RuntimeError(
                    "Miles ModelExpress publisher must be configured before publishing"
                )
            assert self._registration_signature is not None
            _model, worker_id, cohort_id, geometry_items, logical_groups = (
                self._registration_signature
            )
            if (
                str(request.worker_id) != worker_id
                or str(request.cohort_id) != cohort_id
                or tuple(sorted(dict(request.source_geometry).items()))
                != geometry_items
                or str(request.logical_group) not in logical_groups
            ):
                raise RuntimeError(
                    "Miles publish request changed worker/cohort/source geometry/logical-group session"
                )

            native_tensors = _request_tensors(request)
            signature = _tensor_signature(native_tensors)
            if (
                self._native_signature is not None
                and signature != self._native_signature
            ):
                raise RuntimeError(
                    "Miles native Megatron registration changed names, addresses, shapes, or dtypes"
                )
            published = _build_published(request, str(self._manager.agent_name))
            if not published:
                raise ValueError(
                    "Miles request has no non-padding HF aliases to publish"
                )
            published_signature = _published_signature(published)
            if self._published_signature is not None:
                if published_signature != self._published_signature:
                    raise RuntimeError(
                        "Miles HF alias publication changed names, addresses, shapes, "
                        "dtypes, geometry, or NIXL session"
                    )
            else:
                self._manager.register_tensors(native_tensors)
                self._native_signature = signature
                self._published_signature = published_signature

            kwargs: dict[str, Any] = {
                "manager": self._manager,
                "rendezvous": self._rendezvous,
                "published": published,
                "metadata_endpoint": self._metadata_endpoint,
            }
            parameters = inspect.signature(self._publish_fn).parameters
            if "publisher_step" in parameters:
                kwargs["publisher_step"] = int(request.training_step)
            elif "training_step" in parameters:
                # Compatibility with an adapter build that used the earlier
                # argument spelling for the same wire stamp.
                kwargs["training_step"] = int(request.training_step)
            else:
                raise RuntimeError(
                    "ModelExpress publication API cannot stamp the requested "
                    "training step; refusing an unverifiable refit"
                )
            self._publish_fn(**kwargs)

    def _teardown_locked(self) -> None:
        """Release one cohort-scoped session while keeping this adapter reusable."""
        if self._rendezvous is not None:
            try:
                self._rendezvous.close()
            except Exception:
                logger.warning("failed to close Miles rendezvous", exc_info=True)
            self._rendezvous = None
        if self._manager is not None:
            try:
                self._manager.shutdown()
            except Exception:
                logger.warning(
                    "failed to shut down Miles NIXL manager", exc_info=True
                )
            self._manager = None
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                logger.warning(
                    "failed to close Miles metadata client", exc_info=True
                )
            self._client = None
        self._configured = False
        self._registration_signature = None
        self._native_signature = None
        self._published_signature = None
        self._metadata_endpoint = ""

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._atexit_registered:
                atexit.unregister(self.close)
                self._atexit_registered = False
            self._teardown_locked()


def create_miles_publisher() -> MilesModelExpressPublisher:
    """Zero-argument factory for Miles' explicit adapter path."""

    return MilesModelExpressPublisher()


__all__ = ["MilesModelExpressPublisher", "create_miles_publisher"]
