# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang external weight-update receiver factory for ModelExpress.

SGLang calls ``create_receiver`` when an ``init_weights_update_group`` request
names it in ``receiver`` and the engine allowlisted it in
``--weight-update-receivers``. Returns a receiver with
``receive(payload)``/``destroy()`` over the machinery in ``sglang.py``.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any
from uuid import uuid4

import grpc

from modelexpress import auth

from ..rendezvous import CollectiveRendezvous
from ._common import _counted_names, _endpoint
from .sglang import (
    SglangGeneratorSession,
    build_generator_loader,
    close_generator_resources,
)
from .manifest import manifest_from_wire

logger = logging.getLogger("modelexpress_rl.collective.integrations.sglang_receiver")

#: The dotted path to pass in ``receiver`` and ``--weight-update-receivers``.
RECEIVER_PATH = (
    "modelexpress_rl.collective.integrations.sglang_receiver.create_receiver"
)
_ROUND_KEYS = frozenset({"operation_id", "version"})


def _registered_parameter_aliases(model: Any) -> dict[str, str]:
    """Map each tied-alias parameter name to its load-visible registration.

    A parameter registered under two names (tied embeddings: SGLang ties
    ``lm_head`` to ``model.embed_tokens``) appears once in
    ``named_parameters()`` -- under the FIRST registered name, which is the
    only key the model's ``load_weights`` name resolution can see. Walking
    every registration (``remove_duplicate=False``, the same pattern the RAW
    receiver's alias detection uses) exposes the hidden halves: the returned
    map's keys are the load-hidden alias names and its values are the
    registrations that ``load_weights`` actually resolves.

    A model without a module tree (test doubles) has no aliases to find.
    """
    named_modules = getattr(model, "named_modules", None)
    if not callable(named_modules):
        return {}
    seen: dict[int, str] = {}
    aliases: dict[str, str] = {}
    for path, module in named_modules(remove_duplicate=False):
        parameters = getattr(module, "_parameters", None)
        if not parameters:
            continue
        for local, parameter in parameters.items():
            if parameter is None:
                continue
            name = f"{path}.{local}" if path else local
            owner = seen.setdefault(id(parameter), name)
            if owner != name:
                aliases[name] = owner
    return aliases


def _verify_tied_alias_coverage(plan_names: list[str], model: Any) -> None:
    """Accept tied-alias models; refuse alias writes the engine cannot land.

    The plan's names drive the receiver's ``load_weights`` calls, and the
    engine resolves them through its deduplicated parameter dict. A write
    aimed at a load-HIDDEN tied alias (``lm_head.weight`` when that parameter
    was first registered as ``model.embed_tokens.weight``) is dropped by that
    resolution: the received bytes never land. The round is still correct
    when the load-visible registration is itself in the plan -- the shared
    storage is written through it -- so the only refusal is a hidden alias
    whose registration the plan never covers: a silent-corruption hole.
    """
    aliases = _registered_parameter_aliases(model)
    if not aliases:
        return
    planned = set(plan_names)
    dropped = sorted(
        f"{name} (alias of {aliases[name]})"
        for name in planned
        if name in aliases and aliases[name] not in planned
    )
    if dropped:
        raise ValueError(
            "the M2N plan's tied-alias parameters cannot be written by the "
            "engine's load_weights: its name resolution exposes only the "
            "first registration, so the received bytes would be silently "
            f"dropped: {_counted_names(dropped)}; name the load-visible "
            "registration in the plan (or send both)"
        )
    covered = sorted(
        f"{alias}<-{owner}"
        for alias, owner in aliases.items()
        if alias not in planned and owner in planned
    )
    if covered:
        logger.debug(
            "tied parameter aliases covered by their shared registration: %s",
            ", ".join(covered),
        )


def _tp_coordinate(context: Any, name: str) -> int:
    value = getattr(context, name, None)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"WeightUpdateReceiverContext.{name} must be an integer")
    return value


def mx_server_endpoint(host: object, port: object) -> str:
    """The mx-server ``host:port`` carried in ``master_address``/``master_port``."""
    if not isinstance(host, str) or not host.strip() or "://" in host:
        raise ValueError(
            "master_address must be the ModelExpress server host without a scheme"
        )
    if not isinstance(port, int) or isinstance(port, bool) or not 0 < port < 65536:
        raise ValueError("master_port must be the ModelExpress server port")
    host = host.strip()
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return _endpoint(f"{host}:{port}")


class ModelExpressM2NReceiver:
    """One scheduler's ModelExpress generator session, owned by SGLang.

    ``receive`` returns only after the session's lanes finish on their own
    streams, so SGLang has nothing to drain.
    """

    def __init__(
        self,
        *,
        session: SglangGeneratorSession,
        rendezvous: CollectiveRendezvous,
        channel: Any,
        slot_id: str,
    ) -> None:
        self._session = session
        self._rendezvous = rendezvous
        self._channel = channel
        self.slot_id = slot_id
        self._failed = False
        self._destroyed = False

    def receive(self, payload: Mapping[str, Any] | None = None) -> None:
        """Run one round. SGLang already checked the weight-update session."""
        if self._destroyed:
            raise RuntimeError("the ModelExpress receiver was destroyed")
        if self._failed:
            raise RuntimeError(
                "a failed ModelExpress round poisons this receiver; call "
                "destroy_weights_update_group and init_weights_update_group again"
            )
        if not isinstance(payload, Mapping) or set(payload) != _ROUND_KEYS:
            raise ValueError(
                "receiver_payload must be exactly {'operation_id', 'version'}, "
                f"got {payload!r}"
            )
        version = payload["version"]
        operation_id = payload["operation_id"]
        for key, value in (("version", version), ("operation_id", operation_id)):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"receiver_payload {key} must be a non-empty string")
        try:
            self._session.run_round(version=version, operation_id=operation_id)
        except BaseException:
            # The session already tore itself down and reported the failure.
            # SGLang keeps this receiver registered until destroy, so mark it
            # dead and release the connection instead of dropping state.
            self._failed = True
            close_generator_resources(
                None, self._rendezvous, self._channel, suppress_errors=True
            )
            raise

    def destroy(self) -> None:
        """Release the session, rendezvous and channel; one-shot."""
        if self._destroyed:
            return
        self._destroyed = True
        close_generator_resources(
            self._session, self._rendezvous, self._channel, suppress_errors=False
        )


def create_receiver(context: Any) -> ModelExpressM2NReceiver:
    """The ``WeightUpdateReceiverFactory`` SGLang calls from ``init_weights_update_group``.

    ModelExpress owns its communicators, so SGLang creates no torch process
    group for this group (SG-1 promise for receiver groups).
    """
    plan, topology = manifest_from_wire(context.init_payload)
    if context.world_size != len(topology.generator_slots):
        raise ValueError(
            f"world_size {context.world_size} does not match the plan's "
            f"{len(topology.generator_slots)} generator slots"
        )
    rank_offset = context.rank_offset
    if not isinstance(rank_offset, int) or isinstance(rank_offset, bool):
        raise ValueError("rank_offset must be an integer")
    if rank_offset < 0:
        raise ValueError("rank_offset must be non-negative")
    tp_rank = _tp_coordinate(context, "tp_rank")
    tp_size = _tp_coordinate(context, "tp_size")
    if not 0 <= tp_rank < tp_size:
        raise ValueError(f"tp_rank {tp_rank} is outside tp_size {tp_size}")
    endpoint = mx_server_endpoint(context.master_address, context.master_port)

    # Tied-embedding models (Qwen3-0.6B class) register lm_head.weight as an
    # alias of embed_tokens.weight; verify every plan name is landable before
    # the first round depends on it.
    _verify_tied_alias_coverage(list(plan.parameter_names()), context.model)

    prepared = build_generator_loader(
        plan=plan,
        topology=topology,
        model=context.model,
        device=context.device,
        generator_slot_offset=rank_offset,
        tp_rank=tp_rank,
    )
    channel = rendezvous = session = None
    try:
        channel = auth.with_auth(grpc.insecure_channel(endpoint))
        rendezvous = CollectiveRendezvous(channel)
        session = SglangGeneratorSession.create(
            rendezvous=rendezvous,
            topology=topology,
            loader=prepared.loader,
            slot_id=prepared.slot_id,
            worker_id=f"sglang-{prepared.slot_id}-{uuid4().hex}",
            index_in_role=prepared.local_index,
            device=context.device,
        )
        session.prepare()
    except BaseException:
        close_generator_resources(session, rendezvous, channel, suppress_errors=True)
        raise
    return ModelExpressM2NReceiver(
        session=session,
        rendezvous=rendezvous,
        channel=channel,
        slot_id=prepared.slot_id,
    )


__all__ = [
    "RECEIVER_PATH",
    "ModelExpressM2NReceiver",
    "create_receiver",
    "mx_server_endpoint",
]
