# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang generator-side engine boundary for the MILES collective bridge."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

from ..client import RefitClientGenerator
from ..rendezvous import CollectiveRendezvous, Membership
from ..spi import LocalParamSpec, RefitCtx
from ..types import PlacementKind, ReshardPlan
from ._common import (
    _check_stable,
    _client_device,
    _collective_streams,
    _check_stack_budget,
    _derive_wire_plan,
    _dtype_label,
    _FrozenPlan,
    _layer_groups,
    _local_shape,
    _order_current_cuda_stream_before,
    _single_device,
    _Stack,
    _tensor_signature,
    _text,
)
from .miles import CollectiveTopology
from .sglang_layout import SglangModelFacts, destination_shard_dim

logger = logging.getLogger("modelexpress_rl.collective.integrations.sglang")

# HF projection -> (SGLang fused module, the shard id its weight_loader takes).
# Qwen2/Qwen3 stacked_params_mapping at the pinned fork base.
_FUSED_MEMBERS = {
    "q_proj": ("qkv_proj", "q"),
    "k_proj": ("qkv_proj", "k"),
    "v_proj": ("qkv_proj", "v"),
    "gate_proj": ("gate_up_proj", 0),
    "up_proj": ("gate_up_proj", 1),
}
_ROW_PARALLEL = ("o_proj", "down_proj")
_VOCAB_PARALLEL = {
    "model.embed_tokens.weight": "model.embed_tokens",
    "lm_head.weight": "lm_head",
}


class _NotProvable(ValueError):
    """The engine slice of a sharded destination cannot be established."""


def _attr(module: Any, name: str, label: str) -> Any:
    if not hasattr(module, name):
        raise _NotProvable(f"{label} has no {name!r} attribute")
    return getattr(module, name)


def _require(condition: bool, label: str, message: str) -> None:
    if not condition:
        raise _NotProvable(f"{label}: {message}")


def _plain_bf16_weight(module: Any, label: str, *, split_dim: int) -> Any:
    """The module's dense BF16 weight, loaded by its own stock weight_loader;
    quantized, packed, presharded or fused-foreign weights are not modelled."""
    import torch

    weight = _attr(module, "weight", label)
    _require(weight.dtype is torch.bfloat16, label, f"weight is {weight.dtype}")
    _require(weight.is_contiguous(), label, "weight is not contiguous")
    _require(
        getattr(weight, "weight_loader", None) == module.weight_loader,
        label,
        "weight is not loaded by the module's stock weight_loader",
    )
    _require(
        not getattr(module, "use_presharded_weights", False),
        label,
        "presharded weights",
    )
    for flag in ("packed_dim", "use_bitsandbytes_4bit", "is_gguf_weight"):
        _require(not getattr(weight, flag, None), label, f"weight carries {flag}")
    dim_attr = "input_dim" if split_dim == 1 else "output_dim"
    _require(
        getattr(weight, dim_attr, None) == split_dim,
        label,
        f"weight {dim_attr} is {getattr(weight, dim_attr, None)!r}, not {split_dim}",
    )
    return weight


def _require_tp(module: Any, label: str, tp_rank: int, tp_size: int) -> None:
    _require(
        _attr(module, "tp_size", label) == tp_size,
        label,
        f"module tp_size {module.tp_size} is not the engine TP size {tp_size}",
    )
    if hasattr(module, "tp_rank"):
        _require(
            module.tp_rank == tp_rank,
            label,
            f"module tp_rank {module.tp_rank} is not this rank {tp_rank}",
        )


def _engine_view(
    model: Any,
    name: str,
    global_shape: tuple[int, ...],
    tp_rank: int,
    tp_size: int,
) -> Any:
    """This TP rank's slice of ``name`` inside SGLang's live storage.

    Every number comes from the live module and reproduces where the module's
    own weight_loader would copy rank ``tp_rank``'s even slice. Anything
    unproven raises ``_NotProvable``; this never guesses.
    """
    rows, cols = global_shape
    module_path, _, leaf = name.rpartition(".")
    if leaf != "weight":
        raise _NotProvable(f"{name}: only weights are delivered pre-split")
    parent, _, member = module_path.rpartition(".")
    if name in _VOCAB_PARALLEL:
        label = _VOCAB_PARALLEL[name]
        module = model.get_submodule(label)
        _require_tp(module, label, tp_rank, tp_size)
        weight = _plain_bf16_weight(module, label, split_dim=0)
        _require(rows % tp_size == 0, label, f"{rows} rows over TP {tp_size}")
        piece = rows // tp_size
        indices = _attr(module, "shard_indices", label)
        _require(
            _attr(module, "org_vocab_size", label) == rows
            and _attr(module, "num_embeddings_padded", label) == rows
            and _attr(module, "num_added_embeddings", label) == 0,
            label,
            "the vocabulary is padded or extended",
        )
        _require(
            indices.org_vocab_start_index == tp_rank * piece
            and indices.org_vocab_end_index == (tp_rank + 1) * piece,
            label,
            "the rank's vocabulary range is not its even slice",
        )
        _require(
            tuple(weight.shape) == (piece, cols),
            label,
            f"local weight {tuple(weight.shape)} is not {(piece, cols)}",
        )
        return weight.data
    if member in _FUSED_MEMBERS:
        fused, shard_id = _FUSED_MEMBERS[member]
        label = f"{parent}.{fused}"
        module = model.get_submodule(label)
        _require_tp(module, label, tp_rank, tp_size)
        weight = _plain_bf16_weight(module, label, split_dim=0)
        if fused == "qkv_proj":
            _require(
                _attr(module, "kv_tp_size", label) == tp_size
                and _attr(module, "kv_tp_rank", label) == tp_rank
                and _attr(module, "num_kv_head_replicas", label) == 1,
                label,
                "key/value heads are replicated or on a separate TP group",
            )
            head = _attr(module, "head_size", label)
            v_head = _attr(module, "v_head_size", label)
            heads = _attr(module, "num_heads", label)
            kv_heads = _attr(module, "num_kv_heads", label)
            _require(
                heads * tp_size == _attr(module, "total_num_heads", label)
                and kv_heads * tp_size == _attr(module, "total_num_kv_heads", label),
                label,
                "head counts do not split evenly over the TP size",
            )
            local = {"q": heads * head, "k": kv_heads * head, "v": kv_heads * v_head}
            offsets = {"q": 0, "k": local["q"], "v": local["q"] + local["k"]}
            _require(
                local[shard_id] * tp_size == rows,
                label,
                f"{member} has {rows} rows, the module expects "
                f"{local[shard_id] * tp_size}",
            )
            total = sum(local.values())
            offset, extent = offsets[shard_id], local[shard_id]
        else:
            sizes = list(_attr(module, "output_sizes", label))
            _require(len(sizes) == 2, label, f"output_sizes {sizes}")
            _require(
                all(size % tp_size == 0 for size in sizes),
                label,
                f"output_sizes {sizes} do not split over TP {tp_size}",
            )
            _require(
                sizes[shard_id] == rows,
                label,
                f"{member} has {rows} rows, the module expects {sizes[shard_id]}",
            )
            total = sum(sizes) // tp_size
            offset = sum(sizes[:shard_id]) // tp_size
            extent = sizes[shard_id] // tp_size
        _require(
            tuple(weight.shape) == (total, cols),
            label,
            f"local weight {tuple(weight.shape)} is not {(total, cols)}",
        )
        return weight.data[offset : offset + extent]
    if member in _ROW_PARALLEL:
        label = module_path
        module = model.get_submodule(label)
        _require_tp(module, label, tp_rank, tp_size)
        weight = _plain_bf16_weight(module, label, split_dim=1)
        _require(
            _attr(module, "input_size", label) == cols
            and _attr(module, "output_size", label) == rows
            and cols % tp_size == 0,
            label,
            "module sizes do not match the canonical tensor",
        )
        _require(
            tuple(weight.shape) == (rows, cols // tp_size),
            label,
            f"local weight {tuple(weight.shape)} is not {(rows, cols // tp_size)}",
        )
        return weight.data
    raise _NotProvable(f"{name}: SGLang has no proven split for this tensor")


def _view_signature(view: Any) -> tuple[int, tuple[int, ...], tuple[int, ...]]:
    return (
        int(view.data_ptr()),
        tuple(int(dim) for dim in view.shape),
        tuple(int(stride) for stride in view.stride()),
    )


def _current_stream_key() -> int | None:
    """Identity of the CUDA stream the calling hook runs on, None without CUDA."""
    import torch

    if not torch.cuda.is_available():
        return None
    return int(torch.cuda.current_stream().cuda_stream)


def _check_disjoint(stack: str, views: list[Any]) -> None:
    """Fail unless the views of one stack occupy pairwise disjoint memory."""
    spans = sorted(
        (
            int(view.data_ptr()),
            int(view.data_ptr()) + view.numel() * view.element_size(),
        )
        for view in views
    )
    for (_, end), (start, _) in zip(spans, spans[1:], strict=False):
        if start < end:
            raise ValueError(
                f"{stack}: the engine views of one stack overlap in memory, so "
                "the stacked receive could not install them independently"
            )


class SglangLoader:
    """``Loader`` over a live SGLang model.

    Replicated entries land in persistent scratch buffers installed through
    the model's own ``load_weights``; sharded entries go straight into this
    rank's live-storage slice that ``_engine_view`` proves. A failed install
    poisons the loader: later rounds are refused.
    """

    def __init__(
        self,
        *,
        plan: ReshardPlan,
        model: Any,
        device: Any,
        layer_groups: tuple[tuple[str, ...], ...] = (),
        generator_index: int | None = None,
        tp_rank: int = 0,
        tp_size: int = 1,
    ) -> None:
        import torch

        self._plan = _FrozenPlan(plan)
        self._model = model
        # Equal-geometry stacking: ``plan`` stays the per-tensor contract that
        # view resolution and staging follow; the wire plan (what is digested,
        # published and walked as reshard calls) replaces each stack's members
        # with one stacked entry. Without stack keys the two are the same.
        wire_plan, stacks = _derive_wire_plan(self._plan.capture())
        self._stacks: dict[str, _Stack] = {stack.name: stack for stack in stacks}
        self._wire = _FrozenPlan(wire_plan) if stacks else self._plan
        self._member_stack = {
            member: (stack, index)
            for stack in stacks
            for index, member in enumerate(stack.members)
        }
        self._groups = _layer_groups(layer_groups, self._wire.names())
        self._tp_rank = tp_rank
        self._tp_size = tp_size
        # A stack's size is checked before any buffer exists, at prepare.
        _check_stack_budget(stacks)
        self._alloc_device = device
        self._stack_buffers: dict[str, Any] = {}
        self._stack_signatures: dict[str, Any] = {}
        self._scratch: dict[int | None, Any] = {}
        self._scratch_primary: Any = None
        self._scratch_primary_claimed = False
        self._scratch_signature: Any = None
        self._scratch_elements = 0
        self._buffers: dict[str, Any] = {}
        self._views: dict[str, tuple[int, tuple[int, ...], tuple[int, ...]]] = {}
        self._local_shapes: dict[str, tuple[int, ...]] = {}
        self._signatures = {}
        if self._plan.sharded_destination:
            self._check_engine_rank(generator_index)
            facts = self._model_facts()
        # A stack whose destination is whole on this rank is received into one
        # persistent stack; its members are views of it, so load_weights reads
        # them in place. A sharded destination cannot be a stack (live storage
        # is one tensor per parameter) and is received through scratch instead.
        for stack in stacks:
            if stack.entry.dst_placements[-1].kind is PlacementKind.SHARD:
                continue
            stacked = torch.empty(
                stack.entry.global_shape, dtype=torch.bfloat16, device=device
            )
            self._stack_buffers[stack.name] = stacked
            self._stack_signatures[stack.name] = _tensor_signature(
                stack.name,
                stacked,
                expected_shape=stack.entry.global_shape,
                expected_dtype=stack.entry.dtype,
            )
        for name in self._plan.names():
            entry = self._plan.entry(name)
            placement = entry.dst_placements[-1]
            if placement.kind is PlacementKind.SHARD:
                generator_rank = self._plan.dst_mesh.rank_offset + generator_index
                local_shape = _local_shape(
                    self._plan.destination_index(name, generator_rank)
                )
                expected_dim = destination_shard_dim(
                    name, entry.global_shape, tp_size, facts
                )
                if expected_dim != placement.dim:
                    raise ValueError(
                        f"{name}: the plan splits dim {placement.dim}, but SGLang "
                        f"at TP {tp_size} needs {expected_dim}; trainer and "
                        "receiver disagree on the destination rule"
                    )
                buffer = self._resolve_view(name)
                if tuple(int(dim) for dim in buffer.shape) != local_shape:
                    raise ValueError(
                        f"{name}: engine slice {tuple(buffer.shape)} is not the "
                        f"plan's destination shard {local_shape}"
                    )
                self._views[name] = _view_signature(buffer)
            else:
                local_shape = entry.global_shape
                if name in self._member_stack:
                    stack, index = self._member_stack[name]
                    buffer = self._stack_buffers[stack.name][index]
                else:
                    buffer = torch.empty(
                        entry.global_shape, dtype=torch.bfloat16, device=device
                    )
            self._buffers[name] = buffer
            self._local_shapes[name] = local_shape
            self._signatures[name] = _tensor_signature(
                name,
                buffer,
                expected_shape=local_shape,
                expected_dtype=entry.dtype,
            )
        self._device = _single_device(self._signatures, "SGLang loader")
        for stack in stacks:
            if stack.name in self._stack_buffers:
                continue
            # The scratch below is hardcoded bfloat16. Refuse a non-bf16 wire
            # dtype here, with the stack named, so the sharded path fails at
            # prepare like the replicated path's buffer signature instead of
            # only at the native transfer's dtype rejection.
            if _dtype_label(stack.entry.dtype) != "bfloat16":
                raise ValueError(
                    f"{stack.name}: a sharded stack is received through a "
                    "bfloat16 scratch, but its wire dtype is "
                    f"{_dtype_label(stack.entry.dtype)}"
                )
            _check_disjoint(stack.name, [self._buffers[m] for m in stack.members])
            elements = len(stack.members) * math.prod(
                self._local_shapes[stack.members[0]]
            )
            self._scratch_elements = max(self._scratch_elements, elements)
        if self._scratch_elements:
            # Stacks run in order on a lane stream and the post hook's copy
            # out of the scratch is enqueued on that stream before the next
            # stack's receive, so one scratch per stream suffices. The
            # primary scratch is allocated here so its failure fails prepare;
            # which stream a hook runs on is only known when the engine drives
            # the round, so a hook on another lane stream allocates that
            # stream's own scratch on first use and two streams never share.
            self._scratch_primary = torch.empty(
                self._scratch_elements, dtype=torch.bfloat16, device=device
            )
            self._scratch_signature = _tensor_signature(
                "stacked receive scratch",
                self._scratch_primary,
                expected_shape=(self._scratch_elements,),
                expected_dtype="bfloat16",
            )
        self._round_version: str | None = None
        self._poisoned = False

    def _scratch_for_current_stream(self) -> Any:
        """The receive scratch for the stream the hook runs on.

        The single-lane integration has one stream, which claims the scratch
        allocated at prepare. Should a hook ever run on another stream, it
        gets its own scratch, so two streams never share one.
        """
        import torch

        key = _current_stream_key()
        scratch = self._scratch.get(key)
        if scratch is None:
            if not self._scratch_primary_claimed:
                scratch = self._scratch_primary
                self._scratch_primary_claimed = True
            else:
                scratch = torch.empty(
                    self._scratch_elements,
                    dtype=torch.bfloat16,
                    device=self._alloc_device,
                )
            self._scratch[key] = scratch
        return scratch

    def _scratch_spec(self, stack: _Stack) -> LocalParamSpec:
        """Receive a sharded stack into scratch, then copy each member into place."""
        views = [self._buffers[member] for member in stack.members]
        shape = (len(stack.members), *self._local_shapes[stack.members[0]])
        elements = math.prod(shape)

        def pre(_base: Any) -> RefitCtx:
            scratch = self._scratch_for_current_stream()
            return RefitCtx(buf=scratch[:elements].view(shape))

        def post(ctx: RefitCtx) -> None:
            # Each copy_ enqueues on the current stream in order, so the
            # members land before whatever the stream runs next.
            for view, received in zip(views, ctx.buf.unbind(0), strict=True):
                view.copy_(received)

        return LocalParamSpec(base=None, pre=pre, post=post)

    def _check_engine_rank(self, generator_index: int | None) -> None:
        dst_mesh = self._plan.dst_mesh
        if generator_index is None or not 0 <= generator_index < dst_mesh.size:
            raise ValueError(
                "a sharded-destination plan needs this generator's index in "
                f"[0, {dst_mesh.size}), got {generator_index!r}"
            )
        engine_tp = dst_mesh.shape[-1]
        if engine_tp != self._tp_size:
            raise ValueError(
                f"the plan splits destinations over TP {engine_tp}, but this "
                f"engine runs TP {self._tp_size}"
            )
        if generator_index % engine_tp != self._tp_rank:
            raise ValueError(
                f"generator index {generator_index} is TP coordinate "
                f"{generator_index % engine_tp} in the destination mesh, but "
                f"this scheduler is TP rank {self._tp_rank}"
            )

    def _model_facts(self) -> SglangModelFacts:
        config = getattr(self._model, "config", None)
        if config is None:
            raise ValueError(
                "a sharded-destination plan needs the SGLang model's HF config"
            )
        return SglangModelFacts.from_config(config)

    def _resolve_view(self, name: str) -> Any:
        entry = self._plan.entry(name)
        try:
            return _engine_view(
                self._model, name, entry.global_shape, self._tp_rank, self._tp_size
            )
        except (_NotProvable, AttributeError) as error:
            raise ValueError(
                f"{name}: cannot prove this TP rank's slice of SGLang's live "
                f"storage, so the sharded destination is refused: {error}"
            ) from error

    @property
    def source_partition_count(self) -> int:
        return self._plan.source_partition_count

    @property
    def device(self) -> str:
        return self._device

    @property
    def poisoned(self) -> bool:
        return self._poisoned

    @property
    def layer_groups(self) -> tuple[tuple[str, ...], ...]:
        return tuple(tuple(group) for group in self._groups)

    def validate_topology(self, topology: CollectiveTopology) -> None:
        self._wire.validate_topology(topology)

    def _require_healthy(self) -> None:
        if self._poisoned:
            raise RuntimeError(
                "the SGLang loader is poisoned by a failed install; the live "
                "model may hold a partial weight set"
            )

    def _validate_stable(self) -> None:
        for name, buffer in self._buffers.items():
            entry = self._plan.entry(name)
            _check_stable(
                name,
                buffer,
                self._signatures[name],
                expected_shape=self._local_shapes[name],
                expected_dtype=entry.dtype,
            )
        for name, stacked in self._stack_buffers.items():
            _check_stable(
                name,
                stacked,
                self._stack_signatures[name],
                expected_shape=self._stacks[name].entry.global_shape,
                expected_dtype=self._stacks[name].entry.dtype,
            )
        if self._scratch_signature is not None and self._scratch_primary is not None:
            _check_stable(
                "stacked receive scratch",
                self._scratch_primary,
                self._scratch_signature,
                expected_shape=(self._scratch_elements,),
                expected_dtype="bfloat16",
            )
        for name, frozen in self._views.items():
            try:
                current = _view_signature(self._resolve_view(name))
            except ValueError as error:
                raise RuntimeError(str(error)) from error
            if current != frozen:
                raise RuntimeError(
                    f"{name}: SGLang moved or reshaped the live storage this "
                    f"receiver aliases (address, shape, stride {frozen} -> "
                    f"{current})"
                )

    # --- Loader protocol -------------------------------------------------

    def capture(self) -> ReshardPlan:
        self._validate_stable()
        return self._wire.capture()

    def parameter_names(self) -> list[str]:
        return self._wire.names()

    def local_params(self) -> dict[str, LocalParamSpec]:
        self._require_healthy()
        self._validate_stable()
        specs: dict[str, LocalParamSpec] = {}
        for name in self._wire.names():
            stack = self._stacks.get(name)
            if stack is None:
                specs[name] = LocalParamSpec(base=self._buffers[name])
            elif name in self._stack_buffers:
                specs[name] = LocalParamSpec(base=self._stack_buffers[name])
            else:
                specs[name] = self._scratch_spec(stack)
        return specs

    def _staged_members(self, layer_group_id: int) -> list[str]:
        """The per-tensor names of a group that install reads from a receive buffer.

        A group names wire entries; a stacked entry stands for its members.
        Aliased members already landed in the live storage during receive.
        """
        names: list[str] = []
        for wire_name in self._groups[layer_group_id]:
            stack = self._stacks.get(wire_name)
            members = stack.members if stack is not None else (wire_name,)
            names.extend(member for member in members if member not in self._views)
        return names

    def start_new_round(self, version: str) -> None:
        self._require_healthy()
        version = _text(version, "version")
        if self._round_version is not None:
            raise RuntimeError(
                f"a round for version {self._round_version!r} is already in flight"
            )
        self._validate_stable()
        self._round_version = version

    def install(self, layer_group_id: int) -> None:
        """Hand one layer group's received tensors to SGLang's own loader."""
        self._require_healthy()
        if self._round_version is None:
            raise RuntimeError("start_new_round must run before install")
        if not 0 <= layer_group_id < len(self._groups):
            raise ValueError(
                f"layer_group_id {layer_group_id} is outside the "
                f"{len(self._groups)} declared layer groups"
            )
        names = self._staged_members(layer_group_id)
        if not names:
            return
        try:
            self._model.load_weights([(name, self._buffers[name]) for name in names])
        except BaseException:
            self._poisoned = True
            raise

    def finish(self) -> None:
        self._require_healthy()
        if self._round_version is None:
            raise RuntimeError("start_new_round must run before finish")
        self._round_version = None

    def fail_round(self, *, possibly_mutated: bool) -> None:
        """Retire a failed round; a round that may have written the model poisons."""
        if possibly_mutated:
            self._poisoned = True
        self._round_version = None

    def cleanup(self) -> None:
        self._round_version = None
        self._buffers.clear()
        self._stack_buffers.clear()
        self._scratch.clear()
        self._scratch_primary = None
        self._views.clear()


class SglangGeneratorSession:
    """Own one generator rank's collective membership and round lifecycle.

    The session closes its rendezvous on every teardown path; the receiver's
    teardown (``close_generator_resources``) closes it again as the final
    owner, which is safe because ``CollectiveRendezvous.close()`` is
    idempotent.
    """

    def __init__(
        self,
        *,
        client: RefitClientGenerator,
        rendezvous: CollectiveRendezvous,
        loader: SglangLoader,
        worker_id: str,
        device: Any = None,
        streams: list[Any] | None = None,
    ) -> None:
        self._client = client
        self._rendezvous = rendezvous
        self._loader = loader
        self._worker_id = _text(worker_id, "worker_id")
        self._groups = [list(group) for group in loader.layer_groups]
        self._device = device
        self._streams = streams
        self._membership: Membership | None = None
        self._prepared = False
        self._closed = False

    @classmethod
    def create(
        cls,
        *,
        rendezvous: CollectiveRendezvous,
        topology: CollectiveTopology,
        loader: SglangLoader,
        slot_id: str,
        worker_id: str,
        index_in_role: int,
        device: Any = None,
        streams: list[Any] | None = None,
    ) -> SglangGeneratorSession:
        loader.validate_topology(topology)
        device = _client_device(device, loader.device, "SGLang generator")
        streams = _collective_streams(streams, device=device)
        client = RefitClientGenerator(
            rendezvous=rendezvous,
            model_name=topology.model_name,
            trainer_slots=list(topology.trainer_slots),
            generator_slots=list(topology.generator_slots),
            source_partition_count=topology.source_partition_count,
            slot_id=_text(slot_id, "slot_id"),
            worker_id=_text(worker_id, "worker_id"),
            index_in_role=index_in_role,
            receiver_protocol=topology.receiver_protocol,
            m2n_abi_version=topology.m2n_abi_version,
            device=device,
            streams=streams,
        )
        return cls(
            client=client,
            rendezvous=rendezvous,
            loader=loader,
            worker_id=worker_id,
            device=device,
            streams=streams,
        )

    @property
    def membership(self) -> Membership:
        if self._membership is None:
            raise RuntimeError("prepare must complete before reading membership")
        return self._membership

    def prepare(self) -> Membership:
        if self._closed:
            raise RuntimeError("the SGLang generator session is closed")
        if self._prepared and self._membership is not None:
            return self._membership
        try:
            self._client.initialize(self._loader)
            self._client.setup_layer_groups(self._groups)
            self._membership = self._client.compute_plan()
            self._prepared = True
            return self._membership
        except BaseException:
            self.close()
            raise

    def run_round(self, *, version: str, operation_id: str) -> None:
        if not self._prepared or self._membership is None:
            raise RuntimeError("prepare must complete before a generator round")
        if self._closed:
            raise RuntimeError("the SGLang generator session is closed")
        version = _text(version, "version")
        operation_id = _text(operation_id, "operation_id")
        update_attempted = False
        try:
            _order_current_cuda_stream_before(self._streams, device=self._device)
            self._client.start_weight_update(version)
            for layer_group_id in range(len(self._groups)):
                update_attempted = True
                self._client.update_weights(version, layer_group_id)
            self._client.finish_weight_update(version, operation_id=operation_id)
        except BaseException as error:
            self._loader.fail_round(possibly_mutated=update_attempted)
            self._fail_round(error, operation_id=operation_id)
            raise

    def _fail_round(self, error: BaseException, *, operation_id: str) -> None:
        # The round failure itself propagates; this is the teardown path. The
        # failure report is best effort: when finish_weight_update already
        # reported it, the server keeps the first terminal result.
        membership = self._membership
        try:
            self._client.cleanup()
        except BaseException:
            logger.warning(
                "generator cleanup failed after the round error", exc_info=True
            )
        if membership is not None:
            try:
                self._rendezvous.report(
                    operation_id=operation_id,
                    group_id=membership.group_id,
                    epoch=membership.epoch,
                    worker_id=self._worker_id,
                    succeeded=False,
                    message=repr(error),
                )
            except BaseException:
                logger.warning(
                    "reporting the generator round failure also failed",
                    exc_info=True,
                )
        try:
            self._rendezvous.close()
        except BaseException:
            logger.warning(
                "closing generator rendezvous after the round error failed",
                exc_info=True,
            )
        self._closed = True

    def close(self) -> None:
        if self._closed:
            return
        # One-shot, like the trainer session: a failed teardown must not rerun.
        self._closed = True
        try:
            self._client.cleanup()
        finally:
            self._rendezvous.close()


__all__ = [
    "GeneratorLoader",
    "SglangGeneratorSession",
    "SglangLoader",
    "build_generator_loader",
    "close_generator_resources",
]


@dataclass(frozen=True)
class GeneratorLoader:
    """A generator rank's loader with the session inputs derived from its plan."""

    loader: SglangLoader
    slot_id: str
    local_index: int


def build_generator_loader(
    *,
    plan: ReshardPlan,
    topology: CollectiveTopology,
    model: Any,
    device: Any,
    generator_slot_offset: int,
    tp_rank: int,
    tp_size: int,
) -> GeneratorLoader:
    """Build one SGLang scheduler's loader and its session inputs from a plan.

    Used by the public receiver factory; it owns the slot and the layer
    groups. An engine's generator slots are its TP ranks in order, so
    ``local_index`` is also this scheduler's coordinate in a sharded
    destination mesh.
    """
    local_index = generator_slot_offset + tp_rank
    try:
        slot_id = topology.generator_slots[local_index]
    except IndexError as error:
        raise RuntimeError(
            "the explicit generator topology does not contain generator index "
            f"{local_index}"
        ) from error

    # Both boundaries require canonical plan order; contiguous trainer groups
    # and receiver singletons therefore traverse the same per-lane sequence.
    # A stacking plan (MX_NCCL_REFIT_STACK_BYTES on the trainer) carries its
    # stacks in group_key; each wire entry is then one publish group.
    wire_plan, stacks = _derive_wire_plan(plan)
    if stacks:
        publish_groups = tuple((entry.name,) for entry in wire_plan.bulk)
    else:
        publish_groups = tuple((entry.name,) for entry in plan.bulk)
    loader = SglangLoader(
        plan=plan,
        model=model,
        device=device,
        layer_groups=publish_groups,
        generator_index=local_index,
        tp_rank=tp_rank,
        tp_size=tp_size,
    )
    return GeneratorLoader(
        loader=loader,
        slot_id=slot_id,
        local_index=local_index,
    )


def close_generator_resources(
    session: SglangGeneratorSession | None,
    rendezvous: CollectiveRendezvous | None,
    channel: Any,
    *,
    suppress_errors: bool,
) -> None:
    """Close a generator's session, rendezvous and channel, in that order.

    Every resource is attempted. The first failure propagates after the rest
    settle unless ``suppress_errors`` is set; each failure is logged.
    """
    first_error = None
    for name, resource in (
        ("session", session),
        ("rendezvous", rendezvous),
        ("channel", channel),
    ):
        if resource is None:
            continue
        try:
            resource.close()
        except BaseException as error:
            if first_error is None:
                first_error = error
            logger.warning(
                "closing ModelExpress collective %s failed", name, exc_info=True
            )
    if first_error is not None and not suppress_errors:
        raise first_error
