# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generator-side engine boundary for a live vLLM engine.

Installed as vLLM's ``worker_extension_cls``, so every method here runs inside
a real vLLM worker process with ``self.model_runner.model`` being the engine's
own tensor-parallel model. The ModelExpress client is driven from there, which
is the only place a generator rank can be: the destination storage is the
engine's, and it exists nowhere else.

The split of labour is deliberate. This class owns exactly one framework fact
-- that vLLM's own ``load_weights`` knows how to fold a canonical checkpoint
tensor into its fused, tensor-parallel parameters -- and the plan therefore
delivers whole tensors. Re-deriving vLLM's fusion layout here would duplicate
the engine's loader and silently rot against it.
"""

from __future__ import annotations

import os
import time
from typing import Any

import torch

from modelexpress_rl.collective import (
    LocalParamSpec,
    RefitClientGenerator,
    RefitCtx,
)
from modelexpress_rl.collective.rendezvous import CollectiveRendezvous

from .plan_from_hf import build_plan, engine_placement


class VllmLoader:
    """``Loader`` over a live vLLM model.

    Received tensors land in a scratch buffer per canonical parameter, and one
    layer group at a time is handed to the engine's loader. Scratch is kept
    between rounds: a refit loop runs many times and re-allocating the same
    buffers every round would measure the allocator, not the refit.
    """

    def __init__(
        self,
        model: Any,
        plan,
        groupings: list[list[str]],
        device: torch.device,
        *,
        tp_size: int = 1,
        dst_layout: str = "replicate",
    ) -> None:
        self._model = model
        self._plan = plan
        self._groupings = groupings
        self._device = device
        self._specs: dict[str, LocalParamSpec] = {}
        self._staged: dict[int, list[str]] = {}
        self.installed_rounds = 0
        self.install_calls = 0
        self.direct = 0
        self.staged = 0

        dtypes = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        shapes = {entry.name: entry.global_shape for entry in plan.bulk}
        named = dict(model.named_parameters())
        group_of = {
            name: gid for gid, names in enumerate(groupings) for name in names
        }

        for entry in plan.bulk:
            target = None
            if dst_layout == "sharded" and engine_placement(
                entry.name, entry.global_shape, tp_size
            ) is not None:
                target = self._engine_view(named, shapes, entry.name, tp_size)
            if target is not None:
                # The wire op writes into the engine's live storage. No scratch,
                # no copy, and install() has nothing left to do for this one.
                self._specs[entry.name] = LocalParamSpec(base=target)
                self.direct += 1
                continue
            self._specs[entry.name] = LocalParamSpec(
                base=torch.empty(
                    entry.global_shape, dtype=dtypes[entry.dtype], device=device
                )
            )
            self._staged.setdefault(group_of[entry.name], []).append(entry.name)
            self.staged += 1

    @staticmethod
    def _engine_view(named: dict, shapes: dict, name: str, tp_size: int):
        """This rank's slice of ``name`` inside the engine's fused storage.

        Returns None when the slice cannot be established, which is the answer
        whenever the arithmetic is not provably right: a fused parameter whose
        constituents do not account for its local extent means the engine split
        it some other way -- replicated key/value heads, a padded vocabulary --
        and guessing would land wrong bytes with nothing erroring.
        """
        fused = {
            "q_proj.weight": ("qkv_proj.weight", ("q_proj.weight", "k_proj.weight", "v_proj.weight")),
            "k_proj.weight": ("qkv_proj.weight", ("q_proj.weight", "k_proj.weight", "v_proj.weight")),
            "v_proj.weight": ("qkv_proj.weight", ("q_proj.weight", "k_proj.weight", "v_proj.weight")),
            "gate_proj.weight": ("gate_up_proj.weight", ("gate_proj.weight", "up_proj.weight")),
            "up_proj.weight": ("gate_up_proj.weight", ("gate_proj.weight", "up_proj.weight")),
        }
        for suffix, (fused_suffix, members) in fused.items():
            if not name.endswith(suffix):
                continue
            prefix = name[: -len(suffix)]
            param = named.get(prefix + fused_suffix)
            if param is None:
                return None
            rows = []
            for member in members:
                global_shape = shapes.get(prefix + member)
                if global_shape is None or global_shape[0] % tp_size:
                    return None
                rows.append(global_shape[0] // tp_size)
            if sum(rows) != param.shape[0]:
                return None
            offset = sum(rows[: members.index(suffix)])
            extent = rows[members.index(suffix)]
            return param.data[offset : offset + extent]

        param = named.get(name)
        if param is None:
            return None
        expected = shapes[name]
        # An unfused parameter is delivered whole only if the engine's local
        # extents are exactly the declared split; a padded vocabulary is the
        # common way this fails.
        if len(param.shape) != len(expected):
            return None
        for axis, (local, declared) in enumerate(zip(param.shape, expected)):
            if local == declared:
                continue
            if local * tp_size != declared:
                return None
        return param.data

    # --- Loader protocol -------------------------------------------------

    def capture(self):
        return self._plan

    def parameter_names(self) -> list[str]:
        return self._plan.parameter_names()

    def local_params(self) -> dict[str, LocalParamSpec]:
        return self._specs

    def start_new_round(self, version: str) -> None:
        pass

    def install(self, layer_group_id: int) -> None:
        """Hand one layer group's received tensors to vLLM's own loader."""
        names = self._staged.get(layer_group_id, [])
        if not names:
            return
        loaded = self._model.load_weights(
            (name, self._specs[name].base) for name in names
        )
        self.install_calls += 1
        if loaded is not None and len(loaded) == 0 and names:
            raise RuntimeError(
                f"vLLM accepted none of the {len(names)} tensor(s) in layer group "
                f"{layer_group_id}; the canonical names do not match this engine"
            )

    def finish(self) -> None:
        self.installed_rounds += 1

    def cleanup(self) -> None:
        self._specs.clear()


class MxRefitWorker:
    """vLLM ``worker_extension_cls``: the refit control surface of one worker.

    Every method is invoked through ``LLM.collective_rpc`` and runs on all
    tensor-parallel workers at once, which is exactly the cohort the plan
    declares as the generator side.
    """

    def mx_describe(self) -> dict[str, Any]:
        model = self.model_runner.model
        params = list(model.named_parameters())
        return {
            "rank": self._mx_rank(),
            "device": str(next(model.parameters()).device),
            "params": len(params),
            "example": [name for name, _ in params[:3]],
        }

    def _mx_rank(self) -> int:
        from vllm.distributed.parallel_state import get_tensor_model_parallel_rank

        return get_tensor_model_parallel_rank()

    def mx_snapshot(self) -> dict[str, float]:
        """Checksum of the live weights, so a refit can be seen to change them."""
        model = self.model_runner.model
        total = 0.0
        for _, param in model.named_parameters():
            total += float(param.detach().float().abs().sum().item())
        return {"abs_sum": total}

    def mx_corrupt(self, seed: int = 0) -> bool:
        """Overwrite every live weight, so a later match cannot be a no-op.

        A refit that moved nothing is indistinguishable from a correct one
        unless the destination is known to have held something else first.
        """
        model = self.model_runner.model
        generator = torch.Generator(device="cpu").manual_seed(seed + self._mx_rank())
        with torch.no_grad():
            for _, param in model.named_parameters():
                noise = torch.randn(
                    param.shape, generator=generator, dtype=torch.float32
                )
                param.copy_(noise.to(param.dtype).to(param.device) * 0.02)
        torch.cuda.synchronize()
        return True

    def mx_join(
        self,
        *,
        model_dir: str,
        endpoint: str,
        trainers: int,
        generators: int,
        model_name: str,
        run_id: str,
        dst_layout: str = "replicate",
    ) -> dict[str, Any]:
        """Build the plan, construct the client, and join the refit group."""
        import grpc

        rank = self._mx_rank()
        device = next(self.model_runner.model.parameters()).device
        torch.cuda.set_device(device)

        plan, groupings = build_plan(
            model_dir,
            trainers=trainers,
            generators=generators,
            dst_layout=dst_layout,
        )
        loader = VllmLoader(
            self.model_runner.model,
            plan,
            groupings,
            device,
            tp_size=generators,
            dst_layout=dst_layout,
        )

        channel = grpc.insecure_channel(endpoint)
        grpc.channel_ready_future(channel).result(timeout=120)
        rendezvous = CollectiveRendezvous(channel)

        client = RefitClientGenerator(
            rendezvous=rendezvous,
            model_name=model_name,
            trainer_slots=[f"t{i}" for i in range(trainers)],
            generator_slots=[f"g{i}" for i in range(generators)],
            source_partition_count=1,
            slot_id=f"g{rank}",
            worker_id=f"g{rank}-{run_id}",
            index_in_role=rank,
            device=device,
            streams=[torch.cuda.Stream(device=device)],
        )
        client.setup_layer_groups(groupings)
        client.initialize(loader)

        started = time.perf_counter()
        membership = client.compute_plan()
        elapsed = time.perf_counter() - started

        self._mx_client = client
        self._mx_loader = loader
        self._mx_groupings = groupings
        return {
            "rank": rank,
            "group_id": membership.group_id,
            "epoch": membership.epoch,
            "bootstrap_s": elapsed,
            "bulk_params": len(plan.bulk),
            "layer_groups": len(groupings),
            "direct": loader.direct,
            "staged": loader.staged,
        }

    def mx_refit(self, version: str) -> dict[str, Any]:
        """One full refit round, timed on this worker."""
        client = self._mx_client
        timings: dict[str, float] = {}

        started = time.perf_counter()
        client.start_weight_update(version)
        timings["start_s"] = time.perf_counter() - started

        transfer_start = time.perf_counter()
        for group_id in range(len(self._mx_groupings)):
            client.update_weights(version, group_id)
        timings["transfer_s"] = time.perf_counter() - transfer_start

        finish_start = time.perf_counter()
        client.finish_weight_update(version)
        timings["finish_s"] = time.perf_counter() - finish_start

        torch.cuda.synchronize()
        timings["total_s"] = time.perf_counter() - started
        timings["rank"] = float(self._mx_rank())
        timings["install_calls"] = float(self._mx_loader.install_calls)
        return timings

    def mx_cleanup(self) -> bool:
        client = getattr(self, "_mx_client", None)
        if client is not None:
            client.cleanup()
            self._mx_client = None
        return True


def _mx_diff_against_checkpoint(self, path: str, top: int = 12) -> dict[str, Any]:
    """Which live parameters disagree with the checkpoint, and by how much.

    Generation is the right acceptance test but a poor diagnostic: greedy
    decoding diverges from a difference too small to see in the first tokens,
    so a FAIL localizes nothing. This compares every live parameter against
    what the engine's own loader produces from the checkpoint, which names the
    parameter instead.
    """
    import glob

    from safetensors.torch import load_file

    model = self.model_runner.model
    before = {name: param.detach().clone() for name, param in model.named_parameters()}
    device = next(model.parameters()).device
    for shard in sorted(glob.glob(path + "/*.safetensors")):
        state = load_file(shard)
        model.load_weights((name, tensor.to(device)) for name, tensor in state.items())

    diffs = []
    for name, param in model.named_parameters():
        delta = (param.detach().float() - before[name].float()).abs().max().item()
        if delta > 0:
            diffs.append((delta, name, tuple(param.shape)))
    diffs.sort(reverse=True)
    return {
        "rank": self._mx_rank(),
        "params": len(before),
        "differing": len(diffs),
        "worst": [(n, s, round(d, 6)) for d, n, s in diffs[:top]],
    }


MxRefitWorker.mx_diff_against_checkpoint = _mx_diff_against_checkpoint
