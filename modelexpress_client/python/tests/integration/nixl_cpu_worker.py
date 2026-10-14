# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real NIXL workers. Control messages use pipes, payloads only use NIXL."""

import json
import math
import os
import sys
import traceback
from time import perf_counter

import torch
from nixl._bindings import nixlNotFoundError

import modelexpress
from modelexpress.nixl_transfer import NixlTransferManager
from modelexpress.refit.reshard.geometry import capture_geometry
from modelexpress.refit.reshard.slice_plan import Shard
from modelexpress.refit.reshard import transfer_plan
from modelexpress.refit.reshard.transfer_plan import SourceInfo
from modelexpress.refit.reshard.transport import NixlReshardTransport


class CpuBackend:
    """The manager's supported accelerator seam with synchronous CPU semantics."""

    name = "cpu-test"
    torch_device_type = "cpu"
    nixl_mem_type = "DRAM"

    def set_device(self, device_id):
        pass

    def synchronize(self, device_id=None):
        pass


def weights(config, version):
    dtype = getattr(torch, config["dtype"])
    shapes = {"col": (12, 5), "row": (5, 12), "q": (12, 5),
              "k": (12, 5), "v": (12, 5), "gate": (12, 5),
              "up": (12, 5), "expert0": (12, 5), "expert1": (12, 5)}
    if config.get("quantized"):
        # Two signed 4-bit values per byte, with an independent per-row scale.
        packed_version = min(version, 1)
        packed = ((torch.arange(60).reshape(12, 5) + packed_version * 37) % 256).to(torch.uint8)
        scales = (torch.arange(12).reshape(12, 1) + version * 16 + 1).float() / 16
        return {"packed": packed, "scale": scales}
    return {
        name: ((1 + torch.arange(math.prod(shape)).reshape(shape) / 64)
               * 2 ** (index + version * 9 - 12)).to(dtype)
        for index, (name, shape) in enumerate(shapes.items())
    }


class Model(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        rank, count = config["rank"], config["receivers"]
        self.start, self.size = rank * (12 // count), 12 // count
        self.guards = {}
        for name, full in weights(config, 0).items():
            if name in ("k", "v", "up"):
                continue
            target = {"q": "qkv", "gate": "gate_up"}.get(name, name)
            shape = (5, self.size) if name == "row" else (self.size, full.shape[1])
            if name in ("q", "gate"):
                shape = (shape[0] * (3 if name == "q" else 2), shape[1])
            # Guard unused storage around the logical parameter view.
            storage = torch.full((torch.Size(shape).numel() + 2,), 251, dtype=full.dtype)
            self.guards[target] = storage
            offset = config.get("storage_offset", 0)
            parameter = torch.nn.Parameter(storage[offset:offset + torch.Size(shape).numel()].view(shape), requires_grad=False)
            parameter.weight_loader = self.load_parameter
            self.register_parameter(target, parameter)

    def load_weights(self, items):
        for name, value in items:
            target = {"q": "qkv", "k": "qkv", "v": "qkv",
                      "gate": "gate_up", "up": "gate_up"}.get(name, name)
            parameter = getattr(self, target)
            parameter.weight_loader(parameter, value, name)

    def load_parameter(self, parameter, value, name):
        sliced = value.narrow(1 if name == "row" else 0, self.start, self.size)
        if name in ("q", "k", "v", "gate", "up"):
            slot = {"q": 0, "k": 1, "v": 2, "gate": 0, "up": 1}[name]
            parameter = parameter.narrow(0, slot * self.size, self.size)
        parameter.data.copy_(sliced)

    def output(self):
        if self.config.get("quantized"):
            low = (self.packed & 15).float() - 8
            high = (self.packed >> 4).float() - 8
            dense = torch.stack((low, high), dim=-1).flatten(1) * self.scale
            return dense @ torch.arange(1, 11, dtype=torch.float32)
        return torch.cat([p.float().flatten() for p in self.parameters()]) @ torch.arange(
            1, sum(p.numel() for p in self.parameters()) + 1, dtype=torch.float32
        )


def expected(config, version):
    """Independent slices/concatenation, never capture/planner output."""
    full = weights(config, version)
    start = config["rank"] * (12 // config["receivers"])
    end = start + 12 // config["receivers"]
    if config.get("quantized"):
        return {name: tensor[start:end].clone() for name, tensor in full.items()}
    result = {name: tensor[:, start:end].clone() if name == "row" else tensor[start:end].clone()
              for name, tensor in full.items() if name not in ("q", "k", "v", "gate", "up")}
    result["qkv"] = torch.cat([full[name][start:end] for name in ("q", "k", "v")])
    result["gate_up"] = torch.cat([full[name][start:end] for name in ("gate", "up")])
    return result


def emit(value):
    print("MX_TEST_RESULT " + json.dumps(value), flush=True)


def execute(plan, sources, destination, transport):
    started = perf_counter()
    # Only the public planner execution signature changed across the stack.
    if hasattr(transfer_plan, "bind_transfer"):
        def resolve_source(buffer):
            shard = next(shard for shard in sources[buffer.src_name].shards
                         if shard.session == buffer.source_slot_id
                         and tuple(shard.shard_offset) == buffer.shard_offset
                         and tuple(shard.shape) == buffer.shape)
            return shard.session, shard.addr
        physical = transfer_plan.bind_transfer(
            plan, resolve_source, lambda kind, name: destination[name].data_ptr()
        )
        bound = perf_counter()
        transfer_plan.execute_transfer(physical, transport)
        return {"binding_s": bound - started, "transfer_s": perf_counter() - bound}
    else:
        transfer_plan.execute_transfer(plan, lambda name: destination[name].data_ptr(), transport)
        return {"binding_s": None, "transfer_s": perf_counter() - started}


def main(role, config):
    manager = NixlTransferManager(agent_name=f"{role}-{config['rank']}-{os.getpid()}",
                                  device_id=0, accelerator_backend=CpuBackend())
    manager.initialize()
    registrations, buffers = [], {}
    model = None
    retained_plan = None
    selector = None
    if role == "receiver":
        model = Model(config)
        for parameter in model.parameters():
            registrations.append(manager.register_dram_buffer(parameter.view(torch.uint8).flatten()))
        with torch.device("meta"):
            twin = Model(config)
        capture = capture_geometry(twin, [(name, value.dtype, list(value.shape))
                                         for name, value in weights(config, 0).items()])
    try:
        for line in sys.stdin:
            request = json.loads(line)
            if request.get("stop"):
                break
            version = request["version"]
            if role == "source":
                if config.get("lifecycle") == "reallocate":
                    for registration in registrations:
                        manager.deregister_memory(registration)
                    registrations.clear()
                    buffers = {}
                entries = []
                for name, full in weights(config, version).items():
                    axis = 1 if name == "row" else 0
                    size = full.shape[axis] // config["publishers"]
                    offset = config["rank"] * size
                    shard = full.narrow(axis, offset, size).contiguous()
                    if name not in buffers:
                        buffers[name] = shard.clone()
                        registrations.append(manager.register_dram_buffer(buffers[name].view(torch.uint8).flatten()))
                    else:
                        buffers[name].copy_(shard)
                    offsets = [0] * full.ndim
                    offsets[axis] = offset
                    entries.append(dict(name=name, global_shape=list(full.shape),
                                        shape=list(shard.shape), offset=offsets,
                                        dtype=str(shard.dtype).removeprefix("torch."),
                                        address=buffers[name].data_ptr()))
                emit(dict(session=f"source-{config['rank']}-{config.get('replica', 'default')}", slot=f"source-{config['rank']}", metadata=manager.nixl_metadata.hex(), entries=entries))
                continue
            if config.get("replicas") and selector is None:
                from replica_selection import ReplicaSelector
                selector = ReplicaSelector(request["all_sources"])
            cycle_started = perf_counter()
            selection_s = None
            selected = request["sources"]
            if config.get("replicas"):
                selected = selector.select(version, request["selected_workers"])
                assert {item["session"] for item in selected} == set(request["selected_workers"])
                selection_s = perf_counter() - cycle_started
            prepare_started = perf_counter()
            sources, agents = {}, {}
            for published in selected:
                session = published["slot"] if config.get("replicas") else published["session"]
                agents[session] = manager.add_remote_agent(bytes.fromhex(published["metadata"]))
                for entry in published["entries"]:
                    dtype = getattr(torch, entry["dtype"])
                    item = sources.setdefault(entry["name"], SourceInfo(
                        tuple(entry["global_shape"]), dtype, torch.empty((), dtype=dtype).element_size(), []))
                    item.shards.append(Shard(tuple(entry["offset"]), tuple(entry["shape"]),
                                             session, entry["address"], item.elsize))
            if retained_plan is None or not hasattr(transfer_plan, "bind_transfer"):
                plan = transfer_plan.plan_transfer(capture, sources)
                if config.get("replicas"):
                    retained_plan = plan
            else:
                plan = retained_plan
            assert not plan.fallback, plan.fallback
            destination = dict(model.named_parameters())
            transport = NixlReshardTransport(manager, agents, {s: 0 for s in agents},
                                            mem_type="DRAM", local_mem_type="DRAM", timeout_seconds=5)
            if (config.get("retry") and version == 0) or request.get("retry"):
                bad = NixlReshardTransport(manager, {s: "absent-peer" for s in agents},
                                          {s: 0 for s in agents}, mem_type="DRAM", local_mem_type="DRAM", timeout_seconds=1)
                try:
                    execute(plan, sources, destination, bad)
                except nixlNotFoundError:
                    pass
                else:
                    raise AssertionError("an unknown peer was reported successful")
            if request.get("publisher_died"):
                try:
                    execute(plan, sources, destination, transport)
                except Exception as error:
                    if not isinstance(error, (RuntimeError, TimeoutError)) and not type(error).__module__.startswith("nixl"):
                        raise
                else:
                    raise AssertionError("dead publisher transfer reported success")
                emit(dict(validated=True, production=modelexpress.__file__))
                continue
            preparation_s = perf_counter() - prepare_started
            timing = execute(plan, sources, destination, transport)
            timing.update(selection_metadata_s=selection_s, preparation_s=preparation_s,
                          total_s=perf_counter() - cycle_started)
            reference = expected(config, version)
            for name, tensor in destination.items():
                if not torch.equal(tensor, reference[name]):
                    index = tuple(torch.nonzero(tensor != reference[name])[0].tolist())
                    raise AssertionError(
                        f"version={version} rank={config['rank']} tensor={name} index={index} "
                        f"actual={tensor[index].item()} expected={reference[name][index].item()}"
                    )
                offset = config.get("storage_offset", 0)
                assert torch.all(model.guards[name][:offset] == 251)
                assert torch.all(model.guards[name][offset + tensor.numel():] == 251)
            fresh = Model(config)
            fresh.load_weights(weights(config, version).items())
            torch.testing.assert_close(model.output(), fresh.output(), rtol=0, atol=0)
            emit(dict(validated=True, production=modelexpress.__file__, timing=timing))
    finally:
        if selector is not None:
            selector.close()
        for registration in registrations:
            manager.deregister_memory(registration)
        manager.shutdown()


if __name__ == "__main__":
    try:
        main(sys.argv[1], json.loads(sys.argv[2]))
    except Exception:
        emit({"error": traceback.format_exc()})
        raise
