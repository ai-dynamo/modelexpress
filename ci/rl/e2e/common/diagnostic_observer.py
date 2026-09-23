"""Read-only diagnostics; no changes to loader decisions or tensor contents."""

import contextlib
import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path

import torch
from torch.utils._python_dispatch import TorchDispatchMode

ROOT = Path("/refit/benchmark")
COUNTS = {}


def emit(kind, rank, data):
    row = {"kind": kind, "rank": rank, "pid": os.getpid(), "time": time.time(), **data}
    with (ROOT / f"diagnostics-rank{rank}.jsonl").open("a") as f:
        f.write(json.dumps(row) + "\n")
    print("MX_DIAG " + json.dumps(row), flush=True)


def tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from tensors(v)
    elif isinstance(value, (tuple, list)):
        for v in value:
            yield from tensors(v)


def sample(worker, names):
    from vllm.model_executor.model_loader.reload.layerwise import LAYERWISE_INFO

    started = time.perf_counter()
    layers = []
    global_storage = {}
    logical = 0
    calls = 0
    for layer, info in list(LAYERWISE_INFO.items()):
        loaded = list(info.loaded_weights)
        if not loaded:
            continue
        local_storage = {}
        params = {}
        layer_logical = 0
        for param_name, bound in loaded:
            item = params.setdefault(
                param_name, {"calls": 0, "cpu_logical_bytes": 0, "input_examples": []}
            )
            item["calls"] += 1
            target = bound.arguments.get("param")
            if isinstance(target, torch.Tensor):
                item["target_shape"] = list(target.shape)
                item["target_dtype"] = str(target.dtype)
                item["target_numel"] = target.numel()
                if id(target) in COUNTS:
                    item["observed_counts"] = dict(COUNTS[id(target)])
            for arg_name, value in bound.arguments.items():
                for t in tensors(value):
                    if t.device.type != "cpu" or not t.numel():
                        continue
                    n = t.numel() * t.element_size()
                    st = t.untyped_storage()
                    key = int(st._cdata)
                    size = st.nbytes()
                    global_storage[key] = size
                    local_storage[key] = size
                    logical += n
                    layer_logical += n
                    item["cpu_logical_bytes"] += n
                    example = {
                        "argument": arg_name,
                        "shape": list(t.shape),
                        "dtype": str(t.dtype),
                        "storage_bytes": size,
                    }
                    if (
                        len(item["input_examples"]) < 2
                        and example not in item["input_examples"]
                    ):
                        item["input_examples"].append(example)
        calls += len(loaded)
        layers.append(
            {
                "name": names.get(id(layer), type(layer).__name__),
                "type": type(layer).__name__,
                "cached_calls": len(loaded),
                "loaded_numel": info.load_numel,
                "required_numel": info.load_numel_total,
                "can_process": (
                    info.can_process()
                    if hasattr(info, "can_process")
                    else info.load_numel_total is not None
                    and info.load_numel >= info.load_numel_total
                ),
                "cpu_logical_bytes": layer_logical,
                "cpu_unique_storage_bytes": sum(local_storage.values()),
                "parameters": params,
                "restore_metadata": {
                    kind: {
                        n: {
                            "shape": list(t.shape),
                            "dtype": str(t.dtype),
                            "numel": t.numel(),
                        }
                        for n, t in group.items()
                    }
                    for kind, group in zip(
                        ["parameters", "buffers"], info.restore_metadata
                    )
                },
            }
        )
    layers.sort(key=lambda x: x["cpu_unique_storage_bytes"], reverse=True)
    status = dict(
        x.split(":", 1)
        for x in Path("/proc/self/status").read_text().splitlines()
        if ":" in x
    )
    return {
        "sample_seconds": time.perf_counter() - started,
        "pending_layers": len(layers),
        "cached_calls": calls,
        "cpu_logical_bytes": logical,
        "cpu_unique_storage_bytes": sum(global_storage.values()),
        "cpu_unique_storages": len(global_storage),
        "rss": {
            k: status.get(k, "").strip()
            for k in ["VmRSS", "RssAnon", "RssFile", "RssShmem"]
        },
        "cuda_allocated_bytes": torch.cuda.memory_allocated(worker.device),
        "cuda_reserved_bytes": torch.cuda.memory_reserved(worker.device),
        "layers": layers,
    }


def start(worker):
    names = {id(m): n for n, m in worker.model_runner.get_model().named_modules()}
    stop = threading.Event()
    rank = worker.rank

    def loop():
        sequence = 0
        while not stop.is_set():
            try:
                emit("layer_snapshot", rank, sample(worker, names))
                if sequence % 6 == 0:
                    # Only formatted stack text survives this iteration; retain no frame objects.
                    stacks = {
                        str(tid): traceback.format_list(
                            traceback.extract_stack(frame, limit=24)
                        )
                        for tid, frame in sys._current_frames().items()
                        if tid != threading.get_ident()
                    }
                    emit("python_stacks", rank, {"stacks": stacks})
                sequence += 1
            except Exception as e:  # noqa: BLE001 -- Preserve runtime failure evidence.
                emit("observer_error", rank, {"error": repr(e)})
            stop.wait(10)

    thread = threading.Thread(target=loop, name="mx-read-only-observer", daemon=True)
    thread.start()
    return stop


class AllocationTrace(TorchDispatchMode):
    """Record large CPU allocations on rank zero, preserving each operation."""

    def __init__(self, rank):
        super().__init__()
        self.rank = rank
        self.groups = {}
        self.last_emit = time.monotonic()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        name = str(func)
        if any(
            s in name
            for s in [
                "clone",
                "_to_copy",
                "empty",
                "zeros",
                "ones",
                "cat",
                "stack",
                "resize",
            ]
        ):
            try:
                outputs = [
                    t
                    for t in tensors(result)
                    if t.device.type == "cpu"
                    and t.numel() * t.element_size() >= 1024**2
                ]
                if outputs:
                    arg_storages = {
                        int(t.untyped_storage()._cdata)
                        for t in tensors(args)
                        if t.device.type == "cpu"
                    }
                    for t in outputs:
                        st = t.untyped_storage()
                        if int(st._cdata) in arg_storages:
                            continue
                        g = self.groups.setdefault(
                            name, {"count": 0, "bytes": 0, "examples": []}
                        )
                        g["count"] += 1
                        g["bytes"] += st.nbytes()
                        if len(g["examples"]) < 3:
                            g["examples"].append(
                                {
                                    "shape": list(t.shape),
                                    "dtype": str(t.dtype),
                                    "bytes": st.nbytes(),
                                    "stack": traceback.format_list(
                                        traceback.extract_stack(limit=24)[:-1]
                                    ),
                                }
                            )
                    if time.monotonic() - self.last_emit > 10:
                        self.flush()
            except Exception as e:  # noqa: BLE001 -- Preserve runtime failure evidence.
                emit("allocation_observer_error", self.rank, {"error": repr(e)})
        return result

    def flush(self):
        emit("cpu_allocations_cumulative", self.rank, {"operations": self.groups})
        self.last_emit = time.monotonic()


@contextlib.contextmanager
def allocation_trace(rank):
    if rank != 0:
        yield
        return
    from vllm.model_executor.model_loader.reload import layerwise

    original = layerwise.get_numel_loaded

    def counted(weight_loader, args):
        result = original(weight_loader, args)
        target = args.arguments.get("param")
        if isinstance(target, torch.Tensor):
            row = COUNTS.setdefault(
                id(target), {"calls": 0, "loaded_numel": 0, "zero_count_calls": 0}
            )
            row["calls"] += 1
            row["loaded_numel"] += result[0]
            row["zero_count_calls"] += int(result[0] == 0)
        return result

    trace = AllocationTrace(rank)
    layerwise.get_numel_loaded = counted
    try:
        with trace:
            yield
    finally:
        layerwise.get_numel_loaded = original
        trace.flush()
