# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dump weight-derived engine state so a source and a target can be diffed.

Inference output is a weak oracle for this bug class: a poisoned value cache
still produces fluent text for a while. The decisive check is byte equality
of the derived state itself between a source that loaded from disk and a
target that received over RDMA.

Install by putting this file on PYTHONPATH as ``sitecustomize.py``. It hooks
``MxModelLoader.load_model`` and, once the model is fully loaded, writes a
digest of every tensor the manifest historically could not carry:

  ``_attn_res_cw_cache[<dtype>]``   dict-held, read by SGLang's ``get_cw``
  ``_k3_fused_decode_args[<i>]``    tuple-held, fused KDA decode kernel inputs
  quant-method tensors              held on non-Module objects
  control parameters                a fixed sample, to prove the method itself

Both roles dump at the same logical point, so the files are directly
comparable with ``mx_compare_state.py``. A target whose entries match the
source is correct by construction; a mismatch localises to a named tensor
instead of a vague quality regression.

Env:
  MX_DUMP_ROLE   label written into the dump (e.g. "source", "target")
  MX_DUMP_OUT    output path, default /tmp/mx_state_dump.json
  MX_PROBE_OUT   after_rdma_receive telemetry, default /tmp/mx_receive_probe.jsonl
  MX_DUMP_CONTROLS  how many control parameters to digest, default 8
"""

import hashlib
import importlib.abc
import importlib.util
import json
import os
import sys

_ADAPTER = "modelexpress.engines.sglang.adapter"
_LOADER = "modelexpress.engines.sglang.loader"
_PROBE_OUT = os.environ.get("MX_PROBE_OUT", "/tmp/mx_receive_probe.jsonl")
_DUMP_OUT = os.environ.get("MX_DUMP_OUT", "/tmp/mx_state_dump.json")
_ROLE = os.environ.get("MX_DUMP_ROLE", "unset")
_N_CONTROLS = int(os.environ.get("MX_DUMP_CONTROLS", "8"))


def _log(msg):
    print(f"[mx-state-dump] {msg}", file=sys.stderr, flush=True)


def _digest(tensor):
    """sha256 over the tensor's exact bytes, plus shape and dtype."""
    import torch

    t = tensor.detach().cpu().contiguous()
    if t.dtype != torch.uint8:
        try:
            t = t.view(torch.uint8)
        except RuntimeError:
            t = t.to(torch.float32).view(torch.uint8)
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "sha256": hashlib.sha256(t.numpy().tobytes()).hexdigest(),
    }


def _collect(model):
    """Digest the derived state the manifest historically missed."""
    import torch

    entries = {}
    for name, module in model.named_modules():
        cache = getattr(module, "_attn_res_cw_cache", None)
        if isinstance(cache, dict):
            for dtype, tensor in cache.items():
                if isinstance(tensor, torch.Tensor):
                    entries[f"{name}._attn_res_cw_cache[{dtype}]"] = _digest(tensor)

        args = getattr(module, "_k3_fused_decode_args", None)
        if isinstance(args, (tuple, list)):
            for i, item in enumerate(args):
                if isinstance(item, torch.Tensor):
                    entries[f"{name}._k3_fused_decode_args[{i}]"] = _digest(item)

        qm = getattr(module, "quant_method", None)
        if qm is not None and not isinstance(qm, torch.nn.Module):
            for attr, val in list(vars(qm).items()):
                if isinstance(val, torch.Tensor) and val.numel():
                    entries[f"{name}.quant_method.{attr}"] = _digest(val)

    # Controls: a deterministic slice of ordinary parameters. These ride the
    # manifest either way, so a mismatch here means the transfer itself is
    # broken rather than the derived-state coverage.
    for name, param in sorted(model.named_parameters())[:_N_CONTROLS]:
        entries[f"[control] {name}"] = _digest(param.data)
    return entries


def _dump(model):
    entries = _collect(model)
    payload = {"role": _ROLE, "entries": entries}
    with open(_DUMP_OUT, "w") as handle:
        json.dump(payload, handle, indent=1, sort_keys=True)
    kinds = {
        "cw": sum(1 for k in entries if "_attn_res_cw_cache" in k),
        "fused_decode": sum(1 for k in entries if "_k3_fused_decode_args" in k),
        "quant_method": sum(1 for k in entries if ".quant_method." in k),
        "control": sum(1 for k in entries if k.startswith("[control]")),
    }
    _log(f"role={_ROLE} wrote {len(entries)} entries to {_DUMP_OUT} {kinds}")


def _install_loader(module):
    original = module.MxModelLoader.load_model

    def load_model(self, *args, **kwargs):
        out = original(self, *args, **kwargs)
        try:
            model = out if hasattr(out, "named_modules") else getattr(out, "model", None)
            if model is None:
                _log("loader returned no module tree; nothing dumped")
            else:
                _dump(model)
        except Exception as exc:
            _log(f"dump failed: {exc!r}")
        return out

    module.MxModelLoader.load_model = load_model
    _log("loader hook installed")


def _install_adapter(module):
    """Record what after_rdma_receive does, if anything overrides it."""
    original = module.SglangAdapter.after_rdma_receive

    def probed(self, result):
        before = {}
        try:
            before = {
                n: (t.data_ptr(), t.numel() * t.element_size())
                for n, t in self.discover_tensors(result).items()
            }
        except Exception:
            pass
        out = original(self, result)
        record = {"registered": len(before)}
        try:
            after = {n: t.data_ptr() for n, t in self.discover_tensors(out).items()}
            moved = [n for n, (a, _) in before.items() if n in after and after[n] != a]
            record["moved"] = len(moved)
            record["moved_names"] = sorted(moved)[:20]
            record["orphan_bytes"] = sum(before[n][1] for n in moved)
        except Exception as exc:
            record["error"] = repr(exc)
        line = json.dumps(record, sort_keys=True)
        print(f"[mx-receive-probe] {line}", file=sys.stderr, flush=True)
        try:
            with open(_PROBE_OUT, "a") as handle:
                handle.write(line + "\n")
        except OSError:
            pass
        return out

    module.SglangAdapter.after_rdma_receive = probed


class _Hook(importlib.abc.MetaPathFinder):
    """Patch each target module once, immediately after its body executes."""

    def __init__(self):
        self.pending = {_ADAPTER: _install_adapter, _LOADER: _install_loader}

    def find_spec(self, fullname, path=None, target=None):
        install = self.pending.get(fullname)
        if install is None:
            return None
        del self.pending[fullname]
        if not self.pending and self in sys.meta_path:
            sys.meta_path.remove(self)
        was_present = self in sys.meta_path
        if was_present:
            sys.meta_path.remove(self)
        spec = importlib.util.find_spec(fullname)
        if was_present:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return None
        original_exec = spec.loader.exec_module

        def exec_module(module):
            original_exec(module)
            try:
                install(module)
            except Exception as exc:
                _log(f"install failed for {fullname}: {exc!r}")

        spec.loader.exec_module = exec_module
        return spec


if os.environ.get("MX_PROBE", "1") != "0":
    sys.meta_path.insert(0, _Hook())
