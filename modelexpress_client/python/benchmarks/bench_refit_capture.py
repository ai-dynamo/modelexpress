# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU/meta capture benchmark; not a vLLM or end-to-end refit measurement.

Run from modelexpress_client/python with its dependencies installed:
    PYTHONPATH=. python benchmarks/bench_refit_capture.py

The synthetic loader rebuilds a model-wide parameter dictionary on every call.
Both arms capture the same sources into the same destination layouts. The serial
reference keeps one stamp-installation scope, matching the previous capture path.
"""

import argparse
import json
import statistics
import time

import torch
from modelexpress.refit.reshard import geometry


class LookupModel(torch.nn.Module):
    def __init__(self, parameters):
        super().__init__()
        for index in range(parameters):
            self.register_parameter(
                f"p{index}", torch.nn.Parameter(torch.empty(4, device="meta"))
            )
        self.load_calls = 0

    def load_weights(self, weights):
        self.load_calls += 1
        params = dict(self.named_parameters())
        for name, tensor in weights:
            param = params[name.split("/", 1)[0]]
            param.weight_loader(param, tensor)


def default_loader(param, tensor):
    param.data.copy_(tensor)


def serial_capture(model, weights):
    recorder = geometry._shared_recorder(weights)
    saved = geometry._install_stamps(model, recorder, default_loader)
    unsupported, reasons = [], {}
    try:
        for name, tensor in weights.items():
            try:
                model.load_weights([(name, tensor)])
            except geometry.UnsupportedReshard as error:
                source = getattr(tensor, "_name", name)
                unsupported.append(source)
                reasons[source] = str(error)
    finally:
        geometry._restore_stamps(saved)
    return geometry.CaptureResult(
        copies=recorder.copies,
        unsupported=unsupported,
        unattributed=recorder.unattributed,
        unsupported_reasons=reasons,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=int, default=4096)
    parser.add_argument("--parameters", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if min(args.sources, args.parameters, args.repeats) < 1:
        parser.error("sources, parameters, and repeats must be positive")
    manifest = [
        (f"p{i % args.parameters}/source{i}", torch.float32, (4,))
        for i in range(args.sources)
    ]
    records = {"serial": [], "bulk": []}
    expected = None
    for repeat in range(args.repeats):
        # Alternate order to avoid assigning all warm process state to one arm.
        for arm in ("serial", "bulk") if repeat % 2 == 0 else ("bulk", "serial"):
            model = LookupModel(args.parameters)
            weights = geometry.build_lazy_weights(manifest)
            started = time.perf_counter()
            result = (
                serial_capture(model, weights)
                if arm == "serial"
                else geometry.capture_weights(model, weights, default_loader)
            )
            elapsed = time.perf_counter() - started
            if expected is None:
                expected = result
            if result != expected or result.unsupported or result.unattributed:
                raise RuntimeError("Capture arms differ or contain incomplete geometry")
            records[arm].append(
                {
                    "seconds": elapsed,
                    "loader_calls": model.load_calls,
                    "copies": len(result.copies),
                }
            )
    medians = {
        arm: statistics.median(r["seconds"] for r in rows)
        for arm, rows in records.items()
    }
    print(
        json.dumps(
            {
                "scope": "synthetic CPU/meta layout capture only; no vLLM, GPU, or transfer",
                "torch": torch.__version__,
                "sources": args.sources,
                "parameters": args.parameters,
                "identical_capture": True,
                "samples": records,
                "median_seconds": medians,
                "median_speedup": medians["serial"] / medians["bulk"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
