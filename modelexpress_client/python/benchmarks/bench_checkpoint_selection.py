# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only source staging benchmark; does not measure vLLM refit latency."""

import json
import statistics
import tempfile
import time
from pathlib import Path

import torch
from modelexpress_rl.inference.checkpoint_selection import (
    CheckpointChanges,
    CheckpointSelection,
    DependencyGroup,
    read_selected_tensors,
    select_checkpoint_sources,
)
from safetensors.torch import save_file


def main():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        weight_map = {f"w{i}": f"shard-{i}.safetensors" for i in range(64)}
        for name, filename in weight_map.items():
            save_file({name: torch.ones(256 * 1024)}, root / filename)
        selection = select_checkpoint_sources(
            CheckpointChanges("base", "target", frozenset({"w0", "w1"})),
            serving_version="base",
            target_version="target",
            checkpoint_names=frozenset(weight_map),
            groups=(
                DependencyGroup("independent", frozenset({"w0"})),
                DependencyGroup("fused", frozenset({"w1", "w2"})),
            ),
        )
        plans = {
            "full": CheckpointSelection(sources=frozenset(weight_map)),
            "selected": selection,
        }
        elapsed = {key: [] for key in plans}
        # Warm both paths, then alternate order to reduce ordering bias.
        for iteration in range(7):
            order = list(plans) if iteration % 2 else list(reversed(plans))
            for key in order:
                started = time.perf_counter()
                staged = read_selected_tensors(
                    plans[key], checkpoint=root, weight_map=weight_map
                )
                duration = time.perf_counter() - started
                assert set(staged) == plans[key].sources
                del staged
                if iteration:
                    elapsed[key].append(duration)
        print(
            json.dumps(
                {
                    "scope": "synthetic CPU checkpoint staging, warm filesystem cache",
                    "torch": torch.__version__,
                    "changed_tensors": 2,
                    "dependency_expanded_tensors": len(selection.sources),
                    "full_source_payload_bytes": 64 * 1024 * 1024,
                    "selected_source_payload_bytes": 3 * 1024 * 1024,
                    "full_shards_opened": 64,
                    "selected_shards_opened": 3,
                    "seconds": elapsed,
                    "median_seconds": {
                        k: statistics.median(v) for k, v in elapsed.items()
                    },
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
