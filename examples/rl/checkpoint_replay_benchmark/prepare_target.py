# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Create a changed BF16 checkpoint fixture in owned scratch, never in shared cache."""

import argparse
import json
import mmap
import shutil
import struct
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=Path, required=True)
parser.add_argument("--target", type=Path, required=True)
args = parser.parse_args()
source = args.model.resolve()
target = args.target.resolve()
if not source.is_dir() or not list(source.glob("*.safetensors")):
    parser.error("--model must contain local safetensors shards")
if target == source or source in target.parents:
    parser.error("--target must be separate from the source directory")
target.mkdir(exist_ok=False)
changed_tensors = changed_bytes = 0
for p in sorted(source.iterdir()):
    if not p.is_file():
        continue
    out = target / p.name
    shutil.copy2(p, out)
    if p.suffix != ".safetensors":
        continue
    with out.open("r+b") as f:
        size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(size))
        with mmap.mmap(f.fileno(), 0) as mm:
            for name, meta in header.items():
                if name == "__metadata__":
                    continue
                if meta["dtype"] != "BF16":
                    raise RuntimeError((name, meta["dtype"]))
                begin, end = meta["data_offsets"]
                count = min(end - begin, 1024)
                for offset in range(8 + size + begin, 8 + size + begin + count, 2):
                    mm[offset] = mm[offset] ^ 1
                    changed_bytes += 1
                changed_tensors += 1
            mm.flush()
print(
    json.dumps(
        {
            "fixture": "synthetic low-mantissa-bit changes; not optimizer updates",
            "changed_tensors": changed_tensors,
            "changed_bytes": changed_bytes,
        }
    ),
    flush=True,
)
(target / "fixture-receipt.json").write_text(
    json.dumps(
        {
            "changed_tensors": changed_tensors,
            "changed_bytes": changed_bytes,
            "source": str(source),
            "synthetic": True,
        },
        indent=2,
    )
)
