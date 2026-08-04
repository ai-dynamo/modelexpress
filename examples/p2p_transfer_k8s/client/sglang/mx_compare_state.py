#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Diff two mx_receive_probe state dumps: source (disk load) vs target (RDMA).

    mx_compare_state.py source.json target.json

The source loaded from disk, so its derived state is correct by
construction. Every entry the target holds must be byte-identical. A
mismatch names the exact tensor rather than showing up as degraded output
several thousand tokens later.

Entries are grouped so the verdict is readable on a 92-layer model:

  cw             `_attn_res_cw_cache` -- the state PR #585 re-derives and
                 PR #589 transfers; a poisoned target differs here
  fused_decode   `_k3_fused_decode_args` -- the second container-held case,
                 which has no torch reference implementation to fall back on
  quant_method   tensors on non-Module objects
  control        ordinary parameters; these ride the manifest either way, so
                 a mismatch means the transfer itself is broken

Exit status is 1 on any mismatch or missing entry.
"""

import json
import sys


def _group(name):
    if name.startswith("[control]"):
        return "control"
    if "_attn_res_cw_cache" in name:
        return "cw"
    if "_k3_fused_decode_args" in name:
        return "fused_decode"
    if ".quant_method." in name:
        return "quant_method"
    return "other"


def main(src_path, tgt_path):
    src = json.load(open(src_path))
    tgt = json.load(open(tgt_path))
    se, te = src["entries"], tgt["entries"]
    print(f"source role={src.get('role')} entries={len(se)}")
    print(f"target role={tgt.get('role')} entries={len(te)}")

    groups = {}
    for name in sorted(set(se) | set(te)):
        g = groups.setdefault(
            _group(name), {"match": 0, "differ": [], "src_only": [], "tgt_only": []}
        )
        if name not in te:
            g["src_only"].append(name)
        elif name not in se:
            g["tgt_only"].append(name)
        elif se[name] == te[name]:
            g["match"] += 1
        else:
            g["differ"].append(name)

    print()
    print(f"{'group':<14}{'match':>7}{'DIFFER':>8}{'src-only':>10}{'tgt-only':>10}")
    print("-" * 49)
    bad = 0
    for name in ("cw", "fused_decode", "quant_method", "other", "control"):
        g = groups.get(name)
        if not g:
            continue
        bad += len(g["differ"]) + len(g["src_only"]) + len(g["tgt_only"])
        print(
            f"{name:<14}{g['match']:>7}{len(g['differ']):>8}"
            f"{len(g['src_only']):>10}{len(g['tgt_only']):>10}"
        )

    for name, g in groups.items():
        for entry in g["differ"][:10]:
            print(f"\nDIFFER {entry}")
            print(f"  source {se[entry]}")
            print(f"  target {te[entry]}")
        if len(g["differ"]) > 10:
            print(f"  ... and {len(g['differ']) - 10} more in {name}")
        for entry in (g["src_only"] + g["tgt_only"])[:10]:
            side = "source" if entry in se else "target"
            print(f"MISSING on the other side ({side} only): {entry}")

    print()
    if bad:
        print(f"VERDICT: FAIL -- {bad} entries differ or are missing")
    elif not any(groups.get(k, {}).get("match") for k in ("cw", "fused_decode")):
        print(
            "VERDICT: INCONCLUSIVE -- no cw or fused_decode entries were "
            "dumped; this model has no container-held derived state"
        )
    else:
        print("VERDICT: PASS -- every derived-state entry is byte-identical")
    return 1 if bad else 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))
