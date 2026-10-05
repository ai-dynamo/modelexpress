"""Record package installation changes and reject replacement GPU libraries."""

import argparse
import importlib.metadata
import json
from pathlib import Path
import re

p = argparse.ArgumentParser()
p.add_argument("--output", type=Path, required=True)
p.add_argument("--compare", type=Path)
args = p.parse_args()
packages = {
    re.sub(r"[-_.]+", "-", dist.metadata["Name"]).lower(): dist.version
    for dist in importlib.metadata.distributions()
}
receipt = {"packages": packages}
if args.compare:
    before = json.loads(args.compare.read_text())["packages"]
    changes = {
        name: {"before": version, "after": packages.get(name)}
        for name, version in before.items()
        if packages.get(name) != version
    }
    additions = {
        name: version for name, version in packages.items() if name not in before
    }
    receipt.update(changes=changes, additions=additions)
    unexpected = set(changes) - {"modelexpress", "safetensors"}
    added_cuda12 = [name for name in additions if "cu12" in name or name == "nixl"]
    receipt.update(
        passed=not unexpected and not added_cuda12,
        unexpected_changes=sorted(unexpected),
        added_cuda12=added_cuda12,
    )
args.output.write_text(json.dumps(receipt, indent=2) + "\n")
if args.compare:
    assert receipt["passed"], receipt
