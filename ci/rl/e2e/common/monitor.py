import importlib.metadata as md
import json
import time
from pathlib import Path


def version(name):
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


root = Path("/refit/benchmark")
root.mkdir(exist_ok=True)
(root / "versions.json").write_text(
    json.dumps(
        {
            d: version(d)
            for d in [
                "vllm",
                "torch",
                "modelexpress",
                "transformers",
                "nixl",
                "nixl-cu13",
                "runai-model-streamer",
            ]
        },
        indent=2,
    )
)
with (root / "host-memory.jsonl").open("w", buffering=1) as f:
    while True:
        r = {"time": time.time()}
        for k in [
            "memory.current",
            "memory.peak",
            "memory.max",
            "memory.events",
            "memory.stat",
        ]:
            p = Path("/sys/fs/cgroup") / k
            if p.exists():
                r[k] = p.read_text()
        f.write(json.dumps(r) + "\n")
        print("HOST_MEMORY " + json.dumps(r), flush=True)
        time.sleep(1)
