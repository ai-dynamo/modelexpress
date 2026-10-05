"""Prepare a fresh Eos launch directory without submitting jobs."""

import argparse
import hashlib
from pathlib import Path
import re
import shlex
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--reservation", required=True)
    parser.add_argument("--gate", type=Path)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9-]{1,24}", args.run_prefix):
        parser.error("run-prefix must be 1-24 lowercase letters, digits or hyphens")
    assets = args.assets.resolve()
    gate = (args.gate or assets / "python-warm-final-20261002/gpu-gate-v1").resolve()
    baseline = assets / "runs/glm030-clean-v2"
    server = assets / "python-direct-validation-20261001/server-build"
    required = [
        baseline / "SHA256SUMS", gate / "evidence/validation.json",
        gate / "SHA256SUMS", server / "evidence/modelexpress-server",
        assets / "optimized-pr-completion-20260929-v1/runtime.sqsh",
        assets / "models", assets / "data",
    ]
    for path in required:
        if not path.exists():
            parser.error(f"required shared asset is missing: {path}")
    correct = args.run_prefix + "-correct"
    performance = args.run_prefix + "-clean"
    for name in (correct, performance):
        for parent in (assets / "runs", assets / "data/outputs", assets / "evidence"):
            if (parent / name).exists():
                parser.error(f"run ID already exists: {parent / name}")
    output = args.output.resolve()
    if output.exists():
        parser.error(f"output already exists: {output}")
    shutil.copytree(Path(__file__).parent / "followup", output)
    settings = {
        "TASK_ROOT": str(assets), "BASELINE": str(baseline), "GATE": str(gate),
        "SERVER_BUILD": str(server), "SBATCH_RESERVATION": args.reservation,
        "CORRECTNESS_RUN": correct, "PERFORMANCE_RUN": performance,
    }
    (output / "run.env").write_text("".join(
        f"export {key}={shlex.quote(value)}\n" for key, value in settings.items()
    ))
    paths = sorted(p for p in output.rglob("*") if p.is_file())
    (output / "SHA256SUMS").write_text("".join(
        f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(output)}\n"
        for p in paths
    ))
    print(f"Prepared {output}; no job submitted.")
    print(f"cd {shlex.quote(str(output))}")
    print(f"sbatch --parsable --reservation={shlex.quote(args.reservation)} launch.sbatch correctness")


if __name__ == "__main__":
    main()
