"""Copy verified historical build inputs into a fresh shared build directory."""

import argparse
import hashlib
from pathlib import Path
import re
import shlex
import shutil


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kind", choices=("runtime", "server"), required=True)
    args = parser.parse_args()
    assets = args.assets.resolve()
    relative = (
        "optimized-pr-completion-20260929-v1" if args.kind == "runtime"
        else "python-direct-validation-20261001/server-build"
    )
    source = assets / relative
    output = args.output.resolve()
    if output.exists():
        parser.error(f"output already exists: {output}")
    entries = []
    for line in (source / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split("  ", 1)
        path = Path(name)
        if path.is_absolute() or ".." in path.parts:
            parser.error("unsafe path in source manifest")
        original = source / path
        if original.is_symlink() or digest(original) != expected:
            parser.error(f"source checksum mismatch: {name}")
        entries.append(path)
    output.mkdir(parents=True)
    for path in entries:
        target = output / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / path, target)
    if args.kind == "runtime":
        source_files = sorted(p for p in (output / "sources").iterdir() if p.is_file())
        (output / "sources/SHA256SUMS").write_text("".join(
            f"{digest(p)}  {p.name}\n" for p in source_files
        ))
        entries.append(Path("sources/SHA256SUMS"))
        build = output / "build.sh"
        build.write_text(build.read_text().replace("CARGO_BUILD_JOBS=16", "CARGO_BUILD_JOBS=1"))
        settings = {
            "BUILD_ROOT": str(output),
            "BASE_IMAGE": str(assets / "pr-completion-20260929-v1/runtime.sqsh"),
            "BASE_IMAGE_SHA256": "fb3ec91e3c34972f4f2e01973344e8448963256ef5d6d8fb902331d7abe65a26",
            "CACHE_ROOT": str(output / "cache"),
            "PRIME_SHA": "700c8c4979805eb53fd87e75d9895bc67bf5450e",
            "MX_SHA": "e562c84988325a242d7a9d4a8834243bd4f7e8c8",
            "BASE_PRIME_SHA": "26533e688262e13dc5dc3a259f675568eb3f4a7d",
            "BASE_MX_SHA": "33dbb5721d594de3188bf46c5bfd3fd987963522",
            "MX_RDMA_NIC_PIN": "auto", "MX_RESHARD_MIN_GBPS": "25",
        }
        launcher = output / "build.sbatch"
    else:
        settings = {
            "VALIDATION_IMAGE": str(assets / "optimized-pr-completion-20260929-v1/runtime.sqsh"),
            "VALIDATION_IMAGE_SHA256": "56b4c754ec16c6a202482acffcfecb29dd412f7b0c2068ff59fe6729c65b8803",
            "BUILD_CACHE": str(output / "cache/cargo"),
        }
        launcher = output / "run.sbatch"
        text, count = re.subn(
            r",/lustre/[^\s\"]+/cache/cargo:/cargo-cache",
            ",$BUILD_CACHE:/cargo-cache", launcher.read_text(),
        )
        if count != 1:
            raise ValueError("unexpected server build cache mount")
        launcher.write_text(text)
    launcher.write_text("\n".join(
        line for line in launcher.read_text().splitlines()
        if not line.startswith("#SBATCH --reservation=")
    ) + "\n")
    (output / "deployment.env").write_text("".join(
        f"export {key}={shlex.quote(value)}\n" for key, value in settings.items()
    ))
    (output / "cache/cargo").mkdir(parents=True)
    (output / "cache/uv").mkdir()
    (output / "SHA256SUMS").write_text("".join(
        f"{digest(output / path)}  {path}\n" for path in sorted(entries)
    ))
    print(f"Prepared {output}; no job submitted.")
    print(f"cd {shlex.quote(str(output))}")
    print(f'sbatch --parsable --reservation="$GLM_RESERVATION" {launcher.name}')


if __name__ == "__main__":
    main()
