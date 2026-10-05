"""Freeze clean commits for the container overlay; never snapshot live edits."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

SERVER_SOURCE = "abc75671aff09043874f825c88d11f8c05305dfa"


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot(source, destination, prefixes):
    names = (
        subprocess.check_output(
            ["git", "-C", str(source), "ls-files", "-z", "--", *prefixes]
        )
        .decode()
        .split("\0")
    )
    destination.mkdir(exist_ok=False)
    hashes = {}
    for name in sorted(filter(None, names)):
        path = source / name
        if not path.is_file():
            continue
        assert not path.is_symlink(), name
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        hashes[name] = digest(target)
    return hashes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prime", type=Path, required=True)
    parser.add_argument("--mx", type=Path, required=True)
    parser.add_argument("--pr749-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for source in (args.prime, args.mx):
        assert not git(source, "status", "--porcelain"), (
            f"Commit candidate first: {source}"
        )
    subprocess.run(
        [
            "git",
            "-C",
            str(args.mx),
            "merge-base",
            "--is-ancestor",
            args.pr749_sha,
            "HEAD",
        ],
        check=True,
    )
    changed = git(args.mx, "diff", "--name-only", SERVER_SOURCE, "HEAD").splitlines()
    rust_inputs = [
        name
        for name in changed
        if name.endswith((".rs", ".proto"))
        or Path(name).name in ("Cargo.toml", "Cargo.lock", "build.rs")
    ]
    assert not rust_inputs, f"Build a new server binary: {rust_inputs}"
    root = args.output.resolve()
    root.mkdir(exist_ok=False, parents=True)
    for path in Path(__file__).parent.iterdir():
        if path.is_file() and path.name not in ("SHA256SUMS", "snapshot.json"):
            shutil.copy2(path, root / path.name)
    receipt = {
        "schema": "generic-direct-candidate-snapshot-v1",
        "prime_base_head": git(args.prime, "rev-parse", "HEAD"),
        "prime_status": "",
        "mx_source_head": git(args.mx, "rev-parse", "HEAD"),
        "pr749_source_head": args.pr749_sha,
        "server_binary_source_head": SERVER_SOURCE,
        "server_inputs_identical": True,
        "mx_changed_paths_from_server_source": changed,
        "scope": "Python warm installation plus trainer COPY_TO_HOST. Rust server inputs match the previously qualified server build. One-node correctness gate; not a full GLM result.",
        "candidate_files": snapshot(
            args.prime,
            root / "candidate",
            ["src", "packages", "tests", "pyproject.toml", "uv.lock"],
        ),
        "mx_files": snapshot(
            args.mx, root / "mx-source", ["modelexpress_client/python"]
        ),
        "installation_helper": {
            "path": "install_modelexpress_client.sh",
            "sha256": digest(root / "install_modelexpress_client.sh"),
        },
    }
    (root / "snapshot.json").write_text(json.dumps(receipt, indent=2) + "\n")
    paths = sorted(path for path in root.rglob("*") if path.is_file())
    (root / "SHA256SUMS").write_text(
        "".join(f"{digest(path)}  {path.relative_to(root)}\n" for path in paths)
    )
    print(
        json.dumps(
            {
                "output": str(root),
                "prime": receipt["prime_base_head"],
                "mx": receipt["mx_source_head"],
            }
        )
    )


if __name__ == "__main__":
    main()
