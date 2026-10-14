#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run CPU NIXL refit tests against pinned worktrees without copying production code."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

INTEGRATION_DIR = Path(__file__).resolve().parent
COMPONENT_TESTS = (
    "tests/test_refit_vllm_installer.py",
    "tests/test_refit_generator_runtime.py",
    "tests/test_reshard_refit_coverage.py",
    "tests/test_reshard_refit_verify.py",
    "tests/test_refit_nixl_staged_transfer.py",
)
CPU_TESTS = (
    "tests/integration/test_refit_nixl_cpu.py",
    "tests/integration/test_refit_nixl_quantized.py",
    "tests/integration/test_refit_replica_selection.py",
)
LIMITATIONS = (
    ("missing_tensor_and_shard_qualification", "full-refit qualification lives in GPU-coupled receiver/staging; planner alone permits partial plans"),
    ("bounded_staging", "CUDA-required constructor in the production staging path"),
    ("ordered_replay", "CUDA-required constructor in the production replay path"),
    ("descriptor_coarsening", "CUDA-required constructor in the production coarsening path"),
    ("dtype_conversion_stage", "the full-stage production path unconditionally synchronizes CUDA"),
    ("version_serving_safety", "requires full runtime readiness and mixed-version serving"),
    ("gpu_quantizer", "CPU tests do not instantiate a real GPU quantization engine"),
    ("partial_install_serving_safety", "multi-peer transfer failure is tested; atomic installation/readiness needs GPU-coupled runtime"),
    ("checksum_corruption", "production digest rejection is inside GPU-coupled staging; existing CPU component verification tests run separately"),
    ("high_level_warm_cache", "production high-level cached staging/replay requires CUDA; repeated transport updates are tested"),
)


def dependency_paths(python):
    paths = subprocess.check_output(
        [str(python), "-c", "import json,sys; print(json.dumps(sys.path))"], text=True
    )
    return [p for p in json.loads(paths) if p.endswith("site-packages") and Path(p).is_dir()]


def versions(python):
    code = """import importlib.metadata as m, json, sys
versions = {}
for name in ('torch', 'nixl', 'nixl-cu12'):
    try:
        versions[name] = m.version(name)
    except m.PackageNotFoundError:
        versions[name] = None
print(json.dumps({'python': sys.version.split()[0], **versions}))
"""
    return json.loads(subprocess.check_output([str(python), "-c", code], text=True))


def junit_results(path):
    root = ET.parse(path).getroot()
    cases = root.findall(".//testcase")
    results = []
    for case in cases:
        outcome = "passed"
        detail = ""
        for child in case:
            if child.tag in ("failure", "error"):
                outcome, detail = "failed", child.get("message", child.text or "")
                break
            if child.tag == "skipped":
                outcome, detail = "blocked", child.get("message", child.text or "skipped")
        results.append({
            "test": f"{case.get('classname', '')}::{case.get('name', '')}".lstrip(":"),
            "status": outcome,
            **({"reason": detail} if detail else {}),
        })
    return results


def run_suite(python, target_dir, output_dir, name, tests, env, timeout=300):
    xml_path = output_dir / f"{name}.xml"
    log_path = output_dir / f"{name}.log"
    argv = [str(python), "-m", "pytest", *tests, "-v", f"--junitxml={xml_path}"]
    status = None
    timed_out = False
    with log_path.open("w") as log:
        process = subprocess.Popen(
            argv, cwd=target_dir, env=env, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            status = process.wait(timeout=timeout)
        except BaseException as error:
            try:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                time.sleep(1)
            finally:
                # The leader may exit before a worker that ignores SIGTERM.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            if not isinstance(error, subprocess.TimeoutExpired):
                raise
            timed_out = True
            log.write(f"\nTimed out after {timeout} seconds.\n")
    if timed_out:
        results = [{"test": test, "status": "blocked", "reason": f"timeout after {timeout} seconds"}
                   for test in tests]
    elif xml_path.exists():
        try:
            results = junit_results(xml_path)
        except ET.ParseError as error:
            results = [{"test": test, "status": "blocked", "reason": f"invalid JUnit XML: {error}"}
                       for test in tests]
    else:
        results = [{"test": test, "status": "blocked", "reason": "JUnit XML missing"}
                   for test in tests]
    if status not in (0, None) and not any(row["status"] == "failed" for row in results):
        results.append({"test": name, "status": "blocked", "reason": f"pytest exited {status}"})
    return {
        "argv": argv, "cwd": str(target_dir), "exit_code": status,
        "timed_out": timed_out, "log": str(log_path), "junit": str(xml_path),
        "tests": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    args = parser.parse_args()
    interpreter = args.python.absolute()
    args.output = args.output.absolute()
    args.output.mkdir(parents=True, exist_ok=True)
    shared_paths = dependency_paths(interpreter)
    shared_versions = versions(interpreter)
    report = {
        "python": str(interpreter), "versions": shared_versions,
        "ucx_tls": os.environ.get("UCX_TLS", "tcp,self"), "targets": {},
    }
    overlay_files = [p for p in INTEGRATION_DIR.glob("*.py") if p.name != Path(__file__).name]
    report["test_file_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in overlay_files}

    for name, target in json.loads(args.targets.read_text()).items():
        target_root = Path(target["worktree"]).absolute()
        target_dir = target_root / "modelexpress_client" / "python"
        target_output = args.output / name
        target_output.mkdir(parents=True, exist_ok=True)
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=target_root, text=True,
            capture_output=True, check=False,
        )
        actual_commit = commit.stdout.strip() if commit.returncode == 0 else None
        entry = {
            "ref": target.get("ref"), "expected_commit": target["commit"],
            "git_hash": actual_commit, "git_exit_code": commit.returncode,
            "versions": shared_versions, "suites": {},
            "known_cpu_qualification_limits": [
                {"test": limitation, "status": "blocked", "reason": reason}
                for limitation, reason in LIMITATIONS
            ],
        }
        report["targets"][name] = entry
        if actual_commit != target["commit"]:
            reason = f"worktree commit mismatch (expected {target['commit']}, got {actual_commit})"
            for suite in ("cpu_nixl", "components"):
                entry["suites"][suite] = {
                    "argv": [], "cwd": str(target_dir), "exit_code": None,
                    "log": None, "junit": None,
                    "tests": [{"test": suite, "status": "blocked", "reason": reason}],
                }
            continue

        integration_dir = target_dir / "tests" / "integration"
        integration_dir.mkdir(parents=True, exist_ok=True)
        for source in overlay_files:
            shutil.copy2(source, integration_dir / source.name)
        venv_dir = target_output / "venv"
        subprocess.run(
            [str(interpreter), "-m", "venv", "--without-pip", str(venv_dir)],
            check=True,
        )
        target_python = venv_dir / "bin" / "python"
        site_packages = Path(subprocess.check_output(
            [str(target_python), "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
            text=True,
        ).strip())
        site_packages.mkdir(parents=True, exist_ok=True)
        (site_packages / "shared-dependencies.pth").write_text(
            "\n".join(shared_paths) + "\n"
        )
        env = dict(os.environ)
        env.update({
            "PYTHONPATH": str(target_dir),
            "MX_REQUIRE_CPU_NIXL": "1",
            "UCX_TLS": os.environ.get("UCX_TLS", "tcp,self"),
            "UCX_NET_DEVICES": "lo",
            "CUDA_VISIBLE_DEVICES": "",
        })
        entry["dependency_paths"] = shared_paths
        entry["suites"]["cpu_nixl"] = run_suite(
            target_python, target_dir, target_output, "cpu_nixl", CPU_TESTS, env
        )
        entry["suites"]["components"] = run_suite(
            target_python, target_dir, target_output, "components", COMPONENT_TESTS, env
        )

    report_path = args.output / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    failed = any(
        row["status"] in ("failed", "blocked")
        for target in report["targets"].values()
        for suite in target["suites"].values()
        for row in suite["tests"]
    )
    print(report_path)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
