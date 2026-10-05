"""Render an isolated full-GLM comparison from the completed 6120166 bundle.

The base image and Prime code remain fixed. Read-only mounts replace both MX
Python packages; their hashes and clean commit are checked on every host.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def replace_once(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


def verify_frozen_gate(gate, snapshot, sources):
    """Bind mounted bytes to the files actually installed in the completed gate."""
    manifest = {}
    for line in (gate / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split("  ", 1)
        relative = Path(name)
        assert re.fullmatch(r"[0-9a-f]{64}", expected), name
        assert not relative.is_absolute() and ".." not in relative.parts, name
        assert name not in manifest, f"Duplicate frozen input: {name}"
        path = gate / relative
        assert path.is_file() and not path.is_symlink(), name
        assert digest(path) == expected, f"Frozen gate checksum mismatch: {name}"
        manifest[name] = expected
    assert "snapshot.json" in manifest
    # Job outputs are intentionally absent from the pre-run manifest. Mounted
    # package inputs must match it exactly, including any non-Python resources.
    actual = {
        str(path.relative_to(gate))
        for path in (gate / "mx-source").rglob("*")
        if path.is_file()
    }
    expected = {name for name in manifest if name.startswith("mx-source/")}
    assert actual == expected, "MX source file set differs from frozen inputs"
    declared = {"mx-source/" + name: sha for name, sha in snapshot["mx_files"].items()}
    assert declared == {name: manifest[name] for name in expected}, (
        "MX snapshot differs from frozen inputs"
    )
    qualified = json.loads((gate / "evidence/source-verification.json").read_text())
    assert qualified == sources, "Completed gate source receipt differs from validation"
    original_snapshot = json.loads((gate / "evidence/snapshot.json").read_text())
    assert original_snapshot == snapshot, (
        "Snapshot changed after the qualified job started"
    )
    for name in (
        "mx_source_head",
        "pr749_source_head",
        "server_binary_source_head",
        "server_inputs_identical",
    ):
        assert sources[name] == snapshot[name], (
            f"Qualified source identity differs: {name}"
        )
    mounted_python = {
        name.removeprefix("modelexpress_client/python/"): sha
        for name, sha in snapshot["mx_files"].items()
        if name.startswith(
            (
                "modelexpress_client/python/modelexpress/",
                "modelexpress_client/python/modelexpress_rl/",
            )
        )
        and name.endswith(".py")
    }
    installed_python = {
        name: item["sha256"] for name, item in sources["installed_mx_sources"].items()
    }
    assert mounted_python == installed_python, (
        "Mounted MX source differs from the qualified installation"
    )
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--server-build", type=Path)
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--arm", choices=("correctness", "performance"), required=True)
    parser.add_argument("--qualification", type=Path)
    parser.add_argument("--warm-helper", type=Path, required=True)
    parser.add_argument("--install-profile-step", type=int)
    parser.add_argument("--install-profile-cprofile", action="store_true")
    parser.add_argument("--install-profile-helper", type=Path)
    args = parser.parse_args()
    if args.install_profile_step is not None:
        assert args.arm == "correctness", "Installation profiling is diagnostic only"
        assert args.install_profile_step in (1, 5, 6, 7, 8, 9, 10, 11), (
            "Select warmup step 1 or warm steps 5-11, outside cold/replay steps 0/2/3/4"
        )
    assert not args.install_profile_cprofile or args.install_profile_step is not None
    assert args.install_profile_helper is None or args.install_profile_step is not None
    assert re.fullmatch(r"[a-z0-9-]{1,32}", args.run_id)
    gate = args.gate.resolve()
    snapshot = json.loads((gate / "snapshot.json").read_text())
    qualification = json.loads((gate / "evidence/validation.json").read_text())
    assert qualification["passed"] is True
    assert (gate / "evidence/exit-code.txt").read_text().strip() == "0"
    gate_job = (gate / "evidence/job-id.txt").read_text().strip()
    assert gate_job.isdigit()
    state = subprocess.check_output(
        ["sacct", "-X", "-n", "-j", gate_job, "--format=State,ExitCode", "-P"],
        text=True,
    ).strip()
    assert state == "COMPLETED|0:0", f"GPU gate job is not complete: {state}"
    sources = qualification["checks"]["source-verification"]
    verify_frozen_gate(gate, snapshot, sources)
    server_binary = None
    server_receipt = None
    if args.server_build is not None:
        server_evidence = args.server_build.resolve() / "evidence"
        assert (server_evidence / "exit-code.txt").read_text().strip() == "0"
        server_receipt = json.loads((server_evidence / "source-receipt.json").read_text())
        assert server_receipt["mx_source_head"] == snapshot["server_binary_source_head"]
        assert snapshot["server_inputs_identical"] is True
        server_binary = server_evidence / "modelexpress-server"
        server_hash = digest(server_binary)
        assert server_hash == (server_evidence / "server-binary.sha256").read_text().split()[0]
        assert server_hash == (gate / "evidence/server-binary.sha256").read_text().split()[0]
    else:
        assert snapshot["server_binary_source_head"] == "e562c84988325a242d7a9d4a8834243bd4f7e8c8", "A matching server build is required"
    baseline = args.baseline.resolve()
    run = json.loads((baseline / "run.json").read_text())
    assert run["run_id"] == "glm030-clean-v2"
    assert (
        run["config"]["revisions"]["prime"]
        == "700c8c4979805eb53fd87e75d9895bc67bf5450e"
    )
    output = args.task_root.resolve() / "runs" / args.run_id
    output.mkdir(parents=True, exist_ok=False)
    for line in (baseline / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split("  ", 1)
        path = baseline / name
        assert digest(path) == expected, name
        if name in ("correctness-qualification.json",):
            continue
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    for name in ("recover_ray_json.py", "recover_and_replay.py"):
        shutil.copy2(Path(__file__).with_name(name), output / name)
    shutil.copytree(gate / "mx-source", output / "mx-source")
    shutil.copy2(gate / "snapshot.json", output / "mx-snapshot.json")
    shutil.copy2(gate / "evidence/validation.json", output / "generic-gpu-gate.json")
    shutil.copy2(
        gate / "evidence/source-verification.json",
        output / "generic-gpu-source-verification.json",
    )
    shutil.copy2(gate / "SHA256SUMS", output / "generic-gpu-input-SHA256SUMS")
    (output / "generic-gpu-gate-job.txt").write_text(f"{gate_job}|{state}\n")
    if server_binary is not None:
        shutil.copy2(server_binary, output / "modelexpress-server")
        write_json(output / "server-source-receipt.json", server_receipt)
    cfg = run["config"]
    flags = cfg["env"]
    flags.pop("MX_REFIT_GLM_DIRECT", None)
    flags["MX_REFIT_TIMING"] = "1"
    flags["MX_REFIT_TIMING_STDOUT"] = "1"
    if args.install_profile_step is not None:
        flags["MX_BENCH_INSTALL_PROFILE_STEP"] = str(args.install_profile_step)
        flags["MX_BENCH_INSTALL_CPROFILE"] = str(int(args.install_profile_cprofile))
    installer = (
        "modelexpress_client/python/modelexpress_rl/inference/engines/vllm/installer.py"
    )
    reference_sha = snapshot["mx_files"][installer]
    flags["MX_REFERENCE_INSTALLER_SHA256"] = reference_sha
    cfg["revisions"]["mx"] = snapshot["mx_source_head"]
    cfg["run_root"] = str(args.task_root.resolve() / "runs")
    cfg["data_dir"] = str(args.task_root.resolve() / "data")
    cfg["model_dir"] = str(args.task_root.resolve() / "models")
    cfg["reservation"] = ""
    run.update(
        run_id=args.run_id,
        arm=args.arm,
        remote_run_dir=str(output),
        output_dir=str(args.task_root.resolve() / "data/outputs" / args.run_id),
        created_at=datetime.now(timezone.utc).isoformat(),
        correctness_status="PENDING",
        performance_status="PENDING",
    )
    for key in (
        "full_model_admission",
        "build_qualification",
        "correctness_qualification",
    ):
        run.pop(key, None)
    run["base_image_sources"] = {
        "prime_sha": "700c8c4979805eb53fd87e75d9895bc67bf5450e",
        "modelexpress_sha": "e562c84988325a242d7a9d4a8834243bd4f7e8c8",
        "server_build_sha": "e562c84988325a242d7a9d4a8834243bd4f7e8c8",
    }
    run["mx_source_overlay"] = {
        "source_head": snapshot["mx_source_head"],
        "pr749_head": snapshot["pr749_source_head"],
        "snapshot_sha256": digest(output / "mx-snapshot.json"),
        "gate_sha256": digest(output / "generic-gpu-gate.json"),
        "installed_source_receipt_sha256": digest(
            output / "generic-gpu-source-verification.json"
        ),
        "gate_input_manifest_sha256": digest(output / "generic-gpu-input-SHA256SUMS"),
        "server_inputs_identical": True,
        "scope": "Read-only Python package mounts over the qualified base image; image provenance retains its original source identities.",
    }
    run["validation_identity"]["mx"] = snapshot["mx_source_head"]
    run["validation_identity"]["flags"] = {
        key: value for key, value in flags.items()
        if key not in {"MX_BENCH_INSTALL_PROFILE_STEP", "MX_BENCH_INSTALL_CPROFILE"}
    }
    run["validation_identity"]["mx_source_overlay"] = run["mx_source_overlay"]
    expected = run["qualification_749"]
    expected["python_sources"] = {
        name.removeprefix("modelexpress_client/python/"): value
        for name, value in snapshot["mx_files"].items()
        if name.startswith(
            (
                "modelexpress_client/python/modelexpress/",
                "modelexpress_client/python/modelexpress_rl/",
            )
        )
        and name.endswith(".py")
    }
    expected["server_source_sha"] = snapshot["server_binary_source_head"]
    if server_binary is not None:
        expected["server_sha256"] = server_hash
        run["server_overlay"] = {"source_head": snapshot["server_binary_source_head"], "sha256": server_hash}
        run["validation_identity"]["server_overlay"] = run["server_overlay"]
    expected["reference_installer_sha256"] = reference_sha
    expected["scope"] = (
        "Frozen generic MX source overlay; full-model qualification remains pending until this run completes."
    )
    run["diagnostic_source_overlay"] = {
        "base_prime_sha": cfg["revisions"]["prime"],
        "scope": "fresh native reload replay on steps2/3/4"
        if args.arm == "correctness"
        else "no warm replay instrumentation",
    }
    if args.arm == "performance":
        assert args.qualification is not None
        proof = json.loads(args.qualification.read_text())
        assert proof["passed"] is True and proof["arm"] == "correctness"
        assert proof["warm"]["comparisons"] == 96
        assert proof["identity"] == run["validation_identity"]
        shutil.copy2(args.qualification, output / "correctness-qualification.json")
        run["correctness_qualification"] = {
            "bundle_relative_path": "correctness-qualification.json",
            "sha256": digest(output / "correctness-qualification.json"),
            "source_result_path": str(args.qualification.resolve()),
        }
    else:
        helper = args.warm_helper.read_text()
        helper = replace_once(
            helper,
            """        session._start_lease = borrow
        try:
            yield
        finally:
            session._start_lease = acquire
            lease.close()
""",
            """        session._start_lease = borrow
        primary_error = None
        try:
            yield
        except BaseException as error:
            primary_error = error
            raise
        finally:
            session._start_lease = acquire
            try:
                for method in client._runtime.methods:
                    method.validate_close()
            except BaseException as cleanup_error:
                retained = getattr(client, "_comparison_retained_leases", [])
                retained.append(lease)
                client._comparison_retained_leases = retained
                client._engine_state = type(client._engine_state).UNCERTAIN

                def require_process_reset():
                    raise RuntimeError(
                        "Diagnostic refit cleanup is unproven; restart the worker "
                        "before closing its renewing source leases"
                    )

                client.close = require_process_reset
                if primary_error is None:
                    raise
                raise primary_error from cleanup_error
            else:
                try:
                    lease.close()
                except BaseException as cleanup_error:
                    if primary_error is not None:
                        raise primary_error from cleanup_error
                    raise
""",
        )
        start = helper.index("            def replay():\n")
        end = helper.index("            def replay_generic():\n", start)
        helper = helper[:start] + helper[end:]
        helper = helper.replace(
            "self.model_runner.get_model(), replay,",
            "self.model_runner.get_model(), replay_generic,",
        )
        helper = helper.replace(
            "Optimized DIRECT values compared to generic layerwise installation, with the same transport/layout; not independent transport equivalence",
            "Cached generic DIRECT values compared to a fresh native reload installer, with the same transport/layout; not independent transport equivalence",
        )
        helper = replace_once(
            helper,
            "            result = original(self, weight_dir=weight_dir, version_uid=version_uid)",
            """            torch.cuda.synchronize()
            memory_before = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            result = original(self, weight_dir=weight_dir, version_uid=version_uid)
            torch.cuda.synchronize()
            update_memory = {
                "allocated_before_bytes": memory_before,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_incremental_bytes": torch.cuda.max_memory_allocated() - memory_before,
                "allocated_after_bytes": torch.cuda.memory_allocated(),
                "scope": "Candidate update only, before diagnostic snapshot/replay.",
            }""",
        )
        helper = replace_once(
            helper,
            "            record.update(rank=self.rank, step=step, version_uid=version_uid,",
            "            record.update(update_memory=update_memory, rank=self.rank, step=step, version_uid=version_uid,",
        )
        destination = (
            output / "prime-overlay/src/prime_rl/utils/mx_749_warm_reference.py"
        )
        destination.write_text(helper)
        worker = output / "prime-overlay/src/prime_rl/inference/vllm/worker/mx_refit.py"
        with worker.open("a") as stream:
            stream.write(
                "\n\nfrom prime_rl.utils.mx_749_warm_reference import instrument_worker\n\ninstrument_worker(MXRefitUpdateWorker)\n"
            )

    if args.install_profile_step is not None:
        helper = output / "prime-overlay/src/prime_rl/utils/mx_bench_install_profile.py"
        shutil.copy2(
            args.install_profile_helper
            or Path(__file__).with_name("profile_installation.py"),
            helper,
        )
        worker = output / "prime-overlay/src/prime_rl/inference/vllm/worker/mx_refit.py"
        with worker.open("a") as stream:
            stream.write(
                "\n\nfrom prime_rl.utils.mx_bench_install_profile import instrument_worker as profile_installation\n\nprofile_installation(MXRefitUpdateWorker)\n"
            )
        run["diagnostic_source_overlay"]["installation_profile"] = {
            "rank": 0,
            "step": args.install_profile_step,
            "cprofile": args.install_profile_cprofile,
            "helper_sha256": digest(helper),
            "scope": "External benchmark-only instance wrappers; imported MX source bytes are unchanged. Diagnostic sample includes observer overhead and must not be used for performance comparison.",
        }
        run["performance_status"] = "DIAGNOSTIC_ONLY"

    runtime = output / "verify_runtime.py"
    text = runtime.read_text()
    start = text.index('    for name, pin in (("prime_sha",')
    end = text.index('    if digest(Path("/app/modelexpress-server"))', start)
    text = (
        text[:start]
        + """    for name, pin in run["base_image_sources"].items():
        if provenance.get(name) != pin:
            raise ValueError(f"Base image provenance mismatch: {name}")
    overlay = run["mx_source_overlay"]
    if digest(Path("/bundle/mx-snapshot.json")) != overlay["snapshot_sha256"]:
        raise ValueError("MX overlay snapshot changed")
    if digest(Path("/bundle/generic-gpu-gate.json")) != overlay["gate_sha256"]:
        raise ValueError("MX overlay prerequisite changed")
"""
        + text[end:]
    )
    text = replace_once(
        text,
        '    reference = Path("/opt/modelexpress-build/modelexpress_client/python/modelexpress_rl/inference/engines/vllm/installer.py")',
        '    reference = Path(importlib.import_module("modelexpress_rl.inference.engines.vllm.installer").__file__)',
    )
    runtime.write_text(text)
    ray_ready = output / "ray_ready.py"
    text = ray_ready.read_text()
    text = replace_once(
        text,
        """    assert provenance['prime_sha'] == os.environ['PRIME_SHA']
    assert provenance['modelexpress_sha'] == os.environ['MX_SHA']
    return {'host': socket.gethostname(), 'python': sys.executable, 'ray': ray.__version__,
            'torch': torch.__version__, 'vllm': vllm.__version__, 'provenance': provenance}
""",
        """    import hashlib
    import importlib
    import sysconfig

    run = json.loads(Path('/bundle/run.json').read_text())
    for name, expected in run['base_image_sources'].items():
        if provenance.get(name) != expected:
            raise ValueError(f'Ray worker base image provenance mismatch: {name}')
    assert os.environ['PRIME_SHA'] == run['config']['revisions']['prime']
    assert os.environ['MX_SHA'] == run['mx_source_overlay']['source_head']
    assert run['config']['revisions']['mx'] == run['mx_source_overlay']['source_head']
    sites = {Path(sysconfig.get_paths()[key]).resolve() for key in ('purelib', 'platlib')}
    package_roots = {
        name: Path(importlib.import_module(name).__file__).resolve().parent
        for name in ('modelexpress', 'modelexpress_rl')
    }
    for name, root in package_roots.items():
        if root.parent not in sites:
            raise ValueError(f'Ray worker imports {name} outside the mounted installation')
    checked = {}
    for relative, expected in run['qualification_749']['python_sources'].items():
        package = relative.split('/', 1)[0]
        source = package_roots[package].parent / relative
        observed = hashlib.sha256(source.read_bytes()).hexdigest()
        if observed != expected:
            raise ValueError(f'Ray worker MX source mismatch: {relative}')
        checked[relative] = observed
    if not checked:
        raise ValueError('Ray worker has no qualified MX source manifest')
    return {'host': socket.gethostname(), 'python': sys.executable, 'ray': ray.__version__,
            'torch': torch.__version__, 'vllm': vllm.__version__, 'provenance': provenance,
            'mx_source_head': run['mx_source_overlay']['source_head'],
            'mx_python_sources': checked}
""",
    )
    ray_ready.write_text(text)
    gate_prime = output / "vendor/gate_prime.py"
    gate_prime.write_text('''"""Verify unchanged base image and explicit candidate Python overlay."""
import hashlib
import json
import os
from pathlib import Path
import modelexpress_rl
run = json.loads(Path("/bundle/run.json").read_text())
values = dict(line.split("=", 1) for line in Path("/etc/primerl-pr3487-provenance").read_text().splitlines())
for key, expected in run["base_image_sources"].items():
    assert values[key] == expected, key
assert os.environ["MX_SHA"] == run["mx_source_overlay"]["source_head"]
assert os.environ["PRIME_SHA"] == run["config"]["revisions"]["prime"]
source = Path(modelexpress_rl.__file__)
expected = run["qualification_749"]["python_sources"]["modelexpress_rl/__init__.py"]
assert hashlib.sha256(source.read_bytes()).hexdigest() == expected
print("MX_SOURCE_OVERLAY_PROVENANCE_OK", run["mx_source_overlay"], flush=True)
''')
    validator = output / "verify_cache_arm.py"
    text = validator.read_text()
    start = text.index('            "retention_batch_scans": 0,')
    end = text.index("        if step >= 2:", start)
    text = (
        text[:start]
        + """        }.items():
            if number(marks[name], name, integer=True) != value:
                raise ValueError(f"Unexpected {name}: {marks[name]} != {value}")
"""
        + text[end:]
    )
    validator.write_text(text)
    (output / "report/verify_correctness.py").write_text(text)
    exporter = output / "report/export_results.py"
    text = exporter.read_text()
    start = text.index("RECEIVER_MARKS = (")
    end = text.index("GLOSSARY = {", start)
    text = (
        text[:start]
        + """RECEIVER_MARKS = (
    "streaming_prepare_s", "source_metadata_s", "layout_capture_s", "transfer_planning_s",
    "connection_registration_s", "wire_s", "wire_wait_s", "reconstruct_s", "install_commit_s",
    "reload_s", "post_install_sync_s", "streaming_release_s", "streaming_apply_s",
    "streaming_total_s", "bytes_received", "batches", "staging_peak_bytes",
    "source_cache_hits", "plan_cache_hits",
)
ADDITIVE = ("streaming_prepare_s", "wire_s", "reconstruct_s", "install_commit_s",
            "reload_s", "post_install_sync_s", "streaming_release_s")
"""
        + text[end:]
    )
    obsolete = (
        "retention_arena_setup_s",
        "retention_batch_scan_s",
        "retention_final_scan_s",
        "retention_scan_s",
        "derived_refresh_s",
        "direct_guard_s",
        "glm_direct_install",
    )
    for name in obsolete:
        text, count = re.subn(rf'^    "{name}": .*\n', "", text, flags=re.MULTILINE)
        assert count == 1, name
    text = replace_once(
        text,
        '    "install_commit_s": ("MX internal", "GLM DIRECT per-batch geometry validation, copies into live destinations and synchronization. Excludes MLA refresh and pre-install guard."),',
        '    "install_commit_s": ("MX internal", "Per-batch coverage/geometry validation, engine-owned materialization, receive copies, native module post-load processing, copying/restoring live destinations and synchronization. Deferred attention processing is in reload_s."),',
    )
    text = replace_once(
        text,
        '    "receiver_other_s": ("Derived", "Receiver E2E minus disjoint prepare, wire, reconstruct, commit, final scan, reload, final synchronization and release. Cold includes verification."),',
        '    "post_install_sync_s": ("MX internal", "Final CUDA synchronization after native reload finalization; separate from reload_s and install_commit_s."),\n    "receiver_other_s": ("Derived", "Receiver E2E minus disjoint prepare, wire, reconstruct, commit, reload, final synchronization and release. Cold includes verification."),',
    )
    text = replace_once(
        text,
        '    "read_api_gbps": ("Derived",',
        '    "installation_total_s": ("Derived", "Same-rank per-update sum of install_commit_s + reload_s + post_install_sync_s. Includes engine native materialization, post-load processing, live destination restoration and all installation CUDA drains; excludes wire/reconstruction."),\n    "read_api_gbps": ("Derived",',
    )
    text = replace_once(
        text,
        "    result = {key: m[key] for key in RECEIVER_MARKS}",
        '    result = {key: value for key, value in m.items() if key not in ("rank", "replica")}\n    for key in RECEIVER_MARKS:\n        if key not in result:\n            raise ValueError(f"Missing expected generic receiver metric: {key}")',
    )
    text = replace_once(
        text,
        '    result["receiver_e2e_s"] = record["elapsed_s"]',
        '    result["installation_total_s"] = sum(m[key] for key in ("install_commit_s", "reload_s", "post_install_sync_s"))\n    result["receiver_e2e_s"] = record["elapsed_s"]',
    )
    exporter.write_text(text)
    shutil.copy2(
        Path(__file__).with_name("export_native_timings.py"),
        output / "report/export_native_timings.py",
    )
    report = output / "report/export.sh"
    with report.open("a") as stream:
        stream.write(
            '\nuv run --no-sync python "$PACKAGE/report/export_native_timings.py" --logs "$LOGS" --results "$REPORT/results"\n'
        )

    env = output / "env.sh"
    text = env.read_text()
    values = {
        "RUN_ID": args.run_id,
        "RUN_HOST": str(output),
        "MX_SHA": snapshot["mx_source_head"],
        "DATA_HOST": cfg["data_dir"],
        "MODELS_HOST": cfg["model_dir"],
    }
    for name, value in values.items():
        text, count = re.subn(
            rf"^export {name}=.*$",
            lambda _: f"export {name}={shlex.quote(value)}",
            text,
            flags=re.MULTILINE,
        )
        assert count == 1, name
    env.write_text(text)
    (output / "fabric.sh").write_text(
        "".join(
            f"export {name}={shlex.quote(value)}\n" for name, value in flags.items()
        )
    )
    launcher = output / "run.sbatch"
    text = launcher.read_text()
    text = text.replace("mx.glm030-clean-v2", "mx." + args.run_id)
    line = next(line for line in text.splitlines() if line.startswith('MOUNTS="'))
    mounts = "".join(
        f",$RUN_HOST/mx-source/modelexpress_client/python/{package}:/app/.venv/lib/python3.12/site-packages/{package}:ro"
        for package in ("modelexpress", "modelexpress_rl")
    )
    if server_binary is not None:
        mounts += ",$RUN_HOST/modelexpress-server:/app/modelexpress-server:ro"
    text = replace_once(text, line, line[:-1] + mounts + '"')
    launcher.write_text(text)
    manifest = json.loads((output / "overlay-manifest.json").read_text())
    overlay = output / "prime-overlay/src/prime_rl"
    manifest["files"] = [
        {"path": str(path.relative_to(overlay)), "sha256": digest(path)}
        for path in sorted(overlay.rglob("*"))
        if path.is_file()
    ]
    manifest["patch_sha256"] = hashlib.sha256(
        json.dumps(manifest["files"], sort_keys=True).encode()
    ).hexdigest()
    manifest["expected_mx_flags"] = flags
    write_json(output / "overlay-manifest.json", manifest)
    write_json(output / "run.json", run)
    paths = sorted(
        path
        for path in output.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    )
    (output / "SHA256SUMS").write_text(
        "".join(f"{digest(path)}  {path.relative_to(output)}\n" for path in paths)
    )
    print(output)


if __name__ == "__main__":
    main()
