"""Optional benchmark-only host attribution; never imported by production MX."""

from collections import defaultdict
from contextlib import contextmanager
from functools import wraps
import hashlib
import inspect
import json
import os
from pathlib import Path
import time

METHODS = (
    "install_streaming",
    "_reload",
    "_process_and_commit",
    "_restore_parameter_aliases",
    "_validate_alias_owners",
    "_drain_streaming",
)


@contextmanager
def observe_installer(installer, record):
    """Restore instance attributes even when the observed operation fails."""
    missing = object()
    saved = {}
    stack = []
    totals = defaultdict(
        lambda: {"calls": 0, "errors": 0, "inclusive_s": 0.0, "unwrapped_s": 0.0}
    )
    record["calls"] = []
    record["totals"] = totals
    contexts = defaultdict(lambda: {"calls": 0, "inclusive_s": 0.0})
    record["contexts"] = contexts

    def wrap(name, original):
        @wraps(original)
        def measured(*args, **kwargs):
            parent = stack[-1] if stack else None
            context = (
                "inside_process"
                if any(item["method"] == "_process_and_commit" for item in stack)
                else "outside_process"
            )
            frame = {"method": name, "child_s": 0.0}
            stack.append(frame)
            started = time.perf_counter()
            failed = False
            try:
                result = original(*args, **kwargs)
                if name == "install_streaming":
                    record["streaming_metrics"] = dict(args[0].metrics)
                return result
            except BaseException:
                failed = True
                raise
            finally:
                elapsed = time.perf_counter() - started
                assert stack.pop() is frame
                if parent is not None:
                    parent["child_s"] += elapsed
                unwrapped = max(0.0, elapsed - frame["child_s"])
                total = totals[name]
                total["calls"] += 1
                total["errors"] += int(failed)
                total["inclusive_s"] += elapsed
                total["unwrapped_s"] += unwrapped
                contexts[f"{name}:{context}"]["calls"] += 1
                contexts[f"{name}:{context}"]["inclusive_s"] += elapsed
                record["calls"].append(
                    {
                        "method": name,
                        "parent": parent["method"] if parent else None,
                        "context": context,
                        "inclusive_s": elapsed,
                        "unwrapped_s": unwrapped,
                        "failed": failed,
                    }
                )

        return measured

    try:
        for name in METHODS:
            original = getattr(installer, name)
            saved[name] = installer.__dict__.get(name, missing)
            setattr(installer, name, wrap(name, original))
        yield
    finally:
        for name, value in saved.items():
            if value is missing:
                installer.__dict__.pop(name, None)
            else:
                setattr(installer, name, value)


def alias_census(installer):
    """Describe the pre-update live tree, outside all installation intervals."""
    started = time.perf_counter()
    aliases = installer._parameter_aliases(installer._model)
    return {
        "groups": len(aliases),
        "alias_paths": sum(len(group) for group in aliases),
        "distinct_parameter_slots": sum(
            len({(id(module), leaf) for _, module, leaf in group}) for group in aliases
        ),
        "groups_with_repeated_slot_paths": sum(
            len(group) > len({(id(module), leaf) for _, module, leaf in group})
            for group in aliases
        ),
        "host_elapsed_s": time.perf_counter() - started,
        "scope": "Pre-update live model; alias layout may differ inside native reload.",
    }


def instrument_worker(worker_type):
    """Time one selected rank/update without adding CUDA fences or changing MX."""
    original = worker_type.update_weights_from_path

    @wraps(original)
    def update(self, weight_dir=None, version_uid=None):
        step = int(version_uid.rsplit(":", 1)[-1]) if version_uid else -1
        target = int(os.environ["MX_BENCH_INSTALL_PROFILE_STEP"])
        if self.rank != 0 or step != target:
            return original(self, weight_dir=weight_dir, version_uid=version_uid)
        installer = self._generator._runtime.engine.installer
        source = Path(inspect.getfile(type(installer))).resolve()
        expected = os.environ["MX_REFERENCE_INSTALLER_SHA256"]
        actual = hashlib.sha256(source.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError("Profile installer does not match frozen candidate source")
        output = Path(os.environ["OUT"]) / "installation-profile"
        output.mkdir(exist_ok=True)
        record = {
            "record": "mx-benchmark-installation-profile-v1",
            "rank": self.rank,
            "step": step,
            "version_uid": version_uid,
            "pid": os.getpid(),
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "installer_sha256": actual,
            "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "scope": "Diagnostic host call intervals on rank 0 only; observer overhead and rank skew make this unsuitable for performance comparison. No additional CUDA synchronization.",
            "nesting": "Inclusive totals overlap. unwrapped_s subtracts directly wrapped children only; it still contains uninstrumented code and wrapper bookkeeping. _drain_streaming contains existing CUDA synchronization.",
            "alias_census": alias_census(installer),
            "passed": False,
        }
        profiler = None
        if os.environ.get("MX_BENCH_INSTALL_CPROFILE") == "1":
            import cProfile

            profiler = cProfile.Profile()
        started = time.perf_counter()
        primary_error = None
        try:
            with observe_installer(installer, record):
                if profiler is not None:
                    profiler.enable()
                try:
                    result = original(
                        self, weight_dir=weight_dir, version_uid=version_uid
                    )
                finally:
                    if profiler is not None:
                        profiler.disable()
            drains = [
                call for call in record["calls"] if call["method"] == "_drain_streaming"
            ]
            metrics = record["streaming_metrics"]
            expected_drains = int(metrics["batches"]) + 1
            if len(drains) != expected_drains:
                raise ValueError(
                    "Successful streaming drain count differs from batches + final fence"
                )
            record["drain_split"] = {
                "batch_count": expected_drains - 1,
                "batch_drains_s": sum(call["inclusive_s"] for call in drains[:-1]),
                "final_drain_s": drains[-1]["inclusive_s"],
                "scope": "Successful exact-source installation: one drain per batch followed by the final drain; final drain belongs to post_install_sync_s, not install_commit_s.",
            }
            record["passed"] = True
            return result
        except BaseException as error:
            primary_error = error
            record["error_type"] = type(error).__name__
            raise
        finally:
            record["observed_update_s"] = time.perf_counter() - started
            try:
                if profiler is not None:
                    profiler.dump_stats(str(output / f"rank-000-step-{step:03}.pstats"))
                with (output / f"rank-000-step-{step:03}.json").open("x") as stream:
                    json.dump(record, stream, indent=2)
                    stream.write("\n")
                print(
                    json.dumps(
                        {key: value for key, value in record.items() if key != "calls"}
                    ),
                    flush=True,
                )
            except BaseException as output_error:
                if primary_error is None:
                    raise
                print(
                    f"Installation profile output failed: {type(output_error).__name__}",
                    flush=True,
                )

    worker_type.update_weights_from_path = update
