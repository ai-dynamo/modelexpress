"""Exact warm poison/restore replay with the same pinned installer; diagnostic only."""

import hashlib
import importlib.util
import json
import os
import sys
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import torch
from prime_rl.utils.mx_verification import (
    _chunks,
    perturb_weights,
    snapshot_weights,
    verify_weights,
)

REFERENCE_INSTALLER_SHA = os.environ["MX_REFERENCE_INSTALLER_SHA256"]


class _BorrowedLease:
    def close(self):
        pass


@contextmanager
def comparison_lease(client, version_uid):
    """One server lease covers both applies; nested session closes only borrow it."""
    # The server keys leases by worker/version, not by acquisition call. Keep
    # session cleanup from deleting the outer lease between diagnostic phases.
    with client._operation_lock:
        session = client._runtime.session
        acquire = session._start_lease
        lease = client._start_version_lease(version_uid)

        def borrow(version_id):
            if version_id != version_uid:
                raise ValueError("Diagnostic lease cannot cover a different version")
            return _BorrowedLease()

        session._start_lease = borrow
        try:
            yield
        finally:
            session._start_lease = acquire
            lease.close()


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def reference_installer_type():
    from modelexpress_rl.inference.engines.vllm import installer
    source = Path(installer.__file__)
    if hashlib.sha256(source.read_bytes()).hexdigest() != REFERENCE_INSTALLER_SHA:
        raise ValueError("Replay installer does not match the pinned MX source")
    return installer._VllmInstaller


def fingerprints(snapshot):
    result = {}
    for name, tensor in sorted(snapshot.values.items()):
        digest = hashlib.sha256()
        for chunk in _chunks(tensor):
            digest.update(chunk.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        result[name] = digest.hexdigest()
    return result


@torch.no_grad()
def compare_with_reference(model, install_reference, *, max_cpu_bytes, previous=None):
    """Poison all candidate weights and require the reference to restore them."""
    snapshot = snapshot_weights(model, max_bytes=max_cpu_bytes)
    hashes = fingerprints(snapshot)
    if previous is not None:
        if set(previous) != set(hashes):
            raise ValueError("Warm comparison tensor coverage changed")
        changed = sum(previous[name] != digest for name, digest in hashes.items())
        if not changed:
            raise ValueError("Warm version has no observed installed-weight change")
    else:
        changed = None
    perturb_weights(model, snapshot)
    install_reference()
    result = verify_weights(model, snapshot)
    result.update(
        record="glm749-warm-reference-comparison-v1",
        changed_since_previous=changed,
        cpu_snapshot_bytes=sum(t.numel() * t.element_size() for t in snapshot.values.values()),
        reference_installer_sha256=REFERENCE_INSTALLER_SHA,
        content_sha256=hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
        scope="Optimized DIRECT values compared to generic layerwise installation, with the same transport/layout; not independent transport equivalence",
    )
    if not result["passed"]:
        raise ValueError(f"Warm installed-weight comparison failed: {json.dumps(result)}")
    return result, hashes


def instrument_worker(worker_type):
    """Keep the RPC paused through diagnostic replay and fail closed on error."""
    original = worker_type.update_weights_from_path

    @torch.no_grad()
    def update(self, weight_dir=None, version_uid=None):
        step = int(version_uid.rsplit(":", 1)[-1]) if version_uid else -1
        if step not in (2, 3, 4):
            return original(self, weight_dir=weight_dir, version_uid=version_uid)
        client = self._generator
        try:
            with comparison_lease(client, version_uid):
                return compare_update(self, client, version_uid, weight_dir)
        except BaseException:
            client._engine_state = type(client._engine_state).UNCERTAIN
            raise

    def compare_update(self, client, version_uid, weight_dir):
        step = int(version_uid.rsplit(":", 1)[-1])
        try:
            result = original(self, weight_dir=weight_dir, version_uid=version_uid)
            runtime = client._runtime
            runtime.unpublish_runtime_tensors()

            def replay():
                previous_direct = os.environ["MX_REFIT_GLM_DIRECT"]
                os.environ["MX_REFIT_GLM_DIRECT"] = "0"
                try:
                    replay_generic()
                finally:
                    os.environ["MX_REFIT_GLM_DIRECT"] = previous_direct

            def replay_generic():
                prepared = runtime.session.prepare_streaming(
                    client._get_ready_version(version_uid),
                    max_staging_bytes=int(os.environ["MX_REFIT_STAGING_BYTES"]),
                    staging_device="cuda", staging_buffers=1,
                )
                current = prepared.plan.installer
                reference = reference_installer_type()(
                    model=current._model, vllm_config=current._vllm_config,
                    model_config=current._model_config, device=current._device,
                    convert_native_to_hf=current._convert_native_to_hf,
                )
                prepared.plan = replace(prepared.plan, installer=reference)
                try:
                    runtime.session.apply(prepared)
                finally:
                    runtime.session.release(prepared)

            record, hashes = compare_with_reference(
                self.model_runner.get_model(), replay,
                max_cpu_bytes=int(os.environ.get("MX_VERIFY_CPU_BYTES", str(64 * 1024**3))),
                previous=getattr(self, "_glm749_previous_hashes", None),
            )
            self._glm749_previous_hashes = hashes
            runtime.publish_runtime_tensors(version_uid)
            record.update(rank=self.rank, step=step, version_uid=version_uid,
                          job_id=os.environ["SLURM_JOB_ID"], pid=os.getpid())
            out = Path(os.environ["OUT"]) / "warm-reference"
            out.mkdir(exist_ok=True)
            with (out / f"rank-{self.rank:03}-step-{step:03}.json").open("x") as stream:
                json.dump(record, stream, indent=2)
                stream.write("\n")
            print(json.dumps(record), flush=True)
            return result
        except BaseException:
            client._engine_state = type(client._engine_state).UNCERTAIN
            raise

    worker_type.update_weights_from_path = update
