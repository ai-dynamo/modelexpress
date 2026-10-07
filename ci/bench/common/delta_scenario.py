# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""S3 XOR-delta publication and optional generator-peer benchmark scenario."""

import hashlib
import json
import re
import time

import validation

publication_path = "/tmp/mx-delta/report.json"


def configure(config, paths, environment):
    for key in ["seed_prefix", "delta_prefix"]:
        if key in environment:
            config[key] = environment[key]
    if paths not in ["s3", "both"]:
        raise ValueError("paths must be s3 or both")
    if paths == "both" and not (
        environment.get("extra_worker_resources")
        and environment.get("worker_env", {}).get("MX_NIXL_BACKEND")
        and environment.get("peer_transfer_marker")
    ):
        raise ValueError(
            "Peer coverage requires explicit extra_worker_resources, "
            "worker_env.MX_NIXL_BACKEND, and peer_transfer_marker"
        )
    config.update(
        scenario="delta",
        initial_version=config["run"] + "-base",
        target_version=config["run"] + "-d1",
        roles=["s3", "peer"] if paths == "both" else ["s3"],
        sources={
            "s3": "OBJECT_STORAGE",
            **({"peer": "GENERATOR"} if paths == "both" else {}),
        },
    )


def control_env(config):
    return {"DELTA_RUN": config["run"]}


def worker_env(config, environment):
    return {
        "MX_GENERATOR_SOURCE_ORDER": "OBJECT_STORAGE",
        **environment.get("worker_env", {}),
        "MX_MODEL_URI": f"s3://{config['bucket']}/{config['seed_prefix'].rstrip('/')}",
        "BENCH_PREFIX": config["seed_prefix"],
    }


def worker_evidence(text, config):
    marker = config.get("peer_transfer_marker", "RDMA transfer complete:")
    return {
        "rdma_transfer_records": [line for line in text.splitlines() if marker in line]
    }


def generator_kwargs(config, source):
    from modelexpress_rl.inference.receiver import ObjectStorageGeneratorConfig
    from modelexpress_rl.object_storage import ObjectStorageType

    return {
        "object_storage": ObjectStorageGeneratorConfig(
            storage_type=ObjectStorageType.S3,
            initial_base_version_id=config["initial_version"],
            seed_checkpoint_path="/models",
            refit_checkpoint_dir="/refit",
            refit_checkpoint_max_size_gb=config["refit_checkpoint_max_size_gb"],
            endpoint_url=config["storage"]["endpoint_url"],
            region_name=config["storage"]["region"],
        )
        if source == "OBJECT_STORAGE"
        else None,
    }


def refit_evidence(config, version_id):
    import torch
    from modelexpress_rl.inference.checkpoint_store import LocalCheckpointStore
    from safetensors import safe_open

    checkpoint = LocalCheckpointStore(
        root="/refit", model_name=config["model"]
    ).checkpoint_path(version_id)
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())
    name = config["embedding"]
    with safe_open(str(checkpoint / index["weight_map"][name]), framework="pt") as sf:
        raw = sf.get_tensor(name)
        digest = hashlib.sha256(
            raw.contiguous().view(torch.uint8).numpy().tobytes()
        ).hexdigest()
    return {
        "version": version_id,
        "checkpoint_tensor": name,
        "checkpoint_sha256": digest,
        "shape": list(raw.shape),
        "dtype": str(raw.dtype),
        "verified": True,
    }


def checkpoint(rows, config, publication):
    assert publication["run"] == config["run"]
    assert publication["model_revision"] == config["revision"]
    assert publication["tensor"] == config["embedding"]
    assert re.fullmatch(r"[0-9a-f]{64}", publication["expected_sha256"]), (
        "Invalid published digest"
    )
    by_rank = validation.ranks(rows, config)
    for row in by_rank.values():
        assert row["version"] == config["target_version"]
        assert row["checkpoint_tensor"] == publication["tensor"]
        assert row["verified"]
        assert row["checkpoint_sha256"] == publication["expected_sha256"], (
            "Reconstructed checkpoint differs from published embedding"
        )


def verify_refit(rpc, role, config, publication):
    if role == "s3":
        rows = rpc(
            role,
            "verify-checkpoint",
            "hotload_verify_checkpoint",
            {"version_id": config["target_version"]},
        )
        checkpoint(rows, config, publication)


def validate_inventory(baseline, updated):
    assert updated != baseline, "No updated tensors"


def compare_workers(inventories, config):
    if "peer" in config["roles"]:
        assert inventories["s3"] == inventories["peer"], (
            "Peer tensors differ per TP rank"
        )


def validate_report(result, report, root):
    config = report["config"]
    checkpoint(result("s3", "verify-checkpoint"), config, report["publication"])
    for role, worker in report["workers"].items():
        text = (root / f"{role}-worker.log").read_text()
        if role == "s3":
            assert "Streaming weights from s3://" in text
        else:
            assert worker["rdma_transfer_records"], "No RDMA completion evidence"
            assert not any(
                x in text
                for x in [
                    "Trying strategy: model_streamer",
                    "Streaming weights from s3://",
                    "Trying strategy: instant_tensor",
                ]
            ), "Peer cold load fell back"
    if "peer" in config["roles"]:
        assert (
            report["workers"]["s3"]["pod"]["spec"]["nodeName"]
            != report["workers"]["peer"]["pod"]["spec"]["nodeName"]
        )


def pending_publications(processes):
    pending = []
    for process in processes:
        code = process.poll()
        if code is None:
            pending.append(process)
        elif code:
            raise RuntimeError("Seed download or publication failed; inspect logs")
    return pending


def wait_for_publications(processes, deadline):
    while pending_publications(processes):
        if time.monotonic() >= deadline:
            raise TimeoutError("Seed download or publication timed out")
        time.sleep(5)


def prepare_run(benchmark):
    for name in ["harness.json", "control.yaml", "worker-s3.yaml"]:
        benchmark.apply(name)
    source = benchmark.prefix + "-s3"
    benchmark.k.call(
        "wait",
        "--for=condition=Ready",
        "pod/" + benchmark.control,
        "pod/" + source,
        "--timeout=10m",
    )
    worker = benchmark.start(source, "native_server.py", "s3-worker.log")
    seed = benchmark.start(source, "download_seed.py", "seed.log")
    publisher = benchmark.start(
        benchmark.control, "publish_delta.py", "publication.log"
    )
    deadline = time.monotonic() + 7200
    while (
        "BENCH_READY"
        not in (benchmark.root / "s3-worker.log")
        .read_text(errors="replace")
        .splitlines()
    ):
        pending_publications([seed, publisher])
        if worker.poll() is not None:
            raise RuntimeError(
                "S3 worker exited before readiness; inspect s3-worker.log"
            )
        if time.monotonic() >= deadline:
            raise TimeoutError("S3 worker readiness timed out")
        time.sleep(5)
    if "peer" in benchmark.roles:
        benchmark.apply("worker-peer.yaml")
        peer = benchmark.prefix + "-peer"
        benchmark.k.call(
            "wait", "--for=condition=Ready", "pod/" + peer, "--timeout=10m"
        )
        benchmark.start(peer, "native_server.py", "peer-worker.log")
    wait_for_publications([seed, publisher], deadline)
    benchmark.k.call(
        "exec",
        benchmark.control,
        "-c",
        "main",
        "--",
        "cat",
        publication_path,
        output=benchmark.root / "publication.json",
    )


def cleanup(benchmark):
    benchmark.k.call(
        "exec",
        benchmark.control,
        "-c",
        "main",
        "--",
        "python3",
        "-u",
        "/opt/benchmark/cleanup_objects.py",
        output=benchmark.root / "cleanup-objects.json",
    )
    if not json.loads((benchmark.root / "cleanup-objects.json").read_text())[
        "verified_absent"
    ]:
        raise RuntimeError("Object cleanup was not verified")


def collect_publication(benchmark):
    benchmark.capture(
        "publication.json",
        "exec",
        benchmark.control,
        "-c",
        "main",
        "--",
        "cat",
        publication_path,
    )
