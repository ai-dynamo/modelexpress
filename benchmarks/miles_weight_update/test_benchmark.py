# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for the persistent matched MILES benchmark harness."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

import benchmark
import runtime_harness

ACTOR_ZERO = "(ActorModel pid=42, ip=10.0.0.7, actor_name=actor_cell0_rank0) INFO "
ACTOR_ONE = "(ActorModel pid=43, ip=10.0.0.7, actor_name=actor_cell0_rank1) INFO "


def _timer(
    timestamp: str,
    name: str,
    event: str,
    elapsed: float | None = None,
    *,
    prefix: str = ACTOR_ZERO,
) -> str:
    suffix = "" if elapsed is None else f" (elapsed: {elapsed:.1f}s)"
    return f"{timestamp} {prefix}Timer {name} {event}{suffix}"


def _records() -> list[dict[str, object]]:
    lines = [
        _timer(
            "2026-09-22T19:00:00.000000000Z",
            "update_weights",
            "start",
            prefix=ACTOR_ONE,
        ),
        _timer(
            "2026-09-22T19:00:00.000000000Z",
            "update_weights",
            "start",
        ),
        _timer(
            "2026-09-22T19:00:04.000000000Z",
            "update_weights_implementation",
            "end",
            3.0,
        ),
        _timer(
            "2026-09-22T19:00:05.000000000Z",
            "finalize_and_resume_engines",
            "end",
            0.5,
        ),
        _timer(
            "2026-09-22T19:00:06.000000000Z",
            "update_weights",
            "end",
            5.0,
        ),
        _timer(
            "2026-09-22T19:01:00.000000000Z",
            "update_weights",
            "start",
        ),
        _timer(
            "2026-09-22T19:01:02.000000000Z",
            "update_weights_implementation",
            "end",
            1.4,
        ),
        _timer(
            "2026-09-22T19:01:03.000000000Z",
            "finalize_and_resume_engines",
            "end",
            0.4,
        ),
        _timer(
            "2026-09-22T19:01:04.000000000Z",
            "update_weights",
            "end",
            2.0,
        ),
        _timer(
            "2026-09-22T19:02:00.000000000Z",
            "update_weights",
            "start",
        ),
        _timer(
            "2026-09-22T19:02:02.000000000Z",
            "update_weights_implementation",
            "end",
            1.2,
        ),
        _timer(
            "2026-09-22T19:02:03.000000000Z",
            "finalize_and_resume_engines",
            "end",
            0.3,
        ),
        _timer(
            "2026-09-22T19:02:04.000000000Z",
            "update_weights",
            "end",
            1.8,
        ),
    ]
    return benchmark.parse_timer_log("\n".join(lines), expected_updates=3)


def test_persistent_output_rejects_temporary_roots():
    for path in (Path("/tmp/results"), Path("/var/tmp/results"), Path("/dev/shm/x")):
        with pytest.raises(ValueError, match="persistent"):
            benchmark.validate_persistent_output(path)


def test_persistent_output_accepts_workspace_path():
    assert benchmark.validate_persistent_output(Path("/workspace/results")) == Path(
        "/workspace/results"
    )


def test_example_config_pins_materialized_dapo_dataset():
    payload = json.loads((Path(__file__).with_name("config.json.example")).read_text())

    assert payload["workload"]["dataset_path"] == (
        "/workspace/datasets/DAPO-Math-17k/repo/dapo-math-17k.jsonl"
    )
    assert payload["workload"]["dataset_revision"] == (
        "2e65612930298bde4c5d58fd97b3f23a483aaff9"
    )
    assert payload["workload"]["dataset_sha256"] == (
        "cc9c39c2aa19177abe9464741e121cf4cac90fd25484ef3cdf86535101e3a5b6"
    )
    assert payload["workload"]["model_path"] == (
        "/workspace/models/checkpoints/DeepSeek-V4-Flash-FP8"
    )
    assert payload["workload"]["checkpoint_manifest_path"] == (
        "/workspace/models/checkpoints/DeepSeek-V4-Flash-FP8/repo-files.sha256"
    )
    assert payload["workload"]["checkpoint_revision"] == (
        "ae01d80c06cdfe30581edfd0e1c5449dc7ed7f17"
    )
    assert payload["workload"]["checkpoint_sha256"] == (
        "bae62dda7995e9b8c51fb0cd4b0806f4975ff1fdf72adf5c7b62224f411d251d"
    )
    assert payload["experiment"]["require_symmetric_weight_checker"] is True
    assert payload["experiment"]["verify_tensor_equality"] is False


@pytest.mark.parametrize(
    "image",
    (
        "runtime:latest",
        "runtime@sha256:1234",
        "runtime@sha256:" + "A" * 64,
    ),
)
def test_images_must_be_digest_pinned(image):
    with pytest.raises(ValueError, match="digest-pinned"):
        benchmark.validate_image(image, field="runtime_image")


def test_storage_claim_names_must_be_explicit_kubernetes_names():
    assert benchmark.validate_storage_config(
        {
            "model_claim": "m2n-devbox-workspace",
            "workspace_claim": "m2n-devbox-workspace",
        }
    ) == {
        "model_claim": "m2n-devbox-workspace",
        "workspace_claim": "m2n-devbox-workspace",
    }

    with pytest.raises(ValueError, match="model_claim"):
        benchmark.validate_storage_config(
            {
                "model_claim": "",
                "workspace_claim": "m2n-devbox-workspace",
            }
        )


def test_render_replaces_persistent_paths_with_explicit_claims():
    documents = _rendered_documents()

    benchmark.configure_persistent_storage(
        documents,
        {
            "model_claim": "models-rwo",
            "workspace_claim": "workspace-rwo",
        },
    )

    volumes = documents[0]["spec"]["template"]["spec"]["volumes"]
    assert volumes == [
        {
            "name": "model-cache",
            "persistentVolumeClaim": {"claimName": "models-rwo"},
        },
        {
            "name": "shared-memory",
            "emptyDir": {"medium": "Memory", "sizeLimit": "64Gi"},
        },
        {
            "name": "run-output",
            "persistentVolumeClaim": {"claimName": "workspace-rwo"},
        },
        {
            "name": "benchmark-runtime",
            "configMap": {"name": "miles-bench-run-a-b0-r0"},
        },
        {
            "name": "benchmark-workspace",
            "persistentVolumeClaim": {"claimName": "workspace-rwo"},
        },
    ]
    mounts = documents[0]["spec"]["template"]["spec"]["containers"][0]["volumeMounts"]
    assert {(mount["name"], mount["mountPath"]) for mount in mounts} >= {
        ("benchmark-workspace", "/workspace")
    }


def test_render_rejects_ephemeral_required_mounts():
    documents = _rendered_documents()

    with pytest.raises(ValueError, match="model-cache.*persistentVolumeClaim"):
        benchmark.validate_rendered_documents(
            documents,
            arm="broadcast",
            runtime_image="runtime@sha256:" + "1" * 64,
            server_image="server@sha256:" + "2" * 64,
            storage={
                "model_claim": "models-rwo",
                "workspace_claim": "workspace-rwo",
            },
        )


def test_pvc_receipt_requires_bound_rwo_filesystem_claim():
    pvc = {
        "metadata": {"name": "workspace-rwo"},
        "spec": {
            "accessModes": ["ReadWriteOnce"],
            "volumeMode": "Filesystem",
            "volumeName": "pvc-123",
        },
        "status": {"phase": "Bound"},
    }

    benchmark.validate_pvc_receipt(pvc, claim_name="workspace-rwo")

    pvc["spec"]["accessModes"] = ["ReadWriteMany"]
    with pytest.raises(ValueError, match="ReadWriteOnce"):
        benchmark.validate_pvc_receipt(pvc, claim_name="workspace-rwo")


def test_storage_preflight_records_claims_and_rejects_other_node_consumers(
    tmp_path, monkeypatch
):
    node = {
        "metadata": {
            "name": "gpu-node-a",
            "labels": {"topology.kubernetes.io/zone": "us-east-1e"},
        },
        "status": {"allocatable": {"nvidia.com/gpu": "8"}},
    }
    pvc = {
        "metadata": {"name": "workspace-rwo"},
        "spec": {
            "accessModes": ["ReadWriteOnce"],
            "volumeMode": "Filesystem",
            "volumeName": "pvc-123",
        },
        "status": {"phase": "Bound"},
    }
    pv = {
        "metadata": {"name": "pvc-123"},
        "spec": {
            "nodeAffinity": {
                "required": {
                    "nodeSelectorTerms": [
                        {
                            "matchExpressions": [
                                {
                                    "key": "topology.kubernetes.io/zone",
                                    "operator": "In",
                                    "values": ["us-east-1e"],
                                }
                            ]
                        }
                    ]
                }
            }
        },
    }
    pods = {
        "items": [
            {
                "metadata": {"name": "devbox", "namespace": "mx-bench"},
                "spec": {
                    "nodeName": "gpu-node-a",
                    "containers": [{"name": "devbox"}],
                    "volumes": [
                        {"persistentVolumeClaim": {"claimName": "workspace-rwo"}}
                    ],
                },
                "status": {"phase": "Running"},
            }
        ]
    }
    responses = {
        "node/gpu-node-a": node,
        "pods": pods,
        "pvc/workspace-rwo": pvc,
        "pv/pvc-123": pv,
    }

    def fake_kubectl_json(
        config,
        *,
        args,
        output_path,
        output_dir,
        command_name,
        environment,
        all_namespaces=False,
    ):
        del config, output_dir, command_name, environment
        if args == ["get", "pods"]:
            assert all_namespaces is True
        payload = responses[args[1]]
        output_path.write_text(json.dumps(payload))
        return payload

    monkeypatch.setattr(benchmark, "_kubectl_json", fake_kubectl_json)
    config = {
        "cluster": {"namespace": "mx-bench", "node_name": "gpu-node-a"},
        "storage": {
            "model_claim": "workspace-rwo",
            "workspace_claim": "workspace-rwo",
        },
        "topology": {"total_gpus": 4},
    }

    benchmark.preflight_persistent_storage(config, run_dir=tmp_path)

    receipt = json.loads((tmp_path / "preflight" / "storage.json").read_text())
    assert receipt["claims"]["workspace-rwo"]["roles"] == [
        "model_claim",
        "workspace_claim",
    ]

    pods["items"][0]["spec"]["nodeName"] = "gpu-node-b"
    with pytest.raises(ValueError, match="active on node gpu-node-b"):
        benchmark.preflight_persistent_storage(
            config,
            run_dir=tmp_path / "failure",
        )


def test_schedule_is_deterministic_and_paired():
    first = benchmark.build_schedule(blocks=5, seed=3304)
    second = benchmark.build_schedule(blocks=5, seed=3304)
    different_seed = benchmark.build_schedule(blocks=5, seed=3305)

    assert first == second
    assert first != different_seed
    assert len(first) == 10
    for block in range(5):
        block_items = [item for item in first if item["block"] == block]
        assert {item["arm"] for item in block_items} == {
            "broadcast",
            "modelexpress",
        }
        assert [item["order"] for item in block_items] == [0, 1]
        assert {item["repetition"] for item in block_items} == {block}


def test_parser_accepts_only_rank_zero_complete_triplets():
    records = _records()

    assert len(records) == 3
    assert records[0] == {
        "end_timestamp_utc": "2026-09-22T19:00:06+00:00",
        "e2e_update_s": 5.0,
        "finalize_and_resume_s": 0.5,
        "index": 0,
        "setup_residual_raw_s": 1.5,
        "setup_residual_s": 1.5,
        "start_timestamp_utc": "2026-09-22T19:00:00+00:00",
        "update_weights_implementation_s": 3.0,
    }


def test_parser_rejects_incomplete_updates():
    log = "\n".join(
        (
            _timer(
                "2026-09-22T19:00:00.000000000Z",
                "update_weights",
                "start",
            ),
            _timer(
                "2026-09-22T19:00:01.000000000Z",
                "update_weights_implementation",
                "end",
                1.0,
            ),
            _timer(
                "2026-09-22T19:00:02.000000000Z",
                "update_weights",
                "end",
                2.0,
            ),
        )
    )

    with pytest.raises(ValueError, match="complete timer triplet"):
        benchmark.parse_timer_log(log, expected_updates=1)


def test_classification_matches_pr3304_index_policy():
    samples = benchmark.classify_samples(
        [{"index": index} for index in range(21)],
        arm="modelexpress",
        block=2,
        run_id="run-a",
    )

    assert samples[0]["classification"] == "cold"
    assert samples[1]["classification"] == "steady"
    assert samples[12]["reference_step"] is True
    assert samples[19]["classification"] == "steady"
    assert samples[20]["classification"] == "tail_excluded"
    assert all(item["arm"] == "modelexpress" for item in samples)
    assert samples[20]["block"] == 2


def test_job_summary_separates_setup_and_timing_boundaries():
    records = _records()
    samples = [
        records[0] | {"classification": "cold"},
        records[1] | {"classification": "steady"},
        records[2] | {"classification": "steady"},
    ]

    summary = benchmark.summarize_job(
        samples,
        pod_start_timestamp="2026-09-22T18:59:50Z",
    )

    assert summary["setup_before_first_update_s"] == 10.0
    assert summary["first_update"]["update_weights_implementation_s"] == 3.0
    assert summary["first_update"]["e2e_update_s"] == 5.0
    assert summary["steady_state"]["update_weights_implementation_s"][
        "median"
    ] == pytest.approx(1.3)
    assert summary["steady_state"]["e2e_update_s"]["median"] == pytest.approx(1.9)


def test_paired_summary_uses_implementation_timer_as_headline():
    samples = []
    for block, broadcast, modelexpress in (
        (0, (10.0, 12.0), (2.0, 2.2)),
        (1, (8.0, 8.4), (2.0, 2.0)),
    ):
        for arm, values in (
            ("broadcast", broadcast),
            ("modelexpress", modelexpress),
        ):
            for index, value in enumerate(values):
                samples.append(
                    {
                        "arm": arm,
                        "block": block,
                        "classification": "steady",
                        "update_weights_implementation_s": value,
                        "e2e_update_s": value + 1,
                        "timing_valid": True,
                        "receipt_complete": True,
                        "correctness": {
                            "all_names_installed": True,
                            "all_rollout_ready": True,
                            "digest_equal": True,
                            "training_run_completed": True,
                            "version_agreement": True,
                            "weight_checker_enabled": True,
                        },
                        "bytes_total": 100,
                        "bytes_m2n": 0 if arm == "broadcast" else 80,
                        "bytes_residual": 100 if arm == "broadcast" else 20,
                        "coverage_fraction": 1.0,
                        "status": "ok",
                        "index": index,
                        "weight_version": f"v{index}",
                    }
                )

    summary = benchmark.summarize_pairs(samples, bootstrap_samples=200, seed=7)

    assert len(summary["blocks"]) == 2
    assert summary["blocks"][0]["implementation"]["speedup"] == pytest.approx(
        11.0 / 2.1
    )
    assert summary["headline"]["paired_blocks"] == 2
    assert summary["headline"]["median_implementation_speedup"] > 4
    assert summary["headline"]["exact_index_contract"] is False
    assert summary["headline"]["acceptance_ready"] is False
    assert summary["diagnostics"]["median_rollout_ready_e2e_speedup"] > 3


def test_acceptance_requires_exact_steady_indices_1_through_19():
    samples = []
    for arm, value in (("broadcast", 10.0), ("modelexpress", 2.0)):
        for index in range(1, 20):
            samples.append(
                {
                    "arm": arm,
                    "block": 0,
                    "classification": "steady",
                    "update_weights_implementation_s": value,
                    "e2e_update_s": value + 1,
                    "timing_valid": True,
                    "receipt_complete": True,
                    "correctness": {"all_rollout_ready": True},
                    "bytes_total": 100,
                    "bytes_m2n": 0 if arm == "broadcast" else 80,
                    "bytes_residual": 100 if arm == "broadcast" else 20,
                    "coverage_fraction": 1.0,
                    "status": "ok",
                    "index": index,
                    "weight_version": f"v{index}",
                }
            )

    summary = benchmark.summarize_pairs(samples, bootstrap_samples=20, seed=7)

    assert summary["headline"]["exact_index_contract"] is True
    assert summary["headline"]["acceptance_ready"] is True


def test_missing_structured_receipts_keep_timing_but_block_acceptance():
    samples = []
    for arm, value in (("broadcast", 10.0), ("modelexpress", 2.0)):
        for index in range(20):
            samples.append(
                {
                    "arm": arm,
                    "block": 0,
                    "classification": "steady",
                    "update_weights_implementation_s": value,
                    "e2e_update_s": value + 1,
                    "timing_valid": True,
                    "receipt_complete": False,
                    "status": "ok",
                    "index": index,
                    "bytes_total": None,
                    "bytes_m2n": None,
                    "bytes_residual": None,
                    "coverage_fraction": None,
                }
            )

    summary = benchmark.summarize_pairs(samples, bootstrap_samples=20, seed=7)

    assert summary["headline"]["median_implementation_speedup"] == 5.0
    assert summary["headline"]["acceptance_ready"] is False
    assert summary["modes"]["broadcast"]["receipt_complete"] is False


def test_pr3304_reference_window_excludes_cold_and_tail():
    records = [
        {"index": index, "update_weights_implementation_s": float(index)}
        for index in range(21)
    ]

    classified = benchmark.classify_pr3304_reference(records)

    assert classified[0]["classification"] == "cold"
    assert [
        item["index"] for item in classified if item["classification"] == "steady"
    ] == list(range(1, 20))
    assert classified[12]["reference_step"] is True
    assert classified[20]["classification"] == "tail_excluded"


def test_jsonl_writer_is_deterministic(tmp_path):
    output = tmp_path / "records.jsonl"
    records = [{"z": 2, "a": 1}, {"b": 3}]

    benchmark.write_jsonl(output, records)

    assert (
        output.read_text()
        == "\n".join(
            json.dumps(item, sort_keys=True, separators=(",", ":")) for item in records
        )
        + "\n"
    )


def test_source_receipts_reject_dirty_provenance(monkeypatch):
    monkeypatch.setattr(
        benchmark,
        "git_receipt",
        lambda path: {
            "branch": "main",
            "commit": "a" * 40,
            "dirty": path.name == "dirty",
            "path": str(path),
            "remote": "origin",
        },
    )

    with pytest.raises(ValueError, match="dirty repositories: dirty"):
        benchmark.source_receipts(
            {
                "source_repositories": {
                    "clean": "/repos/clean",
                    "dirty": "/repos/dirty",
                }
            }
        )


def test_job_environment_contains_only_benchmark_contract():
    config = {
        "cluster": {
            "context": "aws-dev-02",
            "namespace": "mx-bench",
            "gpu_product": "NVIDIA-H100-80GB-HBM3",
            "node_name": "node-a",
            "runtime_pull_secret": "pull-secret",
        },
        "experiment": {
            "nccl_debug": "INFO",
            "num_rollouts": 20,
            "require_symmetric_weight_checker": True,
            "verify_tensor_equality": False,
        },
        "images": {
            "runtime": "runtime@sha256:" + "1" * 64,
            "server": "server@sha256:" + "2" * 64,
        },
        "topology": {
            "actor_gpus": 2,
            "m2n_pp_concurrency": 2,
            "pipeline_parallel_size": 2,
            "rollout_gpus": 2,
            "rollout_gpus_per_engine": 1,
            "total_gpus": 4,
        },
        "workload": {
            "checkpoint_manifest_path": (
                "/workspace/models/Qwen2.5-0.5B-Instruct/repo-files.sha256"
            ),
            "checkpoint_revision": "checkpoint-revision-a",
            "checkpoint_sha256": "4" * 64,
            "dataset_path": "/workspace/datasets/dapo.jsonl",
            "dataset_revision": "revision-a",
            "dataset_sha256": "3" * 64,
            "miles_model_type": "qwen2.5-0.5B",
            "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
            "model_path": "/workspace/models/Qwen2.5-0.5B-Instruct",
        },
    }
    item = {"arm": "modelexpress", "block": 1, "repetition": 0}

    environment = benchmark.job_environment(config, item, run_id="bench")

    assert environment == {
        "ACTOR_GPUS": "2",
        "BENCH_NODE_NAME": "node-a",
        "GPU_PRODUCT": "NVIDIA-H100-80GB-HBM3",
        "KUBE_CONTEXT": "aws-dev-02",
        "MILES_BENCH_BLOCK": "1",
        "MILES_BENCH_E2E_RECEIPTS": (
            "/root/shared_data/miles-benchmark/bench/block-1-0-external.e2e.jsonl"
        ),
        "MILES_BENCH_RAW_RECEIPTS": (
            "/root/shared_data/miles-benchmark/bench/block-1-0-external.raw.jsonl"
        ),
        "MILES_BENCH_REPETITION": "0",
        "MILES_BENCH_REQUIRE_WEIGHT_CHECKER": "1",
        "MILES_BENCH_ROLLOUTS": "20",
        "MILES_BENCH_RUN_LABEL": "bench",
        "MILES_CHECKPOINT_MANIFEST_PATH": (
            "/workspace/models/Qwen2.5-0.5B-Instruct/repo-files.sha256"
        ),
        "MILES_CHECKPOINT_REVISION": "checkpoint-revision-a",
        "MILES_CHECKPOINT_SHA256": "4" * 64,
        "MILES_DATASET_PATH": "/workspace/datasets/dapo.jsonl",
        "MILES_DATASET_REVISION": "revision-a",
        "MILES_DATASET_SHA256": "3" * 64,
        "MILES_MODEL_PATH": "/workspace/models/Qwen2.5-0.5B-Instruct",
        "MILES_MODEL_TYPE": "qwen2.5-0.5B",
        "MILES_WEIGHT_TRANSFER_MODE": "external",
        "MODEL_ID": "Qwen/Qwen2.5-0.5B-Instruct",
        "MX_MILES_VERIFY_TENSOR_EQUALITY": "0",
        "MX_NCCL_REFIT_NUM_STREAMS": "2",
        "MX_RESHARD_REQUIRE_FULL_COVERAGE": "1",
        "MX_SERVER_IMAGE": "server@sha256:" + "2" * 64,
        "NCCL_DEBUG": "INFO",
        "NAMESPACE": "mx-bench",
        "PIPELINE_PARALLEL_SIZE": "2",
        "ROLLOUT_GPUS": "2",
        "ROLLOUT_GPUS_PER_ENGINE": "1",
        "RUN_ID": "bench",
        "RUNTIME_PULL_SECRET": "pull-secret",
        "RUNTIME_IMAGE": "runtime@sha256:" + "1" * 64,
        "TOTAL_GPUS": "4",
    }


def test_validate_structured_trial_requires_complete_hybrid_receipts():
    trial = {
        "schema": benchmark.SCHEMA,
        "mode": "modelexpress_m2n",
        "trial": 4,
        "step": 4,
        "cold": False,
        "headline_start_boundary": benchmark.HEADLINE_START_BOUNDARY,
        "headline_end_boundary": benchmark.HEADLINE_END_BOUNDARY,
        "update_weights_implementation_s": 1.5,
        "e2e_start_boundary": benchmark.E2E_START_BOUNDARY,
        "e2e_end_boundary": benchmark.E2E_END_BOUNDARY,
        "e2e_update_s": 2.0,
        "transport_s": 1.0,
        "residual_s": 0.2,
        "install_s": 0.3,
        "bytes_total": 100,
        "bytes_m2n": 80,
        "bytes_residual": 20,
        "coverage_fraction": 1.0,
        "status": "ok",
        "correctness": {
            "all_names_installed": True,
            "all_rollout_ready": True,
            "digest_equal": True,
            "training_run_completed": True,
            "version_agreement": True,
            "weight_checker_enabled": True,
        },
        "excluded_names": [],
        "receipt_complete": True,
        "weight_version": "v9",
    }

    benchmark.validate_structured_trial(trial, mode="modelexpress_m2n")

    trial["correctness"]["digest_equal"] = None
    trial["correctness"]["all_names_installed"] = None
    trial["receipt_complete"] = False
    benchmark.validate_structured_trial(trial, mode="modelexpress_m2n")

    trial["correctness"]["digest_equal"] = False
    with pytest.raises(ValueError, match="failed a correctness gate"):
        benchmark.validate_structured_trial(trial, mode="modelexpress_m2n")

    trial["correctness"]["digest_equal"] = True
    trial["correctness"]["all_names_installed"] = True
    trial["receipt_complete"] = True
    trial["step"] = 9
    with pytest.raises(ValueError, match="step must equal"):
        benchmark.validate_structured_trial(trial, mode="modelexpress_m2n")

    trial["step"] = 4
    trial["bytes_residual"] = 19
    with pytest.raises(ValueError, match="byte accounting"):
        benchmark.validate_structured_trial(trial, mode="modelexpress_m2n")


def test_rendered_broadcast_omits_modelexpress_resources():
    documents = _rendered_documents()
    storage = {
        "model_claim": "models-rwo",
        "workspace_claim": "workspace-rwo",
    }
    benchmark.configure_persistent_storage(documents, storage)

    benchmark.validate_rendered_documents(
        documents,
        arm="broadcast",
        runtime_image="runtime@sha256:" + "1" * 64,
        server_image="server@sha256:" + "2" * 64,
        storage=storage,
    )

    documents.append(
        {
            "kind": "Deployment",
            "metadata": {
                "name": "mx-server",
                "labels": {"app.kubernetes.io/name": "modelexpress-server"},
            },
            "spec": {"template": {"spec": {"containers": []}}},
        }
    )
    with pytest.raises(ValueError, match="must omit"):
        benchmark.validate_rendered_documents(
            documents,
            arm="broadcast",
            runtime_image="runtime@sha256:" + "1" * 64,
            server_image="server@sha256:" + "2" * 64,
            storage=storage,
        )


def test_runtime_configmap_propagates_pinned_workload_contract():
    documents = [
        {
            "kind": "Job",
            "metadata": {"name": "job"},
            "spec": {
                "template": {
                    "spec": {
                        "containers": [
                            {
                                "name": "miles",
                                "env": [
                                    {
                                        "name": "PYTHONPATH",
                                        "value": "/workspace/runtime",
                                    }
                                ],
                                "volumeMounts": [],
                            }
                        ],
                        "volumes": [],
                    }
                }
            },
        }
    ]
    environment = benchmark.job_environment(
        _runtime_config(),
        {"arm": "broadcast", "block": 0, "repetition": 0},
        run_id="run-a",
    )

    benchmark.configure_runtime_harness(documents, environment=environment)

    configmap = next(
        document for document in documents if document["kind"] == "ConfigMap"
    )
    job = next(document for document in documents if document["kind"] == "Job")
    container = job["spec"]["template"]["spec"]["containers"][0]
    rendered_environment = {entry["name"]: entry["value"] for entry in container["env"]}
    assert job["metadata"]["labels"] == {
        "modelexpress.nvidia.com/benchmark-block": "0",
        "modelexpress.nvidia.com/benchmark-mode": "broadcast",
        "modelexpress.nvidia.com/benchmark-repetition": "0",
        "modelexpress.nvidia.com/benchmark-run": "run-a",
    }
    assert rendered_environment["MILES_DATASET_PATH"] == (
        "/workspace/datasets/dapo.jsonl"
    )
    assert rendered_environment["MILES_DATASET_REVISION"] == "revision-a"
    assert rendered_environment["MILES_DATASET_SHA256"] == "3" * 64
    assert rendered_environment["MILES_MODEL_PATH"] == "/workspace/models/model"
    assert rendered_environment["MILES_CHECKPOINT_MANIFEST_PATH"] == (
        "/workspace/models/model/repo-files.sha256"
    )
    assert rendered_environment["MILES_CHECKPOINT_REVISION"] == (
        "checkpoint-revision-a"
    )
    assert rendered_environment["MILES_CHECKPOINT_SHA256"] == "4" * 64
    assert rendered_environment["MILES_BENCH_REQUIRE_WEIGHT_CHECKER"] == "1"
    assert rendered_environment["PYTHONPATH"] == (
        "/opt/miles-benchmark:/workspace/runtime"
    )
    assert container["command"] == [
        "python3",
        "/opt/miles-benchmark/runtime_harness.py",
    ]
    assert "gsm8k" not in configmap["data"]["runtime_harness.py"].lower()


def test_launcher_scaffold_does_not_mutate_configured_workload():
    environment = {
        "MODEL_ID": "sgl-project/DeepSeek-V4-Flash-FP8",
        "MILES_MODEL_TYPE": "deepseek-v4-flash",
        "RUN_ID": "run-a",
    }

    rendered = benchmark.launcher_render_environment(environment)

    assert rendered["MODEL_ID"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert rendered["MILES_MODEL_TYPE"] == "qwen2.5-0.5B"
    assert rendered["RUN_ID"] == "run-a"
    assert environment["MODEL_ID"] == "sgl-project/DeepSeek-V4-Flash-FP8"
    assert environment["MILES_MODEL_TYPE"] == "deepseek-v4-flash"


def test_render_validation_rejects_stale_runtime_configmap():
    documents = _rendered_documents()
    storage = {
        "model_claim": "models-rwo",
        "workspace_claim": "workspace-rwo",
    }
    benchmark.configure_persistent_storage(documents, storage)
    configmap = next(
        document for document in documents if document["kind"] == "ConfigMap"
    )
    configmap["data"]["runtime_harness.py"] += "\n# stale render\n"

    with pytest.raises(ValueError, match="current frozen source"):
        benchmark.validate_rendered_documents(
            documents,
            arm="broadcast",
            runtime_image="runtime@sha256:" + "1" * 64,
            server_image="server@sha256:" + "2" * 64,
            storage=storage,
        )


def test_config_accepts_performance_default_with_symmetric_checker():
    payload = json.loads((Path(__file__).with_name("config.json.example")).read_text())
    payload["images"] = {
        "runtime": "runtime@sha256:" + "0123456789abcdef" * 4,
        "server": "server@sha256:" + "fedcba9876543210" * 4,
    }
    payload["output_root"] = "/workspace/test-benchmark-results"
    payload["run_id"] = "test-run"

    normalized = benchmark.normalize_config(payload)

    assert normalized["experiment"]["verify_tensor_equality"] is False
    assert normalized["experiment"]["require_symmetric_weight_checker"] is True

    payload["experiment"]["require_symmetric_weight_checker"] = False
    with pytest.raises(ValueError, match="symmetric MILES weight checker"):
        benchmark.normalize_config(payload)


def test_external_server_resources_are_unique_to_the_benchmark_job():
    documents = _rendered_documents()
    documents[:0] = [
        {
            "kind": "Service",
            "metadata": {
                "name": "mx-server",
                "labels": {"app.kubernetes.io/name": "modelexpress-server"},
            },
            "spec": {"selector": {"app": "mx-server"}},
        },
        {
            "kind": "Deployment",
            "metadata": {
                "name": "mx-server",
                "labels": {"app.kubernetes.io/name": "modelexpress-server"},
            },
            "spec": {
                "selector": {"matchLabels": {"app": "mx-server"}},
                "template": {
                    "metadata": {"labels": {"app": "mx-server"}},
                    "spec": {
                        "containers": [
                            {
                                "name": "modelexpress-server",
                                "image": "server@sha256:" + "2" * 64,
                            }
                        ]
                    },
                },
            },
        },
    ]
    job = next(document for document in documents if document["kind"] == "Job")
    job_container = job["spec"]["template"]["spec"]["containers"][0]
    job_container["env"].append(
        {"name": "MX_SERVER_ADDRESS", "value": "mx-server:8000"}
    )
    storage = {
        "model_claim": "models-rwo",
        "workspace_claim": "workspace-rwo",
    }
    benchmark.configure_persistent_storage(documents, storage)
    benchmark.configure_scoped_server_resources(
        documents,
        server_name="mx-run-a-b0-r0",
        run_id="run-a",
        block="0",
        repetition="0",
    )

    benchmark.validate_rendered_documents(
        documents,
        arm="modelexpress",
        runtime_image="runtime@sha256:" + "1" * 64,
        server_image="server@sha256:" + "2" * 64,
        storage=storage,
        server_name="mx-run-a-b0-r0",
    )
    assert {
        document["metadata"]["name"]
        for document in documents
        if document["kind"] in {"Service", "Deployment"}
    } == {"mx-run-a-b0-r0"}


def test_observed_topology_attests_node_gpu_and_writable_claims():
    config = {
        "cluster": {
            "gpu_product": "NVIDIA-H100-80GB-HBM3",
            "node_name": "gpu-node-a",
        },
        "storage": {
            "model_claim": "workspace-rwo",
            "workspace_claim": "workspace-rwo",
        },
        "topology": {"total_gpus": 4},
    }
    pods = {
        "items": [
            {
                "metadata": {"name": "job-pod"},
                "spec": {
                    "nodeName": "gpu-node-a",
                    "containers": [
                        {
                            "name": "miles",
                            "resources": {
                                "limits": {"nvidia.com/gpu": "4"},
                                "requests": {"nvidia.com/gpu": "4"},
                            },
                            "volumeMounts": [
                                {
                                    "name": "model-cache",
                                    "mountPath": "/models",
                                },
                                {
                                    "name": "run-output",
                                    "mountPath": "/root/shared_data",
                                },
                                {
                                    "name": "benchmark-workspace",
                                    "mountPath": "/workspace",
                                },
                            ],
                        }
                    ],
                    "volumes": [
                        {
                            "name": "model-cache",
                            "persistentVolumeClaim": {"claimName": "workspace-rwo"},
                        },
                        {
                            "name": "run-output",
                            "persistentVolumeClaim": {"claimName": "workspace-rwo"},
                        },
                        {
                            "name": "benchmark-workspace",
                            "persistentVolumeClaim": {"claimName": "workspace-rwo"},
                        },
                    ],
                },
            }
        ]
    }
    nodes = {
        "gpu-node-a": {
            "metadata": {
                "name": "gpu-node-a",
                "labels": {"nvidia.com/gpu.product": "NVIDIA-H100-80GB-HBM3"},
            },
            "status": {"allocatable": {"nvidia.com/gpu": "8"}},
        }
    }

    receipt = benchmark.validate_observed_topology(
        config,
        pods=pods,
        nodes=nodes,
    )

    assert receipt["same_node"] is True
    assert receipt["gpu_count_requested"] == 4
    assert len(receipt["writers"]) == 3


def test_observed_topology_rejects_zero_mismatched_and_sidecar_gpu_requests():
    config, pods, nodes = _topology_fixture()
    workload = pods["items"][0]["spec"]["containers"][0]

    workload["resources"]["requests"]["nvidia.com/gpu"] = "0"
    with pytest.raises(ValueError, match="exactly match"):
        benchmark.validate_observed_topology(config, pods=pods, nodes=nodes)

    workload["resources"]["requests"]["nvidia.com/gpu"] = "4"
    pods["items"][0]["spec"]["containers"].append(
        {
            "name": "sidecar",
            "resources": {"limits": {"nvidia.com/gpu": "1"}},
        }
    )
    with pytest.raises(ValueError, match="only the miles workload"):
        benchmark.validate_observed_topology(config, pods=pods, nodes=nodes)


def test_gpu_headroom_rejects_fully_consumed_node():
    node = {"status": {"allocatable": {"nvidia.com/gpu": "8"}}}
    pods = {
        "items": [
            {
                "metadata": {"name": "devbox"},
                "spec": {
                    "nodeName": "gpu-node-a",
                    "containers": [
                        {
                            "name": "devbox",
                            "resources": {
                                "limits": {"nvidia.com/gpu": "8"},
                            },
                        }
                    ],
                },
                "status": {"phase": "Running"},
            }
        ]
    }

    with pytest.raises(ValueError, match="0 free GPUs"):
        benchmark.validate_node_gpu_headroom(
            node,
            pods,
            node_name="gpu-node-a",
            required_gpus=4,
        )


def test_runtime_driver_uses_pinned_dapo_without_implicit_download():
    arguments = runtime_harness.build_train_args(
        model_path=Path("/workspace/models/model"),
        dataset_path=Path("/workspace/datasets/dapo.jsonl"),
        transfer_mode="external",
        num_rollouts=20,
        actor_gpus=2,
        pipeline_parallel_size=2,
        rollout_gpus=2,
        rollout_gpus_per_engine=1,
    )

    assert "--prompt-data /workspace/datasets/dapo.jsonl" in arguments
    assert (
        "modelexpress_rl.collective.integrations.miles_pr3304:build_protocol"
        in arguments
    )
    assert "gsm8k" not in arguments.lower()
    assert "download" not in arguments.lower()
    assert "--ci-test" in arguments


def test_runtime_state_accounts_hybrid_routed_and_residual_bytes(tmp_path):
    class Entry:
        name = "model.layers.0.weight"
        global_shape = (2, 4)
        dtype = "bfloat16"

    class Plan:
        bulk = (Entry(),)

    class TopologyPlan:
        plan = Plan()

    class Projection:
        topology_plan = TopologyPlan()

    class Protocol:
        _projection = Projection()

    class Tensor:
        @staticmethod
        def numel():
            return 4

        @staticmethod
        def element_size():
            return 2

    state = runtime_harness.RuntimeReceiptState(
        mode="external",
        output_path=tmp_path / "raw.jsonl",
    )
    state.record_hybrid_plan(Protocol())
    state.record_bucket([("model.norm.weight", Tensor())])
    state.weight_version = 1
    state.implementation_s = 1.0
    state.implementation_started_at = 100.0
    state.digest_verified = True

    record = state.raw_record(e2e_update_s=2.0)

    assert record["bytes_m2n"] == 16
    assert record["bytes_residual"] == 8
    assert record["bytes_total"] == 24
    assert record["names"] == [
        "model.layers.0.weight",
        "model.norm.weight",
    ]


def test_runtime_instrumentation_rejects_legacy_whole_round_protocol(tmp_path):
    class Protocol:
        owns_update_round = True

    state = runtime_harness.RuntimeReceiptState(
        mode="external",
        output_path=tmp_path / "raw.jsonl",
    )

    with pytest.raises(RuntimeError, match="normal WeightTransferProtocol lifecycle"):
        runtime_harness._instrument_protocol(Protocol(), state)


def test_runtime_receipts_merge_pp_local_residuals_with_one_routed_plan():
    common = {
        "digest_verified": True,
        "excluded_names": [],
        "implementation_started_at_epoch_s": 100.0,
        "install_s": 0.2,
        "m2n_name_sizes": {"expert.weight": 16},
        "residual_s": 0.1,
        "transport_s": 1.0,
        "trial": 0,
        "update_weights_implementation_s": 1.5,
        "weight_version": "1",
    }
    records = [
        common
        | {
            "name_sizes": {"expert.weight": 16, "layer.0.norm": 8},
            "names": ["expert.weight", "layer.0.norm"],
            "residual_name_sizes": {"layer.0.norm": 8},
        },
        common
        | {
            "name_sizes": {"expert.weight": 16, "layer.1.norm": 8},
            "names": ["expert.weight", "layer.1.norm"],
            "residual_name_sizes": {"layer.1.norm": 8},
        },
    ]

    combined = runtime_harness._combine_rank_records(
        records,
        mode="external",
    )

    assert combined["names"] == [
        "expert.weight",
        "layer.0.norm",
        "layer.1.norm",
    ]
    assert combined["bytes_m2n"] == 16
    assert combined["bytes_residual"] == 16
    assert combined["bytes_total"] == 32


def test_runtime_driver_validates_checkpoint_receipt_dataset_digest_and_revision(
    tmp_path,
):
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "config.json").write_text("{}\n")
    (model_path / "repository.txt").write_text("Qwen/Qwen2.5-0.5B-Instruct\n")
    (model_path / ".prewarm-complete").touch()
    metadata = model_path / ".cache" / "huggingface" / "download"
    metadata.mkdir(parents=True)
    (metadata / "config.json.metadata").write_text("checkpoint-revision-a\n")
    manifest_path = model_path / "repo-files.sha256"
    manifest_path.write_text(
        hashlib.sha256((model_path / "config.json").read_bytes()).hexdigest()
        + "  ./config.json\n"
    )
    checkpoint_digest = hashlib.sha256(
        "".join(sorted(manifest_path.read_text().splitlines(keepends=True))).encode()
    ).hexdigest()
    dataset_repo = tmp_path / "dataset"
    dataset_repo.mkdir()
    dataset_path = dataset_repo / "dapo.jsonl"
    dataset_path.write_text('{"prompt":"1+1","label":"2"}\n')
    subprocess.run(["git", "init", "-q", str(dataset_repo)], check=True)
    subprocess.run(
        ["git", "-C", str(dataset_repo), "add", "dapo.jsonl"],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(dataset_repo),
            "-c",
            "user.name=Benchmark Test",
            "-c",
            "user.email=benchmark@example.com",
            "commit",
            "-q",
            "-m",
            "fixture",
        ],
        check=True,
    )
    revision = subprocess.run(
        ["git", "-C", str(dataset_repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    digest = hashlib.sha256(dataset_path.read_bytes()).hexdigest()

    runtime_harness.validate_workload_files(
        checkpoint_manifest_path=manifest_path,
        checkpoint_revision="checkpoint-revision-a",
        checkpoint_sha256=checkpoint_digest,
        model_path=model_path,
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        dataset_path=dataset_path,
        dataset_revision=revision,
        dataset_sha256=digest,
    )

    (model_path / "config.json").write_text('{"mutated":true}\n')
    with pytest.raises(ValueError, match="checkpoint file SHA-256 mismatch"):
        runtime_harness.validate_workload_files(
            checkpoint_manifest_path=manifest_path,
            checkpoint_revision="checkpoint-revision-a",
            checkpoint_sha256=checkpoint_digest,
            model_path=model_path,
            model_id="Qwen/Qwen2.5-0.5B-Instruct",
            dataset_path=dataset_path,
            dataset_revision=revision,
            dataset_sha256=digest,
        )
    (model_path / "config.json").write_text("{}\n")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        runtime_harness.validate_workload_files(
            checkpoint_manifest_path=manifest_path,
            checkpoint_revision="checkpoint-revision-a",
            checkpoint_sha256=checkpoint_digest,
            model_path=model_path,
            model_id="Qwen/Qwen2.5-0.5B-Instruct",
            dataset_path=dataset_path,
            dataset_revision=revision,
            dataset_sha256="0" * 64,
        )

    with pytest.raises(ValueError, match="checkpoint manifest SHA-256 mismatch"):
        runtime_harness.validate_workload_files(
            checkpoint_manifest_path=manifest_path,
            checkpoint_revision="checkpoint-revision-a",
            checkpoint_sha256="0" * 64,
            model_path=model_path,
            model_id="Qwen/Qwen2.5-0.5B-Instruct",
            dataset_path=dataset_path,
            dataset_revision=revision,
            dataset_sha256=digest,
        )


def test_runtime_driver_rejects_checkpoint_manifest_outside_model_tree(tmp_path):
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "config.json").write_text("{}\n")
    manifest_path = tmp_path / "repo-files.sha256"
    manifest_path.write_text("0" * 64 + "  ./config.json\n")

    with pytest.raises(ValueError, match="must be inside model checkpoint"):
        runtime_harness.validate_checkpoint_receipt(
            checkpoint_manifest_path=manifest_path,
            checkpoint_revision="revision-a",
            checkpoint_sha256="0" * 64,
            model_path=model_path,
            model_id="model-a",
        )


def test_external_runtime_contract_accepts_normal_lifecycle_loader():
    class Protocol:
        def __init__(self, args):
            self.args = args

    def get_protocol(args):
        module_name, attribute = args.update_weight_transfer_protocol.split(":")
        return getattr(__import__(module_name), attribute)(args)

    runtime_harness.validate_external_runtime_contract(
        get_protocol=get_protocol,
        protocol_type=Protocol,
        module_factory=lambda args: Protocol(args),
    )


def test_external_runtime_contract_rejects_stale_whole_round_loader():
    class Protocol:
        def __init__(self, args):
            self.args = args

    def stale_loader(args):
        module_name, attribute = args.update_weight_transfer_protocol.split(":")
        protocol = getattr(__import__(module_name), attribute)(args)
        if not getattr(protocol, "owns_update_round", False):
            raise TypeError("external protocols must own the update round")
        return protocol

    with pytest.raises(RuntimeError, match="normal WeightTransferProtocol lifecycle"):
        runtime_harness.validate_external_runtime_contract(
            get_protocol=stale_loader,
            protocol_type=Protocol,
            module_factory=lambda args: Protocol(args),
        )


def test_runtime_receipt_producer_emits_acceptance_schema(tmp_path, capsys):
    raw_path = tmp_path / "raw.jsonl"
    e2e_path = tmp_path / "e2e.jsonl"
    raw_records = []
    e2e_records = []
    for index in range(21):
        raw_records.append(
            {
                "bytes_m2n": 80,
                "bytes_residual": 20,
                "bytes_total": 100,
                "digest_verified": True,
                "e2e_update_s": 99.0,
                "excluded_names": [],
                "install_s": 0.2,
                "implementation_started_at_epoch_s": 1000.0 + index,
                "names": ["layer.weight"],
                "residual_s": 0.1,
                "transport_s": 1.0,
                "trial": index,
                "update_weights_implementation_s": 1.5,
                "weight_version": str(index + 1),
            }
        )
        e2e_records.append(
            {
                "all_names_installed": True,
                "all_rollout_ready": True,
                "all_rollout_ready_epoch_s": 1002.0 + index,
                "weight_checker_enabled": True,
                "version_agreement": True,
                "versions": [str(index + 1)],
                "weight_version": str(index + 1),
            }
        )
    e2e_records.append(
        {
            "all_names_installed": True,
            "digest_verified": True,
            "record_type": "weight_compare",
            "weight_version": "1",
        }
    )
    raw_path.write_text("".join(json.dumps(record) + "\n" for record in raw_records))
    e2e_path.write_text("".join(json.dumps(record) + "\n" for record in e2e_records))

    trials = runtime_harness.finalize_receipts(
        raw_path,
        e2e_path=e2e_path,
        mode="external",
        expected_updates=21,
    )

    emitted = [
        json.loads(line)
        for line in capsys.readouterr().out.splitlines()
        if line.strip()
    ]
    assert emitted == trials
    assert [trial["trial"] for trial in trials] == list(range(21))
    assert all(trial["schema"] == benchmark.SCHEMA for trial in trials)
    assert trials[0]["receipt_complete"] is True
    assert all(trial["receipt_complete"] is False for trial in trials[1:])
    assert trials[0]["correctness"]["digest_equal"] is True
    assert all(trial["correctness"]["digest_equal"] is None for trial in trials[1:])
    assert all(
        trial["correctness"]["training_run_completed"] is True for trial in trials
    )
    assert all("generation_ok" not in trial["correctness"] for trial in trials)
    assert all(trial["e2e_update_s"] == 2.0 for trial in trials)


def test_weight_compare_rejects_corruption_and_incomplete_results():
    assert runtime_harness.weight_compare_succeeded([{"success": True, "equal": True}])
    assert not runtime_harness.weight_compare_succeeded(
        [{"success": True, "equal": False}]
    )
    assert not runtime_harness.weight_compare_succeeded([{"success": False}])
    assert not runtime_harness.weight_compare_succeeded([])
    assert not runtime_harness.weight_compare_succeeded([None])


def test_runtime_receipt_rejects_duplicate_compare_for_one_version(tmp_path):
    raw_path = tmp_path / "raw.jsonl"
    e2e_path = tmp_path / "e2e.jsonl"
    raw_path.write_text(
        json.dumps(
            {
                "bytes_m2n": 0,
                "bytes_residual": 100,
                "bytes_total": 100,
                "digest_verified": False,
                "e2e_update_s": 99.0,
                "excluded_names": [],
                "install_s": 0.2,
                "implementation_started_at_epoch_s": 1000.0,
                "names": ["layer.weight"],
                "residual_s": 0.0,
                "transport_s": 1.0,
                "trial": 0,
                "update_weights_implementation_s": 1.5,
                "weight_version": "1",
            }
        )
        + "\n"
    )
    update = {
        "all_names_installed": False,
        "all_rollout_ready": True,
        "all_rollout_ready_epoch_s": 1002.0,
        "weight_checker_enabled": True,
        "version_agreement": True,
        "versions": ["1"],
        "weight_version": "1",
    }
    compare = {
        "all_names_installed": True,
        "digest_verified": True,
        "record_type": "weight_compare",
        "weight_version": "1",
    }
    e2e_path.write_text(
        "\n".join(json.dumps(record) for record in (update, compare, compare)) + "\n"
    )

    with pytest.raises(RuntimeError, match="duplicate weight-compare receipt"):
        runtime_harness.finalize_receipts(
            raw_path,
            e2e_path=e2e_path,
            mode="broadcast",
            expected_updates=1,
        )


def test_runtime_receipt_rejects_observed_weight_corruption(tmp_path):
    raw_path = tmp_path / "raw.jsonl"
    e2e_path = tmp_path / "e2e.jsonl"
    raw_path.write_text(
        json.dumps(
            {
                "bytes_m2n": 0,
                "bytes_residual": 100,
                "bytes_total": 100,
                "digest_verified": False,
                "e2e_update_s": 99.0,
                "excluded_names": [],
                "install_s": 0.2,
                "implementation_started_at_epoch_s": 1000.0,
                "names": ["layer.weight"],
                "residual_s": 0.0,
                "transport_s": 1.0,
                "trial": 0,
                "update_weights_implementation_s": 1.5,
                "weight_version": "1",
            }
        )
        + "\n"
    )
    e2e_path.write_text(
        "\n".join(
            json.dumps(record)
            for record in (
                {
                    "all_rollout_ready": True,
                    "all_rollout_ready_epoch_s": 1002.0,
                    "weight_checker_enabled": True,
                    "version_agreement": True,
                    "versions": ["1"],
                    "weight_version": "1",
                },
                {
                    "all_names_installed": False,
                    "digest_verified": False,
                    "record_type": "weight_compare",
                    "weight_version": "1",
                },
            )
        )
        + "\n"
    )

    with pytest.raises(RuntimeError, match="weight comparison failed"):
        runtime_harness.finalize_receipts(
            raw_path,
            e2e_path=e2e_path,
            mode="broadcast",
            expected_updates=1,
        )


def test_cleanup_requires_matching_uid_and_labels(tmp_path, monkeypatch):
    config = _runtime_config()
    item = {
        "arm": "broadcast",
        "block": 0,
        "order": 0,
        "repetition": 0,
    }
    job_dir = tmp_path / "jobs" / "block-00-order-0-broadcast"
    job_dir.mkdir(parents=True)
    labels = {
        "modelexpress.nvidia.com/benchmark-run": "run-a",
        "modelexpress.nvidia.com/benchmark-block": "0",
        "modelexpress.nvidia.com/benchmark-repetition": "0",
    }
    benchmark.write_json(
        job_dir / "ownership.json",
        {
            "resources": [
                {
                    "kind": "Job",
                    "labels": labels,
                    "name": "owned-job",
                    "uid": "uid-a",
                }
            ]
        },
    )
    current = {
        "metadata": {
            "labels": labels,
            "name": "owned-job",
            "uid": "uid-a",
        }
    }
    monkeypatch.setattr(
        benchmark,
        "_kubectl_get_optional",
        lambda *args, **kwargs: current,
    )
    commands = []

    def fake_run(args, **kwargs):
        commands.append((args, kwargs))
        return {"returncode": 0}

    monkeypatch.setattr(benchmark, "_run_command", fake_run)

    benchmark.cleanup_job(config, item, run_dir=tmp_path)

    assert any(
        any(argument.startswith("--raw=/apis/batch/v1/") for argument in command[0])
        for command in commands
    )
    assert any(
        json.loads(command[1]["input_text"])["preconditions"]["uid"] == "uid-a"
        for command in commands
    )

    current["metadata"]["uid"] = "uid-b"
    commands.clear()
    benchmark.cleanup_job(config, item, run_dir=tmp_path)
    assert commands == []
    cleanup = json.loads((job_dir / "cleanup.json").read_text())
    assert cleanup["refused"] == ["job/owned-job: UID changed"]

    current["metadata"]["uid"] = "uid-a"
    current["metadata"]["labels"] = labels | {
        "modelexpress.nvidia.com/benchmark-block": "other"
    }
    benchmark.cleanup_job(config, item, run_dir=tmp_path)
    assert commands == []
    cleanup = json.loads((job_dir / "cleanup.json").read_text())
    assert cleanup["refused"][0].startswith("job/owned-job: labels changed")


def test_preapply_not_found_is_the_only_absent_resource_result(
    tmp_path,
    monkeypatch,
):
    config = _runtime_config()
    resource = {"kind": "Job", "name": "missing-job"}

    def fake_run(args, *, output_dir, name, **kwargs):
        del kwargs
        assert "--ignore-not-found=true" in args
        stdout = output_dir / f"{name}.stdout.log"
        stderr = output_dir / f"{name}.stderr.log"
        stdout.write_text("")
        stderr.write_text("")
        return {
            "returncode": 0,
            "stdout": str(stdout),
            "stderr": str(stderr),
            "timed_out": False,
        }

    monkeypatch.setattr(benchmark, "_run_command", fake_run)

    assert (
        benchmark._kubectl_get_optional(
            config,
            resource=resource,
            environment={},
            output_dir=tmp_path,
            command_name="not-found",
            fail_on_error=True,
        )
        is None
    )


@pytest.mark.parametrize(
    ("returncode", "timed_out", "stderr_text"),
    (
        (1, False, "Error from server (Forbidden): jobs is forbidden"),
        (None, True, ""),
        (1, False, "Unable to connect to the server: connection refused"),
    ),
)
def test_preapply_lookup_errors_abort_before_apply(
    tmp_path,
    monkeypatch,
    returncode,
    timed_out,
    stderr_text,
):
    config = _runtime_config()
    item = {"arm": "broadcast", "block": 0, "order": 0, "repetition": 0}
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text("---\n")
    monkeypatch.setattr(
        benchmark,
        "_render_manifest",
        lambda *args, **kwargs: (manifest, []),
    )
    monkeypatch.setattr(
        benchmark,
        "validate_rendered_documents",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        benchmark,
        "_capture_failure_evidence",
        lambda **kwargs: None,
    )
    command_names = []

    def fake_run(args, *, output_dir, name, **kwargs):
        del args, kwargs
        command_names.append(name)
        stdout = output_dir / f"{name}.stdout.log"
        stderr = output_dir / f"{name}.stderr.log"
        stdout.write_text("")
        stderr.write_text(stderr_text)
        return {
            "returncode": returncode,
            "stdout": str(stdout),
            "stderr": str(stderr),
            "timed_out": timed_out,
        }

    monkeypatch.setattr(benchmark, "_run_command", fake_run)

    with pytest.raises(RuntimeError, match="pre-apply resource lookup"):
        benchmark.run_job(config, item, run_dir=tmp_path)

    assert not any(name.startswith("apply") for name in command_names)


@pytest.mark.parametrize(
    ("arm", "failed_command"),
    (
        ("broadcast", "apply"),
        ("modelexpress", "server-rollout"),
        ("broadcast", "job-wait"),
    ),
)
def test_apply_rollout_and_wait_failures_capture_evidence(
    tmp_path,
    monkeypatch,
    arm,
    failed_command,
):
    config = _runtime_config()
    item = {"arm": arm, "block": 0, "order": 0, "repetition": 0}
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text("---\n")
    monkeypatch.setattr(
        benchmark,
        "_render_manifest",
        lambda *args, **kwargs: (manifest, []),
    )
    monkeypatch.setattr(
        benchmark,
        "validate_rendered_documents",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        benchmark,
        "assert_owned_resources_absent",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        benchmark,
        "capture_owned_resources",
        lambda *args, **kwargs: {"resources": []},
    )
    evidence = []
    monkeypatch.setattr(
        benchmark,
        "_capture_failure_evidence",
        lambda **kwargs: evidence.append(kwargs),
    )

    def fake_run(args, *, name, **kwargs):
        del args, kwargs
        if name == failed_command:
            raise RuntimeError(f"{name} failed")
        return {"returncode": 0}

    monkeypatch.setattr(benchmark, "_run_command", fake_run)

    with pytest.raises(RuntimeError, match="failed"):
        benchmark.run_job(config, item, run_dir=tmp_path)

    assert len(evidence) == 1
    assert evidence[0]["job_name_value"].startswith("miles-m2n-")


def test_rollout_failure_captures_server_pod_and_init_container_logs(
    tmp_path,
    monkeypatch,
):
    command_names = []

    def fake_run(args, *, output_dir, name, **kwargs):
        del args, kwargs
        command_names.append(name)
        stdout = output_dir / f"{name}.stdout.log"
        payload = {"items": []}
        if name == "failed-server-pods-json":
            payload = {
                "items": [
                    {
                        "metadata": {"name": "mx-server-pod"},
                        "spec": {
                            "containers": [{"name": "modelexpress-server"}],
                            "initContainers": [{"name": "redis-init"}],
                        },
                    }
                ]
            }
        stdout.write_text(json.dumps(payload))
        return {"returncode": 0, "stdout": str(stdout)}

    monkeypatch.setattr(benchmark, "_run_command", fake_run)

    benchmark._capture_failure_evidence(
        base=["kubectl", "-n", "mx-bench"],
        environment={"BENCH_NODE_NAME": "gpu-node-a"},
        job_dir=tmp_path,
        job_name_value="benchmark-job",
        launcher=Path(__file__),
        server_name="mx-server",
        timeout_s=30,
    )

    assert "failed-describe-mx-server-pod" in command_names
    assert "failed-log-mx-server-pod-container-modelexpress-server" in command_names
    assert "failed-log-mx-server-pod-init-redis-init" in command_names


def _runtime_config() -> dict[str, object]:
    return {
        "cluster": {
            "context": "aws-dev-02",
            "gpu_product": "NVIDIA-H100-80GB-HBM3",
            "namespace": "mx-bench",
            "node_name": "gpu-node-a",
        },
        "experiment": {
            "command_timeout_s": 30,
            "nccl_debug": "WARN",
            "num_rollouts": 20,
            "require_symmetric_weight_checker": True,
            "verify_tensor_equality": False,
        },
        "images": {
            "runtime": "runtime@sha256:" + "1" * 64,
            "server": "server@sha256:" + "2" * 64,
        },
        "launcher": str(Path(__file__)),
        "run_id": "run-a",
        "storage": {
            "model_claim": "workspace-rwo",
            "workspace_claim": "workspace-rwo",
        },
        "topology": {
            "actor_gpus": 2,
            "m2n_pp_concurrency": 2,
            "pipeline_parallel_size": 2,
            "rollout_gpus": 2,
            "rollout_gpus_per_engine": 1,
            "total_gpus": 4,
        },
        "workload": {
            "checkpoint_manifest_path": ("/workspace/models/model/repo-files.sha256"),
            "checkpoint_revision": "checkpoint-revision-a",
            "checkpoint_sha256": "4" * 64,
            "dataset_path": "/workspace/datasets/dapo.jsonl",
            "dataset_revision": "revision-a",
            "dataset_sha256": "3" * 64,
            "miles_model_type": "qwen2.5-0.5B",
            "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
            "model_path": "/workspace/models/model",
        },
    }


def _topology_fixture():
    config = {
        "cluster": {
            "gpu_product": "NVIDIA-H100-80GB-HBM3",
            "node_name": "gpu-node-a",
        },
        "storage": {
            "model_claim": "workspace-rwo",
            "workspace_claim": "workspace-rwo",
        },
        "topology": {"total_gpus": 4},
    }
    pods = {
        "items": [
            {
                "metadata": {"name": "job-pod"},
                "spec": {
                    "nodeName": "gpu-node-a",
                    "containers": [
                        {
                            "name": "miles",
                            "resources": {
                                "limits": {"nvidia.com/gpu": "4"},
                                "requests": {"nvidia.com/gpu": "4"},
                            },
                            "volumeMounts": [
                                {"name": "model-cache", "mountPath": "/models"},
                                {
                                    "name": "run-output",
                                    "mountPath": "/root/shared_data",
                                },
                                {
                                    "name": "benchmark-workspace",
                                    "mountPath": "/workspace",
                                },
                            ],
                        }
                    ],
                    "volumes": [
                        {
                            "name": "model-cache",
                            "persistentVolumeClaim": {"claimName": "workspace-rwo"},
                        },
                        {
                            "name": "run-output",
                            "persistentVolumeClaim": {"claimName": "workspace-rwo"},
                        },
                        {
                            "name": "benchmark-workspace",
                            "persistentVolumeClaim": {"claimName": "workspace-rwo"},
                        },
                    ],
                },
            }
        ]
    }
    nodes = {
        "gpu-node-a": {
            "metadata": {
                "name": "gpu-node-a",
                "labels": {"nvidia.com/gpu.product": "NVIDIA-H100-80GB-HBM3"},
            },
            "status": {"allocatable": {"nvidia.com/gpu": "8"}},
        }
    }
    return config, pods, nodes


def _rendered_documents() -> list[dict[str, object]]:
    runtime_source = Path(runtime_harness.__file__).read_text()
    sitecustomize_source = Path(__file__).with_name("sitecustomize.py").read_text()
    return [
        {
            "kind": "Job",
            "metadata": {"name": "job"},
            "spec": {
                "template": {
                    "spec": {
                        "containers": [
                            {
                                "name": "miles",
                                "image": "runtime@sha256:" + "1" * 64,
                                "command": [
                                    "python3",
                                    "/opt/miles-benchmark/runtime_harness.py",
                                ],
                                "env": [
                                    {
                                        "name": "MILES_DATASET_PATH",
                                        "value": "/workspace/datasets/dapo.jsonl",
                                    },
                                    {
                                        "name": "MILES_DATASET_REVISION",
                                        "value": "revision-a",
                                    },
                                    {
                                        "name": "MILES_DATASET_SHA256",
                                        "value": "3" * 64,
                                    },
                                    {
                                        "name": "MILES_MODEL_PATH",
                                        "value": "/workspace/models/model",
                                    },
                                    {
                                        "name": "MILES_MODEL_TYPE",
                                        "value": "qwen2.5-0.5B",
                                    },
                                    {
                                        "name": "MODEL_ID",
                                        "value": "Qwen/Qwen2.5-0.5B-Instruct",
                                    },
                                    {
                                        "name": "MILES_CHECKPOINT_MANIFEST_PATH",
                                        "value": (
                                            "/workspace/models/model/repo-files.sha256"
                                        ),
                                    },
                                    {
                                        "name": "MILES_CHECKPOINT_REVISION",
                                        "value": "checkpoint-revision-a",
                                    },
                                    {
                                        "name": "MILES_CHECKPOINT_SHA256",
                                        "value": "4" * 64,
                                    },
                                    {
                                        "name": "MILES_BENCH_REQUIRE_WEIGHT_CHECKER",
                                        "value": "1",
                                    },
                                ],
                                "volumeMounts": [
                                    {"name": "model-cache", "mountPath": "/models"},
                                    {
                                        "name": "shared-memory",
                                        "mountPath": "/dev/shm",
                                    },
                                    {
                                        "name": "run-output",
                                        "mountPath": "/root/shared_data",
                                    },
                                    {
                                        "name": "benchmark-runtime",
                                        "mountPath": "/opt/miles-benchmark",
                                        "readOnly": True,
                                    },
                                ],
                            }
                        ],
                        "volumes": [
                            {"name": "model-cache", "emptyDir": {}},
                            {
                                "name": "shared-memory",
                                "emptyDir": {
                                    "medium": "Memory",
                                    "sizeLimit": "64Gi",
                                },
                            },
                            {"name": "run-output", "emptyDir": {}},
                            {
                                "name": "benchmark-runtime",
                                "configMap": {"name": "miles-bench-run-a-b0-r0"},
                            },
                        ],
                    }
                }
            },
        },
        {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {"name": "miles-bench-run-a-b0-r0"},
            "data": {
                "runtime_harness.py": runtime_source,
                "sitecustomize.py": sitecustomize_source,
            },
        },
    ]
