# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""S3 XOR-delta publication and optional generator-peer benchmark scenario."""

import re

from harness import validation


class DeltaScenario:
    def __init__(self, config):
        self.config = config

    def configure(self, paths, environment):
        config = self.config
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
            publication_path="/tmp/mx-delta/report.json",
            artifact_prefix=config["delta_prefix"] + config["run"] + "/",
            preparation=[
                {
                    "role": "s3",
                    "module": "scenarios.delta.download_seed",
                    "log": "seed.log",
                },
                {
                    "role": "control",
                    "module": "scenarios.delta.publish",
                    "log": "publication.log",
                },
            ],
            control_env={},
            worker_env={
                "MX_GENERATOR_SOURCE_ORDER": "OBJECT_STORAGE",
                **environment.get("worker_env", {}),
                "MX_MODEL_URI": f"s3://{config['bucket']}/{config['seed_prefix'].rstrip('/')}",
            },
            sources={
                "s3": "OBJECT_STORAGE",
                **({"peer": "GENERATOR"} if paths == "both" else {}),
            },
        )

    def _checkpoint(self, rows, publication):
        config = self.config
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

    def verify_refit(self, rpc, role, publication):
        config = self.config
        if config["sources"][role] == "OBJECT_STORAGE":
            rows = rpc(
                role,
                "verify-checkpoint",
                "verify_checkpoint",
                {
                    "version_id": config["target_version"],
                    "tensor_name": publication["tensor"],
                },
            )
            self._checkpoint(rows, publication)
            self._runtime_tensor(
                rpc(
                    role,
                    "verify-runtime-tensor",
                    "verify_runtime_tensor",
                    {
                        "version_id": config["target_version"],
                        "tensor_name": publication["tensor"],
                    },
                ),
                publication,
            )

    def validate_inventory(self, baseline, updated, publication):
        config = self.config
        assert set(baseline) == set(updated) == set(config["roles"])
        for role in config["roles"]:
            before, after = baseline[role], updated[role]
            assert set(before) == set(after), "Changed rank inventory"
            assert before != after, "No updated tensors"
            for rank in before:
                assert set(before[rank]) == set(after[rank]), "Changed tensor inventory"
                for name, tensor in before[rank].items():
                    installed = after[rank][name]
                    assert tensor["shape"] == installed["shape"], "Changed tensor shape"
                    assert tensor["dtype"] == installed["dtype"], "Changed tensor dtype"
        for inventories in [baseline, updated]:
            values = list(inventories.values())
            assert all(value == values[0] for value in values[1:]), (
                "Worker tensors differ per TP rank"
            )

    def validate_report(self, result, report, root):
        config = self.config
        for role in config["roles"]:
            if config["sources"][role] == "OBJECT_STORAGE":
                self._checkpoint(
                    result(role, "verify-checkpoint"), report["publication"]
                )
                self._runtime_tensor(
                    result(role, "verify-runtime-tensor"), report["publication"]
                )
        for role, worker in report["workers"].items():
            text = (root / f"{role}-worker.log").read_text()
            if config["sources"][role] == "OBJECT_STORAGE":
                assert "Streaming weights from s3://" in text
                completed = {
                    int(rank)
                    for rank in re.findall(
                        r"\[Worker (\d+)\] Model streamer weight loading complete", text
                    )
                }
                assert completed == set(range(config["tp"])), (
                    "S3 cold load did not complete on every rank"
                )
                assert "Model streamer loading failed, falling through:" not in text, (
                    "S3 cold load fell back"
                )
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

    def _runtime_tensor(self, rows, publication):
        config = self.config
        for row in validation.ranks(rows, config).values():
            assert row["version"] == config["target_version"]
            assert row["checkpoint_tensor"] == publication["tensor"]
            assert row["verified"] and row["actual_sha256"] == row["expected_sha256"], (
                "Installed runtime tensor differs from reconstructed checkpoint"
            )
