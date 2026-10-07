# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Manage trusted benchmark CI setup, execution, and cleanup."""

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness.lifecycle import Benchmark, Kubernetes
from harness.render import prepare

OWNER_LABEL = "ci.modelexpress.nvidia.com/run-id"


def ci(mode):
    env = os.environ
    for key in [
        "MODEL_PROFILE",
        "NAMESPACE",
        "KUBE_CONTEXT",
        "RUN_ID",
        "RESULTS_DIR",
        "SERVER_IMAGE",
        "WORKER_IMAGE",
        "MX_BENCH_S3_ROLE_ARN",
        "GITHUB_RUN_ID",
        "GITHUB_RUN_ATTEMPT",
    ]:
        if not env.get(key):
            raise ValueError(f"Missing {key}")
    root = Path(env["RESULTS_DIR"])
    namespace = env["NAMESPACE"]
    owner = env["GITHUB_RUN_ID"] + "-" + env["GITHUB_RUN_ATTEMPT"]
    k = Kubernetes(
        env["KUBE_CONTEXT"],
        namespace,
        env.get("KUBECONFIG", "/teleport/kubeconfig.yaml"),
    )

    def render():
        with tempfile.TemporaryDirectory(prefix="mx-bench-render-") as temporary:
            rendered = Path(temporary) / "run"
            config = prepare(
                env["MODEL_PROFILE"],
                rendered,
                env["RUN_ID"],
                "s3",
                environment="aws-ci",
                scenario_name=env.get("SCENARIO", "delta"),
                service_account="mx-bench",
            )
            root.mkdir(parents=True, exist_ok=True)
            shutil.copytree(rendered, root, dirs_exist_ok=True)
            return config

    def owned():
        actual = k.call(
            "get",
            "namespace",
            namespace,
            "--ignore-not-found",
            "-o",
            'go-template={{ index .metadata.labels "' + OWNER_LABEL + '" }}',
        ).strip()
        if actual != owner:
            raise RuntimeError("Namespace ownership mismatch; refusing operation")

    if mode == "setup":
        k.manifest(
            "create",
            {
                "apiVersion": "v1",
                "kind": "Namespace",
                "metadata": {"name": namespace, "labels": {OWNER_LABEL: owner}},
            },
        )
        config = render()
        count = config["tp"]
        k.call(
            "create",
            "quota",
            "bench-gpu-budget",
            f"--hard=requests.nvidia.com/gpu={count},limits.nvidia.com/gpu={count}",
        )
        k.call("create", "serviceaccount", "mx-bench")
        k.call(
            "annotate",
            "serviceaccount",
            "mx-bench",
            "eks.amazonaws.com/role-arn=" + env["MX_BENCH_S3_ROLE_ARN"],
        )
        # Send credentials through stdin, so subprocess errors cannot print them.
        import base64

        auth = base64.b64encode(("$oauthtoken:" + env["NGC_API_KEY"]).encode()).decode()
        k.manifest(
            "create",
            {
                "apiVersion": "v1",
                "kind": "Secret",
                "metadata": {"name": "nvcr-imagepullsecret"},
                "type": "kubernetes.io/dockerconfigjson",
                "stringData": {
                    ".dockerconfigjson": json.dumps(
                        {"auths": {"nvcr.io": {"auth": auth}}}
                    )
                },
            },
        )
        for path in root.glob("*.yaml"):
            manifest = yaml.safe_load(path.read_text())
            for item in manifest["items"]:
                if item["kind"] == "Pod":
                    item["spec"]["activeDeadlineSeconds"] = 3300
            path.write_text(yaml.safe_dump(manifest))
    elif mode == "run":
        owned()
        Benchmark(root).run()
    else:
        if not k.call(
            "get", "namespace", namespace, "--ignore-not-found", "-o", "name"
        ).strip():
            return
        owned()
        render()
        Benchmark(root).cleanup()
        k.call("delete", "namespace", namespace, "--wait=true", "--timeout=2m")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["setup", "run", "collect", "cleanup"])
    args = parser.parse_args()
    if args.command == "collect":
        directory = Path(os.environ["RESULTS_DIR"])
        if (directory / "environment.json").exists():
            Benchmark(directory).collect()
    else:
        ci(args.command)


if __name__ == "__main__":
    main()
