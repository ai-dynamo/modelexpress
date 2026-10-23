# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run vLLM single-Service discovery on real, distinct GPU nodes.

Creates an isolated namespace, warms a source instance, starts a target, and
checks rank-local RDMA, node-local artifacts and deterministic inference. The
normal engine rendezvous Services are separate from the sole MX discovery
Service. Logs and results are retained locally before namespace cleanup.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path


class ClusterTest:
    def __init__(self, args):
        self.args = args
        self.output = Path(args.output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.created = False

    def kubectl(self, *args, payload=None, timeout=45):
        result = subprocess.run(
            [
                "kubectl",
                "--context",
                self.args.context,
                "-n",
                self.args.namespace,
                *args,
            ],
            input=payload,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        if result.returncode:
            raise RuntimeError(result.stderr.strip())
        return result.stdout

    def apply(self, obj):
        return self.kubectl("apply", "-f", "-", payload=json.dumps(obj))

    def log(self, message):
        print(time.strftime("%H:%M:%S"), message, flush=True)

    def save(self, name, value):
        if not isinstance(value, str):
            value = json.dumps(value, indent=2)
        (self.output / name).write_text(value)

    def pods(self):
        return json.loads(self.kubectl("get", "pods", "-o", "json"))["items"]

    def capture(self):
        for args, name in [
            (("get", "pods", "-o", "json"), "pods.json"),
            (("get", "services", "-o", "json"), "services.json"),
            (("get", "events", "--sort-by=.lastTimestamp"), "events.txt"),
        ]:
            try:
                self.save(name, self.kubectl(*args))
            except (RuntimeError, subprocess.TimeoutExpired):
                pass
        for pod in self.pods():
            name = pod["metadata"]["name"]
            try:
                self.save(f"{name}.log", self.kubectl("logs", name, timeout=60))
            except (RuntimeError, subprocess.TimeoutExpired):
                pass

    def wait(self, description, predicate):
        deadline = time.monotonic() + self.args.timeout
        last_report = 0
        while time.monotonic() < deadline:
            result = predicate()
            if result:
                self.log(f"Ready: {description}")
                return result
            if time.monotonic() - last_report > 30:
                status = [
                    (
                        p["metadata"]["name"],
                        p["status"].get("phase"),
                        p["spec"].get("nodeName"),
                    )
                    for p in self.pods()
                ]
                self.log(f"Waiting for {description}: {status}")
                last_report = time.monotonic()
            time.sleep(5)
        raise TimeoutError(description)

    def wait_group(self, group):
        def ready():
            pods = [
                p for p in self.pods() if p["metadata"]["labels"].get("group") == group
            ]
            for pod in pods:
                for status in pod["status"].get("containerStatuses", []):
                    if "terminated" in status.get("state", {}):
                        raise RuntimeError(
                            f"{pod['metadata']['name']} terminated: {status['state']}"
                        )
            return len(pods) == self.args.nodes and all(
                any(
                    c["type"] == "Ready" and c["status"] == "True"
                    for c in p["status"].get("conditions", [])
                )
                for p in pods
            )

        return self.wait(f"{group} Pods", ready)

    def service(self, name, selector, *, headless=False):
        spec = {
            "selector": selector,
            "ports": [
                {
                    "name": "worker" if not headless else "engine",
                    "port": 6555 if not headless else 29500,
                }
            ],
        }
        if headless:
            spec.update(clusterIP="None", publishNotReadyAddresses=True)
        return {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {"name": name},
            "spec": spec,
        }

    def pod(self, group, node_rank):
        a = self.args
        local_gpus = a.tp * a.pp // a.nodes
        start = """set -euo pipefail
mkdir -p /cache/triton
if [ "$TEST_ROLE" = source ]; then
  printf 'source-node-%s' "$TEST_NODE_RANK" > /cache/triton/mx-node-marker.txt
fi
exec vllm serve "$TEST_MODEL" --revision "$MX_MODEL_REVISION" \\
  --load-format modelexpress --served-model-name mx-leader-test \\
  --dtype bfloat16 --max-model-len 1024 --max-num-seqs 4 \\
  --max-num-batched-tokens 1024 --gpu-memory-utilization 0.2 \\
  --tensor-parallel-size "$TEST_TP" --pipeline-parallel-size "$TEST_PP" \\
  --distributed-executor-backend mp --nnodes "$TEST_NNODES" \\
  --node-rank "$TEST_NODE_RANK" --master-addr "$TEST_HEAD" --master-port 29500 \\
  --port 8000 "${@:1}"
"""
        env = {
            "TEST_ROLE": group,
            "TEST_NODE_RANK": str(node_rank),
            "TEST_MODEL": a.model,
            "TEST_TP": str(a.tp),
            "TEST_PP": str(a.pp),
            "TEST_NNODES": str(a.nodes),
            "TEST_HEAD": f"{group}-head",
            "VLLM_PLUGINS": "modelexpress",
            "MX_METADATA_BACKEND": "k8s-service",
            "MX_K8S_LEADER_SERVICE": "mx-sources:6555",
            "MX_MODEL_REVISION": a.revision,
            "MX_ARTIFACT_TRANSFER": "1",
            "MX_WORKER_GRPC_PORT": "6555",
            "MX_METADATA_PORT": "5555",
            "MX_ARTIFACT_BUNDLE_ROOT": "/cache/bundles",
            "VLLM_CACHE_ROOT": "/cache/vllm",
            "TRITON_CACHE_DIR": "/cache/triton",
            "FLASHINFER_WORKSPACE_BASE": "/cache/flashinfer",
            "DG_JIT_CACHE_DIR": "/cache/deepgemm",
            "TILELANG_CACHE_DIR": "/cache/tilelang",
            "CUTE_DSL_CACHE_DIR": "/cache/cute",
            "HF_HOME": "/cache/huggingface",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "2",
            "MX_RDMA_NIC_PIN": "auto",
            "UCX_LOG_LEVEL": "WARN",
            "NIXL_LOG_LEVEL": "WARN",
            "MODEL_EXPRESS_LOG_LEVEL": "INFO",
        }
        if group == "source":
            env.update(MX_K8S_SOURCE_RETRIES="2", MX_K8S_SOURCE_BACKOFF_SECONDS="0.1")
        if a.topology_zone:
            env.update(
                MX_P2P_TOPOLOGY_DOMAIN="zone",
                MX_P2P_TOPOLOGY_ENFORCEMENT="required",
                MX_P2P_TOPOLOGY=json.dumps({"zone": a.topology_zone}),
            )
        container = {
            "name": "vllm",
            "image": a.image,
            "imagePullPolicy": "IfNotPresent",
            "command": [
                "bash",
                "-c",
                start,
                "--",
                *(["--enforce-eager"] if a.enforce_eager else []),
                *(["--headless"] if node_rank else []),
            ],
            "env": [{"name": k, "value": v} for k, v in env.items()]
            + [
                {
                    "name": "MX_WORKER_HOST",
                    "valueFrom": {"fieldRef": {"fieldPath": "status.podIP"}},
                },
            ],
            "resources": {
                "requests": {
                    "cpu": "4",
                    "memory": "16Gi",
                    "nvidia.com/gpu": str(local_gpus),
                    "rdma/shared_ib": str(local_gpus),
                },
                "limits": {
                    "memory": "48Gi",
                    "nvidia.com/gpu": str(local_gpus),
                    "rdma/shared_ib": str(local_gpus),
                },
            },
            "securityContext": {"capabilities": {"add": ["IPC_LOCK"]}},
            "volumeMounts": [
                {"name": "cache", "mountPath": "/cache"},
                {"name": "shm", "mountPath": "/dev/shm"},
            ],
            "readinessProbe": {
                **(
                    {"httpGet": {"path": "/health", "port": 8000}}
                    if node_rank == 0
                    else {"tcpSocket": {"port": 6555}}
                ),
                "periodSeconds": 5,
                "timeoutSeconds": 5,
                "failureThreshold": 240,
            },
        }
        return {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {
                "name": f"{group}-{node_rank}",
                "labels": {
                    "app": "mx-leader-test",
                    "group": group,
                    "node-rank": str(node_rank),
                },
            },
            "spec": {
                "restartPolicy": "Never",
                "imagePullSecrets": [{"name": "registry"}],
                "tolerations": [
                    {"key": key, "operator": "Exists", "effect": "NoSchedule"}
                    for key in ("nvidia.com/gpu", "dra")
                ],
                "affinity": {
                    "podAntiAffinity": {
                        "requiredDuringSchedulingIgnoredDuringExecution": [
                            {
                                "labelSelector": {
                                    "matchLabels": {
                                        "app": "mx-leader-test",
                                        "group": group,
                                    }
                                },
                                "topologyKey": "kubernetes.io/hostname",
                            },
                            {
                                "labelSelector": {
                                    "matchLabels": {
                                        "app": "mx-leader-test",
                                        "node-rank": str(node_rank),
                                    }
                                },
                                "topologyKey": "kubernetes.io/hostname",
                            },
                        ]
                    }
                },
                "containers": [container],
                "volumes": [
                    {"name": "cache", "emptyDir": {}},
                    {
                        "name": "shm",
                        "emptyDir": {"medium": "Memory", "sizeLimit": "8Gi"},
                    },
                ],
            },
        }

    def exec_python(self, pod, code):
        return self.kubectl("exec", pod, "--", "python3", "-c", code, timeout=90)

    def inventory(self):
        logs = self.kubectl("logs", "source-0", timeout=60)
        ids = re.findall(
            r"Published P2P metadata to MX server \(mx_source_id=([a-f0-9]+)", logs
        )
        if not ids:
            return None
        code = f"""import json,grpc
from google.protobuf.json_format import MessageToDict
from modelexpress import p2p_pb2,p2p_pb2_grpc
with grpc.insecure_channel('mx-sources:6555') as channel:
 r=p2p_pb2_grpc.WorkerServiceStub(channel).GetInstanceMetadata(p2p_pb2.GetInstanceMetadataRequest(mx_source_id={ids[0]!r}),timeout=15)
 print(json.dumps(MessageToDict(r,preserving_proto_field_name=True)))
"""
        try:
            result = json.loads(self.exec_python("source-0", code))
        except RuntimeError:
            return None
        nodes = {
            w["worker"].get("node_rank", 0)
            for w in result["workers"]
            if any(
                a["mx_source_type"] == "MX_SOURCE_TYPE_TRITON_CACHE"
                for a in w.get("artifacts", [])
            )
        }
        return result if nodes == set(range(self.args.nodes)) else None

    def inference(self, group):
        payload = {
            "model": "mx-leader-test",
            "prompt": "The capital of France is",
            "max_tokens": 16,
            "temperature": 0,
            "seed": 42,
        }
        code = f"""import urllib.request
r=urllib.request.Request('http://127.0.0.1:8000/v1/completions',data={json.dumps(payload).encode()!r},headers={{'Content-Type':'application/json'}})
with urllib.request.urlopen(r,timeout=60) as f: print(f.read().decode())
"""
        result = json.loads(self.exec_python(f"{group}-0", code))
        self.save(f"{group}-inference.json", result)
        return result["choices"][0]["text"]

    def fault_checks(self, inventory):
        code = f"""import grpc,json
from google.protobuf.json_format import ParseDict
from modelexpress import p2p_pb2,p2p_pb2_grpc
inventory=json.loads({json.dumps(inventory)!r})
leader=next(w['worker'] for w in inventory['workers'] if w['worker'].get('global_rank',0)==0)
worker=ParseDict(leader,p2p_pb2.InstanceWorkerEndpoint())
request=p2p_pb2.RegisterInstanceWorkerRequest(mx_source_id=inventory['mx_source_id'],world_size=inventory['world_size'],worker=worker,ready=False)
metadata_request=p2p_pb2.GetInstanceMetadataRequest(mx_source_id=inventory['mx_source_id'])
checks={{}}
with grpc.insecure_channel('mx-sources:6555') as channel:
 stub=p2p_pb2_grpc.WorkerServiceStub(channel)
 stub.RegisterInstanceWorker(request,timeout=5)
 try:
  try:
   stub.GetInstanceMetadata(metadata_request,timeout=10)
   raise AssertionError('withdrawn worker was advertised')
  except grpc.RpcError as exc:
   assert exc.code()==grpc.StatusCode.UNAVAILABLE,exc
   checks['withdrawn_worker_rejected']=True
 finally:
  request.ready=True
  stub.RegisterInstanceWorker(request,timeout=5)
 stub.GetInstanceMetadata(metadata_request,timeout=10)
 checks['readiness_restored']=True
with grpc.insecure_channel(worker.worker_grpc_endpoint) as channel:
 try:
  p2p_pb2_grpc.WorkerServiceStub(channel).GetTensorManifest(p2p_pb2.GetTensorManifestRequest(mx_source_id=inventory['mx_source_id'],worker_id='__old_generation__'),timeout=5)
  raise AssertionError('old worker generation accepted')
 except grpc.RpcError as exc:
  assert exc.code()==grpc.StatusCode.FAILED_PRECONDITION,exc
  checks['old_generation_rejected']=True
print(json.dumps(checks))
"""
        result = json.loads(self.exec_python("source-0", code))
        self.save("fault-checks.json", result)
        return result

    @staticmethod
    def compile_cache_reused(logs):
        if "was not reused" in logs:
            raise AssertionError("vLLM did not reuse the transferred compile cache")
        return (
            "Directly load AOT compilation from path" in logs
            and "which ModelExpress installed" in logs
        )

    def run(self):
        a = self.args
        if a.resume:
            ns = json.loads(self.kubectl("get", "namespace", a.namespace, "-o", "json"))
            if (
                ns["metadata"].get("labels", {}).get("modelexpress-test")
                != "leader-discovery"
            ):
                raise ValueError(
                    "refusing to resume a namespace not owned by this test"
                )
        else:
            self.kubectl("create", "namespace", a.namespace)
            self.kubectl(
                "label", "namespace", a.namespace, "modelexpress-test=leader-discovery"
            )
        self.created = True
        try:
            raw = self.kubectl(
                "get",
                "secret",
                a.pull_secret,
                "-n",
                a.pull_secret_namespace,
                "-o",
                "json",
            )
            secret = json.loads(raw)
            self.apply(
                {
                    "apiVersion": "v1",
                    "kind": "Secret",
                    "metadata": {"name": "registry"},
                    "type": secret["type"],
                    "data": secret["data"],
                }
            )
            self.apply(
                self.service("mx-sources", {"group": "source", "node-rank": "0"})
            )
            for group in ("source", "target"):
                self.apply(
                    self.service(
                        f"{group}-head",
                        {"group": group, "node-rank": "0"},
                        headless=True,
                    )
                )
            manifests = [
                self.pod(group, rank)
                for group in ("source", "target")
                for rank in range(a.nodes)
            ]
            self.save("pod-manifests.json", manifests)
            existing = {p["metadata"]["name"]: p for p in self.pods()}
            for pod in existing.values():
                if pod["spec"]["containers"][0]["image"] != a.image:
                    raise ValueError("existing test Pod uses a different image")
            for manifest in manifests[: a.nodes]:
                if manifest["metadata"]["name"] not in existing:
                    self.apply(manifest)
            self.wait_group("source")
            inventory = self.wait("published per-node artifacts", self.inventory)
            self.save("source-inventory.json", inventory)
            source_text = self.inference("source")
            for manifest in manifests[a.nodes :]:
                if manifest["metadata"]["name"] not in existing:
                    self.apply(manifest)
            self.wait_group("target")
            target_text = self.inference("target")
            if source_text != target_text:
                raise AssertionError(
                    f"inference mismatch: {source_text!r} != {target_text!r}"
                )
            pods = self.pods()
            placements = {p["metadata"]["name"]: p["spec"]["nodeName"] for p in pods}
            for group in ("source", "target"):
                if (
                    len({placements[f"{group}-{rank}"] for rank in range(a.nodes)})
                    != a.nodes
                ):
                    raise AssertionError(
                        "each instance must span distinct physical nodes"
                    )
            if any(
                placements[f"source-{rank}"] == placements[f"target-{rank}"]
                for rank in range(a.nodes)
            ):
                raise AssertionError(
                    "corresponding source/target ranks must use different nodes"
                )
            ranks = set()
            selections = 0
            compile_cache_nodes = {
                w["worker"].get("node_rank", 0)
                for w in inventory["workers"]
                if any(
                    a["mx_source_type"] == "MX_SOURCE_TYPE_TORCH_COMPILE_CACHE"
                    for a in w.get("artifacts", [])
                )
            }
            reused = []
            for rank in range(a.nodes):
                name = f"target-{rank}"
                if rank in compile_cache_nodes:
                    self.wait(
                        f"{name} AOT cache reuse",
                        lambda name=name: self.compile_cache_reused(
                            self.kubectl("logs", name, timeout=60)
                        ),
                    )
                    reused.append(name)
                logs = self.kubectl("logs", name, timeout=60)
                ranks.update(
                    int(x)
                    for x in re.findall(
                        r"\[Worker (\d+)\].*\[TIMING\] RDMA transfer complete", logs
                    )
                )
                selections += logs.count("Selected instance through mx-sources:6555")
                if "Leader discovery through mx-sources:6555 failed" in logs:
                    raise AssertionError(
                        f"healthy source discovery unexpectedly retried: {name}"
                    )
                marker = self.kubectl(
                    "exec", name, "--", "cat", "/cache/triton/mx-node-marker.txt"
                )
                if marker != f"source-node-{rank}":
                    raise AssertionError(
                        f"wrong node artifact installed on {name}: {marker!r}"
                    )
                if "vLLM artifact install complete:" not in logs:
                    raise AssertionError(f"missing artifact install on {name}")
            if ranks != set(range(a.tp * a.pp)):
                raise AssertionError(f"not all ranks loaded via RDMA: {ranks}")
            if selections != a.tp * a.pp:
                raise AssertionError(
                    f"expected one directory per target worker, got {selections}"
                )
            service = json.loads(
                self.kubectl("get", "service", "mx-sources", "-o", "json")
            )
            if len(service["spec"]["ports"]) != 1:
                raise AssertionError("discovery must expose exactly one port")
            fault_checks = self.fault_checks(inventory)
            result = {
                "status": "passed",
                "namespace": a.namespace,
                "image": a.image,
                "tp": a.tp,
                "pp": a.pp,
                "nodes_per_instance": a.nodes,
                "topology_zone": a.topology_zone,
                "placements": placements,
                "rdma_ranks": sorted(ranks),
                "directory_requests": selections,
                "compile_cache_reused_on": reused,
                "inference_text": target_text,
                "fault_checks": fault_checks,
            }
            self.save("result.json", result)
            self.log(json.dumps(result))
        finally:
            try:
                self.capture()
            finally:
                if self.created and not a.keep_namespace:
                    self.kubectl("delete", "namespace", a.namespace, "--wait=false")
                    self.log(f"Deleted namespace {a.namespace}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", default="nebius-2")
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pull-secret-namespace", required=True)
    parser.add_argument("--pull-secret", default="nvcr-secret")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--keep-namespace", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--topology-zone", help="Exercise required same-zone topology enforcement"
    )
    args = parser.parse_args()
    if args.tp < 1 or args.pp < 1 or args.nodes < 1 or args.tp * args.pp % args.nodes:
        parser.error("positive TP * PP must be divisible by nodes")
    ClusterTest(args).run()


if __name__ == "__main__":
    main()
