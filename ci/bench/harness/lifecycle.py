# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run, collect, and clean up rendered benchmarks."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml


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


class Kubernetes:
    def __init__(self, context, namespace, kubeconfig=None):
        if not context or not namespace:
            raise ValueError("Explicit Kubernetes context and namespace required")
        self.command = ["kubectl", "--context", context, "-n", namespace]
        if kubeconfig:
            self.command += ["--kubeconfig", kubeconfig]

    def call(self, *args, payload=None, output=None, timeout=900):
        command = self.command + list(args)
        if output is not None:
            with Path(output).open("wb") as stream:
                subprocess.run(
                    command,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=timeout,
                )
            return ""
        return subprocess.run(
            command,
            input=payload,
            text=True,
            stdout=subprocess.PIPE,
            check=True,
            timeout=timeout,
        ).stdout

    def manifest(self, verb, value):
        return self.call(verb, "-f", "-", payload=json.dumps(value))


class Benchmark:
    def __init__(self, directory):
        self.root = Path(directory).resolve()
        self.config = json.loads((self.root / "config.json").read_text())
        env = json.loads((self.root / "environment.json").read_text())
        self.k = Kubernetes(env["context"], env["namespace"], env.get("kubeconfig"))
        self.prefix = self.config["resource_prefix"]
        self.control = self.prefix + "-control"
        self.roles = self.config["roles"]
        self.processes = []

    def capture(self, name, *args):
        destination = self.root / name
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        # Binary artifacts must not contain kubectl error messages.
        try:
            with temporary.open("wb") as stream:
                subprocess.run(
                    self.k.command + list(args), stdout=stream, check=True, timeout=180
                )
            temporary.replace(destination)
        except (subprocess.SubprocessError, OSError) as error:
            print(f"Could not collect {name}: {error}", file=sys.stderr)

    def collect(self):
        for role in self.roles:
            pod = self.prefix + "-" + role
            self.capture(role + "-pod.json", "get", "pod", pod, "-o", "json")
            self.capture(
                role + "-evidence.tar.gz",
                "exec",
                pod,
                "-c",
                "main",
                "--",
                "tar",
                "-C",
                "/refit",
                "-czf",
                "-",
                "benchmark",
            )
        self.capture(
            "publication.json",
            "exec",
            self.control,
            "-c",
            "main",
            "--",
            "cat",
            self.config["publication_path"],
        )
        self.capture(
            "bench-evidence.tar.gz",
            "exec",
            self.control,
            "-c",
            "main",
            "--",
            "tar",
            "-C",
            "/tmp",
            "-czf",
            "-",
            "mx-bench",
        )
        self.capture("server.log", "logs", self.control, "-c", "server")
        subprocess.run(
            [sys.executable, "-m", "harness.report", str(self.root)],
            check=True,
            timeout=180,
            cwd=self.root,
        )

    def start(self, pod, module, log):
        with (self.root / log).open("wb") as stream:
            process = subprocess.Popen(
                self.k.command
                + [
                    "exec",
                    pod,
                    "-c",
                    "main",
                    "--",
                    "python3",
                    "-u",
                    "-m",
                    module,
                ],
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
        self.processes.append(process)
        return process

    def apply(self, name):
        path = str(self.root / name)
        self.k.call("apply", "--dry-run=server", "-f", path)
        self.k.call("apply", "-f", path)

    def prepare_run(self):
        self.apply("harness.json")
        self.apply("control.yaml")
        self.k.call(
            "wait", "--for=condition=Ready", "pod/" + self.control, "--timeout=10m"
        )
        preparations = []
        deadline = time.monotonic() + 7200
        for role in self.roles:
            self.apply(f"worker-{role}.yaml")
            pod = self.prefix + "-" + role
            self.k.call("wait", "--for=condition=Ready", "pod/" + pod, "--timeout=10m")
            worker = self.start(pod, "engines.vllm.server", role + "-worker.log")
            for task in self.config["preparation"]:
                if task["role"] == role or (
                    task["role"] == "control" and role == self.roles[0]
                ):
                    target = self.control if task["role"] == "control" else pod
                    preparations.append(self.start(target, task["module"], task["log"]))
            while (
                "BENCH_READY"
                not in (self.root / (role + "-worker.log"))
                .read_text(errors="replace")
                .splitlines()
            ):
                pending_publications(preparations)
                if worker.poll() is not None:
                    raise RuntimeError(f"{role} worker exited before readiness")
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"{role} worker readiness timed out")
                time.sleep(5)
        wait_for_publications(preparations, deadline)
        self.k.call(
            "exec",
            self.control,
            "-c",
            "main",
            "--",
            "cat",
            self.config["publication_path"],
            output=self.root / "publication.json",
        )

    def run(self):
        if (self.root / "started").exists():
            raise ValueError("Render a fresh run; this directory was already started")
        self.k.call("cluster-info")
        (self.root / "started").touch(exist_ok=False)
        try:
            self.prepare_run()
            self.k.call(
                "exec",
                self.control,
                "-c",
                "main",
                "--",
                "python3",
                "-u",
                "-m",
                "harness.runner",
                output=self.root / "bench-driver.log",
                timeout=28800,
            )
            if (
                "BENCH_PASS"
                not in (self.root / "bench-driver.log").read_text().splitlines()
            ):
                raise RuntimeError("Benchmark driver did not pass")
        finally:
            try:
                self.collect()
            finally:
                for process in self.processes:
                    if process.poll() is None:
                        process.terminate()
                for process in self.processes:
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                print(
                    f"Resources retained. Cleanup: python3 {Path(__file__).resolve().parents[1] / 'scripts/lifecycle.py'} cleanup {self.root}"
                )

    def cleanup(self):
        try:
            self.collect()
        except (subprocess.SubprocessError, OSError) as error:
            print(f"Collection failed: {error}", file=sys.stderr)
        cleanup_pod = self.control + "-cleanup"
        self.k.call(
            "delete",
            "pod",
            self.control,
            *(self.prefix + "-" + role for role in self.roles),
            cleanup_pod,
            "--ignore-not-found",
            "--wait=true",
            "--timeout=2m",
        )
        self.k.call("apply", "-f", str(self.root / "harness.json"))
        items = yaml.safe_load((self.root / "control.yaml").read_text())["items"]
        configmap = next(item for item in items if item["kind"] == "ConfigMap")
        pod = next(item for item in items if item["kind"] == "Pod")
        pod["metadata"]["name"] = cleanup_pod
        pod["spec"]["activeDeadlineSeconds"] = 300
        main = pod["spec"]["containers"][0]
        main["command"] = ["python3", "-u", "-m", "harness.cleanup"]
        main["resources"] = {
            "requests": {"cpu": "100m", "memory": "256Mi"},
            "limits": {"cpu": "1", "memory": "1Gi"},
        }
        pod["spec"]["containers"] = [main]
        try:
            self.k.manifest(
                "apply", {"apiVersion": "v1", "kind": "List", "items": [configmap, pod]}
            )
            self.k.call(
                "wait",
                "--for=jsonpath={.status.phase}=Succeeded",
                "pod/" + cleanup_pod,
                "--timeout=6m",
            )
        finally:
            self.k.call("logs", cleanup_pod, output=self.root / "cleanup-objects.log")
        report = json.loads((self.root / "cleanup-objects.log").read_text())
        (self.root / "cleanup-objects.json").write_text(json.dumps(report))
        if not report["verified_absent"]:
            raise RuntimeError("Object cleanup was not verified")
        self.k.call("delete", "pod", cleanup_pod, "--wait=true", "--timeout=2m")
        pods = ["pod/" + self.control]
        for role in self.roles:
            self.k.call(
                "delete",
                "-f",
                str(self.root / f"worker-{role}.yaml"),
                "--ignore-not-found",
                "--wait=false",
            )
            pods.append("pod/" + self.prefix + "-" + role)
        self.k.call(
            "delete",
            "-f",
            str(self.root / "control.yaml"),
            "-f",
            str(self.root / "harness.json"),
            "--ignore-not-found",
            "--wait=false",
        )
        self.k.call("wait", "--for=delete", *pods, "--timeout=2m")
        self.k.call(
            "get",
            "pods,services,configmaps",
            "-l",
            "mx-benchmark=" + self.config["run"],
            "-o",
            "json",
            output=self.root / "cleanup-kubernetes.json",
        )
        if json.loads((self.root / "cleanup-kubernetes.json").read_text())["items"]:
            raise RuntimeError("Kubernetes resources remain")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run", "collect", "cleanup"])
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    getattr(Benchmark(args.directory), args.command)()


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
