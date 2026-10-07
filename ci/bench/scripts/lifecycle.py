# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run, collect, and clean up rendered benchmarks."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import scenario


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
        self.scenario = scenario.load(self.config)
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
        self.scenario.collect_publication(self)
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
            [sys.executable, str(self.root / "report.py"), str(self.root)],
            check=True,
            timeout=180,
        )

    def start(self, pod, script, log):
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
                    "/opt/benchmark/" + script,
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

    def run(self):
        if (self.root / "started").exists():
            raise ValueError("Render a fresh run; this directory was already started")
        self.k.call("cluster-info")
        (self.root / "started").touch(exist_ok=False)
        try:
            self.scenario.prepare_run(self)
            self.k.call(
                "exec",
                self.control,
                "-c",
                "main",
                "--",
                "python3",
                "-u",
                "/opt/benchmark/run_bench.py",
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
                    f"Resources retained. Cleanup: python3 {Path(__file__).resolve()} cleanup {self.root}"
                )

    def cleanup(self):
        try:
            self.collect()
        except (subprocess.SubprocessError, OSError) as error:
            print(f"Collection failed: {error}", file=sys.stderr)
        self.scenario.cleanup(self)
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
