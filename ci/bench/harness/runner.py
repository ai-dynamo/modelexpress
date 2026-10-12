# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the same pause/refit/verify/resume protocol for the selected model profile."""

import json
import time
import traceback
from pathlib import Path

from harness import scenario, validation


class RefitRunner:
    """Run one refit protocol and retain its evidence before resuming workers."""

    def __init__(self, config, root, case=None):
        import requests

        self.config = config
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.case = case if case is not None else scenario.load(config)
        self.requests = requests
        self.urls = {
            role: f"http://{config['resource_prefix']}-{role}:8080/"
            for role in config["roles"]
        }

    def call(self, role, name, route, body):
        started = time.time()
        t = time.perf_counter()
        response = self.requests.post(self.urls[role] + route, json=body, timeout=7200)
        record = {
            "started_unix": started,
            "finished_unix": time.time(),
            "seconds": time.perf_counter() - t,
            "http_status": response.status_code,
            "response": response.json(),
        }
        (self.root / f"{role}-{name}.json").write_text(json.dumps(record, indent=2))
        print("RESULT", role, name, json.dumps(record), flush=True)
        response.raise_for_status()
        assert record["response"]["ok"], record
        result = record["response"]["result"]
        return result

    def rpc(self, role, name, method, kwargs):
        rows = self.call(role, name, "rpc", {"method": method, "kwargs": kwargs})
        validation.ranks(rows, self.config)
        return rows

    def audit(self, role, name):
        if self.config["expected_host_scales_per_rank"] is not None:
            validation.scales(
                self.rpc(role, name, "verify_host_scales", {}),
                self.config,
            )

    def tensor_hashes(self, role, name):
        return validation.hashes(
            self.rpc(role, name, "tensor_hashes", {"phase": name}),
            self.config,
        )

    def run(self):
        try:
            publication = json.loads(Path(self.config["publication_path"]).read_text())
            base, updated = {}, {}
            for role, url in self.urls.items():
                deadline = time.monotonic() + 7200
                while time.monotonic() < deadline:
                    try:
                        if (
                            self.requests.get(url + "health", timeout=5).status_code
                            == 200
                        ):
                            break
                    except self.requests.RequestException:
                        pass
                    time.sleep(5)
                else:
                    raise TimeoutError(role + " readiness")
                validation.inference(
                    self.call(
                        role,
                        "baseline",
                        "generate",
                        {"prompt": "The capital of France is"},
                    )
                )
                self.call(role, "pause", "pause", {})
                self.audit(role, "baseline-host-scales")
                base[role] = self.tensor_hashes(role, "base-hashes")
            for role in self.urls:
                rows = self.rpc(
                    role,
                    "init",
                    "hotload_init",
                    {
                        "initial_version_id": self.config["initial_version"],
                        "source": self.config["sources"][role],
                    },
                )
                assert all(x["phase"] == "init" for x in rows)
            for role in self.urls:
                rows = self.rpc(
                    role,
                    "refit",
                    "hotload",
                    {"version_id": self.config["target_version"]},
                )
                validation.refit(rows, self.config, role)
                self.case.verify_refit(self.rpc, role, publication)
                self.audit(role, "immediate-post-refit-host-scales")
                updated[role] = self.tensor_hashes(role, "updated-hashes")
            self.case.validate_inventory(base, updated, publication)
            for role in self.urls:
                self.audit(role, "host-scales")
            for role in self.urls:
                self.call(role, "resume", "resume", {})
                validation.inference(
                    self.call(
                        role,
                        "post-refit-inference",
                        "generate",
                        {"prompt": "The capital of France is"},
                    )
                )
            for role in self.urls:
                self.call(role, "final-pause", "pause", {})
                self.audit(role, "post-inference-host-scales")
                self.call(role, "final-resume", "resume", {})
            (self.root / "PASS").write_text(
                "Requested paths, all TP ranks, refit and resumed inference verified\n"
            )
            print("BENCH_PASS", flush=True)
        except Exception:
            (self.root / "FAIL").write_text(traceback.format_exc())
            traceback.print_exc()
            raise


def main():
    from harness.config import load_config

    RefitRunner(load_config(), "/tmp/mx-bench").run()


if __name__ == "__main__":
    main()
