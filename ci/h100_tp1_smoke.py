# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run a ModelExpress NIXL transfer between two Kubernetes TP=1 GPU pods."""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
import time
import urllib.request
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

TRANSFER_ELEMENTS = 8 * 1024 * 1024
TRANSFER_VALUE = 7.0
METADATA_PORT = 8080
SOURCE_TIMEOUT_SECONDS = 20 * 60
TARGET_TIMEOUT_SECONDS = 5 * 60


class _MetadataHandler(BaseHTTPRequestHandler):
    payload = b""
    target_done = False

    def do_GET(self) -> None:
        if self.path != "/metadata":
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(self.payload)))
        self.end_headers()
        self.wfile.write(self.payload)

    def do_POST(self) -> None:
        if self.path != "/done":
            self.send_error(404)
            return
        type(self).target_done = True
        self.send_response(204)
        self.end_headers()

    def log_message(self, format: str, *args: Any) -> None:
        return


def _validate_h100(role: str) -> None:
    import torch

    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"{role} TP=1 pod must see exactly one GPU")
    gpu_name = torch.cuda.get_device_name(0)
    if not gpu_name.startswith("NVIDIA H100"):
        raise RuntimeError(f"{role} expected NVIDIA H100, found {gpu_name}")
    gpu_uuids = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True
    ).splitlines()
    if len(gpu_uuids) != 1:
        raise RuntimeError(f"{role} expected one GPU UUID, found {len(gpu_uuids)}")
    print(
        f"H100_GPU_OK role={role} name={gpu_name} uuid={gpu_uuids[0].strip()}",
        flush=True,
    )


def _run_source() -> None:
    import torch

    from modelexpress.nixl_transfer import NixlTransferManager

    _validate_h100("source")
    manager = NixlTransferManager("h100-tp1-source-rank0", device_id=0)
    server: ThreadingHTTPServer | None = None
    try:
        manager.initialize()
        source_tensor = torch.full(
            (TRANSFER_ELEMENTS,),
            TRANSFER_VALUE,
            dtype=torch.float32,
            device="cuda",
        )
        torch.cuda.synchronize()
        metadata = manager.register_tensors({"smoke.weight": source_tensor})
        _MetadataHandler.payload = json.dumps(
            {
                "metadata": base64.b64encode(metadata).decode("ascii"),
                "descriptors": [
                    asdict(descriptor) for descriptor in manager.tensor_descriptors
                ],
            }
        ).encode("utf-8")
        _MetadataHandler.target_done = False
        server = ThreadingHTTPServer(("0.0.0.0", METADATA_PORT), _MetadataHandler)
        server.timeout = 1
        print("H100_TP1_SOURCE_READY", flush=True)

        deadline = time.monotonic() + SOURCE_TIMEOUT_SECONDS
        while not _MetadataHandler.target_done and time.monotonic() < deadline:
            server.handle_request()
        if not _MetadataHandler.target_done:
            raise TimeoutError("target did not finish before source timeout")
        print("H100_TP1_SOURCE_OK", flush=True)
    finally:
        if server is not None:
            server.server_close()
        manager.shutdown()


def _fetch_metadata(source_url: str) -> dict[str, Any]:
    deadline = time.monotonic() + TARGET_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{source_url}/metadata", timeout=5) as response:
                return json.load(response)
        except (OSError, json.JSONDecodeError):
            time.sleep(2)
    raise TimeoutError(f"timed out fetching source metadata from {source_url}")


def _run_target(source_url: str) -> None:
    import torch

    from modelexpress.nixl_transfer import NixlTransferManager
    from modelexpress.types import TensorDescriptor

    _validate_h100("target")
    source_payload = _fetch_metadata(source_url.rstrip("/"))
    manager = NixlTransferManager("h100-tp1-target-rank0", device_id=0)
    try:
        manager.initialize()
        target_tensor = torch.zeros(
            (TRANSFER_ELEMENTS,),
            dtype=torch.float32,
            device="cuda",
        )
        torch.cuda.synchronize()
        manager.register_tensors({"smoke.weight": target_tensor})
        descriptors = [
            TensorDescriptor(**descriptor)
            for descriptor in source_payload["descriptors"]
        ]
        byte_count, tensor_count, duration = manager.receive_from_source(
            source_metadata=base64.b64decode(source_payload["metadata"], validate=True),
            source_tensors=descriptors,
            timeout_seconds=60,
            require_exact_match=True,
        )
        expected_bytes = TRANSFER_ELEMENTS * target_tensor.element_size()
        if byte_count != expected_bytes or tensor_count != 1:
            raise RuntimeError(
                f"unexpected transfer result: bytes={byte_count}, tensors={tensor_count}"
            )
        if not bool(torch.all(target_tensor == TRANSFER_VALUE).item()):
            raise RuntimeError("target tensor does not match the source tensor")
    finally:
        manager.shutdown()

    request = urllib.request.Request(f"{source_url.rstrip('/')}/done", method="POST")
    with urllib.request.urlopen(request, timeout=5):
        pass
    gib_per_second = byte_count / duration / (1024**3)
    print(
        "H100_TP1_TO_TP1_OK "
        f"bytes={byte_count} tensors={tensor_count} "
        f"seconds={duration:.3f} GiB/s={gib_per_second:.2f}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("role", choices=("source", "target"))
    parser.add_argument("--source-url", default="http://h100-tp1-source:8080")
    args = parser.parse_args()

    if args.role == "source":
        _run_source()
    else:
        _run_target(args.source_url)


if __name__ == "__main__":
    main()
