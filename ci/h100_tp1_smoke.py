# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run a same-node ModelExpress NIXL transfer between two TP=1 GPU ranks."""

from __future__ import annotations

import multiprocessing as mp
import os
import time
import traceback
from dataclasses import asdict
from multiprocessing.connection import Connection
from typing import Any

TRANSFER_ELEMENTS = 8 * 1024 * 1024
TRANSFER_VALUE = 7.0
PROCESS_TIMEOUT_SECONDS = 180


def _source_rank(
    gpu_uuid: str,
    metadata_pipe: Connection,
    target_done: mp.synchronize.Event,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_uuid

    import torch

    from modelexpress.nixl_transfer import NixlTransferManager

    manager = NixlTransferManager("h100-tp1-source-rank0", device_id=0)
    metadata_sent = False
    try:
        if torch.cuda.device_count() != 1:
            raise RuntimeError("source TP=1 rank must see exactly one GPU")

        manager.initialize()
        source_tensor = torch.full(
            (TRANSFER_ELEMENTS,),
            TRANSFER_VALUE,
            dtype=torch.float32,
            device="cuda",
        )
        torch.cuda.synchronize()
        metadata = manager.register_tensors({"smoke.weight": source_tensor})
        metadata_pipe.send(
            {
                "metadata": metadata,
                "descriptors": [
                    asdict(descriptor) for descriptor in manager.tensor_descriptors
                ],
            }
        )
        metadata_sent = True

        if not target_done.wait(PROCESS_TIMEOUT_SECONDS):
            raise TimeoutError("target rank did not finish before source timeout")
    except Exception:
        if not metadata_sent:
            metadata_pipe.send({"error": traceback.format_exc()})
        raise
    finally:
        metadata_pipe.close()
        manager.shutdown()


def _target_rank(
    gpu_uuid: str,
    source_payload: dict[str, Any],
    result_pipe: Connection,
    target_done: mp.synchronize.Event,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_uuid

    import torch

    from modelexpress.nixl_transfer import NixlTransferManager
    from modelexpress.types import TensorDescriptor

    manager = NixlTransferManager("h100-tp1-target-rank0", device_id=0)
    try:
        if torch.cuda.device_count() != 1:
            raise RuntimeError("target TP=1 rank must see exactly one GPU")

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
            source_metadata=source_payload["metadata"],
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

        result_pipe.send(
            {
                "bytes": byte_count,
                "tensors": tensor_count,
                "duration": duration,
            }
        )
    except Exception:
        result_pipe.send({"error": traceback.format_exc()})
        raise
    finally:
        manager.shutdown()
        target_done.set()
        result_pipe.close()


def _receive(
    pipe: Connection,
    process: mp.Process,
    label: str,
) -> dict[str, Any]:
    deadline = time.monotonic() + PROCESS_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if pipe.poll(1):
            return pipe.recv()
        if not process.is_alive():
            raise RuntimeError(f"{label} exited with code {process.exitcode}")
    raise TimeoutError(f"timed out waiting for {label}")


def main() -> None:
    """Transfer one tensor from the source H100 rank to the target H100 rank."""
    source_gpu = os.environ["MX_H100_SOURCE_GPU"]
    target_gpu = os.environ["MX_H100_TARGET_GPU"]
    if source_gpu == target_gpu:
        raise RuntimeError("source and target ranks must use different GPUs")

    context = mp.get_context("spawn")
    metadata_recv, metadata_send = context.Pipe(duplex=False)
    target_done = context.Event()
    source = context.Process(
        target=_source_rank,
        args=(source_gpu, metadata_send, target_done),
        name="h100-tp1-source-rank0",
    )
    target: mp.Process | None = None
    result_recv: Connection | None = None
    result: dict[str, Any] | None = None

    source.start()
    metadata_send.close()
    try:
        source_payload = _receive(metadata_recv, source, "source metadata")
        if "error" in source_payload:
            raise RuntimeError(f"source rank failed:\n{source_payload['error']}")

        result_recv, result_send = context.Pipe(duplex=False)
        target = context.Process(
            target=_target_rank,
            args=(target_gpu, source_payload, result_send, target_done),
            name="h100-tp1-target-rank0",
        )
        target.start()
        result_send.close()
        result = _receive(result_recv, target, "target result")
        if "error" in result:
            raise RuntimeError(f"target rank failed:\n{result['error']}")
    finally:
        if target is not None:
            target.join(30)
            if target.is_alive():
                target.terminate()
                target.join(10)
        target_done.set()
        source.join(30)
        if source.is_alive():
            source.terminate()
            source.join(10)
        metadata_recv.close()
        if result_recv is not None:
            result_recv.close()

    if source.exitcode != 0:
        raise RuntimeError(f"source rank exited with code {source.exitcode}")
    if target is None or target.exitcode != 0:
        code = None if target is None else target.exitcode
        raise RuntimeError(f"target rank exited with code {code}")
    if result is None:
        raise RuntimeError("target rank returned no result")

    gib_per_second = result["bytes"] / result["duration"] / (1024**3)
    print(
        "H100_TP1_TO_TP1_OK "
        f"bytes={result['bytes']} tensors={result['tensors']} "
        f"seconds={result['duration']:.3f} GiB/s={gib_per_second:.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
