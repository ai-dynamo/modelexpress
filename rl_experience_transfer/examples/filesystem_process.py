# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Minimal persistent multiprocess filesystem transfer."""

from __future__ import annotations

import argparse
import multiprocessing
import tempfile
from pathlib import Path

import numpy as np

from rlxfer import (
    ExperienceBatch,
    ExperienceConsumer,
    ExperienceMetadata,
    ExperienceProducer,
    TensorPayload,
)
from rlxfer.transports import FileSystemTransport


def consume(path: str) -> None:
    transport = FileSystemTransport(path)
    delivery = ExperienceConsumer(transport).receive(timeout=5.0)
    if delivery is None:
        raise RuntimeError("receive timed out")
    np.testing.assert_array_equal(delivery.batch.tensors["rewards"].data, [1.0, -1.0])
    delivery.ack()
    transport.close()


def run(path: Path) -> None:
    transport = FileSystemTransport(path)
    producer = ExperienceProducer(transport)
    batch = ExperienceBatch(
        metadata=ExperienceMetadata("rollout-0", "canonical", "1"),
        tensors={"rewards": TensorPayload(np.asarray([1.0, -1.0], dtype=np.float32))},
    )
    receipt = producer.publish(batch)
    process = multiprocessing.get_context("spawn").Process(target=consume, args=(str(path),))
    process.start()
    process.join(10.0)
    if process.exitcode != 0 or receipt.wait(1.0).state.value != "acked":
        raise RuntimeError("multiprocess transfer failed")
    transport.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path)
    arguments = parser.parse_args()
    if arguments.path is not None:
        run(arguments.path)
        return
    with tempfile.TemporaryDirectory(prefix="rlxfer-filesystem-") as temporary:
        run(Path(temporary))


if __name__ == "__main__":
    main()
