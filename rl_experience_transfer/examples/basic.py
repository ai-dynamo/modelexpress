# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Minimal transport-independent producer and consumer."""

import numpy as np

from rlxfer import (
    ExperienceBatch,
    ExperienceConsumer,
    ExperienceMetadata,
    ExperienceProducer,
    TensorPayload,
)
from rlxfer.transports import InMemoryTransport

transport = InMemoryTransport()
producer = ExperienceProducer(transport)
consumer = ExperienceConsumer(transport)
batch = ExperienceBatch(
    metadata=ExperienceMetadata("rollout-0", "canonical", "1"),
    tensors={"tokens": TensorPayload(np.asarray([[1, 2, 3]], dtype=np.int64))},
)
receipt = producer.publish(batch)
delivery = consumer.receive(timeout=1.0)
assert delivery is not None
np.testing.assert_array_equal(delivery.batch.tensors["tokens"].data, [[1, 2, 3]])
delivery.ack()
assert receipt.wait(1.0).state.value == "acked"
consumer.close()
