# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generated protobuf bindings for libp2p message types.

Regenerate with:
    python -m grpc_tools.protoc -I proto --python_out=proto proto/*.proto
"""

from .crypto_pb2 import KeyType as KeyType, PublicKey as PublicKey
from .dht_pb2 import Message as Message, Record as Record
from .identify_pb2 import Identify as Identify
from .noise_pb2 import NoiseHandshakePayload as NoiseHandshakePayload
