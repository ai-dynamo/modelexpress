# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Content-addressed source ID computation.

Mirrors the Rust ``compute_mx_source_id`` in
``modelexpress_server/src/p2p/source_identity.rs``. When the central server
is in the loop, it computes the ID and returns it to clients; in
decentralized / K8s-Service-routed deployments where there is no central
server, clients compute the ID locally. The two implementations MUST
produce identical IDs for identical identities, or the handshake
validation that ties ``mx_source_id`` to the content of
``SourceIdentity`` breaks.
"""

from __future__ import annotations

import hashlib
import json

from .. import p2p_pb2

_OPTIONAL_SOURCE_ID_FIELDS = (
    "backend_framework_version",
    "torch_version",
    "cuda_version",
    "triton_version",
    "gpu_arch",
    "compile_config_digest",
)


def compute_mx_source_id(identity: p2p_pb2.SourceIdentity) -> str:
    """Compute the 16-char hex ``mx_source_id`` for a ``SourceIdentity``.

    Model names, revisions and extra parameters preserve case. Compatibility
    labels retain their lowercase normalization. Map insertion order does
    not affect the ID.
    """
    canonical = _canonical_json(identity)
    digest = hashlib.sha256(canonical.encode()).hexdigest()
    return digest[:16]


def _canonical_json(identity: p2p_pb2.SourceIdentity) -> str:
    payload = {
        "mx_version": identity.mx_version.lower(),
        "mx_source_type": identity.mx_source_type,
        "model_name": identity.model_name,
        "backend_framework": identity.backend_framework,
        "tensor_parallel_size": identity.tensor_parallel_size,
        "pipeline_parallel_size": identity.pipeline_parallel_size,
        "expert_parallel_size": identity.expert_parallel_size,
        "dtype": identity.dtype.lower(),
        "quantization": identity.quantization.lower(),
        "extra_parameters": dict(identity.extra_parameters),
        "revision": identity.revision,
    }
    # Optional compatibility fields split artifact/cache sources by framework,
    # compiler, CUDA, and GPU constraints only when set.
    for field in _OPTIONAL_SOURCE_ID_FIELDS:
        value = getattr(identity, field)
        if value:
            payload[field] = value.lower()
    # Rust's serde_json::json! stores keys in a BTreeMap, which sorts
    # them alphabetically on serialization. Match that ordering with
    # sort_keys=True so both sides produce the exact same bytes for
    # SHA256 to hash.
    return json.dumps(
        payload, separators=(",", ":"), sort_keys=True, ensure_ascii=False
    )
