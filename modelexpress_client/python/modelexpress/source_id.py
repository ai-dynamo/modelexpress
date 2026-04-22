# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Content-addressed source ID computation.

Mirrors the Rust ``compute_mx_source_id`` in
``modelexpress_server/src/source_identity.rs``. When the central server
is in the loop, it computes the ID and returns it to clients; in
peer-direct / K8s-Service-routed deployments where there is no central
server, clients compute the ID locally. The two implementations MUST
produce identical IDs for identical identities, or the handshake
validation that ties ``mx_source_id`` to the content of
``SourceIdentity`` breaks.
"""

from __future__ import annotations

import hashlib
import json

from . import p2p_pb2


def compute_mx_source_id(identity: p2p_pb2.SourceIdentity) -> str:
    """Compute the 16-char hex ``mx_source_id`` for a ``SourceIdentity``.

    All string fields are lowercased and ``extra_parameters`` is sorted
    by key so semantically-identical identities produce identical IDs
    regardless of case or insertion order.
    """
    canonical = _canonical_json(identity)
    digest = hashlib.sha256(canonical.encode()).hexdigest()
    return digest[:16]


def _canonical_json(identity: p2p_pb2.SourceIdentity) -> str:
    sorted_extra = {
        k.lower(): v.lower()
        for k, v in sorted(identity.extra_parameters.items())
    }
    payload = {
        "mx_version": identity.mx_version.lower(),
        "mx_source_type": identity.mx_source_type,
        "model_name": identity.model_name.lower(),
        "backend_framework": identity.backend_framework,
        "tensor_parallel_size": identity.tensor_parallel_size,
        "pipeline_parallel_size": identity.pipeline_parallel_size,
        "expert_parallel_size": identity.expert_parallel_size,
        "dtype": identity.dtype.lower(),
        "quantization": identity.quantization.lower(),
        "extra_parameters": sorted_extra,
        "revision": identity.revision.lower(),
    }
    # Rust's serde_json::json! stores keys in a BTreeMap, which sorts
    # them alphabetically on serialization. Match that ordering with
    # sort_keys=True so both sides produce the exact same bytes for
    # SHA256 to hash.
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)
