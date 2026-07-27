# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for WorkerMetadata source_payload migration."""

from __future__ import annotations

from collections.abc import Sequence

from .. import p2p_pb2


def tensor_source_metadata(
    tensors: Sequence[p2p_pb2.TensorDescriptor],
) -> p2p_pb2.TensorSourceMetadata:
    return p2p_pb2.TensorSourceMetadata(tensors=list(tensors))


def worker_tensor_descriptors(worker: p2p_pb2.WorkerMetadata):
    payload = worker.WhichOneof("source_payload")
    if payload == "tensor_source":
        return worker.tensor_source.tensors
    if payload == "artifact_source":
        return []
    return worker.tensors


def worker_tensor_count(worker: p2p_pb2.WorkerMetadata) -> int:
    return len(worker_tensor_descriptors(worker))


# Accelerator-compatibility policy (fail-closed by construction):
#   - Same accelerator family: always allowed.
#   - Different family: allowed only for a hetero-eligible source type, an
#     allowlisted family pair, and explicitly unquantized weights.
#   - Unknown (empty) accelerator: deferred on the pre-fetch check, rejected on
#     the authoritative post-fetch check for a quantized identity.
# Anything undeclared or unrecognized is treated as the unsafe case.

# Accelerator families whose weight bytes are transferable cross-vendor over
# NIXL. Weights are plain tensor bytes whose dtype and size are validated on the
# receive path before any transfer, so they carry across families. Generated
# artifacts (CUDA graphs, torch.compile, Triton, DeepGEMM, TileLang, CuTe,
# FlashInfer caches) are accelerator/arch-specific and must never cross.
HETEROGENEOUS_NIXL_WEIGHT_PAIRS = {
    frozenset(("cuda", "xpu")),
}

# Source types eligible for heterogeneous transfer. WEIGHTS only: LoRA has no
# live tensor publish/transfer path yet, so it stays strict same-family until
# one exists and is tested. Add MX_SOURCE_TYPE_LORA here once that lands.
HETEROGENEOUS_NIXL_SOURCE_TYPES = {
    p2p_pb2.MX_SOURCE_TYPE_WEIGHTS,
}

# Sentinel for the quantization/dtype arguments so an omitted value fails
# closed: a caller that does not explicitly declare the identity's quantization
# is treated as quantized and never gets cross-family compatibility.
_UNSPECIFIED = object()

# Normalized dtype strings that are known-plain (non-quantized) weight storage.
# This is an allowlist, not a denylist: only these dtypes enable a cross-family
# transfer, so an unknown or future quantized dtype fails closed instead of
# silently admitting a hardware-specific layout. Values are compared after
# stripping a leading ``torch.`` and lowercasing (see ``_normalize_dtype``).
_UNQUANTIZED_DTYPES = frozenset(
    {
        "float32",
        "float",
        "f32",
        "float16",
        "half",
        "f16",
        "bfloat16",
        "bf16",
    }
)

# Quantization strings that mean "not quantized" (raw bf16/fp16 weights).
_UNQUANTIZED_QUANTIZATION_VALUES = ("", "none")


def _normalize_dtype(dtype: str) -> str:
    """Lowercase and drop a leading ``torch.`` so ``str(tensor.dtype)`` and the
    already-stripped identity form both normalize to the same token."""
    return dtype.strip().lower().removeprefix("torch.")


def _is_unquantized(quantization: object, dtype: object) -> bool:
    """Return whether an identity describes plain (non-quantized) weights.

    Both signals must affirmatively say "plain": ``SourceIdentity.quantization``
    must be an unquantized value and ``dtype`` must be an allowlisted plain
    dtype. This is "explicitly unquantized only" and fail-closed by
    construction: an argument left ``_UNSPECIFIED``, an empty ``dtype``, or any
    dtype not on the allowlist (int8/int4/qint8/auto and unknown future
    quantized dtypes) is treated as quantized, so a call site that does not
    fully declare a plain identity cannot silently enable a cross-family
    transfer.
    """
    if quantization is _UNSPECIFIED or dtype is _UNSPECIFIED:
        return False
    if str(quantization).strip().lower() not in _UNQUANTIZED_QUANTIZATION_VALUES:
        return False
    return _normalize_dtype(str(dtype)) in _UNQUANTIZED_DTYPES


def accelerators_compatible(
    target: str,
    source: str,
    *,
    mx_source_type: int | None = None,
    quantization: object = _UNSPECIFIED,
    dtype: object = _UNSPECIFIED,
    defer_unknown: bool = True,
) -> bool:
    """Return whether ``target`` may pull from ``source`` given the source type.

    Empty accelerator values mean unknown and are accepted for backward
    compatibility with workers published before accelerator metadata existed.
    Exact family matches are always compatible regardless of quantization.
    Cross-family transfer is allowed only for source types in
    ``HETEROGENEOUS_NIXL_SOURCE_TYPES``, family pairs in
    ``HETEROGENEOUS_NIXL_WEIGHT_PAIRS``, and only for unquantized weights:
    quantized weight layouts (fp8 scale packing, fp4/nvfp4 swizzle, hidden
    quant-config tensors) are hardware/kernel specific and are not proven
    transferable cross-vendor, so they are rejected pending inference-correctness
    validation. This is the single compatibility rule shared by RDMA tensor
    source selection and artifact source discovery, in both their pre-fetch and
    post-fetch checks.

    Args:
        target: Runtime accelerator family of the pulling (target) worker,
            e.g. ``"cuda"`` or ``"xpu"``. Empty means unknown.
        source: Runtime accelerator family of the source worker being
            considered. Empty means unknown.
        mx_source_type: The ``MxSourceType`` of the source. Defaults to ``None``
            so the gate fails closed: a caller that does not pass a
            hetero-eligible source type only ever gets strict same-family
            compatibility. Cross-family transfer requires a type in
            ``HETEROGENEOUS_NIXL_SOURCE_TYPES``.
        quantization: The identity's ``SourceIdentity.quantization`` string.
            Defaults to a sentinel that fails closed (treated as quantized) so a
            caller that does not declare it cannot silently enable a cross-family
            quantized transfer. Empty/``none`` is the unquantized signal.
        dtype: The identity's ``SourceIdentity.dtype`` string. Defaults to a
            sentinel that fails closed. Serves as a defensive net for backends
            that express quantized storage through dtype alone (see
            ``_is_unquantized``).
        defer_unknown: How an unknown (empty) accelerator is treated for
            quantized weights. ``True`` (the pre-fetch default) defers an unknown
            accelerator to the post-fetch check instead of rejecting it: the
            lightweight ``SourceInstanceRef`` may legitimately omit the
            accelerator (the ``k8s-service`` backend publishes a synthetic ref
            and only learns the real accelerator from ``GetTensorManifest``), and
            rejecting there would strand a valid same-family quantized source
            before the accelerator is known. ``False`` (the authoritative
            post-fetch check, where the real ``WorkerMetadata.accelerator`` is in
            hand) fails closed on a still-empty accelerator for a quantized
            identity, since an unknown family could be a different vendor whose
            kernels expect a different quantized layout.

    Returns:
        ``True`` if ``target`` may pull weights from ``source``, else ``False``.
    """
    same_known_family = bool(target) and bool(source) and target == source

    # Quantized weight layouts are hardware/kernel specific and only ride P2P
    # on a verified same-family match. On the authoritative post-fetch path
    # (not defer_unknown), this is checked before the empty-means-unknown
    # affordance below: an empty (unknown) accelerator may belong to a
    # different vendor whose kernels expect a different layout, so trusting it
    # would silently corrupt a quantized transfer. On the pre-fetch path an
    # unknown accelerator is deferred to that post-fetch check instead of being
    # rejected here (see defer_unknown).
    if (
        not defer_unknown
        and mx_source_type in HETEROGENEOUS_NIXL_SOURCE_TYPES
        and not same_known_family
        and not _is_unquantized(quantization, dtype)
    ):
        return False

    if not target or not source:
        return True
    if target == source:
        return True
    if mx_source_type in HETEROGENEOUS_NIXL_SOURCE_TYPES:
        if frozenset((target, source)) not in HETEROGENEOUS_NIXL_WEIGHT_PAIRS:
            return False
        return _is_unquantized(quantization, dtype)
    return False
