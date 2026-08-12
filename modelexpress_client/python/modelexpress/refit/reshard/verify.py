# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Shard digests for the reshard refit path.

A refit that reports plausible timings can still install the wrong bytes: the wire
can deliver correctly while the *plan* points a copy at the wrong sub-box, and
generation-level metrics such as KL against a reference are too coarse to localise
that. This module supplies the cheap half of the answer - a digest a publisher can
compute over the shard it owns, and which a receiver can recompute over the bytes it
actually received - so a mismatch names the offending tensor instead of showing up
as a slightly worse loss curve three steps later.

Why not a byte hash: hashing 30 GB per rank per refit on the GPU means either a
device-side hash kernel we do not have, or a copy to host we cannot afford. The
digest below is a *position-sensitive* reduction instead - see :func:`tensor_digest`
for why plain sums are not enough.

This module is the digest and the wire format for it. The receiver-side gate that
recomputes and compares is deliberately not here; it needs the fresh-discovery
refresh to avoid reporting ordinary training updates as corruption, which is its own
change. Publishing is opt-in via ``MX_RESHARD_PUBLISH_DIGEST`` because the reduction
costs a pass over every published tensor.

Scope, because it is narrower than "the refit is verified" and the difference matters
when planning the consumer. A publisher digest covers a whole shard, so a receiver can
compare it directly only when the ordinary plan gives that receiver every byte of the
shard. That includes a coarser inference topology which assembles whole trainer shards
into known destination sub-boxes. A finer inference topology instead reads only part of
each trainer shard, so its partial destination has no value comparable to the published
whole-shard digest.

For strict qualification of that partial-read case, the receiver must full-pull and
verify the publisher's whole shard, derive the expected destination slice from the
verified staging buffer, and compare it with the destination produced by the ordinary
exact-segment read. Both reads must target the same immutable, step-stamped source
version. :func:`shard_region` serves the full-pull reference path; the direct case needs
the transfer plan's source-to-destination provenance to locate each complete publisher
shard in the assembled destination. A fingerprint over the contributing publishers is
useful freshness evidence, but it is not a byte-for-byte expectation for a partial read
and must not be presented as one.
"""

from __future__ import annotations

import hashlib
import logging

logger = logging.getLogger(__name__)

# Words per digest row. Row sums are position-sensitive *between* rows, since the row
# vector is hashed in order, so this is the granularity at which a permutation is
# detected: reordering whole 4 KiB rows is caught, reordering words inside one row is
# not. 1024 int32 words keeps the hashed vector ~4 KiB per MiB of tensor, small enough
# to move to host, while staying coarse enough that the reduction is a single fast
# kernel.
_ROW_WORDS = 1024

_U64 = (1 << 64) - 1

# Position weights are the same vector for every shard of a given row count, and a
# refit digests thousands of shards, so they are built once per (length, device)
# rather than per call.
_WEIGHT_CACHE: dict = {}


def _row_weights(length: int, device):
    """Cached ``[1..length]`` on ``device``, for the position-weighted reduction."""
    import torch

    key = (length, str(device))
    weights = _WEIGHT_CACHE.get(key)
    if weights is None:
        weights = torch.arange(1, length + 1, device=device, dtype=torch.int64)
        _WEIGHT_CACHE[key] = weights
    return weights


def tensor_digest(tensor) -> str:
    """A position-sensitive digest of ``tensor``'s bytes, computed on-device.

    Plain reductions (sum, sum-of-squares) are unusable here because the failure
    mode we most need to catch is a *permutation*: a plan that copies the right
    bytes to the wrong offsets preserves every order-independent statistic. So the
    bytes are viewed as int32 words, reshaped into rows, reduced per row, and the
    resulting row vector is hashed **in order** - reordering rows changes the hash,
    while the reduction itself stays one cheap kernel over the full tensor.

    The tensor is made contiguous first, since a strided view has no meaningful byte
    order to digest, and is rebased to a word-aligned start. Returns a hex string so it
    can ride in JSON alongside the shard table.
    """
    import torch

    flat = tensor.detach().reshape(-1)
    if not flat.is_contiguous():
        flat = flat.contiguous()
    elif (flat.storage_offset() * flat.element_size()) % 4:
        # What a publisher actually passes. ``narrow`` returns a view that is
        # contiguous *and* offset, so the branch above never fires, and
        # ``.contiguous()`` would return self anyway - only a copy rebases the start.
        # Without this the int32 view below raises on any shard whose byte offset is
        # not a multiple of 4, which is every odd bf16 split: the gate/up narrow at an
        # odd half, the QKV narrow at an odd row.
        #
        # Rebasing rather than skipping the leading bytes to reach an aligned
        # boundary, which would be cheaper and wrong: the digest has to be a function
        # of the shard's contents alone. A publisher digesting at offset 2818 and a
        # receiver digesting the same bytes at offset 0 must agree, and they only do
        # if neither one's result depends on where the bytes happen to sit.
        flat = flat.clone()
    raw = flat.view(torch.uint8)
    n = int(raw.numel())

    # int32 view needs a 4-byte-aligned length; digest the aligned body as words and
    # fold the remainder in separately rather than silently dropping it.
    body = n - (n % 4)
    words = raw[:body].view(torch.int32)
    rows = int(words.numel()) // _ROW_WORDS

    # Reduce all the way to scalars on-device. Moving the row vector to the host
    # instead would mean a per-element Python conversion for every one of the
    # thousands of shards in a refit, which is slower than the transfer being
    # verified. Position sensitivity survives the second reduction because the row
    # sums are weighted by their index before being summed; int64 overflow simply
    # makes it arithmetic mod 2**64, which is still order-dependent.
    scalars = [n]
    if rows:
        # dtype=int64 accumulates without materialising an int64 copy of the input.
        row_sums = (
            words[: rows * _ROW_WORDS]
            .view(rows, _ROW_WORDS)
            .sum(dim=1, dtype=torch.int64)
        )
        weights = _row_weights(rows, row_sums.device)
        scalars.append(int(row_sums.sum().item()))
        scalars.append(int((row_sums * weights).sum().item()))
    tail = words[rows * _ROW_WORDS :]
    if tail.numel():
        tail64 = tail.to(torch.int64)
        scalars.append(int(tail64.sum().item()))
        scalars.append(
            int((tail64 * _row_weights(tail64.numel(), tail64.device)).sum().item())
        )
    if n % 4:
        # The ragged bytes past the last whole int32 word. Folded in explicitly
        # because dropping them would make two tensors differing only in their final
        # bytes digest identically.
        scalars.append(int(raw[body:].to(torch.int64).sum().item()))
        scalars.append(n % 4)

    h = hashlib.blake2b(digest_size=16)
    for value in scalars:
        h.update((value & _U64).to_bytes(8, "little"))
    return h.hexdigest()


def shard_region(full_tensor, global_shape, shard_offset, shape):
    """The sub-box of an assembled full-pull staging buffer that one publisher owns.

    ``full_tensor`` is the flat staging buffer holding the whole logical source
    tensor; a publisher's shard occupies the box at ``shard_offset`` with ``shape``.
    Returned as a contiguous copy because :func:`tensor_digest` digests byte order
    and the box is generally strided inside the full tensor.

    Here rather than with the comparison gate because it is the inverse of what a
    publisher digests: the publisher hashes its own contiguous shard, and this is how
    a receiver recovers the same box out of the buffer it assembled.

    The three ranks must agree. A short ``shard_offset`` would otherwise stop the
    narrow loop early and return a larger region than the publisher digested, which
    surfaces as a mismatch on a shard that transferred correctly - the most expensive
    kind of wrong answer for a gate whose whole job is to be trusted.
    """
    if not len(global_shape) == len(shard_offset) == len(shape):
        raise ValueError(
            f"rank mismatch: global_shape {tuple(global_shape)}, shard_offset "
            f"{tuple(shard_offset)} and shape {tuple(shape)} must have equal rank"
        )
    view = full_tensor.reshape(tuple(global_shape))
    for axis, (start, extent) in enumerate(zip(shard_offset, shape, strict=True)):
        view = view.narrow(axis, int(start), int(extent))
    return view.contiguous()


def published_digest(tensor) -> str | None:
    """``tensor_digest`` when digest publication is on, otherwise ``None``.

    Read live rather than captured at import so a publisher can be switched without
    reimporting, and so the tests do not have to reload this module.
    """
    from modelexpress import envs

    if not envs.MX_RESHARD_PUBLISH_DIGEST:
        return None
    return tensor_digest(tensor)
