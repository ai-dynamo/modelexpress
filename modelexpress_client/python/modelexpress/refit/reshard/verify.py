# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Parameter-equality gate for the reshard refit path.

A refit that reports plausible timings can still install the wrong bytes: the wire
can deliver correctly while the *plan* points a copy at the wrong sub-box, and
generation-level metrics (KL against a reference) are too coarse to localise that.
This module supplies the cheap half of the answer - a digest each publisher can
compute over the shard it owns, and which the receiver can recompute over the
bytes it actually received - so a mismatch names the offending tensor instead of
showing up as a slightly worse loss curve three steps later.

Opt-in via ``MX_RESHARD_VERIFY=1``. It costs a few passes over the received bytes,
which is a large relative cost against a ~1.5 s wire, so it is a gate to run when
qualifying a build, not something to leave on in a throughput measurement.

Why not a byte hash: hashing 30 GB per rank per refit on the GPU means either a
device-side hash kernel we do not have, or a copy to host we cannot afford. The
digest below is a *position-sensitive* reduction instead - see :func:`tensor_digest`
for why plain sums are not enough.
"""

from __future__ import annotations

import hashlib
import logging
import os

logger = logging.getLogger(__name__)

VERIFY = os.environ.get("MX_RESHARD_VERIFY", "0") == "1"

# A gate that cannot run must fail, not pass quietly. The blob schema omits the
# digest key when the publisher has none and the receiver then reports the shard as
# *unchecked*, which is correct for a rolling upgrade and wrong for a gate: a fleet
# whose publishers all predate the digest reports zero mismatches and reads as a
# clean run. That is exactly how a whole series of runs came to be labelled verified
# while proving nothing. Since ``MX_RESHARD_VERIFY`` is already opt-in, the default
# here is strict; set ``MX_RESHARD_VERIFY_STRICT=0`` only when deliberately
# qualifying against a mixed fleet, and then do not call the result a pass.
VERIFY_STRICT = os.environ.get("MX_RESHARD_VERIFY_STRICT", "1") == "1"

# Words per digest row. Row sums are position-sensitive *between* rows (the row
# vector is hashed in order), so this is the granularity at which a permutation is
# detected: reordering whole 4 KiB rows is caught, reordering words inside one row
# is not. 1024 int32 words keeps the hashed vector ~4 KiB per MiB of tensor, small
# enough to move to host, while staying coarse enough that the reduction is a
# single fast kernel.
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
    resulting row vector is hashed **in order** - reordering rows changes the
    hash, while the reduction itself stays one cheap kernel over the full tensor.

    The tensor is made contiguous first, since a strided view has no meaningful
    byte order to digest. Returns a hex string so it can ride in JSON alongside
    the shard table.
    """
    import torch

    flat = tensor.detach().reshape(-1)
    if not flat.is_contiguous():
        flat = flat.contiguous()
    raw = flat.view(torch.uint8)
    n = int(raw.numel())

    # int32 view needs a 4-byte-aligned length; digest the aligned body as words
    # and fold the remainder in separately rather than silently dropping it.
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
        row_sums = words[: rows * _ROW_WORDS].view(rows, _ROW_WORDS).sum(
            dim=1, dtype=torch.int64
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
        # because dropping them would make two tensors differing only in their
        # final bytes digest identically.
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
    """
    view = full_tensor.reshape(tuple(global_shape))
    for axis, (start, extent) in enumerate(zip(shard_offset, shape)):
        view = view.narrow(axis, int(start), int(extent))
    return view.contiguous()


SENTINEL_BYTE = int(os.environ.get("MX_RESHARD_SENTINEL_BYTE", "165"))  # 0xA5

# Pre-filling the staging buffers with a byte no weight tensor plausibly contains
# turns "these bytes are wrong" into "these bytes were never written", which are
# different bugs. The buffers are allocated once and reused for every refit step,
# so an unwritten region silently holds the *previous* step's weights - plausible
# values, wrong step - and is indistinguishable from a corrupt transfer without
# this. Diagnostic only: it costs a full memset of the staging arena per step.
FILL_SENTINEL = os.environ.get("MX_RESHARD_FILL_SENTINEL", "0") == "1"


def fill_sentinel(full_staging: dict) -> None:
    """Stamp every full-pull staging buffer with :data:`SENTINEL_BYTE`."""
    import torch

    for tensor in full_staging.values():
        tensor.view(torch.uint8).fill_(SENTINEL_BYTE)


def _sentinel_fraction(region) -> float:
    """Fraction of ``region``'s bytes still holding the sentinel."""
    import torch

    raw = region.reshape(-1).view(torch.uint8)
    return float((raw == SENTINEL_BYTE).sum().item()) / max(1, raw.numel())


def verify_full_pulls(
    *,
    full_staging: dict,
    sources: dict,
    max_report: int = 20,
) -> dict:
    """Recompute digests over received full-pull bytes and compare to the publishers'.

    Only shards whose publisher supplied a digest are checked, so a fleet where the
    publishers predate this gate degrades to "checked 0" rather than failing - the
    caller must therefore treat ``checked == 0`` as *no evidence*, not as a pass.

    Restricted to full pulls because those are the only sources whose whole shard
    lands in a staging buffer the receiver can re-read. Exact-fetch segments are
    scattered straight into live params, so digesting them would mean digesting the
    destination layout instead of the source shard - a different check, not this one.

    Returns a report with the counts and the first few mismatches by name.
    """
    checked = 0
    skipped = 0
    # Counted separately from the reported sample. Returning len(detail) conflates
    # "20 mismatches" with "at least 20 mismatches, truncated", and the difference
    # is the difference between a handful of bad shards and a systematically wrong
    # plan - which is exactly the judgement this report exists to support.
    failed = 0
    mismatches = []
    # Replicated placements (DP, and expert-DP for MoE experts) mean the same box is
    # offered by more than one rank. Those offers must be byte-identical; the planner
    # picks one and is entitled to assume the choice does not matter. When they are
    # not identical, a mismatch says nothing about the wire - the receiver faithfully
    # delivered one replica and is being compared against another - so the two cases
    # have opposite fixes and must not be reported as one number.
    divergent_replicas = []
    # Only meaningful under FILL_SENTINEL; see the module notes on that flag.
    sentinel_total = 0.0
    never_written = 0
    for src_name, staging in full_staging.items():
        info = sources.get(src_name)
        if info is None:
            continue
        by_box: dict = {}
        for shard in info.shards:
            key = (tuple(shard.shard_offset), tuple(shard.shape))
            digest = getattr(shard, "digest", None)
            if digest is not None:
                by_box.setdefault(key, {}).setdefault(digest, []).append(shard.session)
        for (offset, shape), offers in by_box.items():
            if len(offers) > 1 and len(divergent_replicas) < max_report:
                divergent_replicas.append(
                    {
                        "source": src_name,
                        "shard_offset": list(offset),
                        "shape": list(shape),
                        "offers": {d: sorted(s) for d, s in offers.items()},
                    }
                )
        for shard in info.shards:
            want = getattr(shard, "digest", None)
            if want is None:
                skipped += 1
                continue
            region = shard_region(
                staging, info.global_shape, shard.shard_offset, shard.shape
            )
            got = tensor_digest(region)
            checked += 1
            if got != want:
                failed += 1
                if FILL_SENTINEL:
                    fraction = _sentinel_fraction(region)
                    sentinel_total += fraction
                    if fraction > 0.999:
                        never_written += 1
                if len(mismatches) < max_report:
                    entry = {
                        "source": src_name,
                        "session": shard.session,
                        "shard_offset": list(shard.shard_offset),
                        "shape": list(shard.shape),
                        "want": want,
                        "got": got,
                    }
                    if FILL_SENTINEL:
                        entry["sentinel_frac"] = round(fraction, 6)
                    mismatches.append(entry)
    report = {
        "checked": checked,
        "skipped_no_digest": skipped,
        "mismatches": failed,
        "detail": mismatches,
        "detail_truncated": failed > len(mismatches),
        "divergent_replicas": len(divergent_replicas),
        "divergent_detail": divergent_replicas,
    }
    if FILL_SENTINEL:
        # never_written == mismatches says the wire skipped those regions
        # entirely; never_written == 0 with a mismatch says it wrote the wrong
        # bytes there. Anything in between localises a partial write.
        report["never_written"] = never_written
        report["mean_sentinel_frac"] = round(sentinel_total / failed, 6) if failed else 0.0
    return report
