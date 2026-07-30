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


# Digest the assembled destination rather than the received source shards. This is
# the other half of the gate: ``verify_full_pulls`` can only check sources whose
# whole shard lands in a staging buffer, which on Topology B is 6,192 of 18,867.
# The other 12,675 arrive as exact segments written straight into the receive
# buffers and are checked by nothing at all, so a wrong offset or stride in
# ``plan_pull`` delivers plausible bytes to the right place and every gate passes.
#
# Digesting the receive buffers closes that hole because *every* path lands there:
# exact segments directly, full pulls after the local re-slice, and converts after
# the cast. What it cannot do alone is say whether those bytes are *right* - a
# digest is not a reference. It becomes a gate when two runs are compared, one
# planned normally and one with ``MX_RESHARD_FORCE_FULL_PULL=1``, because the
# full-pull path's bytes are independently checked against the publishers'
# digests. A per-parameter difference then localises a segment-planning bug to a
# named tensor. See :mod:`modelexpress.refit.reshard.dest_digest_report`.
#
# The comparison is only valid while the weights hold still between the two runs,
# so run it against a quiesced publisher (step 1, or with training frozen);
# otherwise legitimate training updates read as mismatches, which is the same trap
# Bug 9 fell into.
DEST_DIGEST = os.environ.get("MX_RESHARD_DEST_DIGEST", "0") == "1"


def digest_destination(recv_buffers: dict) -> dict:
    """Digest each assembled destination buffer, keyed by parameter name.

    Called after re-slice and convert but *before* install, so the result
    describes what the fetch pipeline produced rather than what an
    engine-specific installer (which may quantize or derive) made of it. That
    keeps the digest comparable across engines and across the two plan shapes
    being differenced.

    Sorted so the emitted record has a stable key order and two runs diff cleanly.
    """
    return {name: tensor_digest(buf) for name, buf in sorted(recv_buffers.items())}


def source_expectation_digests(
    *,
    dest_sources: dict,
    sources: dict,
    fresh_sources: dict | None = None,
    expectation_is_current: bool = True,
) -> tuple[dict, dict]:
    """Per destination param, a digest over the *publisher* digests behind it.

    This is the counterpart to :func:`digest_destination` and it exists to make a
    single run self-checking. A destination digest on its own says only "these are
    the bytes we assembled"; it becomes evidence when compared against something,
    and the obvious something - the same digest from another run - stops being
    available after one training step, because training legitimately moves weights
    and two runs are not bitwise reproducible. Measured, not assumed: two runs
    differing in nothing disagreed on 10 of 4 350 param-steps, against 1 for the
    two runs differing in the path under test.

    Pairing each destination digest with a fingerprint of the source shards that
    fed it removes the second run from the argument. Over consecutive steps of one
    run the two sequences must move together, which separates the two explanations
    a changed destination digest otherwise has:

    * source fingerprint changed too - training moved the weights, expected;
    * source fingerprint held still while the destination moved, or the
      destination reverted to an earlier value while the source did not - the
      receiver installed bytes that are not what the publisher currently holds.

    The second is the staleness class of bug, and it is invisible to every other
    gate here: ``verify_full_pulls`` compares against the publisher's digest for
    the shard it *thinks* it read, so a stale-but-self-consistent read passes it.

    This is a fingerprint of the publishers' claims, not of received bytes, so it
    is not a correctness check by itself - a publisher that lies, or a shard table
    that is stale in the same way, is not caught. It answers "did the thing we were
    told to copy change" and nothing more.

    Params whose sources did not all carry a digest are reported as ``None`` rather
    than omitted, so a caller can tell "unchanged" from "no evidence" - the same
    distinction ``verify_full_pulls`` draws with ``checked == 0``, and for the same
    reason: a fleet of publishers predating the digest must degrade to no evidence
    instead of a silent pass.

    ``expectation_is_current`` must be false when ``sources`` describes weights
    older than the step being recorded and no ``fresh_sources`` was supplied. The
    whole record is then emitted as no-evidence, because a stale expectation here is
    worse than none: it cannot change, so every weight training legitimately moved
    reads as the destination having moved by itself, which is the audit's strongest
    finding. Bug 9 was that mistake made once already, and it cost a run's
    correctness verdict; refusing to emit is the only safe response.

    Returns ``(digests, stats)``.
    """
    if not expectation_is_current:
        return (
            dict.fromkeys(sorted(dest_sources)),
            {
                "params": len(dest_sources),
                "covered": 0,
                "uncovered": len(dest_sources),
                "shard_claims_from_fresh_table": 0,
                "reason": (
                    "shard table is older than this step and was not refreshed; "
                    "emitting no evidence rather than a frozen expectation"
                ),
            },
        )
    fresh_index, fresh_by_box = (
        _fresh_digest_index(fresh_sources) if fresh_sources else ({}, {})
    )
    digests: dict = {}
    covered = 0
    uncovered = 0
    from_fresh = 0
    for param_name, src_names in sorted(dest_sources.items()):
        # Sorted by (source, offset) so the fingerprint describes the set of shards
        # rather than the order the planner happened to visit them in. Replica
        # reselection between discoveries reorders that visit; it is not a change.
        claims: list[str] = []
        missing = False
        for src_name in sorted(src_names):
            info = sources.get(src_name)
            if info is None:
                missing = True
                continue
            for shard in sorted(
                getattr(info, "shards", ()),
                key=lambda s: tuple(int(x) for x in s.shard_offset),
            ):
                offset = tuple(int(x) for x in shard.shard_offset)
                fresh = fresh_index.get((src_name, shard.session, offset))
                if fresh is None:
                    # Same sibling-replica fallback as verify_full_pulls, and
                    # confined the same way: only when the replicas agree, since a
                    # box whose replicas disagree has no single expectation.
                    offers = fresh_by_box.get((src_name, offset))
                    if offers and len(offers) == 1:
                        fresh = next(iter(offers))
                digest = fresh if fresh is not None else getattr(shard, "digest", None)
                if digest is None:
                    missing = True
                    continue
                if fresh is not None:
                    from_fresh += 1
                claims.append(f"{src_name}|{offset}|{digest}")
        if missing or not claims:
            digests[param_name] = None
            uncovered += 1
            continue
        digests[param_name] = hashlib.blake2b(
            "\n".join(claims).encode(), digest_size=16
        ).hexdigest()
        covered += 1
    return digests, {
        "params": len(digests),
        "covered": covered,
        "uncovered": uncovered,
        "shard_claims_from_fresh_table": from_fresh,
    }


# The in-process differential. Everything above compares two *runs*, which only
# works where both hold identical source weights - measured to be step 1 and no
# further, because training is not bitwise reproducible and the run-to-run noise
# floor (10 differing params in 4,350) swamped the effect we were measuring (1).
#
# This compares the two *implementations* inside a single refit instead, over one
# set of received bytes, so there is no second run and no trajectory assumption:
#
#   full-pull path: stage the whole source contiguously, then slice it locally with
#                   torch ops driven by the recorded op-chain (``_replay_ops``).
#   exact path:     compute byte segments from the op-chain and shard table
#                   (``plan_pull``) and write them straight into the destination.
#
# Both consume the same op-chain and produce the same destination bytes if correct,
# but the arithmetic is independent: one materializes and narrows, the other solves
# for offsets and strides. Replaying the exact plan's segments *out of the staging
# buffer* pits them against each other on identical input. A wrong offset or stride
# in ``plan_pull`` - the failure that had no gate at all, on 12,675 of Topology B's
# 18,867 sources - shows up as a per-parameter difference.
#
# What this does not cover: the exact path's segments are executed here as local
# copies rather than as RDMA reads, so it tests the descriptor *computation*, not
# its execution over the wire. Descriptor execution is NIXL's, and is shared - the
# full-pull path issues reads through the same mechanism, only into staging instead
# of into live destinations. State that limit rather than claiming the exact path is
# verified end to end.
#
# Needs the source staged, so it wants ``MX_RESHARD_FORCE_FULL_PULL=1`` to reach
# every bounded source. Without it only the sources that were full-pulled anyway
# are covered, and the rest are reported uncovered rather than passing silently.
EXACT_REPLAY = os.environ.get("MX_RESHARD_EXACT_REPLAY", "0") == "1"


def exact_replay_digests(
    *,
    plan,
    sources: dict,
    full_staging: dict,
    recv_buffers: dict,
) -> tuple[dict, dict]:
    """Digest what the exact segment plan would have written, from staged bytes.

    Returns ``(digests, stats)`` where ``digests`` is per destination param and
    directly comparable to :func:`digest_destination` over the same
    ``recv_buffers``. A param is present only when every source feeding it was
    staged; otherwise the scratch buffer would have holes that read as mismatches,
    so it is omitted and counted in ``uncovered_params``.
    """
    import torch

    from modelexpress.refit.reshard.slice_plan import plan_pull
    from modelexpress.refit.reshard.types import UnsupportedReshard

    # Where each shard's bytes live inside its staging buffer. Read off the plan
    # rather than recomputed: these are the very segments that filled staging, so
    # the mapping cannot drift from what actually happened.
    staged_shards: dict = {}
    for full_pull in plan.full_pulls:
        for segment in full_pull.segments:
            staged_shards.setdefault(full_pull.src_name, []).append(
                (segment.session, segment.src_addr, segment.nbytes, segment.dst_byte)
            )

    # Destination params fed entirely by staged sources, with the copies that feed
    # them. A param whose sources are split between staged and unstaged is dropped.
    copies_by_param: dict = {}
    incomplete: set = set()
    for param, src_names in getattr(plan, "dest_sources", {}).items():
        if any(name not in staged_shards for name in src_names):
            incomplete.add(param)
    for full_pull in plan.full_pulls:
        for copy in full_pull.copies:
            copies_by_param.setdefault(copy.param_name, []).append(
                (full_pull.src_name, copy)
            )

    digests: dict = {}
    unplannable = 0
    unmapped_segments = 0
    for param, entries in sorted(copies_by_param.items()):
        buffer = recv_buffers.get(param)
        if buffer is None or param in incomplete:
            continue
        scratch = torch.zeros_like(buffer)
        scratch_bytes = scratch.reshape(-1).view(torch.uint8)
        failed = False
        for src_name, copy in entries:
            source = sources.get(src_name)
            if source is None:
                failed = True
                break
            try:
                segments = plan_pull(
                    copy,
                    source.global_shape,
                    source.dtype,
                    source.elsize,
                    source.shards,
                )
            except UnsupportedReshard:
                # The exact path could not plan this copy at all, so there is no
                # second implementation to compare against for this param.
                unplannable += 1
                failed = True
                break
            staging = full_staging[src_name].reshape(-1).view(torch.uint8)
            for segment in segments:
                offset = _staging_offset(staged_shards[src_name], segment)
                if offset is None:
                    unmapped_segments += 1
                    failed = True
                    break
                scratch_bytes[
                    segment.dst_byte : segment.dst_byte + segment.nbytes
                ] = staging[offset : offset + segment.nbytes]
            if failed:
                break
        if not failed:
            digests[param] = tensor_digest(scratch)

    return digests, {
        "params": len(digests),
        "uncovered_params": len(recv_buffers) - len(digests),
        "params_with_unstaged_sources": len(incomplete),
        "copies_the_exact_path_could_not_plan": unplannable,
        "segments_outside_any_staged_shard": unmapped_segments,
        "forced_full_pull": bool(getattr(plan, "forced_full_pull", False)),
    }


def _staging_offset(staged, segment) -> int | None:
    """Byte offset in the staging buffer for an exact segment's source address.

    ``PullSegment.src_addr`` is absolute in the publisher's address space
    (``shard.addr`` plus an offset), so the owning shard is found by address range
    within the matching session, then the offset carries over into staging. Returns
    ``None`` when no staged shard contains the address, which is a real finding -
    the exact plan would be reading memory the full-pull plan never covered - and is
    reported rather than skipped.
    """
    for session, addr, nbytes, staging_offset in staged:
        if session == segment.session and addr <= segment.src_addr < addr + nbytes:
            delta = segment.src_addr - addr
            if delta + segment.nbytes > nbytes:
                return None
            return staging_offset + delta
    return None


def compare_exact_replay(*, replayed: dict, received: dict) -> dict:
    """Difference the replayed exact plan against what the full-pull path installed.

    Returns a report with ``mismatches`` and the first few offending params. An
    empty ``replayed`` is reported as ``checked == 0``, which callers must read as
    no evidence rather than as a pass - the same rule as
    :func:`verify_full_pulls`.
    """
    detail = []
    checked = 0
    for param, digest in sorted(replayed.items()):
        expected = received.get(param)
        if expected is None:
            continue
        checked += 1
        if digest != expected:
            detail.append(
                {"param": param, "exact_replay": digest, "received": expected}
            )
    return {
        "checked": checked,
        "mismatches": len(detail),
        "detail": detail[:20],
    }


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


def _fresh_digest_index(fresh_sources: dict) -> tuple[dict, dict]:
    """Index a freshly discovered shard table, by exact owner and by box.

    Keyed on session and offset rather than position: discovery order is not
    stable, and ``merge_shard_tables`` keeps the first offer of each geometry, so
    two discoveries legitimately pin the same box to different - but byte-identical
    - replicas. Comparing positionally reports that reshuffle as a change.

    Session-keying alone is not enough, which is what let Bug 9 survive its first
    fix. A replicated box is offered by several ranks, and the two discoveries need
    not settle on the same one - a live run reselected 867 boxes to a different
    rank. The exact key then misses, the caller keeps the stale expectation, and
    the refresh silently does nothing. So a second index maps the box itself to the
    digests its replicas advertise, letting the caller fall back to a sibling offer.
    That fallback is only sound while replicas agree, so it is confined to boxes
    with a single distinct digest; a box whose replicas disagree is a divergence to
    report, not an expectation to adopt.
    """
    exact: dict = {}
    by_box: dict = {}
    for src_name, info in fresh_sources.items():
        for shard in getattr(info, "shards", ()):  # tolerate stubs in tests
            digest = getattr(shard, "digest", None)
            if digest is None:
                continue
            offset = tuple(shard.shard_offset)
            exact[(src_name, shard.session, offset)] = digest
            by_box.setdefault((src_name, offset), set()).add(digest)
    return exact, by_box


def verify_full_pulls(
    *,
    full_staging: dict,
    sources: dict,
    max_report: int = 20,
    fresh_sources: dict | None = None,
    step: int | None = None,
    stale_sessions: frozenset | set | None = None,
    stamps_seen: bool = False,
) -> dict:
    """Recompute digests over received full-pull bytes and compare to the publishers'.

    Only shards whose publisher supplied a digest are checked, so a fleet where the
    publishers predate this gate degrades to "checked 0" rather than failing - the
    caller must therefore treat ``checked == 0`` as *no evidence*, not as a pass.

    Restricted to full pulls because those are the only sources whose whole shard
    lands in a staging buffer the receiver can re-read. Exact-fetch segments are
    scattered straight into live params, so digesting them would mean digesting the
    destination layout instead of the source shard - a different check, not this one.

    ``fresh_sources`` is the fix for Bug 9. ``sources`` comes from ``_prepare()``,
    which runs once, so its digests describe the weights as of the *first* refit.
    Every later step then compares current bytes against a stale expectation, and
    any parameter training legitimately updated is reported as corruption. That is
    not hypothetical: on Topology B it produced exactly one mismatch
    (``model.layers.18.self_attn.o_proj.weight``) whose ``want`` was byte-identical
    across two runs while ``got`` tracked training - a frozen expectation against a
    moving reality. Diagnostics confirmed no source address ever moved, so the wire
    was reading the right memory and delivering current weights the whole time.

    When a freshly discovered table is supplied, its digest wins. The bytes were
    read from an address that has not moved, so what landed in staging is what the
    publisher holds *now*, and now is what the fresh table describes.

    **That fix is only as good as the freshness of the "fresh" table, and on
    2026-07-30 it was not fresh at all.** Re-discovery returned a blob identical to
    the prepare-time one at every step - `addr_changed: 0` *and* `digest_changed: 0`
    over ~18,432 comparisons - so the digests still described the weights at load
    time. Digesting the flagged tensors straight out of the HF checkpoint confirmed
    it: `want` was bit-for-bit the checkpoint value for all three, while `got`
    tracked training. The gate was therefore failing runs for the one reason it must
    never fail them - the reference was wrong, not the bytes - and `VERIFY_STRICT`
    aborted two runs on it.

    So freshness is now something this function *reports on* rather than assumes.
    ``step`` enables that: past the first refit, some weights have moved, so a fresh
    table that refreshed nothing cannot be current, and a mismatch against it is
    unattributable rather than a wire fault. ``reference_is_current`` carries that
    judgement, and the caller must not abort on a mismatch when it is false.

    A frozen reference is not a pass either. It means those shards are *unverified*,
    and the report says so rather than quietly reporting zero problems.

    ``stale_sessions`` supersedes that inference where it is available, and is the
    reason to prefer it. Deducing freshness from "did anything refresh?" is a whole-
    discovery verdict, and publishers propagate independently: when some publishers'
    tables for this step have landed and others' have not, *something* refreshed, the
    reference is pronounced current, and the lagging publisher's shard is then reported
    as a hard defect. That failure grows with publisher count, so it is a corner case
    on two receiver ranks and an expected one at sixteen. Passing the set of sessions
    whose published step did not advance replaces the guess with an observation and
    localises it to the shards actually affected, leaving a real mismatch elsewhere in
    the same report fatal.

    Returns a report with the counts and the first few mismatches by name.
    """
    fresh_index, fresh_by_box = (
        _fresh_digest_index(fresh_sources) if fresh_sources else ({}, {})
    )
    # Sources whose expectation was refreshed. Non-zero means training moved those
    # weights between prepare and this step - which is normal, and which without
    # this refresh would have been reported as that many mismatches.
    refreshed = 0
    # Of those, the ones the exact owner key missed and a sibling replica supplied.
    # Non-zero means the planner reselected the box between the two discoveries, so
    # an owner-keyed refresh alone would have left a stale expectation in place.
    refreshed_via_replica = 0
    checked = 0
    skipped = 0
    # Counted separately from the reported sample. Returning len(detail) conflates
    # "20 mismatches" with "at least 20 mismatches, truncated", and the difference
    # is the difference between a handful of bad shards and a systematically wrong
    # plan - which is exactly the judgement this report exists to support.
    failed = 0
    # Of the failures, the ones whose publisher's step stamp shows its table describes
    # an earlier step. Those are unattributable; the remainder are real.
    failed_stale = 0
    stale_session_set = frozenset(stale_sessions or ())
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
            offset_key = tuple(shard.shard_offset)
            current = fresh_index.get((src_name, shard.session, offset_key))
            via_replica = False
            if current is None:
                offers = fresh_by_box.get((src_name, offset_key))
                if offers and len(offers) == 1:
                    current = next(iter(offers))
                    via_replica = True
            if current is not None:
                if want is not None and current != want:
                    refreshed += 1
                    if via_replica:
                        refreshed_via_replica += 1
                want = current
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
                # A mismatch is only evidence about the wire if the digest it was
                # compared against describes this step. When the publisher's own step
                # stamp says otherwise, the comparison is uninformative for this shard
                # and only for this shard - other sessions in the same report stay
                # fully accountable.
                if stale_session_set and shard.session in stale_session_set:
                    failed_stale += 1
                if FILL_SENTINEL:
                    fraction = _sentinel_fraction(region)
                    sentinel_total += fraction
                    if fraction > 0.999:
                        never_written += 1
                if len(mismatches) < max_report:
                    entry = {
                        "stale_publisher": bool(
                            stale_session_set and shard.session in stale_session_set
                        ),
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
        # 0 with a fresh table supplied means training did not move these weights,
        # which makes the run a weak test of the gate rather than a strong pass.
        "digests_refreshed": refreshed,
        "digests_refreshed_via_replica": refreshed_via_replica,
        "digest_source": "fresh" if fresh_index else "prepare",
    }
    # Past the first refit the optimizer has touched something, so a table that
    # refreshed nothing is describing older weights than the ones just read. Step 1
    # is exempt: nothing has moved yet, so refreshing nothing is the correct answer
    # there and the reference is genuinely current.
    report["mismatches_from_stale_publishers"] = failed_stale
    report["stale_publisher_sessions"] = sorted(stale_session_set)
    if stamps_seen or stale_session_set:
        # An observation beats an inference. With stamps in hand, freshness is decided
        # per shard above, so the whole-discovery guess is not consulted at all - it is
        # the guess that mis-handles partial propagation.
        #
        # ``stamps_seen`` matters separately from a non-empty stale set: stamps present
        # with nothing lagging is a positive statement that the reference is current,
        # whereas the inference would call that same table stale whenever no digest
        # happened to change - which is exactly the situation at a low learning rate,
        # where few weights move per step. Treating "no stamps" and "stamps, none
        # lagging" alike would throw away verification we have earned.
        report["freshness_evidence"] = "publisher_step_stamp"
        attributable = failed - failed_stale
        reference_is_current = not stale_session_set
    else:
        report["freshness_evidence"] = "refresh_inference"
        reference_is_current = bool(step is None or int(step) <= 1 or refreshed > 0)
        # Without stamps the verdict is all-or-nothing, because the evidence is. A
        # reference the inference calls stale makes every mismatch in this report
        # unattributable, not just some - claiming otherwise would contradict
        # ``reference_is_current`` in the same record.
        attributable = failed if reference_is_current else 0
    # The caller's abort signal. It already folds in whichever freshness judgement was
    # available, so a caller aborting on this alone gets the old behaviour when there
    # are no stamps and the per-shard behaviour when there are.
    report["attributable_mismatches"] = attributable
    report["reference_is_current"] = reference_is_current
    if failed_stale:
        report["stale_reference_suspected"] = True
        report["unattributable_reason"] = (
            f"{failed_stale} of {failed} mismatching shard(s) at step {step} come from "
            f"publisher(s) whose own step stamp shows their shard table describes an "
            f"earlier step: {sorted(stale_session_set)}. Those comparisons say nothing "
            f"about the wire and those shards are UNVERIFIED, which is not the same as "
            f"clean. The remaining {failed - failed_stale} mismatch(es), if any, are "
            f"attributable and are a real finding."
        )
    elif failed and not reference_is_current:
        report["stale_reference_suspected"] = True
        report["unattributable_reason"] = (
            f"{failed} shard(s) differ from the publisher's digest at step {step}, "
            f"but the freshly discovered table refreshed 0 of {checked} digests, so "
            f"it cannot describe weights the optimizer has already moved. A "
            f"difference against a stale reference says nothing about the wire. "
            f"These shards are UNVERIFIED, which is not the same as clean. Confirm "
            f"by digesting one of them out of the initial checkpoint: if `want` "
            f"equals the checkpoint value, the reference is frozen, not the bytes. "
            f"No publisher step stamps were available; a publisher carrying them "
            f"would make this per-shard instead of a whole-run guess."
        )
    if FILL_SENTINEL:
        # never_written == mismatches says the wire skipped those regions
        # entirely; never_written == 0 with a mismatch says it wrote the wrong
        # bytes there. Anything in between localises a partial write.
        report["never_written"] = never_written
        report["mean_sentinel_frac"] = (
            round(sentinel_total / failed, 6) if failed else 0.0
        )
    return report
