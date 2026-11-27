# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU SHA-256 leaves and pairwise reduction for registered weight bytes."""

import triton
import triton.language as tl

BLOCK_BYTES = 16 * 1024
_LANES = 128
_K = (
    0x428A2F98,
    0x71374491,
    0xB5C0FBCF,
    0xE9B5DBA5,
    0x3956C25B,
    0x59F111F1,
    0x923F82A4,
    0xAB1C5ED5,
    0xD807AA98,
    0x12835B01,
    0x243185BE,
    0x550C7DC3,
    0x72BE5D74,
    0x80DEB1FE,
    0x9BDC06A7,
    0xC19BF174,
    0xE49B69C1,
    0xEFBE4786,
    0x0FC19DC6,
    0x240CA1CC,
    0x2DE92C6F,
    0x4A7484AA,
    0x5CB0A9DC,
    0x76F988DA,
    0x983E5152,
    0xA831C66D,
    0xB00327C8,
    0xBF597FC7,
    0xC6E00BF3,
    0xD5A79147,
    0x06CA6351,
    0x14292967,
    0x27B70A85,
    0x2E1B2138,
    0x4D2C6DFC,
    0x53380D13,
    0x650A7354,
    0x766A0ABB,
    0x81C2C92E,
    0x92722C85,
    0xA2BFE8A1,
    0xA81A664B,
    0xC24B8B70,
    0xC76C51A3,
    0xD192E819,
    0xD6990624,
    0xF40E3585,
    0x106AA070,
    0x19A4C116,
    0x1E376C08,
    0x2748774C,
    0x34B0BCB5,
    0x391C0CB3,
    0x4ED8AA4A,
    0x5B9CCA4F,
    0x682E6FF3,
    0x748F82EE,
    0x78A5636F,
    0x84C87814,
    0x8CC70208,
    0x90BEFFFA,
    0xA4506CEB,
    0xBEF9A3F7,
    0xC67178F2,
)


@triton.jit
def _initial_state(lanes: tl.constexpr):
    return (
        tl.full((lanes,), 0x6A09E667, tl.uint32),
        tl.full((lanes,), 0xBB67AE85, tl.uint32),
        tl.full((lanes,), 0x3C6EF372, tl.uint32),
        tl.full((lanes,), 0xA54FF53A, tl.uint32),
        tl.full((lanes,), 0x510E527F, tl.uint32),
        tl.full((lanes,), 0x9B05688C, tl.uint32),
        tl.full((lanes,), 0x1F83D9AB, tl.uint32),
        tl.full((lanes,), 0x5BE0CD19, tl.uint32),
    )


@triton.jit
def _ror(value, shift: tl.constexpr):
    return (value >> shift) | (value << (32 - shift))


@triton.jit
def _compress(state, words, constants):
    a, b, c, d, e, f, g, h = state
    schedule = ()
    for index in tl.static_range(16):
        schedule += (tl.broadcast_to(words[index], a.shape),)
    for start in range(0, 64, 4):
        for step in tl.static_range(4):
            word = schedule[0]
            x, y = schedule[1], schedule[14]
            s0 = _ror(x, 7) ^ _ror(x, 18) ^ (x >> 3)
            s1 = _ror(y, 17) ^ _ror(y, 19) ^ (y >> 10)
            schedule = schedule[1:] + (word + s0 + schedule[9] + s1,)
            s1 = _ror(e, 6) ^ _ror(e, 11) ^ _ror(e, 25)
            choice = g ^ (e & (f ^ g))
            constant = tl.load(constants + start + step).to(tl.uint32)
            t1 = h + s1 + choice + constant + word
            s0 = _ror(a, 2) ^ _ror(a, 13) ^ _ror(a, 22)
            majority = (a & b) ^ (a & c) ^ (b & c)
            a, b, c, d, e, f, g, h = t1 + s0 + majority, a, b, c, d + t1, e, f, g
    return (
        state[0] + a,
        state[1] + b,
        state[2] + c,
        state[3] + d,
        state[4] + e,
        state[5] + f,
        state[6] + g,
        state[7] + h,
    )


@triton.jit
def _payload_word(data, offset, pos, length, padding_end, bits, aligned: tl.constexpr):
    word = tl.full(length.shape, 0, tl.uint32)
    if aligned:
        ptr = data.to(tl.pointer_type(tl.uint32)) + (offset + pos) // 4
        raw = tl.load(ptr, mask=pos + 4 <= length, other=0)
        word = (
            ((raw & 255) << 24)
            | ((raw & 65280) << 8)
            | ((raw >> 8) & 65280)
            | (raw >> 24)
        )
    for i in tl.static_range(4):
        index = pos + i
        partial = (
            (pos + 4 > length) if aligned else tl.full(length.shape, True, tl.int1)
        )
        byte = tl.load(data + offset + index, mask=partial & (index < length), other=0)
        byte = tl.where(index == length, 128, byte).to(tl.uint32)
        word |= byte << (24 - i * 8)
    return tl.where(pos == padding_end - 4, bits, word)


@triton.jit(do_not_specialize=("data_bytes", "prefix_blocks", "first_leaf"))
def _hash_leaves(
    data,
    prefix,
    output,
    constants,
    data_bytes,
    prefix_blocks,
    first_leaf,
    chunk_bytes: tl.constexpr,
    lanes: tl.constexpr,
    aligned: tl.constexpr,
):
    leaf = tl.program_id(0) * lanes + tl.arange(0, lanes)
    offset = leaf.to(tl.int64) * chunk_bytes
    length = tl.maximum(0, tl.minimum(chunk_bytes, data_bytes - offset)).to(tl.int32)
    state = _initial_state(lanes)
    for block in range(prefix_blocks):
        words = ()
        for i in tl.static_range(16):
            words += (tl.load(prefix + block * 16 + i).to(tl.uint32),)
        state = _compress(state, words, constants)

    length_word = length.to(tl.uint32)
    zero = tl.cast(0, tl.uint32)
    block_header = (
        (offset >> 32).to(tl.uint32),
        offset.to(tl.uint32),
        length_word,
        zero,
        zero,
        zero,
        zero,
        zero,
        zero,
        zero,
        zero,
        zero,
        zero,
        zero,
        zero,
        zero,
    )
    state = _compress(state, block_header, constants)

    blocks = tl.cdiv(length + 9, 64)
    bits = (((prefix_blocks + 1) * 64 + length) * 8).to(tl.uint32)
    max_length = tl.minimum(
        chunk_bytes, data_bytes - tl.program_id(0).to(tl.int64) * lanes * chunk_bytes
    )
    for block in range(tl.cdiv(max_length + 9, 64)):
        words = ()
        for i in tl.static_range(16):
            words += (
                _payload_word(
                    data, offset, block * 64 + i * 4, length, blocks * 64, bits, aligned
                ),
            )
        updated = _compress(state, words, constants)
        state = (
            tl.where(block < blocks, updated[0], state[0]),
            tl.where(block < blocks, updated[1], state[1]),
            tl.where(block < blocks, updated[2], state[2]),
            tl.where(block < blocks, updated[3], state[3]),
            tl.where(block < blocks, updated[4], state[4]),
            tl.where(block < blocks, updated[5], state[5]),
            tl.where(block < blocks, updated[6], state[6]),
            tl.where(block < blocks, updated[7], state[7]),
        )

    for i in tl.static_range(8):
        tl.store(output + (first_leaf + leaf) * 8 + i, state[i], mask=length > 0)


@triton.jit
def _hash_parents(children, output, constants, count, lanes: tl.constexpr):
    parent = tl.program_id(0) * lanes + tl.arange(0, lanes)
    child = parent * 2
    left, right = (), ()
    for i in tl.static_range(8):
        left += (
            tl.load(children + child * 8 + i, mask=child < count, other=0).to(
                tl.uint32
            ),
        )
        right += (
            tl.load(children + (child + 1) * 8 + i, mask=child + 1 < count, other=0).to(
                tl.uint32
            ),
        )

    first = (tl.cast(1, tl.uint32),) + left + right[:7]
    zero = tl.cast(0, tl.uint32)
    second = (
        right[7],
        tl.cast(0x80000000, tl.uint32),
        zero,
        zero,
        zero,
        zero,
        zero,
        zero,
        zero,
        zero,
        zero,
        zero,
        zero,
        zero,
        zero,
        tl.cast(68 * 8, tl.uint32),
    )
    state = _compress(_initial_state(lanes), first, constants)
    state = _compress(state, second, constants)
    for i in tl.static_range(8):
        word = tl.where(child + 1 < count, state[i], left[i])
        tl.store(output + parent * 8 + i, word, mask=child < count)


def round_constants(device):
    import torch

    signed = [value if value < 2**31 else value - 2**32 for value in _K]
    return torch.tensor(signed, dtype=torch.int32, device=device)


def hash_leaves(data, prefix, output, constants, first_leaf: int) -> None:
    count = triton.cdiv(data.numel(), BLOCK_BYTES)
    _hash_leaves[(triton.cdiv(count, _LANES),)](
        data,
        prefix,
        output,
        constants,
        data.numel(),
        prefix.numel() // 16,
        first_leaf,
        BLOCK_BYTES,
        _LANES,
        data.data_ptr() % 4 == 0,
    )


def hash_parents(children, output, constants) -> None:
    _hash_parents[(triton.cdiv(output.shape[0], _LANES),)](
        children,
        output,
        constants,
        children.shape[0],
        _LANES,
    )
