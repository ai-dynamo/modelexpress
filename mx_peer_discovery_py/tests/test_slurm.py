# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SLURM hostlist expansion tests.

Covers:
- Simple comma-separated hostlists
- Range expansion with and without zero-padding
- Bracket lists (commas inside brackets)
- Mixed ranges and lists in brackets
- Multiple bracket groups (cartesian product)
- Single hostnames and empty input
- Top-level comma splitting that respects brackets
- Whitespace handling
- Malformed input (unterminated bracket)
"""

from mx_peer_discovery.slurm import expand_hostlist


def test_simple_list():
    assert expand_hostlist("host1,host2") == ["host1", "host2"]


def test_simple_list_three():
    assert expand_hostlist("host1,host2,host3") == ["host1", "host2", "host3"]


def test_range():
    assert expand_hostlist("node[01-04]") == ["node01", "node02", "node03", "node04"]


def test_range_no_padding():
    assert expand_hostlist("gpu[1-3]") == ["gpu1", "gpu2", "gpu3"]


def test_bracket_list():
    assert expand_hostlist("gpu[1,3,5]") == ["gpu1", "gpu3", "gpu5"]


def test_mixed_range_list():
    assert expand_hostlist("n[1-3,5,8-9]") == ["n1", "n2", "n3", "n5", "n8", "n9"]


def test_multiple_brackets():
    result = expand_hostlist("rack[1-2]-node[01-02]")
    assert result == [
        "rack1-node01", "rack1-node02",
        "rack2-node01", "rack2-node02",
    ]


def test_no_brackets():
    assert expand_hostlist("singlehost") == ["singlehost"]


def test_empty():
    assert expand_hostlist("") == []


def test_whitespace_only():
    assert expand_hostlist("   ") == []


def test_comma_with_brackets():
    result = expand_hostlist("a[1-2],b[3-4]")
    assert result == ["a1", "a2", "b3", "b4"]


def test_zero_padding_preserved():
    result = expand_hostlist("n[08-12]")
    assert result == ["n08", "n09", "n10", "n11", "n12"]


def test_zero_padding_uses_wider_endpoint():
    # width follows max(len(start), len(end)), so "8-10" pads to width 2
    result = expand_hostlist("n[8-10]")
    assert result == ["n08", "n09", "n10"]


def test_whitespace_around_entries():
    assert expand_hostlist(" host1 , host2 ") == ["host1", "host2"]


def test_mixed_entries():
    # A bare host and a bracketed entry combined at top level
    result = expand_hostlist("seed,gpu[1-2]")
    assert result == ["seed", "gpu1", "gpu2"]


def test_unterminated_bracket_is_literal():
    # Malformed: missing closing bracket. The remainder is kept as literal.
    result = expand_hostlist("node[1-2")
    assert result == ["node[1-2"]


def test_adjacent_brackets():
    result = expand_hostlist("[1-2][a,b]")
    assert result == ["1a", "1b", "2a", "2b"]


def test_single_element_bracket():
    assert expand_hostlist("node[5]") == ["node5"]
