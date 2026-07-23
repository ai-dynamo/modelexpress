# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SLURM hostlist expansion for HPC bootstrap.

Parses SLURM compact hostlist notation (e.g. ``gpu[01-08]``) into individual
hostnames.  Used by DhtNode to auto-detect ``SLURM_JOB_NODELIST`` and bootstrap
from peer hostnames without explicit multiaddrs or K8s DNS.
"""

import itertools
import re


def expand_hostlist(hostlist: str) -> list[str]:
    """Expand a SLURM compact hostlist into individual hostnames.

    Supported syntax:
    - Simple list: ``host1,host2,host3``
    - Ranges: ``node[01-04]`` -> ``node01, node02, node03, node04``
    - Bracket lists: ``gpu[1,3,5]`` -> ``gpu1, gpu3, gpu5``
    - Mixed: ``gpu[1-3,5]`` -> ``gpu1, gpu2, gpu3, gpu5``
    - Multiple bracket groups: ``rack[1-2]-node[01-04]`` -> cartesian product
    - Zero-padding preserved from range endpoints

    Returns an empty list for empty input.
    """
    if not hostlist or not hostlist.strip():
        return []

    result = []
    for entry in _split_top_level(hostlist):
        entry = entry.strip()
        if not entry:
            continue
        segments = _parse_segments(entry)
        for combo in itertools.product(*segments):
            result.append("".join(combo))
    return result


def _split_top_level(hostlist: str) -> list[str]:
    """Split a hostlist string on commas that are NOT inside brackets."""
    parts = []
    depth = 0
    start = 0
    for i, ch in enumerate(hostlist):
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append(hostlist[start:i])
            start = i + 1
    parts.append(hostlist[start:])
    return parts


def _parse_segments(entry: str) -> list[list[str]]:
    """Parse a single hostlist entry into segments for cartesian product.

    Each segment is a list of strings. Literal text becomes a single-element
    list; bracket expressions expand to multiple elements.

    Example: ``rack[1-2]-node[01-04]``
      -> [["rack"], ["1","2"], ["-node"], ["01","02","03","04"]]
    """
    segments: list[list[str]] = []
    i = 0
    while i < len(entry):
        bracket_start = entry.find("[", i)
        if bracket_start == -1:
            # Rest is literal
            segments.append([entry[i:]])
            break
        # Literal before the bracket
        if bracket_start > i:
            segments.append([entry[i:bracket_start]])
        # Find the matching close bracket
        bracket_end = entry.find("]", bracket_start)
        if bracket_end == -1:
            # Malformed: treat rest as literal
            segments.append([entry[bracket_start:]])
            break
        # Expand the bracket expression
        bracket_content = entry[bracket_start + 1 : bracket_end]
        segments.append(_expand_bracket(bracket_content))
        i = bracket_end + 1
    return segments


def _expand_bracket(content: str) -> list[str]:
    """Expand a bracket expression like ``1-3,5,8-9`` into individual strings."""
    result = []
    for part in content.split(","):
        part = part.strip()
        m = re.match(r"^(\d+)-(\d+)$", part)
        if m:
            start_str, end_str = m.group(1), m.group(2)
            width = max(len(start_str), len(end_str))
            start_val, end_val = int(start_str), int(end_str)
            for val in range(start_val, end_val + 1):
                result.append(str(val).zfill(width))
        else:
            result.append(part)
    return result
