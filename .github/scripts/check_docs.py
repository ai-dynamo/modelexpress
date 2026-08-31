#!/usr/bin/env python3
"""Validate local Markdown links and heading anchors without third-party packages."""

from __future__ import annotations

import html
import re
import sys
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[2]
LINK_RE = re.compile(r"(?<!!)\[[^\]]*\]\(([^)]+)\)")
FENCE_RE = re.compile(r"```.*?```|~~~.*?~~~", re.DOTALL)
HEADING_RE = re.compile(r"^ {0,3}#{1,6}\s+(.+?)\s*#*\s*$", re.MULTILINE)
ID_RE = re.compile(r"\bid\s*=\s*[\"']([^\"']+)[\"']", re.IGNORECASE)


def _link_target(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("<"):
        end = raw.find(">")
        return raw[1:end] if end >= 0 else raw
    return raw.split()[0]


def _slugify(heading: str) -> str:
    heading = html.unescape(re.sub(r"<[^>]+>", "", heading)).lower().strip()
    heading = re.sub(r"[^\w\s-]", "", heading)
    return re.sub(r"\s+", "-", heading)


def _anchors(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8")
    anchors = {match.group(1) for match in ID_RE.finditer(text)}
    counts: dict[str, int] = {}
    for match in HEADING_RE.finditer(text):
        slug = _slugify(match.group(1))
        suffix = counts.get(slug, 0)
        anchors.add(slug if suffix == 0 else f"{slug}-{suffix}")
        counts[slug] = suffix + 1
    return anchors


def main() -> int:
    errors: list[str] = []
    checked = 0
    for path in sorted(ROOT.rglob("*.md")):
        if ".git" in path.parts or ".tmp" in path.parts:
            continue
        text = FENCE_RE.sub("", path.read_text(encoding="utf-8"))
        for match in LINK_RE.finditer(text):
            raw = _link_target(match.group(1))
            if raw.startswith(("#", "http://", "https://", "mailto:", "data:")):
                continue

            target, _, fragment = raw.partition("#")
            target = unquote(target.split("?", 1)[0])
            resolved = (path.parent / target).resolve()
            checked += 1
            line = text.count("\n", 0, match.start()) + 1
            if not resolved.exists():
                errors.append(f"{path.relative_to(ROOT)}:{line}: missing target {raw!r}")
                continue
            if fragment and resolved.is_file() and resolved.suffix.lower() == ".md":
                if fragment not in _anchors(resolved):
                    errors.append(
                        f"{path.relative_to(ROOT)}:{line}: missing anchor {raw!r}"
                    )

    if errors:
        print("Documentation link check failed:", file=sys.stderr)
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(f"Documentation link check passed ({checked} local links).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
