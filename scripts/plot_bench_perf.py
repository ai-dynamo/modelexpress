# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plot `bench_perf_history.py --format json` output as a stacked-line SVG.

One subplot per benchmark name. Each subplot shows the median timing as a line
with a shaded band for the [low, high] range from Criterion. Y-axis units
auto-scale per subplot (so a ps-scale bench and a µs-scale bench are both
readable).

Stdlib only. Output is a single .svg file you can open in any browser or paste
into Slack/Confluence/etc.

Usage:
    python3 scripts/bench_perf_history.py --format json \\
      | python3 scripts/plot_bench_perf.py --output perf.svg

    # or from a saved file
    python3 scripts/plot_bench_perf.py --input perf.json --output perf.svg
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from html import escape

UNIT_SCALES: list[tuple[str, float]] = [
    ("ps", 1e-3),
    ("ns", 1.0),
    ("us", 1e3),
    ("ms", 1e6),
    ("s", 1e9),
]
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#17becf"]


def auto_unit(median_ns: float) -> tuple[str, float]:
    for unit, scale in UNIT_SCALES:
        if median_ns < scale * 1000:
            return unit, scale
    return "s", 1e9


def to_dt(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def render_subplot(samples: list[dict], idx: int, name: str, x0: int, y0: int, w: int, h: int) -> str:
    rows = sorted(samples, key=lambda r: r["run_started_at"])
    xs = [to_dt(r["run_started_at"]) for r in rows]
    mids_ns = [r["mid_ns"] for r in rows]
    lows_ns = [r["low_ns"] for r in rows]
    highs_ns = [r["high_ns"] for r in rows]

    median = sorted(mids_ns)[len(mids_ns) // 2]
    unit, scale = auto_unit(median)
    mids = [v / scale for v in mids_ns]
    lows = [v / scale for v in lows_ns]
    highs = [v / scale for v in highs_ns]

    x_min, x_max = min(xs), max(xs)
    x_span = (x_max - x_min).total_seconds() or 1.0
    y_min = min(lows) * 0.95
    y_max = max(highs) * 1.05
    y_span = (y_max - y_min) or 1.0

    pad_l, pad_r, pad_t, pad_b = 70, 20, 26, 36
    plot_x = x0 + pad_l
    plot_y = y0 + pad_t
    plot_w = w - pad_l - pad_r
    plot_h = h - pad_t - pad_b

    def to_px(x_dt: datetime, y_v: float) -> tuple[float, float]:
        if len(xs) == 1:
            sx = plot_x + plot_w / 2
        else:
            sx = plot_x + ((x_dt - x_min).total_seconds() / x_span) * plot_w
        sy = plot_y + plot_h - ((y_v - y_min) / y_span) * plot_h
        return sx, sy

    color = COLORS[idx % len(COLORS)]
    out: list[str] = ['<g class="sub">']

    out.append(
        f'<text x="{x0 + w / 2:.1f}" y="{y0 + 16}" class="title">'
        f"{escape(name)} (n={len(rows)})</text>"
    )

    out.append(
        f'<rect x="{plot_x}" y="{plot_y}" width="{plot_w}" '
        f'height="{plot_h}" fill="white" stroke="#bbb"/>'
    )

    for i in range(5):
        frac = i / 4
        v = y_min + frac * y_span
        ty = plot_y + plot_h - frac * plot_h
        out.append(
            f'<line x1="{plot_x}" y1="{ty:.1f}" x2="{plot_x + plot_w}" '
            f'y2="{ty:.1f}" stroke="#eee"/>'
        )
        out.append(
            f'<text x="{plot_x - 6}" y="{ty + 4:.1f}" class="ylabel">{v:.3g}</text>'
        )

    out.append(
        f'<text x="{x0 + 14}" y="{y0 + h / 2:.1f}" class="yunit" '
        f'transform="rotate(-90 {x0 + 14} {y0 + h / 2:.1f})">{escape(unit)}</text>'
    )

    for frac in (0.0, 0.5, 1.0):
        tx = plot_x + frac * plot_w
        when = x_min + (x_max - x_min) * frac
        out.append(
            f'<text x="{tx:.1f}" y="{plot_y + plot_h + 16:.1f}" '
            f'class="xlabel">{when.strftime("%Y-%m-%d")}</text>'
        )

    if len(xs) >= 2:
        forward = [f"{to_px(x, lo)[0]:.1f},{to_px(x, lo)[1]:.1f}" for x, lo in zip(xs, lows)]
        reverse = [
            f"{to_px(x, hi)[0]:.1f},{to_px(x, hi)[1]:.1f}"
            for x, hi in zip(reversed(xs), reversed(highs))
        ]
        out.append(
            f'<polygon points="{" ".join(forward + reverse)}" '
            f'fill="{color}" opacity="0.15"/>'
        )

    line_pts = [f"{to_px(x, m)[0]:.1f},{to_px(x, m)[1]:.1f}" for x, m in zip(xs, mids)]
    if len(line_pts) >= 2:
        out.append(
            f'<polyline points="{" ".join(line_pts)}" fill="none" '
            f'stroke="{color}" stroke-width="2"/>'
        )

    for x_dt, m in zip(xs, mids):
        sx, sy = to_px(x_dt, m)
        out.append(f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="3" fill="{color}"/>')

    out.append("</g>")
    return "\n".join(out)


def render(samples: list[dict], out_path: str, sub_w: int, sub_h: int) -> None:
    groups: dict[str, list[dict]] = defaultdict(list)
    for s in samples:
        groups[s["bench_name"]].append(s)
    names = sorted(groups)

    margin = 20
    total_w = sub_w + 2 * margin
    total_h = (sub_h + margin) * len(names) + margin

    parts: list[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{total_w}" '
        f'height="{total_h}" font-family="sans-serif" font-size="11">',
        "<style>"
        ".title{font-size:13px;font-weight:bold;text-anchor:middle}"
        ".ylabel{text-anchor:end;fill:#444}"
        ".yunit{text-anchor:middle;fill:#444;font-style:italic}"
        ".xlabel{text-anchor:middle;fill:#444}"
        "</style>",
        f'<rect width="{total_w}" height="{total_h}" fill="white"/>',
    ]
    for i, name in enumerate(names):
        x0 = margin
        y0 = margin + i * (sub_h + margin)
        parts.append(render_subplot(groups[name], i, name, x0, y0, sub_w, sub_h))
    parts.append("</svg>")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input", default="-", help="JSON input file (default: stdin)")
    p.add_argument("--output", default="bench_perf.svg", help="output SVG path")
    p.add_argument("--width", type=int, default=900, help="subplot width in px")
    p.add_argument("--height", type=int, default=200, help="subplot height in px")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])

    if args.input == "-":
        samples = json.load(sys.stdin)
    else:
        with open(args.input, encoding="utf-8") as f:
            samples = json.load(f)

    if not samples:
        print("no samples in input", file=sys.stderr)
        return 1

    render(samples, args.output, args.width, args.height)
    n_benches = len({s["bench_name"] for s in samples})
    print(
        f"wrote {args.output} ({n_benches} bench{'es' if n_benches != 1 else ''}, "
        f"{len(samples)} samples)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
