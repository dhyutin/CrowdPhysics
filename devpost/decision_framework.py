#!/usr/bin/env python3
"""
Render the CrowdPhysics Monitor-pipeline "Multi-Agent Decision Framework".

A zoom-in on the decision hub: every signal the monitor produces (anomaly
status, world-model futures, statistical trend, the RL intervention, and the
counterfactual proof) is fused, reasoned over by Claude, and turned into a
single operator-facing verdict + action.

Reproducible source for devpost/monitor_decision_framework.png. Re-run:
    python devpost/decision_framework.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parent / "monitor_decision_framework.png"

# ── palette (matches architecture.py) ─────────────────────────────────────────
INK = "#1c2533"
GREY = "#5b6b7f"
LINE = "#3a4759"
WM_FC = "#eceff3"
RL_FC = "#cfe2f7"
RL_EC = "#3f78c2"
SUG_FC = "#c9efce"
SUG_EC = "#4caf6a"
CL_FC = "#e7d8f3"
CL_EC = "#9a6cc4"
AM_FC = "#fdf0d2"
AM_EC = "#d8a341"

fig, ax = plt.subplots(figsize=(15, 8.2), dpi=150)
ax.set_xlim(0, 15)
ax.set_ylim(0, 8.2)
ax.axis("off")
ax.set_aspect("equal")


def box(x, y, w, h, *, fc="white", ec=LINE, lw=1.6, rounding=0.06, z=2, ls="-"):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0.02,rounding_size={rounding}",
        fc=fc, ec=ec, lw=lw, ls=ls, zorder=z))
    return (x + w / 2, y + h / 2)


def label(cx, cy, text, *, size=11, weight="normal", color=INK, z=4,
          ha="center", va="center", style="normal"):
    ax.text(cx, cy, text, ha=ha, va=va, fontsize=size, fontweight=weight,
            color=color, zorder=z, style=style, linespacing=1.3)


def arrow(p1, p2, *, color=LINE, lw=1.9, z=1, ls="-", rad=0.0, mut=14):
    ax.add_patch(FancyArrowPatch(
        p1, p2, arrowstyle="-|>", mutation_scale=mut, lw=lw, color=color,
        connectionstyle=f"arc3,rad={rad}", ls=ls, zorder=z, shrinkA=2, shrinkB=2))


# ── title ─────────────────────────────────────────────────────────────────────
label(5.25, 7.78, "CrowdPhysics", size=19, weight="bold", ha="right")
label(5.45, 7.78, "\u2014  Multi-Agent Decision Framework", size=19,
      weight="bold", color=GREY, ha="left")
label(7.5, 7.28, "Monitor pipeline \u00b7 every live signal is fused into one verdict + action",
      size=10.5, color=GREY, style="italic")

# ── input signals (left) ──────────────────────────────────────────────────────
ix, iw, ih = 0.35, 3.05, 0.92
inputs = [
    (6.05, "Anomaly detector", "surprise \u03c3  \u2192  SAFE / WARN / DANGER", "white", LINE),
    (4.85, "World-model futures", "imagined rollout  z(t+1 \u2026 t+H)", "white", LINE),
    (3.65, "Statistical trend", "risk projected minutes ahead", "white", LINE),
    (2.45, "RL agent \u00b7 Dyna + CQL", "recommended intervention", RL_FC, RL_EC),
    (1.25, "Counterfactual", "do-nothing vs act \u2014 proves the fix", "white", LINE),
]
for iy, title, sub, fc, ec in inputs:
    box(ix, iy, iw, ih, fc=fc, ec=ec, lw=2.0 if ec == RL_EC else 1.6)
    label(ix + iw / 2, iy + ih / 2 + 0.2, title, size=10.5, weight="bold")
    label(ix + iw / 2, iy + ih / 2 - 0.22, sub, size=8.2, color=GREY)

# ── decision hub (centre) ─────────────────────────────────────────────────────
hx, hy, hw, hh = 5.45, 3.35, 3.25, 2.35
box(hx, hy, hw, hh, fc=WM_FC, lw=2.0)
label(hx + hw / 2, hy + hh - 0.7, "DECISION", size=17, weight="bold")
label(hx + hw / 2, hy + hh - 1.25, "FRAMEWORK", size=17, weight="bold")
label(hx + hw / 2, hy + 0.55, "fuses every signal,\nresolves one status + risk",
      size=9.2, color=GREY, style="italic")

# input → hub
for iy, *_ in inputs:
    arrow((ix + iw, iy + ih / 2), (hx, hy + hh / 2),
          rad=0.12 if iy > 4 else -0.12, color=LINE, lw=1.7)

# ── Claude (below hub) ────────────────────────────────────────────────────────
cx, cy, cw, ch = 5.25, 0.55, 3.65, 1.95
box(cx, cy, cw, ch, fc=CL_FC, ec=CL_EC, lw=1.9)
label(cx + cw / 2, cy + ch - 0.42, "Claude (Sonnet)", size=12.5, weight="bold")
label(cx + cw / 2, cy + ch / 2 - 0.28,
      "reasons over physics + trend + forecast\n\u2192 agent-decided crush risk %,\nlead time \u00b7 reasoning \u00b7 recommendation",
      size=8.8, color="#5a3f78")
# hub <-> claude (two-way)
arrow((hx + hw * 0.38, hy), (cx + cw * 0.42, cy + ch), color=CL_EC, lw=1.8, rad=0.0)
arrow((cx + cw * 0.6, cy + ch), (hx + hw * 0.62, hy), color=CL_EC, lw=1.8, rad=0.0)

# ── outputs (right) ───────────────────────────────────────────────────────────
ox, ow, oh = 10.6, 3.35, 0.95
outputs = [
    (5.55, "Unified status + crush risk %", "one calibrated verdict", SUG_FC, SUG_EC),
    (4.15, "Operator briefing", "plain-language explanation", "white", LINE),
    (2.75, "Recommended action", "open gates \u00b7 slow ingress \u00b7 redirect", "white", LINE),
    (1.35, "Alerts", "Fetch.ai heartbeat \u00b7 radio \u00b7 SMS", AM_FC, AM_EC),
]
for oy, title, sub, fc, ec in outputs:
    box(ox, oy, ow, oh, fc=fc, ec=ec, lw=2.0 if ec in (SUG_EC, AM_EC) else 1.6)
    label(ox + ow / 2, oy + oh / 2 + 0.2, title, size=10.5, weight="bold")
    label(ox + ow / 2, oy + oh / 2 - 0.22, sub, size=8.5, color=GREY)
    arrow((hx + hw, hy + hh / 2), (ox, oy + oh / 2),
          rad=0.12 if oy > 3.6 else -0.12, color=LINE, lw=1.7)

# footer note
label(7.5, 0.18, "world model \u00b7 RL policy \u00b7 statistical trend \u00b7 Claude  \u2014  no single model decides alone",
      size=9.0, color=GREY, style="italic")

fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white", pad_inches=0.25)
print(f"wrote {OUT}")
