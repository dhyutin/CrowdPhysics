#!/usr/bin/env python3
"""
Render the CrowdPhysics "Simulation -> RAFT bridge" architecture diagram.

Reproducible source for devpost/sim_to_raft_bridge.png. Edit and re-run:
    python devpost/bridge_architecture.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parent / "sim_to_raft_bridge.png"

# ── palette (matches architecture.py) ─────────────────────────────────────────
INK = "#1c2533"
GREY = "#5b6b7f"
LINE = "#3a4759"
WM_FC = "#eceff3"
RL_FC = "#cfe2f7"
RL_EC = "#3f78c2"
SUG_FC = "#c9efce"
SUG_EC = "#4caf6a"
GREEN = "#3aa45b"

fig, ax = plt.subplots(figsize=(15, 4.6), dpi=150)
ax.set_xlim(0, 15)
ax.set_ylim(0, 4.6)
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


def arrow(p1, p2, *, color=LINE, lw=1.9, z=1, ls="-", rad=0.0, mut=15):
    ax.add_patch(FancyArrowPatch(
        p1, p2, arrowstyle="-|>", mutation_scale=mut, lw=lw, color=color,
        connectionstyle=f"arc3,rad={rad}", ls=ls, zorder=z, shrinkA=2, shrinkB=2))


# ── title ─────────────────────────────────────────────────────────────────────
label(7.05, 4.28, "CrowdPhysics", size=18, weight="bold", ha="right")
label(7.25, 4.28, "\u2014  Simulation \u2192 RAFT bridge", size=18,
      weight="bold", color=GREY, ha="left")

cy = 2.15
bw, bh = 2.35, 1.7

# 1) Crowd simulator — velocity field
box(0.3, cy - bh / 2, bw, bh, fc=WM_FC)
label(0.3 + bw / 2, cy + 0.34, "Crowd simulator", size=11, weight="bold")
label(0.3 + bw / 2, cy - 0.16, "pressure +\nvelocity field", size=9.5, color=GREY)

# 2) Seed particles at entries
box(3.25, cy - bh / 2, bw, bh, fc="white")
label(3.25 + bw / 2, cy + 0.34, "Seed particles", size=11, weight="bold")
label(3.25 + bw / 2, cy - 0.16, "at the entry\nports", size=9.5, color=GREY)
rng = np.random.default_rng(3)
px = 3.45 + rng.random(26) * (bw - 0.4)
py = (cy - bh / 2 + 0.18) + rng.random(26) * 0.5
ax.scatter(px, py, s=16, c="#e0746a", zorder=5, edgecolors="none")

# 3) Advect -> synthetic crowd video
box(6.2, cy - bh / 2, bw, bh, fc="white")
label(6.2 + bw / 2, cy + 0.4, "Advect through field", size=10.5, weight="bold")
label(6.2 + bw / 2, cy - 0.05, "\u2192 synthetic\ncrowd video", size=9.5,
      color=GREY)
fx = 6.45 + rng.random(40) * (bw - 0.45)
fy = (cy - bh / 2 + 0.16) + rng.random(40) * 0.46
ax.scatter(fx, fy, s=14, c="#4f7fd6", zorder=5, edgecolors="none")

# 4) RAFT optical flow (same as live)
box(9.15, cy - bh / 2, bw, bh, fc=RL_FC, ec=RL_EC, lw=2.0)
label(9.15 + bw / 2, cy + 0.4, "RAFT optical flow", size=11, weight="bold")
nx = np.array([9.6, 10.2, 10.8])
for j, layer in enumerate([3, 4, 3]):
    ys = np.linspace(cy - 0.5, cy + 0.05, layer)
    ax.scatter([nx[j]] * layer, ys, s=46,
               c=["#7fb1e3", "#7fcf9b", "#a7d18a"][j], zorder=5,
               edgecolors="#3a4759", linewidths=0.5)
label(9.15 + bw / 2, cy - 0.66, "same extractor as Monitor", size=8.0,
      color="#2c5a96", style="italic")

# 5) Expected flow per door
box(12.1, cy - bh / 2, bw, bh, fc=SUG_FC, ec=SUG_EC, lw=2.0)
label(12.1 + bw / 2, cy + 0.42, "Expected flow", size=11, weight="bold")
label(12.1 + bw / 2, cy + 0.06, "per door", size=11, weight="bold")
label(12.1 + bw / 2, cy - 0.42, "inflow / outflow\nintensity \u00b7 share",
      size=9.0, color="#2f6b43")

# arrows
for x0 in (0.3 + bw, 3.25 + bw, 6.2 + bw, 9.15 + bw):
    arrow((x0, cy), (x0 + 0.6, cy))

fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white", pad_inches=0.2)
print(f"wrote {OUT}")
