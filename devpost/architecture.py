#!/usr/bin/env python3
"""
Render the CrowdPhysics "Monitor Pipeline" architecture diagram.

Reproducible source for devpost/Architecture.png. Edit the layout here and
re-run:  python devpost/architecture.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "Architecture.png"

# ── palette ──────────────────────────────────────────────────────────────────
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
GREEN = "#3aa45b"

fig, ax = plt.subplots(figsize=(16, 10), dpi=150)
ax.set_xlim(0, 16)
ax.set_ylim(0, 10)
ax.axis("off")
ax.set_aspect("equal")


# ── helpers ──────────────────────────────────────────────────────────────────
def box(x, y, w, h, *, fc="white", ec=LINE, lw=1.6, rounding=0.06, z=2,
        ls="-"):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.02,rounding_size={rounding}",
        fc=fc, ec=ec, lw=lw, ls=ls, zorder=z,
    )
    ax.add_patch(p)
    return (x + w / 2, y + h / 2)


def label(cx, cy, text, *, size=11, weight="normal", color=INK, z=4,
          ha="center", va="center", style="normal"):
    ax.text(cx, cy, text, ha=ha, va=va, fontsize=size, fontweight=weight,
            color=color, zorder=z, style=style, linespacing=1.3)


def arrow(p1, p2, *, color=LINE, lw=1.8, z=1, ls="-", rad=0.0,
          mut=14):
    a = FancyArrowPatch(
        p1, p2, arrowstyle="-|>", mutation_scale=mut, lw=lw, color=color,
        connectionstyle=f"arc3,rad={rad}", ls=ls, zorder=z,
        shrinkA=2, shrinkB=2,
    )
    ax.add_patch(a)


def heat(x, y, w, h, *, shift=0.0, spread=1.55, z=3, border=True):
    """Draw an 8x8 crowd-density heatmap inside the given rectangle."""
    yy, xx = np.mgrid[0:8, 0:8]
    cy, cx = 3.5 + shift, 3.5
    g = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * spread ** 2)))
    g += 0.06 * np.random.default_rng(int(abs(shift * 100)) + 7).random((8, 8))
    ax.imshow(g, extent=[x, x + w, y, y + h], origin="upper", cmap="turbo",
              vmin=0, vmax=1.05, aspect="auto", zorder=z, interpolation="nearest")
    if border:
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="square,pad=0", fill=False,
            ec="#26303d", lw=1.0, zorder=z + 1))


def brace(x, y0, y1, *, depth=0.22, color=LINE, lw=1.8, z=2, facing="right"):
    """A vertical curly-ish brace spanning y0..y1 at column x."""
    ym = (y0 + y1) / 2
    s = depth if facing == "right" else -depth
    xs = [x, x + s, x + s, x + 2 * s, x + s, x + s, x]
    ys = [y0, y0 + (ym - y0) * 0.25, ym - 0.12, ym, ym + 0.12,
          y1 - (y1 - ym) * 0.25, y1]
    ax.plot(xs, ys, color=color, lw=lw, zorder=z, solid_capstyle="round")


# ════════════════════════════════════════════════════════════════════════════
# TITLE
# ════════════════════════════════════════════════════════════════════════════
label(7.55, 9.62, "CrowdPhysics", size=23, weight="bold", ha="right")
label(7.75, 9.62, "\u2014  Monitor Pipeline", size=23, weight="bold",
      color=GREY, ha="left")

# ════════════════════════════════════════════════════════════════════════════
# TOP ROW — capture → frames → optical flow → grid features → world model
# ════════════════════════════════════════════════════════════════════════════

# 1) Crowd video stream (with real captured frame if available)
vx, vy, vw, vh = 0.35, 6.55, 2.25, 2.05
box(vx, vy - 0.92, vw, vh + 0.92, fc="white")
thumb = ROOT / "tests" / "live_frame_test.png"
if thumb.exists():
    img = plt.imread(str(thumb))
    ax.imshow(img, extent=[vx + 0.12, vx + vw - 0.12, vy + 0.16, vy + vh - 0.12],
              aspect="auto", zorder=3)
    ax.add_patch(FancyBboxPatch((vx + 0.12, vy + 0.16), vw - 0.24, vh - 0.28,
                 boxstyle="square,pad=0", fill=False, ec="#26303d", lw=1.0,
                 zorder=4))
    # play glyph
    ax.scatter([vx + vw / 2], [vy + vh / 2 + 0.02], s=620, c="white",
               zorder=5, alpha=0.92)
    ax.text(vx + vw / 2, vy + vh / 2, "\u25B6", ha="center", va="center",
            fontsize=15, color=INK, zorder=6)
else:
    box(vx + 0.12, vy + 0.16, vw - 0.24, vh - 0.28, fc="#11161d")
label(vx + vw / 2, vy - 0.18, "Crowd video stream\n(live or recorded)",
      size=10.5, weight="bold")
box(vx + 0.14, vy - 0.86, vw - 0.28, 0.5, fc="#f3f5f8", lw=1.2)
label(vx + vw / 2, vy - 0.61, "Browserbase  (live capture)", size=8.2,
      color=GREY)

# 2) Periodic frames (stacked)
fx, fy = 3.15, 6.85
for i in range(4):
    box(fx + i * 0.12, fy + i * 0.12, 1.5, 1.35, fc="#f6f7f9", lw=1.3, z=2 + i)
label(fx + 0.9, fy - 0.28, "Periodic video frames", size=10, weight="bold")

# 3) RAFT optical flow
rx, ry, rw, rh = 5.25, 6.45, 2.35, 2.2
box(rx, ry, rw, rh, fc="white")
label(rx + rw / 2, ry + rh - 0.4, "RAFT-small (~1M)\n— Optical Flow",
      size=11, weight="bold")
# little net glyph
nx = np.array([rx + 0.5, rx + 1.18, rx + 1.85])
for j, layer in enumerate([3, 4, 3]):
    ys = np.linspace(ry + 0.45, ry + 1.25, layer)
    ax.scatter([nx[j]] * layer, ys, s=70,
               c=["#7fb1e3", "#7fcf9b", "#a7d18a"][j], zorder=4,
               edgecolors="#3a4759", linewidths=0.6)
label(rx + rw / 2, ry + 0.18, "fine-tuned on crowd video", size=8.5,
      color=GREY, style="italic")

# 4) Grid feature heatmaps (t-30 … t)
gx = 8.3
hw, hh = 1.15, 0.95
rows = [(8.0, "t-30", -0.7), (6.95, "t-29", -0.2)]
for gy, name, sh in rows:
    heat(gx, gy, hw, hh, shift=sh)
    label(gx - 0.18, gy + hh / 2, name, size=10, ha="right", weight="bold")
label(gx + hw / 2, 6.7, "\u22ee", size=16, weight="bold")
heat(gx, 5.45, hw, hh, shift=0.55, spread=1.25)
label(gx - 0.18, 5.45 + hh / 2, "t", size=10, ha="right", weight="bold")
label(gx + hw / 2 + 0.05, 8.95 + 0.18, "256-dim grid features (8x8)",
      size=10.5, weight="bold")
label(gx + hw / 2, 5.15, "calibrated baseline\non calm footage", size=8.5,
      color=GREY, style="italic")

# brace grouping the grid features → world model (tip feeds the WM arrow)
_bx = gx + hw + 0.12
brace(_bx, 5.45, 8.95, depth=0.16)

# 5) World model
wx, wy, ww, wh = 12.35, 5.55, 3.25, 3.35
box(wx, wy, ww, wh, fc=WM_FC, lw=1.8)
label(wx + ww / 2, wy + wh - 0.85, "WORLD\nMODEL", size=20, weight="bold")
label(wx + ww / 2, wy + 0.95, "encoder\n256 \u2192 64-dim\nlatent  +  LSTM",
      size=11, color=GREY)

# top-row arrows
arrow((vx + vw, 7.6), (fx, 7.5))
arrow((fx + 1.62, 7.5), (rx, 7.55))
arrow((rx + rw, 7.3), (gx - 0.08, 7.3))
arrow((_bx + 0.42, 7.2), (wx, 7.2))

# ════════════════════════════════════════════════════════════════════════════
# WORLD MODEL OUTPUTS — latent history, future rollout, statistical trend
# ════════════════════════════════════════════════════════════════════════════

# Latent history strip (past latents that also feed the RL agent)
lhx, lhy, lhw, lhh = 9.95, 4.55, 2.05, 1.0
box(lhx, lhy, lhw, lhh, fc="#eef3fb", ec=RL_EC, lw=1.5)
label(lhx + lhw / 2, lhy + lhh / 2 + 0.16, "Latent history",
      size=10.5, weight="bold")
label(lhx + lhw / 2, lhy + lhh / 2 - 0.22, "z(t\u2212k \u2026 t) \u00b7 64-dim",
      size=9.5, color=GREY)
arrow((wx, 6.0), (lhx + lhw, 5.2), rad=0.15)

# Future states rollout (green dashed box) — the FULL predicted horizon
frx, fry, frw, frh = 10.95, 1.35, 2.45, 3.05
ax.add_patch(FancyBboxPatch((frx, fry), frw, frh,
             boxstyle="round,pad=0.02,rounding_size=0.06",
             fc="#f3fbf5", ec=GREEN, lw=1.8, ls=(0, (5, 3)), zorder=2))
heat(frx + 0.62, fry + 1.85, 1.2, 1.0, shift=0.7, spread=1.15)
label(frx + 0.5, fry + 1.85 + 0.5, "t+1", size=10, ha="right", weight="bold")
heat(frx + 0.62, fry + 0.62, 1.2, 1.0, shift=1.05, spread=1.0)
label(frx + 0.5, fry + 0.62 + 0.5, "t+2", size=10, ha="right", weight="bold")
label(frx + frw - 0.55, fry + 1.2, "\u22ef  t+H", size=11, weight="bold",
      color=GREEN)
label(frx + frw / 2, fry - 0.28, "Future states (rollout)", size=10.5,
      weight="bold", color=GREEN)
arrow((wx + 0.4, wy), (frx + frw / 2, fry + frh), rad=0.0)

# Statistical trend
stx, sty, stw, sth = 13.85, 2.05, 1.7, 1.35
box(stx, sty, stw, sth, fc="white")
label(stx + stw / 2, sty + sth / 2 + 0.18, "Statistical\ntrend", size=10.5,
      weight="bold")
label(stx + stw / 2, sty + 0.22, "(minutes ahead)", size=8.5, color=GREY,
      style="italic")
arrow((frx + frw, fry + frh / 2 + 0.4), (stx, sty + sth / 2), ls=(0, (4, 3)),
      color=GREY)

# ════════════════════════════════════════════════════════════════════════════
# DECISION SIDE — anomaly detector, status, RL agent, multi-agent, claude
# ════════════════════════════════════════════════════════════════════════════

# Anomaly detector
adx, ady, adw, adh = 4.55, 4.55, 2.7, 1.25
box(adx, ady, adw, adh, fc="white")
label(adx + adw / 2, ady + adh / 2 + 0.22, "Anomaly Detector —", size=10.5,
      weight="bold")
label(adx + adw / 2, ady + adh / 2 - 0.18,
      "predicted vs actual\nlatent = surprise (sigma)", size=9, color=GREY)

# actual latent (t) feeds the detector (single clean arc)
arrow((gx, 5.45 + 0.5), (adx + adw, ady + adh / 2), rad=0.16)

# Status bar
sbx, sby, sbw, sbh = 4.55, 3.85, 2.7, 0.55
box(sbx, sby, sbw, sbh, fc="#f3f5f8", lw=1.3)
label(sbx + sbw / 2, sby + sbh / 2, "Status:  SAFE / WARNING / DANGER",
      size=8.8, weight="bold")
arrow((adx + adw / 2, ady), (sbx + sbw / 2, sby + sbh))

# RL agent
rlx, rly, rlw, rlh = 7.45, 2.15, 2.35, 1.7
box(rlx, rly, rlw, rlh, fc=RL_FC, ec=RL_EC, lw=2.0)
label(rlx + rlw / 2, rly + rlh / 2 + 0.2, "RL Agent", size=15, weight="bold")
label(rlx + rlw / 2, rly + rlh / 2 - 0.32, "(Dyna + CQL)", size=11,
      color="#2c5a96")

# --- RL state = past latents  ⊕  all future predicted frames ---
# (a) all future rollout frames → RL
arrow((frx, fry + frh / 2 + 0.25), (rlx + rlw, rly + rlh / 2 - 0.05),
      color=RL_EC, lw=2.0, rad=0.10)
label(8.6, 1.78, "all predicted futures  z(t+1 \u2026 t+H)", size=8.6,
      color="#2c5a96", weight="bold")
# (b) recent past latents → RL
arrow((lhx + 0.1, lhy), (rlx + rlw - 0.25, rly + rlh), color=RL_EC, lw=2.0,
      rad=0.22)
label(9.4, 4.32, "recent past latents", size=8.2,
      color="#2c5a96", weight="bold", ha="center")
# grouping caption
label(8.35, 1.12,
      "RL state  =  past latents  \u2295  full rollout",
      size=9.2, color="#2c5a96", style="italic")

# Multi-agent decision framework
max_, may, maw, mah = 2.55, 2.35, 1.95, 1.7
box(max_, may, maw, mah, fc="white", lw=1.8)
label(max_ + maw / 2, may + mah / 2, "Multi-agent\ndecision\nframework",
      size=10.5, weight="bold")

# Status → multi-agent ; RL → multi-agent (intervention) ; claude → multi-agent
arrow((sbx, sby + sbh / 2), (max_ + maw / 2, may + mah), rad=0.2)
arrow((rlx, rly + rlh / 2), (max_ + maw, may + mah / 2 - 0.1), rad=0.05)
label(6.0, 2.92, "Intervention\n(open gates, slow ingress,\nredirect\u2026)",
      size=8.6, color=GREY, style="italic")

# Suggestions
sgx, sgy, sgw, sgh = 0.35, 2.45, 1.95, 1.5
box(sgx, sgy, sgw, sgh, fc=SUG_FC, ec=SUG_EC, lw=1.8)
label(sgx + sgw / 2, sgy + sgh / 2, "Suggestions\nto best handle\nthe crowd",
      size=10.5, weight="bold")
arrow((max_, may + mah / 2), (sgx + sgw, sgy + sgh / 2))

# Claude
clx, cly, clw, clh = 2.55, 0.7, 2.15, 1.35
box(clx, cly, clw, clh, fc=CL_FC, ec=CL_EC, lw=1.7)
label(clx + clw / 2, cly + clh / 2 + 0.22, "Claude (Sonnet)", size=11,
      weight="bold")
label(clx + clw / 2, cly + clh / 2 - 0.22, "operator briefing +\nexplains RL choice",
      size=8.8, color=GREY)
arrow((clx + clw / 2, cly + clh), (max_ + maw / 2, may), rad=0.0)

# ════════════════════════════════════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════════════════════════════════════
box(5.6, 0.2, 4.8, 0.55, fc="white", lw=1.3)
ax.text(6.05, 0.475, "Arize", fontsize=10, fontweight="bold", color=INK,
        va="center")
ax.text(6.85, 0.475, "(tracing)   \u00b7   ", fontsize=10, color=GREY,
        va="center")
ax.text(8.4, 0.475, "Fetch.ai", fontsize=10, fontweight="bold", color=INK,
        va="center")
ax.text(9.25, 0.475, "(heartbeat agent)", fontsize=10, color=GREY,
        va="center")

fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white",
            pad_inches=0.25)
print(f"wrote {OUT}")
