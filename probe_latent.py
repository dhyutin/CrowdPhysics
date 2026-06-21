# probe_latent.py
"""
Linear-probe the trained world model's latent space — for real.

Encodes crowd-video features into the 64-dim latent, then fits a linear
probe from the latent to each *measured* physics quantity (velocity,
turbulence, backward pressure, boundary stress). The R² tells us how much
of each concept the model represents, and the standardized coefficients
tell us which latent dimensions carry it.

It also flags "unknown" dimensions: dims poorly explained by any known
concept that nonetheless separate high-surprise (pre-anomaly) frames from
calm frames. This replaces the previously hardcoded Discovery numbers.

Output: probe_results.json  (consumed by backend /api/discover)

Run (unsandboxed — needs torch):
    python probe_latent.py
"""

from __future__ import annotations

import json
import os
import numpy as np
import torch

from flow_extractor import process_video_to_features
from world_model import CrowdWorldModel

WM_HIDDEN = int(os.environ.get("WM_HIDDEN_DIM", "512"))
WM_LAYERS = int(os.environ.get("WM_N_LAYERS", "3"))
WINDOW = 30
MAX_FRAMES = int(os.environ.get("PROBE_MAX_FRAMES", "150"))


def _fit_linear_probe(Z: np.ndarray, y: np.ndarray) -> tuple[float, list[int]]:
    """Least-squares probe Z→y on standardized latents. Returns (R², top dims)."""
    mu, sd = Z.mean(0), Z.std(0) + 1e-8
    Zs = (Z - mu) / sd
    Zb = np.concatenate([Zs, np.ones((len(Zs), 1))], axis=1)
    w, *_ = np.linalg.lstsq(Zb, y, rcond=None)
    pred = Zb @ w
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2)) + 1e-12
    r2 = max(0.0, 1.0 - ss_res / ss_tot)
    top = list(np.argsort(np.abs(w[:-1]))[::-1][:5].astype(int))
    return r2, [int(d) for d in top]


def _physics_targets(feats: np.ndarray) -> dict[str, np.ndarray]:
    """Per-frame measured physics from 256-dim features (matches detector)."""
    fx = feats[:, 0::4]
    fy = feats[:, 1::4]
    mag = np.sqrt(fx ** 2 + fy ** 2)
    boundary = np.abs(np.concatenate([feats[:, :32], feats[:, -32:]], axis=1))
    return {
        "crowd_velocity": mag.mean(1),
        "turbulence": mag.var(1),
        "backward_pressure": -fy.mean(1),
        "boundary_stress": boundary.mean(1),
    }


def main(video_dir: str = "data/videos",
         out_path: str = "probe_results.json") -> dict:
    wm = CrowdWorldModel(hidden_dim=WM_HIDDEN, n_layers=WM_LAYERS)
    wm.load_state_dict(torch.load("models/world_model.pt", map_location="cpu"))
    wm.eval()
    print(f"[probe] world model loaded ({WM_HIDDEN}/{WM_LAYERS})")

    Z_all, err_at_frame, targets_all = [], [], {k: [] for k in
                                               ("crowd_velocity", "turbulence",
                                                "backward_pressure",
                                                "boundary_stress")}

    vids = [f for f in os.listdir(video_dir)
            if f.lower().endswith((".mp4", ".avi", ".mov"))]
    print(f"[probe] {len(vids)} videos")

    for v in vids:
        feats = process_video_to_features(os.path.join(video_dir, v),
                                          max_frames=MAX_FRAMES)
        if len(feats) < WINDOW + 2:
            continue
        with torch.no_grad():
            z = wm.encoder(torch.FloatTensor(feats)).numpy()          # (T,64)
        Z_all.append(z)
        for k, y in _physics_targets(feats).items():
            targets_all[k].append(y)

        # per-window surprise (prediction error), aligned to window end
        errs = np.full(len(feats), np.nan, dtype=np.float32)
        with torch.no_grad():
            for s in range(0, len(feats) - WINDOW - 1, 3):
                x = torch.FloatTensor(feats[s:s + WINDOW + 1]).unsqueeze(0)
                mu, _, zt, _, _ = wm(x)
                errs[s + WINDOW] = float(torch.mean((mu - zt) ** 2))
        err_at_frame.append(errs)
        print(f"  {v}: {len(feats)} frames")

    Z = np.concatenate(Z_all, 0)
    err = np.concatenate(err_at_frame, 0)
    targets = {k: np.concatenate(v, 0) for k, v in targets_all.items()}
    print(f"[probe] total frames: {len(Z)}")

    # ── known concepts ──────────────────────────────────────────────────────
    concepts, used_dims = {}, set()
    descriptions = {
        "crowd_velocity": "Mean crowd movement speed",
        "turbulence": "Chaotic motion intensity",
        "backward_pressure": "Crowd moving against primary flow",
        "boundary_stress": "Compression at walls and barriers",
    }
    for name, y in targets.items():
        r2, top = _fit_linear_probe(Z, y)
        concepts[name] = {"r2": round(r2, 3), "top_dimensions": top,
                          "description": descriptions[name]}
        used_dims.update(top)
        print(f"  {name:18s} R²={r2:.3f} dims={top}")

    # ── unknown dimensions ──────────────────────────────────────────────────
    # Dims not claimed by any known concept, ranked by how strongly they
    # separate high-surprise frames from calm frames.
    valid = ~np.isnan(err)
    Zv, ev = Z[valid], err[valid]
    hi = ev >= np.quantile(ev, 0.75)
    lo = ev <= np.quantile(ev, 0.50)
    candidate = [d for d in range(Z.shape[1]) if d not in used_dims]

    seps = []
    for d in candidate:
        mh, ml = Zv[hi, d].mean(), Zv[lo, d].mean()
        pooled = np.sqrt(0.5 * (Zv[hi, d].var() + Zv[lo, d].var())) + 1e-8
        seps.append((d, abs(mh - ml) / pooled, float(mh), float(ml)))
    seps.sort(key=lambda t: -t[1])
    top_unknown = seps[:5]
    mean_sep = float(np.mean([s[1] for s in top_unknown])) if top_unknown else 0.0

    unknown = {
        "dimensions": [int(s[0]) for s in top_unknown],
        "separation_z_score": round(mean_sep, 2),
        "high_surprise_activation": round(
            float(np.mean([s[2] for s in top_unknown])), 3) if top_unknown else 0.0,
        "calm_activation": round(
            float(np.mean([s[3] for s in top_unknown])), 3) if top_unknown else 0.0,
        "verdict": (f"{mean_sep:.2f}σ separation between pre-anomaly and calm "
                    f"frames on unexplained dimensions"),
    }
    print(f"  unknown dims={unknown['dimensions']} sep={mean_sep:.2f}σ")

    # ── markdown table ──────────────────────────────────────────────────────
    rows = ["| Concept | R² | Key Dimensions | Status |", "|---|---|---|---|"]
    for name, c in concepts.items():
        label = name.replace("_", " ").title()
        rows.append(f"| {label} | **{c['r2']:.2f}** | "
                    f"{c['top_dimensions'][:3]} | ✅ Discovered |")
    rows.append(f"| **UNKNOWN** | — | **{unknown['dimensions']}** | "
                f"⭐ **{mean_sep:.2f}σ surprise separation** |")
    table_md = "\n".join(rows)

    result = {
        "latent_dim": int(Z.shape[1]),
        "n_frames": int(len(Z)),
        "concepts": concepts,
        "unknown": unknown,
        "table_md": table_md,
        "computed": True,
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[probe] ✓ saved {out_path}")
    return result


if __name__ == "__main__":
    main()
