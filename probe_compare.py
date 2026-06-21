# probe_compare.py
"""
Probe two world models and compare which better represents crowd physics.

Compares:
  - BASELINE : CrowdWorldModel (v1)   from models/world_model.pt
  - VLA      : CrowdWorldModelV2 (v2) from models_v2/world_model_vla.pt

Both are probed over the SAME feature sequences (reuses the v2 feature cache
if present, else extracts), so the comparison is apples-to-apples.

For each model we report, per physics concept (velocity / turbulence /
backward-pressure / boundary-stress):
    R²  — how much of that concept the 64-dim latent linearly encodes
and the unknown-dimension surprise separation:
    σ   — how strongly unexplained latent dims separate pre-anomaly from
          calm frames (the signal that powers anomaly detection)

Higher mean R² AND higher σ ⇒ that model represents physics better ⇒ use it.

Run on the box (unsandboxed, needs torch):
    python3 probe_compare.py
"""

from __future__ import annotations

import json
import os
import numpy as np
import torch

from flow_extractor import process_video_to_features
from world_model import CrowdWorldModel
from world_model_v2 import CrowdWorldModelV2
from probe_latent import _fit_linear_probe, _physics_targets

WINDOW = 30
WM_HIDDEN = int(os.environ.get("WM_HIDDEN_DIM", "512"))
WM_LAYERS = int(os.environ.get("WM_N_LAYERS", "3"))
MAX_FRAMES = int(os.environ.get("PROBE_MAX_FRAMES", "300"))
CACHE_PATH = os.environ.get("FEATURE_CACHE", "data/features_cache_v2.pkl")
VIDEO_DIR = os.environ.get("VIDEO_DIR", "data/videos")

CONCEPTS = ("crowd_velocity", "turbulence", "backward_pressure",
            "boundary_stress")


def load_features() -> list[np.ndarray]:
    """Reuse the cached v2 features if available, else extract from videos."""
    if os.path.exists(CACHE_PATH):
        import pickle
        with open(CACHE_PATH, "rb") as f:
            feats = pickle.load(f)
        print(f"[compare] loaded {len(feats)} cached sequences from {CACHE_PATH}")
        return [np.asarray(s, dtype=np.float32) for s in feats]

    vids = sorted(f for f in os.listdir(VIDEO_DIR)
                  if f.lower().endswith((".mp4", ".avi", ".mov")))
    feats = []
    for v in vids:
        seq = process_video_to_features(os.path.join(VIDEO_DIR, v),
                                        max_frames=MAX_FRAMES)
        if len(seq) >= WINDOW + 2:
            feats.append(np.asarray(seq, dtype=np.float32))
            print(f"  {v}: {len(seq)} frames")
    return feats


@torch.no_grad()
def collect(wm, encode_fn, feats_list):
    """Gather latents, per-frame surprise, and physics targets for one model."""
    Z_all, err_all = [], []
    targets_all = {k: [] for k in CONCEPTS}

    for feats in feats_list:
        if len(feats) < WINDOW + 2:
            continue
        Z_all.append(encode_fn(wm, feats))
        for k, y in _physics_targets(feats).items():
            targets_all[k].append(y)

        errs = np.full(len(feats), np.nan, dtype=np.float32)
        for s in range(0, len(feats) - WINDOW - 1, 3):
            x = torch.FloatTensor(feats[s:s + WINDOW + 1]).unsqueeze(0)
            mu, _, zt, _, _ = wm(x)
            errs[s + WINDOW] = float(torch.mean((mu - zt) ** 2))
        err_all.append(errs)

    Z = np.concatenate(Z_all, 0)
    err = np.concatenate(err_all, 0)
    targets = {k: np.concatenate(v, 0) for k, v in targets_all.items()}
    return Z, err, targets


def analyze(Z, err, targets):
    """Concept R² + unknown-dim surprise separation σ for one model."""
    concepts, used = {}, set()
    for name in CONCEPTS:
        r2, top = _fit_linear_probe(Z, targets[name])
        concepts[name] = {"r2": round(r2, 3), "top_dimensions": top}
        used.update(top)

    valid = ~np.isnan(err)
    Zv, ev = Z[valid], err[valid]
    hi = ev >= np.quantile(ev, 0.75)
    lo = ev <= np.quantile(ev, 0.50)
    seps = []
    for d in (d for d in range(Z.shape[1]) if d not in used):
        mh, ml = Zv[hi, d].mean(), Zv[lo, d].mean()
        pooled = np.sqrt(0.5 * (Zv[hi, d].var() + Zv[lo, d].var())) + 1e-8
        seps.append(abs(mh - ml) / pooled)
    seps.sort(reverse=True)
    mean_sep = float(np.mean(seps[:5])) if seps else 0.0

    mean_r2 = float(np.mean([c["r2"] for c in concepts.values()]))
    return {"concepts": concepts, "mean_r2": round(mean_r2, 3),
            "surprise_sep_sigma": round(mean_sep, 3)}


def main():
    feats_list = load_features()
    if not feats_list:
        raise RuntimeError("No features available to probe.")
    total_frames = sum(len(s) for s in feats_list)
    print(f"[compare] {len(feats_list)} sequences, {total_frames} frames\n")

    # ── BASELINE (v1) ──────────────────────────────────────────────────────────
    base = CrowdWorldModel(hidden_dim=WM_HIDDEN, n_layers=WM_LAYERS)
    # strict=False: older baselines predate the feat_mean/feat_std buffers.
    # Leaving them at identity (0/1) makes standardize() a no-op, which matches
    # how a pre-standardization model was trained (raw features).
    missing, _ = base.load_state_dict(
        torch.load("models/world_model.pt", map_location="cpu"), strict=False)
    if any("feat_" in m for m in missing):
        print("[compare] baseline has no standardization buffers "
              "(pre-standardization checkpoint) -> using raw features")
    base.eval()

    def enc_v1(wm, feats):
        return wm.encoder(wm.standardize(torch.FloatTensor(feats))).numpy()

    base_res = analyze(*collect(base, enc_v1, feats_list))
    print("BASELINE (models/world_model.pt):")
    print(f"  mean R² = {base_res['mean_r2']:.3f} | "
          f"surprise σ = {base_res['surprise_sep_sigma']:.3f}")

    # ── VLA (v2) ────────────────────────────────────────────────────────────────
    vla = CrowdWorldModelV2(hidden_dim=WM_HIDDEN, n_layers=WM_LAYERS,
                            transition_type="lstm")
    vla.load_state_dict(
        torch.load("models_v2/world_model_vla.pt", map_location="cpu"))
    vla.eval()

    def enc_v2(wm, feats):
        x = torch.FloatTensor(feats).unsqueeze(0)
        return wm.encode_sequence(x).squeeze(0).numpy()

    vla_res = analyze(*collect(vla, enc_v2, feats_list))
    print("VLA (models_v2/world_model_vla.pt):")
    print(f"  mean R² = {vla_res['mean_r2']:.3f} | "
          f"surprise σ = {vla_res['surprise_sep_sigma']:.3f}")

    # ── SIDE-BY-SIDE ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print(f"{'concept':20s} {'baseline R²':>14s} {'vla R²':>10s}  winner")
    print("-" * 64)
    for name in CONCEPTS:
        b = base_res["concepts"][name]["r2"]
        v = vla_res["concepts"][name]["r2"]
        win = "vla" if v > b else "baseline" if b > v else "tie"
        print(f"{name:20s} {b:>14.3f} {v:>10.3f}  {win}")
    print("-" * 64)
    print(f"{'MEAN R²':20s} {base_res['mean_r2']:>14.3f} "
          f"{vla_res['mean_r2']:>10.3f}")
    print(f"{'SURPRISE σ':20s} {base_res['surprise_sep_sigma']:>14.3f} "
          f"{vla_res['surprise_sep_sigma']:>10.3f}")
    print("=" * 64)

    # Verdict: weight surprise separation (the anomaly signal) alongside R².
    base_score = base_res["mean_r2"] + 0.5 * base_res["surprise_sep_sigma"]
    vla_score = vla_res["mean_r2"] + 0.5 * vla_res["surprise_sep_sigma"]
    winner = "VLA (v2)" if vla_score > base_score else "BASELINE (v1)"
    print(f"\nRECOMMENDATION: use {winner}")
    print(f"  combined score (meanR² + 0.5·σ): "
          f"baseline={base_score:.3f}  vla={vla_score:.3f}")

    out = {"baseline": base_res, "vla": vla_res,
           "recommendation": winner,
           "scores": {"baseline": round(base_score, 3),
                      "vla": round(vla_score, 3)}}
    with open("probe_compare_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\n[compare] saved probe_compare_results.json")


if __name__ == "__main__":
    main()
