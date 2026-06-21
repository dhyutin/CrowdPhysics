# probe_mlp.py
"""
Can a small non-linear MLP probe on the SHIPPED baseline latent recover the
representation quality the VLA gets — without shipping a second model?

Motivation
----------
The linear probe (probe_compare.py) showed the VLA edges the baseline on mean
R² (0.869 vs 0.848), winning 3 of 4 physics concepts. One hypothesis: that gap
is non-linear structure the *linear* probe can't read off the baseline latent,
not information the baseline failed to encode. If so, a small MLP probe on the
baseline latent should close the gap — letting us ship ONE model.

Method (fair, leak-free)
------------------------
The numbers in probe_compare.py are IN-SAMPLE least squares. An MLP fit
in-sample would trivially overfit and look better for free. So here every probe
is scored with **held-out, sequence-grouped cross-validation** (GroupKFold over
whole video sequences -> no temporal leakage between train and test):

  for each concept:
    baseline + linear (Ridge)   held-out R²
    baseline + MLP              held-out R²
    VLA      + linear (Ridge)   held-out R²   <- reference target
    VLA      + MLP              held-out R²

Verdict: if  baseline+MLP  >=  VLA+linear  on mean held-out R², the MLP probe
closes the gap and you ship the baseline alone with a richer probe head.

Run (unsandboxed — needs torch):
    python3 probe_mlp.py
"""

from __future__ import annotations

import json
import os
import pickle

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from world_model import CrowdWorldModel
from world_model_v2 import CrowdWorldModelV2
from probe_latent import _physics_targets

WM_HIDDEN = int(os.environ.get("WM_HIDDEN_DIM", "512"))
WM_LAYERS = int(os.environ.get("WM_N_LAYERS", "3"))
WINDOW = 30
CACHE_PATH = os.environ.get("FEATURE_CACHE", "data/features_cache_v2.pkl")
BASE_PATH = os.environ.get("BASE_PATH", "models/world_model.pt")
VLA_PATH = os.environ.get("VLA_PATH", "models_v2/world_model.pt")
N_SPLITS = int(os.environ.get("PROBE_CV_SPLITS", "3"))
SEED = 0

CONCEPTS = ("crowd_velocity", "turbulence", "backward_pressure",
            "boundary_stress")


def load_features() -> list[np.ndarray]:
    with open(CACHE_PATH, "rb") as f:
        feats = pickle.load(f)
    print(f"[mlp] loaded {len(feats)} cached sequences from {CACHE_PATH}")
    return [np.asarray(s, dtype=np.float32) for s in feats]


@torch.no_grad()
def encode_baseline(wm, feats):
    return wm.encoder(wm.standardize(torch.FloatTensor(feats))).numpy()


@torch.no_grad()
def encode_vla(wm, feats):
    x = torch.FloatTensor(feats).unsqueeze(0)
    return wm.encode_sequence(x).squeeze(0).numpy()


def build_matrix(encode_fn, wm, feats_list):
    """Return Z (N,64), targets dict, and per-frame group ids (sequence index)."""
    Z_all, groups = [], []
    targets_all = {k: [] for k in CONCEPTS}
    for gi, feats in enumerate(feats_list):
        if len(feats) < WINDOW + 2:
            continue
        z = encode_fn(wm, feats)
        Z_all.append(z)
        groups.append(np.full(len(z), gi, dtype=int))
        for k, y in _physics_targets(feats).items():
            targets_all[k].append(y)
    Z = np.concatenate(Z_all, 0)
    groups = np.concatenate(groups, 0)
    targets = {k: np.concatenate(v, 0) for k, v in targets_all.items()}
    return Z, targets, groups


def _linear():
    return make_pipeline(StandardScaler(), Ridge(alpha=1.0))


def _mlp():
    # Small head: 64 -> 128 -> 64 -> 1, L2-regularized to resist overfit.
    return make_pipeline(
        StandardScaler(),
        MLPRegressor(hidden_layer_sizes=(128, 64), activation="relu",
                     alpha=1e-2, max_iter=1500, random_state=SEED),
    )


def heldout_r2(make_model, Z, y, groups):
    """Out-of-fold (sequence-grouped) R² — no train/test leakage."""
    cv = GroupKFold(n_splits=N_SPLITS)
    pred = cross_val_predict(make_model(), Z, y, groups=groups, cv=cv)
    return max(0.0, float(r2_score(y, pred)))


def evaluate(tag, Z, targets, groups):
    print(f"\n[{tag}] held-out (GroupKFold={N_SPLITS}) R² per concept:")
    out = {}
    for name in CONCEPTS:
        y = targets[name]
        lin = heldout_r2(_linear, Z, y, groups)
        mlp = heldout_r2(_mlp, Z, y, groups)
        out[name] = {"linear": round(lin, 3), "mlp": round(mlp, 3)}
        print(f"  {name:18s} linear={lin:.3f}  mlp={mlp:.3f}  "
              f"(+{mlp - lin:+.3f})")
    out["mean_linear"] = round(float(np.mean([out[c]["linear"]
                                              for c in CONCEPTS])), 3)
    out["mean_mlp"] = round(float(np.mean([out[c]["mlp"]
                                           for c in CONCEPTS])), 3)
    print(f"  {'MEAN':18s} linear={out['mean_linear']:.3f}  "
          f"mlp={out['mean_mlp']:.3f}")
    return out


def main():
    feats_list = load_features()

    base = CrowdWorldModel(hidden_dim=WM_HIDDEN, n_layers=WM_LAYERS)
    base.load_state_dict(torch.load(BASE_PATH, map_location="cpu"),
                         strict=False)
    base.eval()
    print(f"[mlp] baseline = {BASE_PATH}")

    vla = CrowdWorldModelV2(hidden_dim=WM_HIDDEN, n_layers=WM_LAYERS,
                            transition_type="lstm")
    vla.load_state_dict(torch.load(VLA_PATH, map_location="cpu"))
    vla.eval()
    print(f"[mlp] vla      = {VLA_PATH}")

    Zb, tb, gb = build_matrix(encode_baseline, base, feats_list)
    Zv, tv, gv = build_matrix(encode_vla, vla, feats_list)
    print(f"[mlp] baseline Z={Zb.shape}  vla Z={Zv.shape}")

    base_res = evaluate("BASELINE", Zb, tb, gb)
    vla_res = evaluate("VLA", Zv, tv, gv)

    # ── verdict ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print(f"{'concept':18s} {'base-lin':>9s} {'base-MLP':>9s} "
          f"{'vla-lin':>9s} {'vla-MLP':>9s}  {'baseMLP≥vlaLin?':>15s}")
    print("-" * 72)
    for name in CONCEPTS:
        bl, bm = base_res[name]["linear"], base_res[name]["mlp"]
        vl, vm = vla_res[name]["linear"], vla_res[name]["mlp"]
        ok = "YES" if bm >= vl else "no"
        print(f"{name:18s} {bl:>9.3f} {bm:>9.3f} {vl:>9.3f} {vm:>9.3f}  "
              f"{ok:>15s}")
    print("-" * 72)
    print(f"{'MEAN':18s} {base_res['mean_linear']:>9.3f} "
          f"{base_res['mean_mlp']:>9.3f} {vla_res['mean_linear']:>9.3f} "
          f"{vla_res['mean_mlp']:>9.3f}")
    print("=" * 72)

    closed = base_res["mean_mlp"] >= vla_res["mean_linear"]
    print(f"\nVERDICT: baseline+MLP {'CLOSES' if closed else 'does NOT close'} "
          f"the gap to VLA+linear "
          f"({base_res['mean_mlp']:.3f} vs {vla_res['mean_linear']:.3f}).")
    if closed:
        print("  -> Ship the baseline alone with an MLP probe head; no 2nd model.")
    else:
        print("  -> MLP probe alone doesn't fully recover it; baseline linear "
              "still ships fine (σ wins), VLA stays the representation story.")

    out = {"baseline": base_res, "vla": vla_res,
           "cv_splits": N_SPLITS, "gap_closed": bool(closed)}
    with open("probe_mlp_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\n[mlp] saved probe_mlp_results.json")


if __name__ == "__main__":
    main()
