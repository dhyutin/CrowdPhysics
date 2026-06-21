# tune_world_model.py
"""
World-model hyperparameter search + final training.

Why this exists:
  train.py picks "best" by lowest *training* loss — that just rewards
  overfitting on our tiny (12-video) dataset. This script instead:

    1. Extracts RAFT flow features ONCE and caches them to disk
       (feature extraction is the slow part; trials then reuse the cache).
    2. Makes a temporal train/val split (first 80% of each video trains,
       last 20% validates) so we measure generalization.
    3. Sweeps candidate TRAINING hyperparameters (lr / KL / recon / seq_len)
       on a short epoch budget, scoring each by validation transition MSE.
    4. Retrains on ALL frames with the best config for the full epoch
       budget and saves models/world_model.pt.

Architecture is fixed at hidden_dim=512 / n_layers=3 because backend/main.py
and train_rl.py load the checkpoint with that shape. Do not tune it here.

Usage (Lambda GPU):
    # full sweep (8 trials, 100 epochs each) then 1500-epoch final train
    TUNE_EPOCHS=100 WM_EPOCHS=1500 python3 tune_world_model.py

    # force re-extraction of features (e.g. after RAFT was re-fine-tuned)
    REBUILD_CACHE=1 python3 tune_world_model.py
"""

import os
import pickle
import numpy as np
import torch
import torch.nn as nn

# --- run from any cwd + import root-level modules ---
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
os.chdir(_ROOT)

from flow_extractor import process_video_to_features
from world_model import CrowdWorldModel
from metrics_logger import MetricsLogger


# ─── CONFIG ───────────────────────────────────────────────────────────────────

HIDDEN_DIM = int(os.environ.get("WM_HIDDEN_DIM", "512"))
N_LAYERS = int(os.environ.get("WM_N_LAYERS", "3"))
TUNE_EPOCHS = int(os.environ.get("TUNE_EPOCHS", "100"))
WM_EPOCHS = int(os.environ.get("WM_EPOCHS", "1500"))
# Cap trials for a quick smoke check (0 = use all candidates).
TUNE_MAX_TRIALS = int(os.environ.get("TUNE_MAX_TRIALS", "0"))
CACHE_PATH = os.environ.get("FEATURE_CACHE", "data/features_cache.pkl")
VIDEO_DIR = os.environ.get("VIDEO_DIR", "data/videos")
WM_OUTPUT = os.environ.get("WM_OUTPUT", "models/world_model.pt")

# Candidate training configs. Architecture is intentionally absent.
# The first entry is the current baseline (train.py defaults).
CANDIDATES = [
    {"lr": 1e-3, "kl": 0.005, "recon": 0.10, "seq_len": 50, "wd": 1e-4},
    {"lr": 5e-4, "kl": 0.005, "recon": 0.10, "seq_len": 50, "wd": 1e-4},
    {"lr": 3e-4, "kl": 0.005, "recon": 0.10, "seq_len": 50, "wd": 1e-4},
    {"lr": 5e-4, "kl": 0.001, "recon": 0.10, "seq_len": 50, "wd": 1e-4},
    {"lr": 5e-4, "kl": 0.005, "recon": 0.05, "seq_len": 50, "wd": 1e-4},
    {"lr": 5e-4, "kl": 0.005, "recon": 0.10, "seq_len": 40, "wd": 1e-4},
    {"lr": 1e-3, "kl": 0.001, "recon": 0.05, "seq_len": 40, "wd": 1e-4},
    {"lr": 5e-4, "kl": 0.010, "recon": 0.10, "seq_len": 50, "wd": 1e-5},
]


def get_device():
    if torch.cuda.is_available():
        d = torch.device("cuda")
    elif torch.backends.mps.is_available():
        d = torch.device("mps")
    else:
        d = torch.device("cpu")
    print(f"[tune] Device: {d}")
    return d

DEVICE = get_device()


# ─── DATA ─────────────────────────────────────────────────────────────────────

def load_or_extract_features():
    """Extract per-video feature sequences once, cache to disk, reuse."""
    if os.path.exists(CACHE_PATH) and not os.environ.get("REBUILD_CACHE"):
        with open(CACHE_PATH, "rb") as f:
            feats = pickle.load(f)
        print(f"[tune] Loaded {len(feats)} cached feature sequences "
              f"from {CACHE_PATH}")
        return feats

    video_files = []
    for root, _, files in os.walk(VIDEO_DIR):
        for f in files:
            if f.endswith((".mp4", ".avi", ".mov")):
                video_files.append(os.path.join(root, f))

    feats = []
    for path in sorted(video_files)[:50]:
        print(f"[tune] Processing {os.path.basename(path)}...")
        try:
            seq = process_video_to_features(path, max_frames=300)
            if len(seq) >= 32:
                feats.append(np.asarray(seq, dtype=np.float32))
                print(f"        → {len(seq)} frames")
        except Exception as e:  # noqa: BLE001
            print(f"        → Error: {e}, skipping")

    if not feats:
        raise RuntimeError(f"No usable videos found in {VIDEO_DIR}")

    os.makedirs(os.path.dirname(CACHE_PATH) or ".", exist_ok=True)
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(feats, f)
    print(f"[tune] Cached {len(feats)} sequences → {CACHE_PATH}")
    return feats


def temporal_split(features_list, val_frac=0.2, min_len=34):
    """First (1-val_frac) of each video → train, last val_frac → val."""
    train, val = [], []
    for seq in features_list:
        n = len(seq)
        cut = int(n * (1 - val_frac))
        tr, va = seq[:cut], seq[cut:]
        if len(tr) >= min_len:
            train.append(tr)
        if len(va) >= min_len:
            val.append(va)
    print(f"[tune] Split: {len(train)} train / {len(val)} val sequences")
    return train, val


# ─── TRAIN / EVAL ───────────────────────────────────────────────────────────--

def build_model():
    return CrowdWorldModel(hidden_dim=HIDDEN_DIM, n_layers=N_LAYERS).to(DEVICE)


def train_one(model, features_list, cfg, epochs, scheduler_T=None, log=None):
    """Train in place for `epochs`. Returns final-epoch avg train loss."""
    # Fit standardization on this split's frames (stored as model buffers).
    all_frames = np.concatenate(
        [np.asarray(s, dtype=np.float32) for s in features_list], axis=0)
    model.set_feature_stats(all_frames.mean(0), all_frames.std(0))

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                                  weight_decay=cfg["wd"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=scheduler_T or epochs, eta_min=1e-5)
    seq_len = cfg["seq_len"]

    last_avg = float("inf")
    for epoch in range(epochs):
        model.train()
        total, n = 0.0, 0
        np.random.shuffle(features_list)
        for features in features_list:
            if len(features) < seq_len + 2:
                continue
            max_start = len(features) - seq_len - 1
            starts = np.random.choice(max_start, size=min(3, max_start),
                                      replace=False)
            for start in starts:
                seq = features[start:start + seq_len + 1]
                x = torch.FloatTensor(seq).unsqueeze(0).to(DEVICE)
                mu, log_var, z_target, z, recon = model(x)

                trans_loss = nn.MSELoss()(mu, z_target)
                lv = log_var.clamp(-6, 2)
                kl_loss = -0.5 * torch.mean(
                    1 + lv - mu.pow(2) - lv.exp())
                recon_loss = nn.MSELoss()(
                    recon[:, :-1], model.standardize(x)[:, 1:].detach())
                loss = (trans_loss + cfg["kl"] * kl_loss
                        + cfg["recon"] * recon_loss)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total += loss.item()
                n += 1
        scheduler.step()
        last_avg = total / max(1, n)
        if log is not None:
            log.log(epoch, train_loss=last_avg,
                    lr=scheduler.get_last_lr()[0])
        if epoch % 10 == 0:
            print(f"        epoch {epoch:4d}/{epochs} | "
                  f"train_loss={last_avg:.5f}")
    return last_avg


@torch.no_grad()
def eval_val(model, val_list, seq_len):
    """Deterministic validation transition MSE (mu vs next-state target)."""
    model.eval()
    total, n = 0.0, 0
    stride = max(1, seq_len // 2)
    for features in val_list:
        if len(features) < seq_len + 2:
            continue
        for start in range(0, len(features) - seq_len - 1, stride):
            seq = features[start:start + seq_len + 1]
            x = torch.FloatTensor(seq).unsqueeze(0).to(DEVICE)
            mu, _, z_target, _, _ = model(x)
            total += nn.MSELoss()(mu, z_target).item()
            n += 1
    return total / max(1, n)


# ─── MAIN ───────────────────────────────────────────────────────────────────--

def main():
    os.makedirs("models", exist_ok=True)
    all_feats = load_or_extract_features()
    train_feats, val_feats = temporal_split(all_feats)
    if not val_feats:
        raise RuntimeError("Validation split empty — videos too short.")

    sweep_log = MetricsLogger("wm_tune", config={
        "hidden_dim": HIDDEN_DIM, "n_layers": N_LAYERS,
        "tune_epochs": TUNE_EPOCHS, "n_candidates": len(CANDIDATES)})

    n_cand = len(CANDIDATES[:TUNE_MAX_TRIALS] if TUNE_MAX_TRIALS
                 else CANDIDATES)
    print("\n" + "=" * 60)
    print(f"HYPERPARAMETER SWEEP — {n_cand} configs × {TUNE_EPOCHS} epochs")
    print("=" * 60)

    candidates = CANDIDATES[:TUNE_MAX_TRIALS] if TUNE_MAX_TRIALS else CANDIDATES
    results = []
    for i, cfg in enumerate(candidates):
        print(f"\n[trial {i}] {cfg}")
        torch.manual_seed(0)          # same init → fair comparison
        np.random.seed(0)
        model = build_model()
        train_one(model, list(train_feats), cfg, epochs=TUNE_EPOCHS)
        val_mse = eval_val(model, val_feats, cfg["seq_len"])
        print(f"[trial {i}] val_transition_mse = {val_mse:.6f}")
        sweep_log.log(i, val_mse=val_mse, lr=cfg["lr"], kl=cfg["kl"],
                      recon=cfg["recon"], seq_len=cfg["seq_len"])
        results.append((val_mse, i, cfg))

    results.sort(key=lambda r: r[0])
    best_val, best_i, best_cfg = results[0]
    sweep_log.close(plot_keys=["val_mse"])

    print("\n" + "=" * 60)
    print("SWEEP RESULTS (best → worst)")
    print("=" * 60)
    for val_mse, i, cfg in results:
        mark = " ← BEST" if i == best_i else ""
        print(f"  trial {i}: val_mse={val_mse:.6f}  {cfg}{mark}")

    # ── FINAL TRAINING on ALL frames with the best config ─────────────────────
    print("\n" + "=" * 60)
    print(f"FINAL TRAINING — best config (trial {best_i}), "
          f"{WM_EPOCHS} epochs on all data")
    print(f"  {best_cfg}")
    print("=" * 60)

    final_log = MetricsLogger("world_model", config={
        **best_cfg, "hidden_dim": HIDDEN_DIM, "n_layers": N_LAYERS,
        "epochs": WM_EPOCHS, "device": DEVICE.type,
        "selected_val_mse": round(best_val, 6)})

    torch.manual_seed(0)
    np.random.seed(0)
    final_model = build_model()
    # Train on train+val combined (maximize the tiny dataset post-selection).
    train_one(final_model, list(all_feats), best_cfg,
              epochs=WM_EPOCHS, log=final_log)
    final_log.close(plot_keys=["train_loss", "lr"])

    torch.save(final_model.state_dict(), WM_OUTPUT)
    print(f"\n✓ Saved best world model → {WM_OUTPUT}")
    print(f"  config: {best_cfg}")
    print(f"  selection val_mse: {best_val:.6f}")
    print("\nNext: train the RL policy on this world model:")
    print("    RL_EPISODES=15000 python3 train_rl.py")


if __name__ == "__main__":
    main()
