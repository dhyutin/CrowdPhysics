# train_v2.py
"""
Train World Model v2 (proper VAE) then auto-chain the Dyna-CQL RL policy.

ISOLATION: writes ONLY to models_v2/ and logs/. It never touches the
baseline's models/world_model.pt or models/rl_policy.pt, so this can run on a
separate box (or alongside the baseline) without risk.

Objective (Dreamer/RSSM-style):
    loss = recon_w  * MSE(recon, x)
         + kl_dyn_w * KL(posterior_t || prior_t)        (with free bits)
         + kl_reg_w * KL(posterior   || N(0,1))         (anchors latent scale)
    (+ optional trans_mse_w * MSE(mu_prior, z_target) as a stabilizer)

Transition backbone is selectable via TRANSITION_TYPE = lstm|gru|tcn|transformer.

Env knobs (with the sweep-selected defaults baked in):
    WM_EPOCHS=1500  WM_SEQ_LEN=40  WM_LR=5e-4
    WM_HIDDEN_DIM=512  WM_N_LAYERS=3  TRANSITION_TYPE=lstm
    RECON_W=0.1  KL_DYN_W=1.0  KL_REG_W=0.005  FREE_BITS=1.0  TRANS_MSE_W=0.0
    RL_EPISODES=15000  RL_STEPS_PER_EPISODE=10
"""

import os
import pickle
import numpy as np
import torch
import torch.nn as nn

from flow_extractor import process_video_to_features
from world_model_v2 import (
    CrowdWorldModelV2, kl_two_gaussians, kl_standard_normal)
from metrics_logger import MetricsLogger

MODEL_DIR = "models_v2"
CACHE_PATH = os.environ.get("FEATURE_CACHE", "data/features_cache_v2.pkl")


def get_device():
    if torch.cuda.is_available():
        d = torch.device("cuda")
    elif torch.backends.mps.is_available():
        d = torch.device("mps")
    else:
        d = torch.device("cpu")
    print(f"[train_v2] Device: {d}")
    return d


DEVICE = get_device()

HIDDEN_DIM = int(os.environ.get("WM_HIDDEN_DIM", "512"))
N_LAYERS = int(os.environ.get("WM_N_LAYERS", "3"))
TRANSITION_TYPE = os.environ.get("TRANSITION_TYPE", "lstm").strip().lower()

EPOCHS = int(os.environ.get("WM_EPOCHS", "1500"))
SEQ_LEN = int(os.environ.get("WM_SEQ_LEN", "40"))
LR = float(os.environ.get("WM_LR", "5e-4"))
BATCH_SIZE = int(os.environ.get("WM_BATCH_SIZE", "8"))

RECON_W = float(os.environ.get("RECON_W", "0.1"))
KL_DYN_W = float(os.environ.get("KL_DYN_W", "1.0"))
KL_REG_W = float(os.environ.get("KL_REG_W", "0.005"))
FREE_BITS = float(os.environ.get("FREE_BITS", "1.0"))   # nats per step, on KL_dyn
TRANS_MSE_W = float(os.environ.get("TRANS_MSE_W", "0.0"))

print(f"[train_v2] transition={TRANSITION_TYPE} hidden={HIDDEN_DIM} "
      f"layers={N_LAYERS} | epochs={EPOCHS} seq_len={SEQ_LEN} lr={LR} "
      f"batch={BATCH_SIZE}")
print(f"[train_v2] loss weights: recon={RECON_W} kl_dyn={KL_DYN_W} "
      f"kl_reg={KL_REG_W} free_bits={FREE_BITS} trans_mse={TRANS_MSE_W}")


def load_all_videos(video_dir="data/videos"):
    """Load feature sequences from videos; cache to disk; synthetic fallback."""
    if os.path.exists(CACHE_PATH) and not os.environ.get("REBUILD_CACHE"):
        with open(CACHE_PATH, "rb") as f:
            feats = pickle.load(f)
        print(f"[train_v2] Loaded {len(feats)} cached feature sequences "
              f"from {CACHE_PATH}")
        return feats

    all_features = []
    video_files = []
    for root, _dirs, files in os.walk(video_dir):
        for f in files:
            if f.endswith((".mp4", ".avi", ".mov")):
                video_files.append(os.path.join(root, f))

    if not video_files:
        print("[train_v2] No videos found. Using synthetic data.")
        for _ in range(30):
            T = np.random.randint(100, 300)
            seq = np.cumsum(
                np.random.randn(T, 256).astype(np.float32) * 0.1, axis=0)
            if np.random.random() > 0.7:
                ds = np.random.randint(T // 2, int(T * 0.8))
                seq[ds:, 1::4] -= 2.0   # backward-flow spike
                seq[ds:, 3::4] += 3.0   # turbulence spike
            all_features.append(seq)
        return all_features

    for path in sorted(video_files)[:50]:
        print(f"[train_v2] Processing {os.path.basename(path)}...")
        try:
            feats = process_video_to_features(path, max_frames=300)
            if len(feats) >= 32:
                all_features.append(np.asarray(feats, dtype=np.float32))
                print(f"  -> {len(feats)} frames")
        except Exception as e:  # noqa: BLE001
            print(f"  -> Error: {e}, skipping")

    print(f"[train_v2] Loaded {len(all_features)} videos")
    if all_features:
        os.makedirs(os.path.dirname(CACHE_PATH) or ".", exist_ok=True)
        with open(CACHE_PATH, "wb") as f:
            pickle.dump(all_features, f)
        print(f"[train_v2] Cached features -> {CACHE_PATH}")
    return all_features


def train_world_model(features_list):
    model = CrowdWorldModelV2(
        hidden_dim=HIDDEN_DIM, n_layers=N_LAYERS,
        transition_type=TRANSITION_TYPE).to(DEVICE)

    # Standardize raw flow features (carried as checkpoint buffers). Without
    # this, recon MSE on raw RAFT magnitudes dwarfs the KL/dynamics terms.
    all_frames = np.concatenate(
        [np.asarray(s, dtype=np.float32) for s in features_list], axis=0)
    model.set_feature_stats(all_frames.mean(0), all_frames.std(0))
    print(f"[train_v2] feature stats set on {all_frames.shape[0]} frames "
          f"(mean|{np.abs(all_frames.mean(0)).mean():.3f}| "
          f"std~{all_frames.std(0).mean():.3f})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-5)

    best_loss = float("inf")
    log = MetricsLogger("world_model_v2", config={
        "epochs": EPOCHS, "seq_len": SEQ_LEN, "batch_size": BATCH_SIZE,
        "hidden_dim": HIDDEN_DIM, "n_layers": N_LAYERS,
        "transition_type": TRANSITION_TYPE, "lr": LR,
        "recon_w": RECON_W, "kl_dyn_w": KL_DYN_W, "kl_reg_w": KL_REG_W,
        "free_bits": FREE_BITS, "trans_mse_w": TRANS_MSE_W,
        "device": DEVICE.type})

    print("\n" + "=" * 50)
    print("WORLD MODEL v2 TRAINING (proper VAE)")
    print("=" * 50)

    for epoch in range(EPOCHS):
        model.train()
        tot = tot_recon = tot_kld = tot_klr = 0.0
        n_batches = 0
        np.random.shuffle(features_list)

        for features in features_list:
            if len(features) < SEQ_LEN + 2:
                continue
            max_start = len(features) - SEQ_LEN - 1
            starts = np.random.choice(
                max_start, size=min(3, max_start), replace=False)

            for start in starts:
                seq = features[start:start + SEQ_LEN + 1]
                x = torch.FloatTensor(seq).unsqueeze(0).to(DEVICE)

                out = model.forward_train(x)

                # Reconstruction in STANDARDIZED space (decoder reconstructs
                # standardized features; target must be standardized too).
                recon_loss = nn.MSELoss()(out["recon"], model.standardize(x))

                # Dynamics: KL( posterior_t || prior_t ), t = 1..T.
                # prior predicts steps 1..T from z_<t.
                post_mu = out["mu_e"][:, 1:]
                post_lv = out["log_var_e"][:, 1:]
                kl_dyn_elem = kl_two_gaussians(
                    post_mu, post_lv, out["mu_p"], out["log_var_p"])
                # Free bits: don't penalise KL below FREE_BITS nats/step/dim-group.
                kl_dyn = kl_dyn_elem.sum(-1).clamp(min=FREE_BITS).mean()

                # Regularizer: anchor latent scale toward N(0,1) so
                # ||z|| stays order-1 (keeps danger_score meaningful).
                kl_reg = kl_standard_normal(
                    out["mu_e"], out["log_var_e"]).sum(-1).mean()

                loss = (RECON_W * recon_loss
                        + KL_DYN_W * kl_dyn
                        + KL_REG_W * kl_reg)

                if TRANS_MSE_W > 0:
                    z_target = out["z"][:, 1:].detach()
                    loss = loss + TRANS_MSE_W * nn.MSELoss()(out["mu_p"], z_target)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                tot += loss.item()
                tot_recon += recon_loss.item()
                tot_kld += float(kl_dyn)
                tot_klr += float(kl_reg)
                n_batches += 1

        scheduler.step()
        nb = max(1, n_batches)
        avg = tot / nb

        if avg < best_loss:
            best_loss = avg
            os.makedirs(MODEL_DIR, exist_ok=True)
            torch.save(model.state_dict(), f"{MODEL_DIR}/world_model.pt")

        log.log(epoch, loss=avg, best=best_loss,
                recon=tot_recon / nb, kl_dyn=tot_kld / nb, kl_reg=tot_klr / nb,
                lr=scheduler.get_last_lr()[0])

        if epoch % 5 == 0:
            print(f"  Epoch {epoch:4d}/{EPOCHS} | Loss {avg:.5f} | "
                  f"Best {best_loss:.5f} | recon {tot_recon/nb:.4f} | "
                  f"kl_dyn {tot_kld/nb:.4f} | kl_reg {tot_klr/nb:.4f}")

    log.close(plot_keys=["loss", "best", "recon", "kl_dyn", "kl_reg", "lr"])
    print(f"\n[train_v2] World model done. Best loss: {best_loss:.5f}")
    print(f"[train_v2] Saved: {MODEL_DIR}/world_model.pt")
    return model


if __name__ == "__main__":
    os.makedirs(MODEL_DIR, exist_ok=True)
    features = load_all_videos()
    model = train_world_model(features)

    print("\n" + "=" * 50)
    print("STARTING DYNA RL TRAINING (v2)")
    print("=" * 50)
    try:
        from dyna_trainer import DynaTrainer
        rl_episodes = int(os.environ.get("RL_EPISODES", "15000"))
        rl_steps = int(os.environ.get("RL_STEPS_PER_EPISODE", "10"))
        rl_log = MetricsLogger("rl_policy_v2", config={
            "episodes": rl_episodes, "steps_per_episode": rl_steps,
            "transition_type": TRANSITION_TYPE})
        # DynaTrainer runs on CPU (latent loop + small Q-net).
        trainer = DynaTrainer(model.to("cpu"))
        trainer.run_dyna_training(n_episodes=rl_episodes,
                                  steps_per_episode=rl_steps, logger=rl_log)
        torch.save(trainer.q_net.state_dict(), f"{MODEL_DIR}/rl_policy.pt")
        rl_log.close(plot_keys=["avg_reward_50", "reward", "loss", "epsilon"])
        print(f"[train_v2] RL policy saved: {MODEL_DIR}/rl_policy.pt")
    except Exception as e:  # noqa: BLE001
        print(f"[train_v2] RL stage failed ({e}). World model is saved; "
              f"run RL separately against {MODEL_DIR}/world_model.pt.")
