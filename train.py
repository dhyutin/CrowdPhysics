# train.py
"""
Training pipeline.
Run immediately: python train.py &
Let it run while you build everything else.
Check progress every hour.

Device auto-detection: CUDA → MPS → CPU
Hyperparams scale automatically with device capability.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from flow_extractor import process_video_to_features
from world_model import CrowdWorldModel
from metrics_logger import MetricsLogger


# ─── DEVICE ───────────────────────────────────────────────────────────────────

def get_device():
    if torch.cuda.is_available():
        d = torch.device("cuda")
    elif torch.backends.mps.is_available():
        d = torch.device("mps")
    else:
        d = torch.device("cpu")
    print(f"[train] Device: {d}")
    return d

DEVICE = get_device()

# Architecture MUST match what loads the checkpoint downstream:
# backend/main.py and train_rl.py both build CrowdWorldModel(512, 3).
# Saving a different size here produces a state_dict that fails to load.
# Override via WM_HIDDEN_DIM / WM_N_LAYERS only if you change BOTH places too.
HIDDEN_DIM = int(os.environ.get("WM_HIDDEN_DIM", "512"))
N_LAYERS = int(os.environ.get("WM_N_LAYERS", "3"))

# Scale epochs/batch to available hardware. EPOCHS is env-overridable
# (WM_EPOCHS) so you can crank a longer run on the GPU box.
if DEVICE.type == "cuda":
    _default_epochs, SEQ_LEN, BATCH_SIZE = 500, 50, 8
elif DEVICE.type == "mps":
    _default_epochs, SEQ_LEN, BATCH_SIZE = 300, 50, 4
else:
    _default_epochs, SEQ_LEN, BATCH_SIZE = 80, 30, 1
EPOCHS = int(os.environ.get("WM_EPOCHS", str(_default_epochs)))

print(f"[train] Hyperparams: epochs={EPOCHS}, seq_len={SEQ_LEN}, "
      f"batch={BATCH_SIZE}, hidden={HIDDEN_DIM}, layers={N_LAYERS}")


def load_all_videos(video_dir="data/videos"):
    """Load and extract features from all videos"""
    all_features = []
    video_files = []

    for root, dirs, files in os.walk(video_dir):
        for f in files:
            if f.endswith(('.mp4', '.avi', '.mov')):
                video_files.append(os.path.join(root, f))

    if not video_files:
        print("No videos found. Using synthetic data.")
        for _ in range(30):
            T = np.random.randint(100, 300)
            seq = np.cumsum(
                np.random.randn(T, 256).astype(np.float32) * 0.1,
                axis=0
            )
            # Add "dangerous" pattern to some videos
            if np.random.random() > 0.7:
                danger_start = np.random.randint(T//2, int(T*0.8))
                seq[danger_start:, 1::4] -= 2.0  # backward flow spike
                seq[danger_start:, 3::4] += 3.0  # turbulence spike
            all_features.append(seq)
        return all_features

    for path in video_files[:50]:  # max 50 videos
        print(f"Processing {os.path.basename(path)}...")
        try:
            feats = process_video_to_features(path, max_frames=300)
            if len(feats) >= 32:
                all_features.append(feats)
                print(f"  → {len(feats)} frames")
        except Exception as e:
            print(f"  → Error: {e}, skipping")

    print(f"\nLoaded {len(all_features)} videos")
    return all_features


def train_world_model(features_list, epochs=None, seq_len=None,
                      batch_size=None, lr=1e-3):
    epochs     = epochs     or EPOCHS
    seq_len    = seq_len    or SEQ_LEN
    batch_size = batch_size or BATCH_SIZE

    model = CrowdWorldModel(hidden_dim=HIDDEN_DIM, n_layers=N_LAYERS).to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5)

    best_loss = float('inf')
    log = MetricsLogger("world_model", config={
        "epochs": epochs, "seq_len": seq_len, "batch_size": batch_size,
        "hidden_dim": HIDDEN_DIM, "n_layers": N_LAYERS, "device": DEVICE.type})
    print("\n" + "="*50)
    print("WORLD MODEL TRAINING")
    print("="*50)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        n_batches = 0

        np.random.shuffle(features_list)

        for features in features_list:
            if len(features) < seq_len + 2:
                continue

            max_start = len(features) - seq_len - 1
            starts = np.random.choice(
                max_start,
                size=min(3, max_start),
                replace=False
            )

            for start in starts:
                seq = features[start:start + seq_len + 1]
                x = torch.FloatTensor(seq).unsqueeze(0).to(DEVICE)

                mu, log_var, z_target, z, recon = model(x)

                # 1. Transition prediction loss (MAIN)
                trans_loss = nn.MSELoss()(mu, z_target)

                # 2. KL divergence (latent regularization)
                log_var_c = log_var.clamp(-6, 2)
                kl_loss = -0.5 * torch.mean(
                    1 + log_var_c - mu.pow(2) - log_var_c.exp()
                )

                # 3. Reconstruction loss (auxiliary)
                recon_loss = nn.MSELoss()(
                    recon[:, :-1],
                    x[:, 1:].detach()
                )

                # Total: transition is most important
                loss = trans_loss + 0.005 * kl_loss + 0.1 * recon_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

        scheduler.step()
        avg = total_loss / max(1, n_batches)

        if avg < best_loss:
            best_loss = avg
            torch.save(model.state_dict(), "models/world_model.pt")

        log.log(epoch, loss=avg, best=best_loss,
                lr=scheduler.get_last_lr()[0])

        if epoch % 5 == 0:
            print(f"  Epoch {epoch:3d}/{epochs} | "
                  f"Loss: {avg:.5f} | Best: {best_loss:.5f}")

    log.close(plot_keys=["loss", "best", "lr"])
    print(f"\n✓ World model done. Best loss: {best_loss:.5f}")
    print("  Saved: models/world_model.pt")
    return model


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    features = load_all_videos()
    model = train_world_model(features, epochs=80)

    print("\n" + "="*50)
    print("STARTING DYNA RL TRAINING")
    print("="*50)
    try:
        from dyna_trainer import DynaTrainer
        rl_log = MetricsLogger("rl_policy", config={"episodes": 300})
        trainer = DynaTrainer(model)
        trainer.run_dyna_training(n_episodes=300, logger=rl_log)
        torch.save(trainer.q_net.state_dict(), "models/rl_policy.pt")
        rl_log.close(plot_keys=["avg_reward_50", "reward", "loss", "epsilon"])
        print("✓ RL policy saved: models/rl_policy.pt")
    except (ImportError, AttributeError) as e:
        print(f"  Dyna RL not ready yet ({e}) — skipping. Run after Phase 3.")
