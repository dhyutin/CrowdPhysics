# finetune_raft.py
"""
Fine-tune RAFT optical flow on crowd videos using self-supervised
photometric consistency loss — no ground-truth flow labels needed.

Run on AWS g4dn.xlarge (CUDA) or locally (MPS):
    python finetune_raft.py

Saves fine-tuned weights to: models/raft_crowd.pt
Then train.py will automatically use them via flow_extractor.extract_flow().

Why self-supervised?
  We don't have ground-truth optical flow annotations for crowd footage.
  Photometric loss warps frame2 using predicted flow and measures pixel
  difference against frame1 — if the flow is correct, the warp is perfect.
  Smoothness regularization prevents degenerate solutions.

Training time:
  ~20 min on g4dn.xlarge (T4)
  ~45 min on MPS (M-series Mac)
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from metrics_logger import MetricsLogger


# ─── DEVICE ───────────────────────────────────────────────────────────────────

def get_device():
    if torch.cuda.is_available():
        d = torch.device("cuda")
    elif torch.backends.mps.is_available():
        d = torch.device("mps")
    else:
        d = torch.device("cpu")
    print(f"[device] Using: {d}")
    return d

DEVICE = get_device()


# ─── DATASET ──────────────────────────────────────────────────────────────────

class CrowdFramePairDataset(Dataset):
    """
    Extracts consecutive frame pairs from all crowd videos.
    Each item: (frame1_tensor, frame2_tensor) in [0,1] RGB float32.
    """

    def __init__(self, video_dir="data/videos", max_pairs_per_video=200,
                 resize=(256, 256)):
        self.pairs = []
        self.resize = resize

        for root, _, files in os.walk(video_dir):
            for f in sorted(files):
                if not f.endswith(('.mp4', '.avi', '.mov')):
                    continue
                path = os.path.join(root, f)
                pairs = self._extract_pairs(path, max_pairs_per_video)
                self.pairs.extend(pairs)
                print(f"  {f}: {len(pairs)} pairs")

        print(f"Total frame pairs: {len(self.pairs)}")

    def _extract_pairs(self, video_path, max_pairs):
        cap = cv2.VideoCapture(video_path)
        pairs = []
        ret, prev = cap.read()
        count = 0
        while count < max_pairs:
            ret, curr = cap.read()
            if not ret:
                break
            # Skip every other frame for variety
            if count % 2 == 0:
                prev_t = self._to_tensor(prev)
                curr_t = self._to_tensor(curr)
                pairs.append((prev_t, curr_t))
            prev = curr
            count += 1
        cap.release()
        return pairs

    def _to_tensor(self, frame):
        H, W = self.resize
        frame = cv2.resize(frame, (W, H))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        return t

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


# ─── LOSSES ───────────────────────────────────────────────────────────────────

def warp_with_flow(frame, flow):
    """
    Warp frame2 backwards using predicted flow to reconstruct frame1.
    frame: (B, C, H, W)
    flow:  (B, 2, H, W)  — (dx, dy) in pixels
    """
    B, C, H, W = frame.shape

    # Build sampling grid
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=frame.device),
        torch.arange(W, dtype=torch.float32, device=frame.device),
        indexing='ij'
    )
    grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)  # (1,2,H,W)
    grid = grid.expand(B, -1, -1, -1)

    # Apply flow offset, normalize to [-1, 1]
    new_grid = grid + flow
    new_grid[:, 0] = 2.0 * new_grid[:, 0] / (W - 1) - 1.0
    new_grid[:, 1] = 2.0 * new_grid[:, 1] / (H - 1) - 1.0
    new_grid = new_grid.permute(0, 2, 3, 1)  # (B, H, W, 2)

    warped = F.grid_sample(frame, new_grid, align_corners=True,
                           padding_mode='border')
    return warped


def photometric_loss(frame1, frame2, flow):
    """L1 + SSIM photometric consistency."""
    warped = warp_with_flow(frame2, flow)
    l1 = F.l1_loss(warped, frame1)
    # Simple SSIM approximation via local mean
    mu1 = F.avg_pool2d(frame1, 3, 1, 1)
    mu2 = F.avg_pool2d(warped, 3, 1, 1)
    ssim_loss = F.l1_loss(mu1, mu2)
    return 0.85 * ssim_loss + 0.15 * l1


def smoothness_loss(flow):
    """Edge-aware flow smoothness — penalizes large flow gradients."""
    dy = flow[:, :, 1:, :] - flow[:, :, :-1, :]
    dx = flow[:, :, :, 1:] - flow[:, :, :, :-1]
    return (dx.abs().mean() + dy.abs().mean())


# ─── TRAINING ─────────────────────────────────────────────────────────────────

def finetune_raft(video_dir="data/videos",
                  output_path="models/raft_crowd.pt",
                  epochs=80,
                  batch_size=12,
                  lr=3e-4,
                  smooth_weight=0.1,
                  weight_decay=1e-5,
                  pct_start=0.1,
                  max_pairs_per_video=400):

    os.makedirs("models", exist_ok=True)

    # Load pretrained RAFT-Small
    from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
    model = raft_small(weights=Raft_Small_Weights.DEFAULT).to(DEVICE)
    model.train()

    # Freeze the feature extractor, only fine-tune the update block
    for name, param in model.named_parameters():
        if "feature_encoder" in name or "context_encoder" in name:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=weight_decay
    )

    # Dataset
    dataset = CrowdFramePairDataset(video_dir=video_dir,
                                    max_pairs_per_video=max_pairs_per_video)
    if len(dataset) == 0:
        print("No video pairs found. Check data/videos/")
        return

    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=True, num_workers=0, drop_last=True)

    steps_per_epoch = max(1, len(loader))
    total_steps = steps_per_epoch * epochs
    # Proper one-cycle schedule: warm up then cosine-anneal across ALL steps
    # (the previous per-epoch stepping barely moved the LR).
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr,
        total_steps=total_steps,
        pct_start=pct_start, anneal_strategy="cos",
        div_factor=10.0, final_div_factor=100.0,
    )

    best_loss = float('inf')
    log = MetricsLogger("raft_finetune", config={
        "epochs": epochs, "batch_size": batch_size, "max_lr": lr,
        "smooth_weight": smooth_weight, "weight_decay": weight_decay,
        "device": DEVICE.type, "n_pairs": len(dataset)})
    print(f"\n{'='*50}")
    print("RAFT FINE-TUNING")
    print(f"  epochs={epochs}  batch_size={batch_size}  max_lr={lr}")
    print(f"  steps/epoch={steps_per_epoch}  total_steps={total_steps}")
    print(f"  smooth_weight={smooth_weight}  weight_decay={weight_decay}")
    print(f"{'='*50}")

    for epoch in range(epochs):
        total_loss = 0
        n_batches = 0

        for frame1, frame2 in loader:
            frame1 = frame1.to(DEVICE)
            frame2 = frame2.to(DEVICE)

            # Pad to be divisible by 8
            B, C, H, W = frame1.shape
            pad_h = (8 - H % 8) % 8
            pad_w = (8 - W % 8) % 8
            if pad_h or pad_w:
                frame1 = F.pad(frame1, (0, pad_w, 0, pad_h))
                frame2 = F.pad(frame2, (0, pad_w, 0, pad_h))

            # Forward: RAFT returns list of flow predictions (coarse→fine)
            flow_preds = model(frame1, frame2)

            loss = 0
            # Multi-scale photometric loss (all RAFT iterations)
            weights = [0.32 ** (len(flow_preds) - 1 - i)
                       for i in range(len(flow_preds))]
            for flow, w in zip(flow_preds, weights):
                # Crop back to original resolution for loss
                flow_crop = flow[:, :, :H, :W]
                f1_crop = frame1[:, :, :H, :W]
                f2_crop = frame2[:, :, :H, :W]
                loss += w * photometric_loss(f1_crop, f2_crop, flow_crop)
                loss += w * smooth_weight * smoothness_loss(flow_crop)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            n_batches += 1

        avg = total_loss / max(1, n_batches)
        cur_lr = scheduler.get_last_lr()[0]

        if avg < best_loss:
            best_loss = avg
            torch.save(model.state_dict(), output_path)

        log.log(epoch, loss=avg, best=best_loss, lr=cur_lr)

        print(f"  Epoch {epoch:3d}/{epochs} | "
              f"Loss: {avg:.5f} | Best: {best_loss:.5f} | lr: {cur_lr:.2e}",
              flush=True)

    log.close(plot_keys=["loss", "best", "lr"])

    # Always keep a copy of the final-epoch weights alongside the best.
    final_path = output_path.replace(".pt", "_final.pt")
    torch.save(model.state_dict(), final_path)

    print(f"\n✓ RAFT fine-tuning done. Best loss: {best_loss:.5f}")
    print(f"  Saved best:  {output_path}")
    print(f"  Saved final: {final_path}")
    print("  train.py will now use fine-tuned RAFT automatically.")


if __name__ == "__main__":
    finetune_raft(
        video_dir="data/videos",
        output_path="models/raft_crowd.pt",
        epochs=80,
        batch_size=12 if DEVICE.type == "cuda" else 2,
    )
