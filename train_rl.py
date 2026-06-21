# train_rl.py
"""
Train ONLY the Dyna RL policy on top of an already-trained world model.

Use this when models/world_model.pt already exists (e.g. trained on the
GPU box with RAFT features) and you just need models/rl_policy.pt.

The world-model architecture must match the checkpoint — set via env vars
WM_HIDDEN_DIM / WM_N_LAYERS (defaults 512/3, the GPU-trained config).
"""

import os
import torch

from world_model import CrowdWorldModel
from dyna_trainer import DynaTrainer
from metrics_logger import MetricsLogger

HIDDEN_DIM = int(os.environ.get("WM_HIDDEN_DIM", "512"))
N_LAYERS = int(os.environ.get("WM_N_LAYERS", "3"))
N_EPISODES = int(os.environ.get("RL_EPISODES", "8000"))
STEPS_PER_EPISODE = int(os.environ.get("RL_STEPS_PER_EPISODE", "10"))
CKPT_EVERY = int(os.environ.get("RL_CKPT_EVERY", "100"))

# Dyna runs entirely on CPU (latent-space loop + small Q-net; DynaTrainer uses
# numpy on CPU tensors). The GPU is not needed here.
print(f"[train_rl] hidden={HIDDEN_DIM} layers={N_LAYERS} "
      f"episodes={N_EPISODES} steps/ep={STEPS_PER_EPISODE}")

wm = CrowdWorldModel(hidden_dim=HIDDEN_DIM, n_layers=N_LAYERS)
wm.load_state_dict(torch.load("models/world_model.pt", map_location="cpu"))
wm.eval()
print("[train_rl] ✓ world model loaded")

log = MetricsLogger("rl_policy", config={
    "hidden_dim": HIDDEN_DIM, "n_layers": N_LAYERS,
    "episodes": N_EPISODES, "steps_per_episode": STEPS_PER_EPISODE})

os.makedirs("models", exist_ok=True)

trainer = DynaTrainer(wm)
trainer.run_dyna_training(n_episodes=N_EPISODES,
                          steps_per_episode=STEPS_PER_EPISODE,
                          logger=log,
                          checkpoint_path="models/rl_policy.pt",
                          checkpoint_every=CKPT_EVERY)

torch.save(trainer.q_net.state_dict(), "models/rl_policy.pt")
print("[train_rl] ✓ RL policy saved: models/rl_policy.pt "
      "(+ models/rl_policy_best.pt)")

log.close(plot_keys=["avg_reward_50", "reward", "loss", "epsilon"])
