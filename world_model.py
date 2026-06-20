# world_model.py
"""
Phase 2: World Model
CNN Encoder + LSTM Transition Model
This is where crowd physics gets discovered.
"""

import torch
import torch.nn as nn
import numpy as np


class CrowdEncoder(nn.Module):
    """
    Compresses 256-dim flow features → 64-dim latent state z.

    The compression bottleneck is intentional:
    It forces the representation to keep only what matters
    for predicting future states — i.e., the physics.
    """
    def __init__(self, input_dim=256, latent_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 192),
            nn.LayerNorm(192),
            nn.SiLU(),              # smoother than ReLU for dynamics
            nn.Linear(192, 128),
            nn.SiLU(),
            nn.Linear(128, latent_dim)
        )

    def forward(self, x):
        # x: (..., 256) → (..., 64)
        return self.net(x)


class CrowdDecoder(nn.Module):
    """
    Reconstructs 256-dim features from 64-dim latent.
    Used only during training as an auxiliary loss.
    Forces the latent space to preserve visual information.
    """
    def __init__(self, latent_dim=64, output_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 192),
            nn.SiLU(),
            nn.Linear(192, output_dim)
        )

    def forward(self, z):
        return self.net(z)


class CrowdTransitionModel(nn.Module):
    """
    LSTM that predicts next latent state from current sequence.

    The stochastic output (mean + variance) is important:
    - It captures uncertainty in crowd dynamics
    - It gives us a way to measure "surprise" at inference time
    - High prediction error = crowd doing something unexpected = anomaly

    Architecture:
        z(t-k:t) → LSTM → μ(t+1), σ(t+1)
        z(t+1) sampled from N(μ, σ)
    """
    def __init__(self, latent_dim=64, hidden_dim=256, n_layers=2):
        super().__init__()
        self.latent_dim = latent_dim

        self.lstm = nn.LSTM(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,    # (batch, seq, features)
            dropout=0.1
        )
        # Output: mean and log-variance for stochastic next state
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.SiLU(),
            nn.Linear(128, latent_dim * 2)  # mu + log_var
        )
        self.hidden = None

    def forward(self, z_seq, reset_hidden=True):
        """
        Args:
            z_seq: (batch, seq_len, latent_dim) — sequence of latent states
            reset_hidden: whether to reset LSTM hidden state

        Returns:
            mu: (batch, seq_len, latent_dim) — predicted next states
            log_var: (batch, seq_len, latent_dim) — uncertainty
        """
        if reset_hidden:
            self.hidden = None

        out, self.hidden = self.lstm(z_seq, self.hidden)

        # Detach hidden from computation graph (prevent BPTT across batches)
        if self.hidden is not None:
            self.hidden = (
                self.hidden[0].detach(),
                self.hidden[1].detach()
            )

        pred = self.output_head(out)
        mu, log_var = pred.chunk(2, dim=-1)
        return mu, log_var

    def sample_next(self, z_seq, reset_hidden=True):
        """Sample next state (with reparameterization trick)"""
        mu, log_var = self.forward(z_seq, reset_hidden)
        log_var = log_var.clamp(-6, 2)
        std = torch.exp(0.5 * log_var)
        # Reparameterization: z = mu + std * epsilon, epsilon ~ N(0,1)
        eps = torch.randn_like(std)
        return mu + std * eps, mu, log_var


class CrowdWorldModel(nn.Module):
    """
    Complete world model: Encoder + Transition + Decoder.

    Training objective: predict z(t+1) from z(t).
    What it learns: crowd fluid dynamics (emergently).

    At inference:
    - Encode frame sequence → latent states
    - Predict next latent state
    - High prediction error = anomaly = danger signal
    """
    def __init__(self, input_dim=256, latent_dim=64,
                 hidden_dim=256, n_layers=2):
        super().__init__()
        self.encoder = CrowdEncoder(input_dim, latent_dim)
        self.decoder = CrowdDecoder(latent_dim, input_dim)
        self.transition = CrowdTransitionModel(latent_dim,
                                               hidden_dim=hidden_dim,
                                               n_layers=n_layers)
        self.latent_dim = latent_dim

    def encode_sequence(self, feature_seq):
        """
        Encode a sequence of flow features to latent states.
        feature_seq: (batch, T, 256) → z: (batch, T, 64)
        """
        b, t, f = feature_seq.shape
        z = self.encoder(feature_seq.reshape(-1, f))
        return z.reshape(b, t, self.latent_dim)

    def forward(self, feature_seq):
        """
        Full forward pass for training.

        feature_seq: (batch, T+1, 256)

        Returns:
            mu, log_var: predicted next latent states (batch, T, 64)
            z_target: actual next latent states (batch, T, 64)
            z: all latent states (batch, T+1, 64)
            recon: reconstructed features from latent (batch, T+1, 256)
        """
        z = self.encode_sequence(feature_seq)
        mu, log_var = self.transition(z[:, :-1], reset_hidden=True)
        z_target = z[:, 1:].detach()
        recon = self.decoder(z.reshape(-1, self.latent_dim))
        recon = recon.reshape(z.shape[0], z.shape[1], -1)
        return mu, log_var, z_target, z, recon

    def rollout_latent(self, z_start, n_steps=20):
        """
        Roll forward in latent space from a starting state.
        Used by RL policy for planning in imagination.

        z_start: (latent_dim,) — starting crowd state
        Returns: (n_steps+1, latent_dim) — trajectory in latent space
        """
        z = z_start.unsqueeze(0).unsqueeze(0)  # (1, 1, 64)
        trajectory = [z_start.detach()]
        self.transition.hidden = None

        with torch.no_grad():
            for _ in range(n_steps):
                mu, log_var = self.transition(z, reset_hidden=False)
                log_var = log_var.clamp(-6, 2)
                std = torch.exp(0.5 * log_var)
                z_next = mu + std * torch.randn_like(std)
                trajectory.append(z_next[0, 0].detach())
                z = z_next

        return torch.stack(trajectory)  # (n_steps+1, 64)
