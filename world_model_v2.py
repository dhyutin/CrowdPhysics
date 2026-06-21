# world_model_v2.py
"""
World Model v2 — proper VAE latent + pluggable transition backbone.

What changed vs world_model.py (v1):
  - v1 encoder was DETERMINISTIC (a point z) yet the KL in train.py was
    applied to the TRANSITION output. That's a "half-VAE": stochasticity
    lived in the wrong place and the latent space was never shaped toward
    a prior.
  - v2 makes the ENCODER a real posterior  q(z_t | x_t) = N(mu_e, sigma_e),
    and the TRANSITION a prior              p(z_t | z_<t) = N(mu_p, sigma_p).
    Training objective (Dreamer/RSSM-style):
        reconstruction  +  KL(posterior || prior)  +  small KL(posterior || N(0,1))
    The last term anchors the latent scale near the origin, which is what
    makes DynaTrainer.danger_score = ||z|| / sqrt(d) meaningful (normal
    states actually cluster near 0), and gives the anomaly signal a
    principled probabilistic form.

Pluggable transition backbone ("apart from LSTM what can I try"):
  transition_type ∈ {"lstm", "gru", "tcn", "transformer"}
  All four expose the SAME interface the rest of the codebase relies on:
      transition(z_seq, reset_hidden=bool) -> (mu, log_var)
      transition.hidden  (set to None to reset recurrent/context state)
  so this model is a drop-in replacement for CrowdWorldModel everywhere
  (DynaTrainer, CrowdPhysicsDetector, backend _forecast_future / rollout).

Interface compatibility with v1 (intentional, do not break):
  - forward(feature_seq) -> (mu, log_var, z_target, z, recon)   [DETERMINISTIC]
  - encode_sequence(feature_seq) -> (b, T, latent)              [posterior mean]
  - decoder(z)                                                  [nn.Module]
  - transition(z_seq, reset_hidden=)                            [nn.Module]
  - rollout_latent(z_start, n_steps)
  - latent_dim attribute
The extra training-only signal (posterior params, prior params) is exposed
via forward_train(); use that from train_v2.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── ENCODER (posterior q(z|x)) ───────────────────────────────────────────────

class CrowdEncoderV2(nn.Module):
    """256-dim flow features -> posterior over 64-dim latent: (mu_e, log_var_e)."""

    def __init__(self, input_dim=256, latent_dim=64):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 192),
            nn.LayerNorm(192),
            nn.SiLU(),
            nn.Linear(192, 128),
            nn.SiLU(),
        )
        self.to_mu = nn.Linear(128, latent_dim)
        self.to_logvar = nn.Linear(128, latent_dim)

    def forward(self, x):
        h = self.backbone(x)
        mu = self.to_mu(h)
        log_var = self.to_logvar(h).clamp(-6, 2)
        return mu, log_var


class CrowdDecoder(nn.Module):
    """Reconstruct 256-dim features from 64-dim latent (keeps latent informative)."""

    def __init__(self, latent_dim=64, output_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 192),
            nn.SiLU(),
            nn.Linear(192, output_dim),
        )

    def forward(self, z):
        return self.net(z)


# ─── TRANSITION BACKBONES (prior p(z_t | z_<t)) ───────────────────────────────
# Each backbone maps a latent sequence -> (mu, log_var) per timestep, and
# supports stateful single-step rollout via `self.hidden`.

class _TCNCore(nn.Module):
    """Dilated causal 1-D conv stack. Parallel in time, fixed receptive field."""

    def __init__(self, latent_dim, hidden_dim, n_layers, kernel_size=3):
        super().__init__()
        self.layers = nn.ModuleList()
        self.dilations = []
        in_ch = latent_dim
        for i in range(n_layers):
            dilation = 2 ** i
            self.dilations.append(dilation)
            self.layers.append(nn.Conv1d(in_ch, hidden_dim, kernel_size,
                                         dilation=dilation))
            in_ch = hidden_dim
        self.kernel_size = kernel_size
        self.act = nn.SiLU()

    def forward(self, x):
        # x: (B, T, latent) -> (B, T, hidden)
        h = x.transpose(1, 2)  # (B, C, T)
        for conv, dilation in zip(self.layers, self.dilations):
            pad = (self.kernel_size - 1) * dilation  # left pad => causal
            h = self.act(conv(F.pad(h, (pad, 0))))
        return h.transpose(1, 2)


class _TransformerCore(nn.Module):
    """Causal self-attention encoder over the latent sequence."""

    def __init__(self, latent_dim, hidden_dim, n_layers, n_heads=4,
                 max_len=64):
        super().__init__()
        self.in_proj = nn.Linear(latent_dim, hidden_dim)
        self.pos = nn.Parameter(torch.zeros(1, max_len, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads,
            dim_feedforward=hidden_dim * 2, dropout=0.1,
            activation="gelu", batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.max_len = max_len

    def forward(self, x):
        # x: (B, T, latent) -> (B, T, hidden)
        T = x.size(1)
        h = self.in_proj(x) + self.pos[:, :T]
        mask = torch.triu(
            torch.full((T, T), float("-inf"), device=x.device), diagonal=1)
        return self.encoder(h, mask=mask)


class CrowdTransitionModelV2(nn.Module):
    """
    Prior model p(z_t | z_<t) with a swappable temporal backbone.

    transition_type:
      "lstm"        recurrent, unbounded memory (default, matches v1 behaviour)
      "gru"         recurrent, lighter than LSTM
      "tcn"         dilated causal conv — parallel training, fixed window
      "transformer" causal self-attention — attention-based context

    Recurrent backbones use a native hidden state. Non-recurrent backbones
    (tcn/transformer) emulate the same stateful single-step API with a rolling
    context buffer so DynaTrainer/forecast rollouts behave identically.
    """

    def __init__(self, latent_dim=64, hidden_dim=256, n_layers=2,
                 transition_type="lstm", max_ctx=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.transition_type = transition_type
        self.max_ctx = max_ctx
        self.recurrent = transition_type in ("lstm", "gru")

        if transition_type == "lstm":
            self.core = nn.LSTM(latent_dim, hidden_dim, num_layers=n_layers,
                                batch_first=True, dropout=0.1)
        elif transition_type == "gru":
            self.core = nn.GRU(latent_dim, hidden_dim, num_layers=n_layers,
                               batch_first=True, dropout=0.1)
        elif transition_type == "tcn":
            self.core = _TCNCore(latent_dim, hidden_dim, n_layers)
        elif transition_type == "transformer":
            self.core = _TransformerCore(latent_dim, hidden_dim, n_layers,
                                         max_len=max_ctx)
        else:
            raise ValueError(f"unknown transition_type: {transition_type}")

        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.SiLU(),
            nn.Linear(128, latent_dim * 2),  # mu + log_var
        )

        # Unified state slot. For recurrent: (h, c) / h. For tcn/transformer:
        # the buffered context tensor. None == fresh start.
        self.hidden = None

    def forward(self, z_seq, reset_hidden=True):
        """
        z_seq: (B, T, latent) -> mu, log_var: (B, T, latent)
        reset_hidden: start a fresh sequence (clears recurrent/context state).
        """
        if reset_hidden:
            self.hidden = None

        if self.recurrent:
            out, self.hidden = self.core(z_seq, self.hidden)
            if isinstance(self.hidden, tuple):
                self.hidden = tuple(h.detach() for h in self.hidden)
            else:
                self.hidden = self.hidden.detach()
        else:
            # Rolling-context emulation for stateful single-step rollout.
            if self.hidden is None:
                ctx = z_seq
            else:
                ctx = torch.cat([self.hidden, z_seq], dim=1)[:, -self.max_ctx:]
            feats = self.core(ctx)
            out = feats[:, -z_seq.size(1):]  # align to the queried steps
            self.hidden = ctx.detach()

        pred = self.output_head(out)
        mu, log_var = pred.chunk(2, dim=-1)
        return mu, log_var

    def sample_next(self, z_seq, reset_hidden=True):
        mu, log_var = self.forward(z_seq, reset_hidden)
        log_var = log_var.clamp(-6, 2)
        std = torch.exp(0.5 * log_var)
        return mu + std * torch.randn_like(std), mu, log_var


# ─── FULL MODEL ───────────────────────────────────────────────────────────────

def _reparameterize(mu, log_var):
    std = torch.exp(0.5 * log_var)
    return mu + std * torch.randn_like(std)


class CrowdWorldModelV2(nn.Module):
    """
    Encoder (posterior) + Transition (prior) + Decoder.

    Drop-in compatible with v1's CrowdWorldModel:
      forward()/encode_sequence() are DETERMINISTIC (use posterior mean) so
      downstream inference (anomaly detector, backend forecast, RL rollouts)
      stays stable. forward_train() exposes the stochastic posterior/prior
      params needed for the proper VAE loss in train_v2.py.
    """

    def __init__(self, input_dim=256, latent_dim=64, hidden_dim=256,
                 n_layers=2, transition_type="lstm"):
        super().__init__()
        self.encoder = CrowdEncoderV2(input_dim, latent_dim)
        self.decoder = CrowdDecoder(latent_dim, input_dim)
        self.transition = CrowdTransitionModelV2(
            latent_dim, hidden_dim=hidden_dim, n_layers=n_layers,
            transition_type=transition_type)
        self.latent_dim = latent_dim

    # ── encoding ──────────────────────────────────────────────────────────────

    def encode_dist(self, feature_seq):
        """Return posterior (mu_e, log_var_e) for a sequence. (b, T, latent)."""
        b, t, f = feature_seq.shape
        mu, log_var = self.encoder(feature_seq.reshape(-1, f))
        return (mu.reshape(b, t, self.latent_dim),
                log_var.reshape(b, t, self.latent_dim))

    def encode_sequence(self, feature_seq, sample=False):
        """
        Deterministic by default (posterior mean) — what inference wants.
        feature_seq: (b, T, 256) -> z: (b, T, 64)
        """
        mu, log_var = self.encode_dist(feature_seq)
        return _reparameterize(mu, log_var) if sample else mu

    # ── inference forward (DETERMINISTIC, v1-compatible 5-tuple) ───────────────

    def forward(self, feature_seq):
        """
        feature_seq: (batch, T+1, 256)
        Returns (mu, log_var, z_target, z, recon) using the posterior MEAN so
        anomaly scores / forecasts are reproducible.
        """
        z = self.encode_sequence(feature_seq, sample=False)        # (b,T+1,64)
        mu, log_var = self.transition(z[:, :-1], reset_hidden=True)
        z_target = z[:, 1:].detach()
        recon = self.decoder(z.reshape(-1, self.latent_dim))
        recon = recon.reshape(z.shape[0], z.shape[1], -1)
        return mu, log_var, z_target, z, recon

    # ── training forward (STOCHASTIC, exposes posterior + prior) ───────────────

    def forward_train(self, feature_seq):
        """
        feature_seq: (batch, T+1, 256)
        Returns dict for the proper VAE loss:
          mu_e, log_var_e : posterior params for all T+1 steps
          z               : sampled latents (reparameterized), (b, T+1, 64)
          mu_p, log_var_p : prior params predicting steps 1..T from z_<t
          recon           : decoded features, (b, T+1, 256)
        """
        mu_e, log_var_e = self.encode_dist(feature_seq)            # (b,T+1,64)
        z = _reparameterize(mu_e, log_var_e)
        mu_p, log_var_p = self.transition(z[:, :-1], reset_hidden=True)
        recon = self.decoder(z.reshape(-1, self.latent_dim))
        recon = recon.reshape(z.shape[0], z.shape[1], -1)
        return {
            "mu_e": mu_e, "log_var_e": log_var_e, "z": z,
            "mu_p": mu_p, "log_var_p": log_var_p, "recon": recon,
        }

    # ── latent rollout (RL planning / forecast) ────────────────────────────────

    def rollout_latent(self, z_start, n_steps=20):
        """Roll forward in latent space from a starting state. (n_steps+1, 64)."""
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
        return torch.stack(trajectory)


# ─── LOSS HELPERS ──────────────────────────────────────────────────────────────

def kl_two_gaussians(mu_q, log_var_q, mu_p, log_var_p):
    """KL( N(mu_q, e^{log_var_q}) || N(mu_p, e^{log_var_p}) ), per-element."""
    return 0.5 * (
        log_var_p - log_var_q
        + (log_var_q.exp() + (mu_q - mu_p) ** 2) / log_var_p.exp()
        - 1.0
    )


def kl_standard_normal(mu_q, log_var_q):
    """KL( N(mu_q, e^{log_var_q}) || N(0, 1) ), per-element."""
    return -0.5 * (1 + log_var_q - mu_q.pow(2) - log_var_q.exp())
