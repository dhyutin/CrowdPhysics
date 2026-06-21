# anomaly_detector.py
"""
Phase 4: Anomaly Detector
Connects world model + RL policy into clean inference interface.

This is the glue layer. It holds a rolling window of recent flow features,
encodes them, asks the world model to predict the next state, measures how
surprised the model is, and returns a complete physics state dict.

Usage:
    detector = CrowdPhysicsDetector(world_model, dyna_trainer)
    detector.calibrate(normal_feature_sequences)

    for each frame:
        features = flow_to_features(optical_flow)   # 256-dim
        state = detector.process_frame(features)
        # state.keys(): status, score, probability, turbulence,
        #               backward_flow, boundary_stress, mean_speed,
        #               z_latent, intervention
"""

import torch
import numpy as np
from collections import deque


class CrowdPhysicsDetector:
    """
    Single inference interface for the complete CrowdPhysics pipeline.

    Anomaly detection mechanism:
      1. Maintain a rolling buffer of the last `window_size` feature vectors
      2. Encode the window → latent sequence z(t-k:t)
      3. Run the LSTM to predict z(t+1), z(t+2), ...
      4. Measure prediction error (MSE between predicted and actual next states)
      5. Normalise against a baseline established on calm footage
      6. Score > threshold → WARNING/DANGER

    The key insight: the world model learned "normal" crowd physics.
    When the crowd does something abnormal (crush forming), prediction error
    spikes — the model is "surprised". That surprise IS the anomaly signal.
    """

    def __init__(self, world_model, dyna_trainer,
                 window_size=30, alert_threshold=2.5):
        self.wm = world_model
        self.wm.eval()
        self.trainer = dyna_trainer

        self.window = window_size
        self.threshold = alert_threshold

        self.buf = deque(maxlen=window_size + 1)
        self.err_history = deque(maxlen=300)

        self.baseline_mean = None
        self.baseline_std = 0.01
        self.calibrated = False

    # ── CALIBRATION ───────────────────────────────────────────────────────────

    def calibrate(self, normal_sequences):
        """
        Establish baseline prediction error on safe crowd footage.
        Should be called once before live monitoring begins.

        Args:
            normal_sequences: list of np.ndarray (T, 256) — calm crowd footage
        """
        errors = []
        with torch.no_grad():
            for seq in normal_sequences:
                if len(seq) < self.window + 1:
                    continue
                for s in range(0, len(seq) - self.window, 5):
                    chunk = seq[s:s + self.window + 1]
                    x = torch.FloatTensor(chunk).unsqueeze(0)
                    mu, lv, zt, z, _ = self.wm(x)
                    errors.append(float(torch.mean((mu - zt)**2)))
                    if len(errors) >= 500:
                        break

        if errors:
            self.baseline_mean = float(np.mean(errors))
            self.baseline_std = float(np.std(errors)) + 1e-6
            self.calibrated = True
            print(f"[calibrated] baseline={self.baseline_mean:.5f} "
                  f"± {self.baseline_std:.5f} ({len(errors)} samples)")
        else:
            self.baseline_mean = 0.01
            self.baseline_std = 0.005
            self.calibrated = True
            print("[calibrated] using defaults (no normal data provided)")

    def calibrate_auto(self, feature_sequence, burn_in=60):
        """
        Auto-calibrate from the first `burn_in` frames of a live feed.
        Assumes the feed starts in a safe state.
        """
        if len(feature_sequence) >= burn_in:
            self.calibrate([np.array(feature_sequence[:burn_in])])

    # ── LIVE INFERENCE ────────────────────────────────────────────────────────

    def process_frame(self, features_256):
        """
        Process one frame's 256-dim flow features.
        Returns a complete physics state dictionary.

        Call this every frame during live monitoring.

        Args:
            features_256: np.ndarray (256,) — output of flow_to_features()

        Returns:
            dict with keys:
                status          str   'SAFE' | 'WARNING' | 'DANGER' | 'CALIBRATING'
                score           float anomaly score (σ above baseline)
                probability     float crush probability [0,1]
                error           float raw prediction MSE
                turbulence      float variance of motion magnitudes
                backward_flow   float mean backward (negative y) pressure
                boundary_stress float stress at spatial boundaries
                mean_speed      float average crowd speed
                z_latent        ndarray (64,) current latent state
                intervention    dict | None  RL recommendation (WARNING/DANGER only)
        """
        self.buf.append(features_256.copy())

        if len(self.buf) < self.window:
            return self._empty_state('CALIBRATING')

        seq = np.array(list(self.buf))
        x = torch.FloatTensor(seq).unsqueeze(0)

        with torch.no_grad():
            mu, lv, z_target, z, _ = self.wm(x)
            error = float(torch.mean((mu - z_target)**2))
            z_now = z[0, -1].cpu().numpy()

        self.err_history.append(error)

        # Anomaly score: σ above calibrated baseline
        if self.calibrated and self.baseline_mean is not None:
            score = (error - self.baseline_mean) / self.baseline_std
        else:
            score = error * 50

        # Crush probability via sigmoid centred at threshold
        prob = float(1 / (1 + np.exp(-score + 1.5)))

        # ── PHYSICS METRICS from raw features ─────────────────────────────
        recent = seq[-10:]
        fx_all  = recent[:, 0::4]   # x-velocity dims
        fy_all  = recent[:, 1::4]   # y-velocity dims
        mag_all = np.sqrt(fx_all**2 + fy_all**2)

        turbulence      = float(np.var(mag_all))
        backward_flow   = float(-np.mean(fy_all))   # negative y = backward
        boundary_stress = float(np.mean(
            np.abs(np.concatenate([recent[:, :32],
                                   recent[:, -32:]], axis=1))
        ))
        mean_speed = float(np.mean(mag_all))

        # ── STATUS ────────────────────────────────────────────────────────
        if score > self.threshold:
            status = 'DANGER'
        elif score > self.threshold * 0.65:
            status = 'WARNING'
        else:
            status = 'SAFE'

        # ── RL RECOMMENDATION (only when elevated) ─────────────────────
        intervention = None
        if status in ['WARNING', 'DANGER']:
            try:
                intervention = self.trainer.get_intervention(z_now)
            except Exception:
                pass

        return {
            'status':          status,
            'score':           round(float(score), 3),
            'probability':     round(prob, 3),
            'error':           round(float(error), 6),
            'turbulence':      round(turbulence, 4),
            'backward_flow':   round(backward_flow, 4),
            'boundary_stress': round(boundary_stress, 4),
            'mean_speed':      round(mean_speed, 4),
            'z_latent':        z_now,
            'intervention':    intervention,
        }

    # ── BATCH / VIDEO ANALYSIS ────────────────────────────────────────────────

    def analyze_sequence(self, feature_sequence, auto_calibrate=True):
        """
        Analyze a full feature sequence (e.g. a pre-recorded video).
        Returns list of state dicts, one per frame.

        Auto-calibrates on first 60 frames if not already calibrated.
        """
        if auto_calibrate and not self.calibrated:
            self.calibrate_auto(feature_sequence)

        self.buf.clear()
        states = []
        for feat in feature_sequence:
            states.append(self.process_frame(feat))
        return states

    def get_danger_timeline(self, states):
        """
        Summarise a sequence of states into a danger timeline.
        Returns list of (frame_idx, status, score) for WARNING/DANGER frames.
        """
        return [
            (i, s['status'], s['score'])
            for i, s in enumerate(states)
            if s['status'] in ('WARNING', 'DANGER')
        ]

    # ── UTILS ─────────────────────────────────────────────────────────────────

    def reset(self):
        """Clear buffer — call when switching to a new camera/scene."""
        self.buf.clear()
        self.err_history.clear()

    def _empty_state(self, status):
        return {
            'status': status, 'score': 0.0, 'probability': 0.0,
            'error': 0.0, 'turbulence': 0.0, 'backward_flow': 0.0,
            'boundary_stress': 0.0, 'mean_speed': 0.0,
            'z_latent': np.zeros(64), 'intervention': None
        }


# ── FACTORY ───────────────────────────────────────────────────────────────────

def load_detector(world_model_path="models/world_model.pt",
                  rl_policy_path="models/rl_policy.pt",
                  window_size=30, alert_threshold=2.5):
    """
    Load a fully trained detector from saved checkpoints.
    Use this in the backend for the live demo.
    """
    from world_model import CrowdWorldModel
    from dyna_trainer import DynaTrainer

    wm = CrowdWorldModel()
    if world_model_path and __import__("os").path.exists(world_model_path):
        wm.load_state_dict(torch.load(world_model_path, map_location="cpu"))
        print(f"[detector] Loaded world model: {world_model_path}")
    else:
        print("[detector] No checkpoint found — using untrained model (demo mode)")

    trainer = DynaTrainer(wm)
    if rl_policy_path and __import__("os").path.exists(rl_policy_path):
        import torch.nn as nn
        trainer.q_net.load_state_dict(
            torch.load(rl_policy_path, map_location="cpu"))
        print(f"[detector] Loaded RL policy: {rl_policy_path}")
    else:
        print("[detector] No RL checkpoint — using random policy (demo mode)")

    return CrowdPhysicsDetector(wm, trainer, window_size, alert_threshold)
