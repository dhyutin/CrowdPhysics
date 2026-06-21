# dyna_trainer.py
"""
Phase 3b: Dyna Training Loop
RL inside the world model. No real disasters needed.
"""

import torch
import torch.nn as nn
import numpy as np
from collections import deque
import random
from rl_policy import CrowdQNetwork, compute_cql_loss, ACTIONS, N_ACTIONS


class DynaTrainer:
    """
    Dyna-style model-based RL trainer.

    The loop:
    1. Start from a latent crowd state
    2. Apply action → modify latent state
    3. Roll world model forward one step
    4. Compute reward from anomaly score
    5. Store (s, a, r, s') in replay buffer
    6. Train CQL policy on replay buffer
    7. Repeat

    The world model IS the simulator.
    This is the same core idea as DreamerV3.
    """

    def __init__(self, world_model, latent_dim=64):
        self.world_model = world_model
        self.world_model.eval()  # freeze world model
        self.latent_dim = latent_dim

        # Q-network and slow-moving target
        self.q_net = CrowdQNetwork(latent_dim, N_ACTIONS)
        self.target_net = CrowdQNetwork(latent_dim, N_ACTIONS)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(
            self.q_net.parameters(), lr=3e-4)

        self.replay_buffer = deque(maxlen=100_000)
        self.step = 0
        self.epsilon = 1.0    # exploration rate
        self.eps_min = 0.05
        self.eps_decay = 0.995

    # ── DANGER SIGNAL ─────────────────────────────────────────────────────────

    def danger_score(self, z):
        """
        How dangerous is this latent crowd state?

        We use L2 norm from origin because the world model
        is trained such that normal states cluster near origin.
        Anomalous states push further out.
        """
        return float(torch.norm(z, dim=-1).mean())

    # ── ACTION EFFECTS ────────────────────────────────────────────────────────

    def apply_action_effect(self, z, action):
        """
        Modify latent state based on intervention action.

        Principled perturbations based on physics intuition:
        - Gate opens: reduce y-compression (backward pressure dims)
        - Stop entry: damp all dims (reduce incoming flow energy)
        - Redirect: increase lateral x-dims
        - Alert: add noise then damp (temporary confusion → dispersal)
        - Emergency: strong global damping
        """
        z_mod = z.clone()

        if action == 0:   # monitor — no change
            pass

        elif action in [1, 2]:   # open gate
            z_mod = z_mod * 0.75
            z_mod[:, 1::4] *= 0.5   # y-velocity dims (backward pressure)

        elif action == 3:   # stop entry
            z_mod = z_mod * 0.65

        elif action == 4:   # redirect flow
            z_mod[:, 0::4] = z_mod[:, 0::4] * 1.3   # x-dims up
            z_mod[:, 1::4] = z_mod[:, 1::4] * 0.6   # y-dims down

        elif action == 5:   # dispersal alert
            noise = torch.randn_like(z_mod) * 0.15
            z_mod = (z_mod + noise) * 0.8

        elif action == 6:   # emergency
            z_mod = z_mod * 0.3

        return z_mod

    # ── EPISODE GENERATION ────────────────────────────────────────────────────

    def generate_episode(self, z_start=None, episode_len=40):
        """
        Generate one synthetic crowd scenario inside the world model.
        Returns list of (action, danger_before, danger_after, reward).
        """
        if z_start is None:
            danger_level = np.random.choice([0.3, 0.8, 1.5, 2.5],
                                            p=[0.3, 0.3, 0.25, 0.15])
            z_start = torch.randn(1, self.latent_dim) * danger_level

        z = z_start
        self.world_model.transition.hidden = None
        episode = []

        for t in range(episode_len):
            # ε-greedy action selection
            if random.random() < self.epsilon:
                action = random.randint(0, N_ACTIONS - 1)
            else:
                action = self.q_net.best_action(z)

            danger_before = self.danger_score(z)

            # Apply intervention
            z_intervened = self.apply_action_effect(z, action)

            # World model rolls forward
            with torch.no_grad():
                z_seq = z_intervened.unsqueeze(1)  # (1, 1, 64)
                mu, log_var = self.world_model.transition(
                    z_seq, reset_hidden=False)
                log_var = log_var.clamp(-6, 2)
                std = torch.exp(0.5 * log_var)
                z_next = (mu + std * 0.3 * torch.randn_like(std)
                          ).squeeze(1)

            danger_after = self.danger_score(z_next)

            # ── REWARD FUNCTION ───────────────────────────────────────────
            reward = (danger_before - danger_after) * 3.0

            if danger_after < 0.8:
                reward += 1.5                        # bonus: staying safe

            if danger_after > 3.5:
                reward -= 15.0                       # crush threshold crossed

            if action == 0 and danger_before > 2.0:
                reward -= 2.0                        # inaction during danger

            if action == 6 and danger_before < 1.0:
                reward -= 3.0                        # overreaction penalty

            action_cost = [0, 0.1, 0.1, 0.3, 0.2, 0.2, 1.0][action]
            reward -= action_cost
            # ──────────────────────────────────────────────────────────────

            done = float(danger_after > 4.5)

            self.replay_buffer.append((
                z.detach().numpy()[0],
                action,
                float(reward),
                z_next.detach().numpy()[0],
                done
            ))
            episode.append((action, float(danger_before),
                            float(danger_after), float(reward)))

            z = z_next
            self.step += 1

            if done:
                break

        return episode

    # ── TRAINING STEP ─────────────────────────────────────────────────────────

    def train_step(self, batch_size=128):
        """One gradient step on the CQL policy"""
        if len(self.replay_buffer) < batch_size:
            return None

        batch_data = random.sample(self.replay_buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch_data)

        batch = (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(dones)
        )

        self.q_net.train()
        loss, metrics = compute_cql_loss(self.q_net, self.target_net, batch)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()

        # Soft update target network (Polyak averaging)
        tau = 0.005
        for p, tp in zip(self.q_net.parameters(),
                         self.target_net.parameters()):
            tp.data.copy_(tau * p.data + (1 - tau) * tp.data)

        return metrics

    # ── MAIN TRAINING LOOP ────────────────────────────────────────────────────

    def run_dyna_training(self, n_episodes=500, steps_per_episode=10,
                          logger=None):
        """
        Full Dyna training loop.
        Generate episodes in world model → train policy → repeat.

        Args:
            logger: optional MetricsLogger — logs per-episode reward/loss curves.
        """
        print(f"\nDyna-CQL Training: {n_episodes} episodes")
        print("=" * 40)
        all_rewards = []
        all_losses = []

        for ep in range(n_episodes):
            episode = self.generate_episode()
            ep_reward = sum(t[3] for t in episode)
            all_rewards.append(ep_reward)

            for _ in range(steps_per_episode):
                metrics = self.train_step()
                if metrics:
                    all_losses.append(metrics['total_loss'])

            # Decay exploration
            self.epsilon = max(self.eps_min,
                               self.epsilon * self.eps_decay)

            avg_r = float(np.mean(all_rewards[-50:])) if all_rewards else 0.0
            avg_l = float(np.mean(all_losses[-100:])) if all_losses else 0.0

            if logger is not None:
                logger.log(ep,
                           reward=ep_reward,
                           avg_reward_50=avg_r,
                           loss=avg_l,
                           epsilon=self.epsilon,
                           buffer=len(self.replay_buffer))

            if ep % 50 == 0:
                print(f"  Ep {ep:4d} | "
                      f"Reward: {avg_r:6.2f} | "
                      f"Loss: {avg_l:.4f} | "
                      f"ε: {self.epsilon:.3f} | "
                      f"Buffer: {len(self.replay_buffer)}")

        print(f"\n✓ Dyna training complete")
        return self.q_net

    def get_intervention(self, z_latent):
        """Get RL recommendation for current crowd state"""
        z = torch.FloatTensor(np.array(z_latent).reshape(1, -1))
        self.q_net.eval()
        return self.q_net.get_full_recommendation(z)
