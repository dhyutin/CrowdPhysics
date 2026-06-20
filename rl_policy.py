# rl_policy.py
"""
Phase 3a: RL Policy
Conservative Q-Learning with Dueling DQN architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ─── ACTION SPACE ─────────────────────────────────────────────────────────────

ACTIONS = {
    0: ("Monitor",         "Continue monitoring. No action needed."),
    1: ("Gate A Open",     "Open exit gate A. Increases east-side capacity."),
    2: ("Gate B Open",     "Open exit gate B. Increases west-side capacity."),
    3: ("Stop Entry",      "Halt all incoming crowd flow immediately."),
    4: ("Redirect Flow",   "Guide crowd away from high-pressure zone."),
    5: ("Dispersal Alert", "Sound audio cue. Signal crowd to spread out."),
    6: ("Emergency",       "Full evacuation. Contact emergency services.")
}
N_ACTIONS = len(ACTIONS)


# ─── Q-NETWORK ────────────────────────────────────────────────────────────────

class CrowdQNetwork(nn.Module):
    """
    Dueling DQN: separate value and advantage streams.

    Why dueling?
    - Value stream: "how dangerous is this situation overall?"
    - Advantage stream: "which action is relatively better?"
    - Combined: Q = V + (A - mean(A))
    - More stable than standard DQN, better at safety-critical decisions

    Input: latent crowd state z (64-dim)
    Output: Q-value per action (7 values)
    """
    def __init__(self, latent_dim=64, n_actions=N_ACTIONS, hidden_dim=256):
        super().__init__()
        self.n_actions = n_actions

        # Shared feature extraction
        self.shared = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 128)
        )

        # Value stream: single scalar
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 1)
        )

        # Advantage stream: one per action
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, n_actions)
        )

    def forward(self, z):
        """
        z: (batch, latent_dim) or (latent_dim,)
        Returns: Q-values (batch, n_actions)
        """
        if z.dim() == 1:
            z = z.unsqueeze(0)

        features = self.shared(z)
        value = self.value_stream(features)          # (batch, 1)
        advantage = self.advantage_stream(features)  # (batch, n_actions)

        # Dueling combination
        q = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return q

    def best_action(self, z):
        """Get best action index"""
        with torch.no_grad():
            q = self.forward(z)
            return int(q.argmax(dim=-1).item())

    def get_full_recommendation(self, z):
        """
        Get complete intervention recommendation with all details.
        This is what goes to Claude for explanation.
        """
        with torch.no_grad():
            q = self.forward(z)
            q_np = q[0].numpy()

        best_idx = int(q_np.argmax())
        probs = F.softmax(torch.FloatTensor(q_np), dim=0).numpy()

        ranked = sorted(
            [(i, float(q_np[i]), ACTIONS[i][0], ACTIONS[i][1])
             for i in range(N_ACTIONS)],
            key=lambda x: -x[1]
        )

        return {
            'action_idx': best_idx,
            'action_name': ACTIONS[best_idx][0],
            'action_description': ACTIONS[best_idx][1],
            'confidence': float(probs[best_idx]),
            'q_values': {
                ACTIONS[i][0]: float(q_np[i]) for i in range(N_ACTIONS)
            },
            'top_3': [
                {
                    'rank': r+1,
                    'action': name,
                    'description': desc,
                    'q_value': round(float(q), 3)
                }
                for r, (i, q, name, desc) in enumerate(ranked[:3])
            ],
            'all_q': q_np.tolist()
        }


# ─── CQL LOSS ─────────────────────────────────────────────────────────────────

def compute_cql_loss(q_net, target_net, batch, gamma=0.99, alpha=0.5):
    """
    Conservative Q-Learning loss.

    = TD loss (standard Q-learning)
    + alpha * CQL penalty (conservative regularization)

    CQL penalty: log(sum(exp(Q(s,a)))) - Q(s, a_taken)
    This penalizes high Q-values for actions not in the dataset.
    Effect: policy only recommends actions it has seen work.
    Perfect for safety-critical systems.

    Args:
        alpha: CQL weight. Higher = more conservative.
               0.5 is good for crowd safety.
    """
    states, actions, rewards, next_states, dones = batch

    # Current Q-values
    q_values = q_net(states)                              # (B, n_actions)
    q_taken = q_values.gather(
        1, actions.unsqueeze(1)
    ).squeeze(1)                                           # (B,)

    # Target Q-values (Double DQN: action from q_net, value from target)
    with torch.no_grad():
        next_actions = q_net(next_states).argmax(dim=1)
        next_q = target_net(next_states).gather(
            1, next_actions.unsqueeze(1)
        ).squeeze(1)
        targets = rewards + gamma * next_q * (1 - dones)

    # TD loss
    td_loss = F.smooth_l1_loss(q_taken, targets)

    # CQL penalty
    cql_penalty = (
        torch.logsumexp(q_values, dim=1) - q_taken
    ).mean()

    total = td_loss + alpha * cql_penalty

    return total, {
        'td_loss': round(float(td_loss), 5),
        'cql_loss': round(float(cql_penalty), 5),
        'total_loss': round(float(total), 5),
        'mean_q': round(float(q_taken.mean()), 4)
    }
