"""
Phase 3 tests — rl_policy.py + dyna_trainer.py
Run from project root: conda run -n crowdphysics python tests/test_rl_policy.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from rl_policy import CrowdQNetwork, compute_cql_loss, ACTIONS, N_ACTIONS
from dyna_trainer import DynaTrainer
from world_model import CrowdWorldModel


def test_q_network_shapes():
    q = CrowdQNetwork()
    z = torch.randn(4, 64)
    out = q(z)
    assert out.shape == (4, N_ACTIONS), f"Expected (4,{N_ACTIONS}), got {out.shape}"
    print(f"  Q-network: {z.shape} → {out.shape}  ✓")


def test_q_network_single():
    q = CrowdQNetwork()
    z = torch.randn(64)           # single state, no batch dim
    out = q(z)
    assert out.shape == (1, N_ACTIONS)
    print(f"  Q-network single: {z.shape} → {out.shape}  ✓")


def test_best_action():
    q = CrowdQNetwork()
    z = torch.randn(64)
    action = q.best_action(z)
    assert 0 <= action < N_ACTIONS
    print(f"  best_action: {action} ({ACTIONS[action][0]})  ✓")


def test_full_recommendation():
    q = CrowdQNetwork()
    z = torch.randn(1, 64) * 2.5  # dangerous-ish state
    rec = q.get_full_recommendation(z)
    assert 'action_name' in rec
    assert 'confidence' in rec
    assert len(rec['top_3']) == 3
    assert len(rec['q_values']) == N_ACTIONS
    print(f"  recommendation: '{rec['action_name']}' "
          f"({rec['confidence']*100:.0f}% conf)  ✓")
    print(f"  top_3: {[t['action'] for t in rec['top_3']]}  ✓")


def test_cql_loss():
    q_net = CrowdQNetwork()
    target_net = CrowdQNetwork()
    target_net.load_state_dict(q_net.state_dict())

    B = 16
    batch = (
        torch.randn(B, 64),
        torch.randint(0, N_ACTIONS, (B,)),
        torch.randn(B),
        torch.randn(B, 64),
        torch.zeros(B)
    )
    loss, metrics = compute_cql_loss(q_net, target_net, batch)
    assert not torch.isnan(loss), "CQL loss is NaN"
    assert 'td_loss' in metrics and 'cql_loss' in metrics
    print(f"  CQL loss: td={metrics['td_loss']}, "
          f"cql={metrics['cql_loss']}  ✓")


def test_dyna_episode():
    wm = CrowdWorldModel()
    trainer = DynaTrainer(wm)
    ep = trainer.generate_episode(episode_len=20)
    assert len(ep) > 0
    action, d_before, d_after, reward = ep[0]
    assert 0 <= action < N_ACTIONS
    print(f"  episode len: {len(ep)}  ✓")
    print(f"  step 0: action={action} ({ACTIONS[action][0]}), "
          f"danger {d_before:.2f}→{d_after:.2f}, reward={reward:.2f}  ✓")


def test_dyna_train_step():
    wm = CrowdWorldModel()
    trainer = DynaTrainer(wm)

    # Fill buffer enough to train
    for _ in range(20):
        trainer.generate_episode(episode_len=10)

    metrics = trainer.train_step(batch_size=32)
    assert metrics is not None
    assert not np.isnan(metrics['total_loss'])
    print(f"  train_step loss: {metrics['total_loss']:.4f}  ✓")


def test_get_intervention():
    wm = CrowdWorldModel()
    trainer = DynaTrainer(wm)
    z = torch.randn(1, 64).numpy() * 2.5
    rec = trainer.get_intervention(z)
    assert 'action_name' in rec
    print(f"  intervention: '{rec['action_name']}' @ "
          f"{rec['confidence']*100:.0f}% conf  ✓")


if __name__ == '__main__':
    print("── Phase 3 Tests: rl_policy + dyna_trainer ──")
    test_q_network_shapes()
    test_q_network_single()
    test_best_action()
    test_full_recommendation()
    test_cql_loss()
    test_dyna_episode()
    test_dyna_train_step()
    test_get_intervention()
    print("\n✓ All Phase 3 tests passed")
