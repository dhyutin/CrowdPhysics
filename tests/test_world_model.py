"""
Phase 2 tests — world_model.py
Run from project root: conda run -n crowdphysics python tests/test_world_model.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from world_model import (
    CrowdEncoder,
    CrowdDecoder,
    CrowdTransitionModel,
    CrowdWorldModel,
)


def test_encoder_shape():
    enc = CrowdEncoder()
    x = torch.randn(4, 256)       # batch of 4 single frames
    z = enc(x)
    assert z.shape == (4, 64), f"Expected (4,64), got {z.shape}"
    print(f"  encoder:    {x.shape} → {z.shape}  ✓")


def test_decoder_shape():
    dec = CrowdDecoder()
    z = torch.randn(4, 64)
    recon = dec(z)
    assert recon.shape == (4, 256), f"Expected (4,256), got {recon.shape}"
    print(f"  decoder:    {z.shape} → {recon.shape}  ✓")


def test_transition_shapes():
    trans = CrowdTransitionModel()
    z_seq = torch.randn(2, 30, 64)   # batch=2, seq=30, latent=64
    mu, log_var = trans(z_seq)
    assert mu.shape == (2, 30, 64), f"Expected mu (2,30,64), got {mu.shape}"
    assert log_var.shape == (2, 30, 64)
    print(f"  transition: {z_seq.shape} → mu {mu.shape}, log_var {log_var.shape}  ✓")


def test_world_model_forward():
    model = CrowdWorldModel()
    x = torch.FloatTensor(np.random.randn(1, 31, 256))
    mu, log_var, z_target, z, recon = model(x)

    assert x.shape      == (1, 31, 256), f"Input wrong: {x.shape}"
    assert z.shape      == (1, 31,  64), f"z wrong: {z.shape}"
    assert mu.shape     == (1, 30,  64), f"mu wrong: {mu.shape}"
    assert z_target.shape == (1, 30, 64), f"z_target wrong: {z_target.shape}"
    assert recon.shape  == (1, 31, 256), f"recon wrong: {recon.shape}"

    print(f"  input:    {x.shape}")
    print(f"  z:        {z.shape}  ✓")
    print(f"  mu:       {mu.shape}  ✓")
    print(f"  z_target: {z_target.shape}  ✓")
    print(f"  recon:    {recon.shape}  ✓")


def test_rollout():
    model = CrowdWorldModel()
    z_start = torch.randn(64)
    traj = model.rollout_latent(z_start, n_steps=10)
    assert traj.shape == (11, 64), f"Expected (11,64), got {traj.shape}"
    print(f"  rollout:  {traj.shape}  ✓")


def test_loss_backward():
    """Make sure the training loss can actually backprop."""
    import torch.nn as nn
    model = CrowdWorldModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    x = torch.FloatTensor(np.random.randn(2, 16, 256))
    mu, log_var, z_target, z, recon = model(x)

    trans_loss = nn.MSELoss()(mu, z_target)
    log_var_c  = log_var.clamp(-6, 2)
    kl_loss    = -0.5 * torch.mean(1 + log_var_c - mu.pow(2) - log_var_c.exp())
    recon_loss = nn.MSELoss()(recon[:, :-1], x[:, 1:].detach())
    loss       = trans_loss + 0.005 * kl_loss + 0.1 * recon_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert not torch.isnan(loss), "Loss is NaN"
    print(f"  loss backward: {loss.item():.5f}  ✓")


if __name__ == '__main__':
    print("── Phase 2 Tests: world_model ──")
    test_encoder_shape()
    test_decoder_shape()
    test_transition_shapes()
    test_world_model_forward()
    test_rollout()
    test_loss_backward()
    print("\n✓ All Phase 2 tests passed")
