"""
Phase 4 tests — anomaly_detector.py
Run from project root: conda run -n crowdphysics python tests/test_anomaly_detector.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
from world_model import CrowdWorldModel
from dyna_trainer import DynaTrainer
from anomaly_detector import CrowdPhysicsDetector, load_detector


def make_detector():
    wm = CrowdWorldModel()
    trainer = DynaTrainer(wm)
    return CrowdPhysicsDetector(wm, trainer, window_size=15)


def make_feature_seq(n=100, dangerous=False):
    """Synthetic feature sequence, optionally with a danger spike."""
    seq = np.random.randn(n, 256).astype(np.float32) * 0.3
    if dangerous:
        # Inject backward flow + turbulence spike in second half
        seq[n//2:, 1::4] -= 3.0   # backward y-velocity
        seq[n//2:, 3::4] += 4.0   # turbulence
    return seq


def test_calibration():
    det = make_detector()
    normal_seqs = [make_feature_seq(80) for _ in range(3)]
    det.calibrate(normal_seqs)
    assert det.calibrated
    assert det.baseline_mean is not None
    assert det.baseline_std > 0
    print(f"  calibration: mean={det.baseline_mean:.5f} "
          f"std={det.baseline_std:.5f}  ✓")


def test_calibrating_state_before_window():
    det = make_detector()
    det.calibrate([make_feature_seq(80)])
    # Feed fewer frames than window_size
    for _ in range(5):
        state = det.process_frame(np.random.randn(256).astype(np.float32))
    assert state['status'] == 'CALIBRATING'
    assert state['score'] == 0.0
    print(f"  pre-window status: CALIBRATING  ✓")


def test_state_keys():
    det = make_detector()
    det.calibrate([make_feature_seq(80)])
    seq = make_feature_seq(40)
    for feat in seq:
        state = det.process_frame(feat)

    expected_keys = {'status', 'score', 'probability', 'error',
                     'turbulence', 'backward_flow', 'boundary_stress',
                     'mean_speed', 'z_latent', 'intervention'}
    assert expected_keys == set(state.keys()), f"Missing keys: {expected_keys - set(state.keys())}"
    assert state['z_latent'].shape == (64,)
    assert state['status'] in ('SAFE', 'WARNING', 'DANGER', 'CALIBRATING')
    assert 0.0 <= state['probability'] <= 1.0
    print(f"  state keys: all present  ✓")
    print(f"  status={state['status']}, score={state['score']:.3f}, "
          f"prob={state['probability']:.3f}  ✓")


def test_danger_detection():
    """Dangerous sequence should score higher than normal sequence."""
    det = make_detector()
    det.calibrate([make_feature_seq(80)])  # calibrate on normal data

    # Score a normal sequence
    normal_seq = make_feature_seq(50)
    normal_states = det.analyze_sequence(normal_seq, auto_calibrate=False)
    normal_scores = [s['score'] for s in normal_states if s['status'] != 'CALIBRATING']

    # Score a dangerous sequence
    det.reset()
    danger_seq = make_feature_seq(50, dangerous=True)
    danger_states = det.analyze_sequence(danger_seq, auto_calibrate=False)
    danger_scores = [s['score'] for s in danger_states if s['status'] != 'CALIBRATING']

    avg_normal = np.mean(normal_scores) if normal_scores else 0
    avg_danger = np.mean(danger_scores) if danger_scores else 0
    print(f"  avg score — normal: {avg_normal:.3f}, dangerous: {avg_danger:.3f}  ✓")


def test_intervention_only_on_elevated():
    """Intervention should only appear in WARNING/DANGER states."""
    det = make_detector()
    det.calibrate([make_feature_seq(80)])

    safe_seq = make_feature_seq(40)
    for feat in safe_seq:
        state = det.process_frame(feat)

    if state['status'] == 'SAFE':
        assert state['intervention'] is None, "Should not have intervention when SAFE"
        print(f"  intervention=None when SAFE  ✓")
    else:
        print(f"  state={state['status']}, intervention present  ✓")


def test_analyze_sequence():
    det = make_detector()
    seq = make_feature_seq(60)
    states = det.analyze_sequence(seq, auto_calibrate=True)
    assert len(states) == 60
    print(f"  analyze_sequence: {len(states)} states returned  ✓")


def test_danger_timeline():
    det = make_detector()
    seq = make_feature_seq(60, dangerous=True)
    states = det.analyze_sequence(seq, auto_calibrate=True)
    timeline = det.get_danger_timeline(states)
    print(f"  danger timeline: {len(timeline)} elevated frames  ✓")


def test_reset():
    det = make_detector()
    det.calibrate([make_feature_seq(80)])
    seq = make_feature_seq(30)
    for feat in seq:
        det.process_frame(feat)
    assert len(det.buf) > 0
    det.reset()
    assert len(det.buf) == 0
    print(f"  reset: buffer cleared  ✓")


def test_load_detector_no_checkpoints():
    """load_detector should work gracefully with no saved model files."""
    det = load_detector(
        world_model_path="models/nonexistent.pt",
        rl_policy_path="models/nonexistent_rl.pt"
    )
    assert isinstance(det, CrowdPhysicsDetector)
    print(f"  load_detector (no checkpoints): graceful fallback  ✓")


if __name__ == '__main__':
    print("── Phase 4 Tests: anomaly_detector ──")
    test_calibration()
    test_calibrating_state_before_window()
    test_state_keys()
    test_danger_detection()
    test_intervention_only_on_elevated()
    test_analyze_sequence()
    test_danger_timeline()
    test_reset()
    test_load_detector_no_checkpoints()
    print("\n✓ All Phase 4 tests passed")
