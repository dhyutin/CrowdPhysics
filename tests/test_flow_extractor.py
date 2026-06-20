"""
Phase 1 tests — flow_extractor.py
Run from project root: conda run -n crowdphysics python tests/test_flow_extractor.py
Output image saved to tests/test_pressure_field.png
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import cv2
import numpy as np
from flow_extractor import (
    extract_farneback_flow,
    flow_to_features,
    render_pressure_field,
    process_video_to_features,
)

OUTPUT_DIR = os.path.dirname(__file__)


def test_flow_shape():
    frame1 = np.random.randint(80, 180, (480, 640, 3), dtype=np.uint8)
    frame2 = np.roll(frame1, 3, axis=1)
    flow = extract_farneback_flow(frame1, frame2)
    assert flow.shape == (480, 640, 2), f"Expected (480,640,2), got {flow.shape}"
    print(f"  flow shape:     {flow.shape}  ✓")


def test_features_shape():
    frame1 = np.random.randint(80, 180, (480, 640, 3), dtype=np.uint8)
    frame2 = np.roll(frame1, 3, axis=1)
    flow = extract_farneback_flow(frame1, frame2)
    features = flow_to_features(flow)
    assert features.shape == (256,), f"Expected (256,), got {features.shape}"
    assert features.dtype == np.float32
    print(f"  features shape: {features.shape}  ✓")


def test_rightward_flow_has_positive_fx():
    """Shifting right should produce positive mean x-velocity in all cells."""
    frame1 = np.random.randint(80, 180, (480, 640, 3), dtype=np.uint8)
    frame2 = np.roll(frame1, 5, axis=1)  # shift right
    flow = extract_farneback_flow(frame1, frame2)
    features = flow_to_features(flow)
    # x-velocity is every 4th feature starting at index 0
    x_vels = features[0::4]
    assert x_vels.mean() > 0, "Expected positive x-drift for rightward shift"
    print(f"  rightward flow mean x-vel: {x_vels.mean():.3f}  ✓")


def test_pressure_field_render():
    frame1 = np.random.randint(80, 180, (480, 640, 3), dtype=np.uint8)
    frame2 = np.roll(frame1, 3, axis=1)
    flow = extract_farneback_flow(frame1, frame2)

    canvas, grid = render_pressure_field(
        flow,
        {'status': 'WARNING', 'score': 1.8, 'probability': 0.4},
        frame_shape=(480, 640)
    )
    assert canvas.shape == (480, 640, 3), f"Expected (480,640,3), got {canvas.shape}"
    assert grid.shape == (8, 8), f"Expected (8,8) pressure grid, got {grid.shape}"

    out_path = os.path.join(OUTPUT_DIR, 'test_pressure_field.png')
    cv2.imwrite(out_path, canvas)
    print(f"  canvas shape:   {canvas.shape}  ✓")
    print(f"  pressure grid:  {grid.shape}  ✓")
    print(f"  saved → {out_path}")


def test_process_video_to_features_synthetic():
    """Generate a tiny synthetic video and check feature extraction."""
    tmp_path = os.path.join(OUTPUT_DIR, '_tmp_test.mp4')
    out = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*'mp4v'), 25, (320, 240))
    base = np.random.randint(80, 180, (240, 320, 3), dtype=np.uint8)
    for i in range(10):
        frame = np.roll(base, i * 2, axis=1).astype(np.uint8)
        out.write(frame)
    out.release()

    feats = process_video_to_features(tmp_path)
    assert feats.ndim == 2, f"Expected 2D array, got {feats.ndim}D"
    assert feats.shape[1] == 256, f"Expected 256 features, got {feats.shape[1]}"
    print(f"  video features: {feats.shape}  ✓")
    os.remove(tmp_path)


if __name__ == '__main__':
    print("── Phase 1 Tests: flow_extractor ──")
    test_flow_shape()
    test_features_shape()
    test_rightward_flow_has_positive_fx()
    test_pressure_field_render()
    test_process_video_to_features_synthetic()
    print("\n✓ All Phase 1 tests passed")
