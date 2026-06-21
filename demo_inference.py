# demo_inference.py
"""
Quick inference demo for the fine-tuned RAFT crowd-flow model.

Runs the model (models/raft_crowd.pt) on a real crowd video frame pair and
renders a side-by-side panel:  frame  →  optical flow  →  pressure field.

    python demo_inference.py [video_path]

Saves: inference_demo.png
"""

import os
import sys
import time
import cv2
import numpy as np

# This script lives in tests/ — make the repo root importable and use it as
# the base for data/model paths so it runs from anywhere.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raft")
sys.path.insert(0, ROOT)

import flow_extractor as fx


def flow_to_rgb(flow):
    """Color-code optical flow: hue = direction, value = magnitude."""
    h, w = flow.shape[:2]
    fx_, fy_ = flow[..., 0], flow[..., 1]
    mag, ang = cv2.cartToPolar(fx_, fy_, angleInDegrees=True)
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 0] = (ang / 2).astype(np.uint8)            # direction
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def label(img, text):
    bar = img.copy()
    cv2.rectangle(bar, (0, 0), (img.shape[1], 30), (12, 12, 16), -1)
    cv2.addWeighted(bar, 0.7, img, 0.3, 0, img)
    cv2.putText(img, text, (10, 21), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (240, 240, 240), 1, cv2.LINE_AA)
    return img


def main():
    video = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        ROOT, "data/videos",
        "Crowd dynamics experiment： Merging crowds during an evacuation.mp4")

    print(f"[demo] Backend: {fx.FLOW_BACKEND}  |  Device: {fx.DEVICE}")
    print(f"[demo] Video : {video}")

    cap = cv2.VideoCapture(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Jump into the middle of the clip where the crowd is dense/moving.
    start = max(0, total // 2)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    ok1, f1 = cap.read()
    ok2, f2 = cap.read()
    cap.release()
    if not (ok1 and ok2):
        print("Could not read frames from video.")
        sys.exit(1)

    # Resize to a sane working resolution (keep aspect-ish, divisible by 8).
    target_w = 640
    scale = target_w / f1.shape[1]
    size = (target_w, int(round(f1.shape[0] * scale / 8) * 8))
    f1 = cv2.resize(f1, size)
    f2 = cv2.resize(f2, size)

    weights = os.path.join(ROOT, "models/raft_crowd.pt")
    # Warm-up (first call loads weights + compiles kernels), then timed run.
    _ = fx.extract_flow(f1, f2, backend="raft", raft_weights=weights)
    t0 = time.time()
    flow = fx.extract_flow(f1, f2, backend="raft", raft_weights=weights)
    dt = time.time() - t0

    mag = np.linalg.norm(flow, axis=2)
    print(f"\n[inference] frame size      : {size[0]}x{size[1]}")
    print(f"[inference] latency         : {dt*1000:.1f} ms  ({1/dt:.1f} fps)")
    print(f"[inference] flow magnitude  : mean={mag.mean():.3f}  "
          f"max={mag.max():.3f} px")
    print(f"[inference] dominant motion : dx={flow[...,0].mean():+.3f}  "
          f"dy={flow[...,1].mean():+.3f} px/frame")

    # Build the physics panels.
    flow_rgb = flow_to_rgb(flow)
    feats = fx.flow_to_features(flow)
    turbulence = float(feats[3::4].mean())
    score = float(min(1.0, turbulence * 2 + mag.mean() * 0.1))
    status = "DANGER" if score > 0.66 else "WARNING" if score > 0.33 else "SAFE"
    pressure_img, _ = fx.render_pressure_field(
        flow, physics_state={"status": status, "score": score,
                             "probability": score},
        frame_shape=(size[1], size[0]))

    panel = np.hstack([
        label(f1.copy(), "INPUT FRAME"),
        label(flow_rgb, "RAFT OPTICAL FLOW"),
        label(pressure_img, "PRESSURE FIELD"),
    ])
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, "inference_demo.png")
    cv2.imwrite(out, panel)
    print(f"\n[demo] Status: {status}  (score {score:.2f})")
    print(f"[demo] Saved visualization -> {out}")


if __name__ == "__main__":
    main()
