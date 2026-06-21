"""Quick smoke test: verify all app.py imports and simulation engine work."""
import sys
sys.path.insert(0, '.')

import gradio as gr
from flow_extractor import extract_farneback_flow, flow_to_features, render_pressure_field
from world_model import CrowdWorldModel
from dyna_trainer import DynaTrainer
from anomaly_detector import CrowdPhysicsDetector
from simulation_engine import VenueConfig, VenueElement, CrowdSimulator, DEFAULT_VENUE
print("All imports OK")

# Simulation engine test
sim = CrowdSimulator(grid_size=20)
sim.configure_from_venue(DEFAULT_VENUE)
sim.run_steps(30, 0.6)
zones  = sim.get_danger_zones()
cap    = sim.estimate_safe_capacity(8000)
feats  = sim.to_features()
canvas = sim.render_simulation()
print(f"Simulation: {len(zones)} danger zones | safe cap {cap:,} | features {feats.shape} | canvas {canvas.shape}")

# World model + detector smoke test
wm      = CrowdWorldModel()
trainer = DynaTrainer(wm)
det     = CrowdPhysicsDetector(wm, trainer)
import numpy as np
for _ in range(35):
    f = np.random.randn(256).astype("float32") * 0.1
    state = det.process_frame(f)
print(f"Detector: status={state['status']} score={state['score']:.3f}")
print("ALL TESTS PASSED — app.py is ready to launch")
