# backend/main.py
"""
CrowdPhysics FastAPI backend.

Endpoints:
  POST /api/analyze   — video file → physics timeline + pressure heatmap
  POST /api/simulate  — venue config → pressure simulation + safety report
  GET  /api/discover  — probe world model latent space → Claude hypothesis
  GET  /api/health    — readiness check
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ── resolve project root so we can import sibling modules ─────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Initialize Arize tracing BEFORE importing claude_interpreter (which builds the
# Anthropic client). Auto-instruments every Claude call.
from instrumentation import setup_tracing
setup_tracing()

from flow_extractor import (
    extract_flow,
    extract_farneback_flow,
    flow_to_features,
    render_pressure_field,
)
from world_model import CrowdWorldModel
from dyna_trainer import DynaTrainer
from anomaly_detector import CrowdPhysicsDetector
from claude_interpreter import (
    interpret_live,
    name_discovered_physics,
    explain_rl_decision,
    generate_safety_report,
)
from simulation_engine import VenueConfig, VenueElement, CrowdSimulator

# ── LOAD MODELS ONCE AT STARTUP ───────────────────────────────────────────────

print("[startup] Loading CrowdPhysics models...")
# Architecture must match the GPU-trained checkpoint (hidden=512, layers=3).
_wm      = CrowdWorldModel(hidden_dim=512, n_layers=3)
_trainer = DynaTrainer(_wm)

_wm_path = ROOT / "models" / "world_model.pt"
_rl_path = ROOT / "models" / "rl_policy.pt"

if _wm_path.exists():
    _wm.load_state_dict(torch.load(_wm_path, map_location="cpu"))
    print(f"[startup] ✓ World model: {_wm_path}")
else:
    print("[startup] ⚠  No world model checkpoint — demo mode")

if _rl_path.exists():
    _trainer.q_net.load_state_dict(torch.load(_rl_path, map_location="cpu"))
    print(f"[startup] ✓ RL policy: {_rl_path}")
else:
    print("[startup] ⚠  No RL policy checkpoint — demo mode")

_detector = CrowdPhysicsDetector(_wm, _trainer)

# ── APP ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="CrowdPhysics API",
    description="Crowd fluid dynamics safety platform",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # restrict to your Vercel domain in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── HELPERS ───────────────────────────────────────────────────────────────────

def _frame_to_b64(frame_bgr: np.ndarray) -> str:
    """Encode a BGR cv2 frame as base64 PNG."""
    _, buf = cv2.imencode(".png", frame_bgr)
    return base64.b64encode(buf).decode()


def _numpy_clean(obj: Any) -> Any:
    """Recursively convert numpy types to Python natives for JSON."""
    if isinstance(obj, dict):
        return {k: _numpy_clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_numpy_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def _analyze_frames(frames: list[np.ndarray], fps: float,
                    venue: str) -> dict:
    """
    Run the crowd-physics pipeline over a list of consecutive BGR frames.

    Shared by /api/analyze (frames decoded from an uploaded video) and
    /api/monitor_url (frames captured live from a web page via Browserbase).

    Returns the response dict used by the Monitor tab.
    """
    _detector.buf.clear()

    # ── CALIBRATION ──────────────────────────────────────────────────────────
    # Establish the anomaly baseline on the opening frames (assumed calm) so the
    # σ-above-baseline score is real instead of the uncalibrated error*50 fallback.
    cal_feats = []
    for i in range(1, min(len(frames), 61)):
        f = extract_flow(cv2.resize(frames[i - 1], (320, 240)),
                         cv2.resize(frames[i], (320, 240)))
        cal_feats.append(flow_to_features(f))
    if cal_feats:
        _detector.calibrated = False
        _detector.calibrate([np.array(cal_feats)])
    _detector.buf.clear()

    timeline: list = []
    peak_frame = None
    peak_score = -999.0
    peak_physics = None
    last_claude = "Calibrating..."
    last_rl = ""
    did_claude = False

    prev_frame = frames[0]
    for step, curr in enumerate(frames[1:]):
        sm_curr = cv2.resize(curr, (320, 240))
        sm_prev = cv2.resize(prev_frame, (320, 240))

        flow = extract_flow(sm_prev, sm_curr)
        features = flow_to_features(flow)
        physics = _detector.process_frame(features)

        disp_flow = extract_farneback_flow(
            cv2.resize(prev_frame, (640, 480)),
            cv2.resize(curr, (640, 480)),
        )
        canvas, _ = render_pressure_field(disp_flow, physics,
                                          frame_shape=(480, 640))

        timeline.append({
            "time":        round(step / fps, 1),
            "status":      physics["status"],
            "score":       physics["score"],
            "probability": round(physics["probability"] * 100, 1),
        })

        if physics["score"] > peak_score:
            peak_score = physics["score"]
            peak_frame = canvas.copy()
            peak_physics = {k: v for k, v in physics.items()
                            if k != "z_latent"}

        if physics["status"] != "CALIBRATING" and (not did_claude or step % 60 == 0):
            try:
                last_claude = interpret_live(physics, venue=venue)
                did_claude = True
                if physics.get("intervention"):
                    last_rl = explain_rl_decision(
                        physics["intervention"], physics)
            except Exception as exc:
                last_claude = f"Claude error: {exc}"

        prev_frame = curr

    danger_n = sum(1 for p in timeline if p["status"] == "DANGER")
    total = len(timeline)
    first_danger = next(
        (p["time"] for p in timeline if p["status"] == "DANGER"), None)

    if first_danger is not None:
        summary = (f"DANGER at T+{first_danger}s | "
                   f"Peak: {peak_score:.2f} | "
                   f"Dangerous: {danger_n}/{total} frames")
    else:
        summary = (f"No crush risk | "
                   f"Peak: {peak_score:.2f} | "
                   f"Analyzed {total} frames")

    return {
        "peak_frame_b64": _frame_to_b64(peak_frame) if peak_frame is not None else None,
        "summary":         summary,
        "claude_briefing": last_claude,
        "rl_explanation":  last_rl,
        "timeline":        timeline[-60:],
        "peak_physics":    peak_physics,
    }


# ── GET /api/health ───────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {
        "status":      "ok",
        "world_model": _wm_path.exists(),
        "rl_policy":   _rl_path.exists(),
    }


# ── POST /api/analyze ─────────────────────────────────────────────────────────

@app.post("/api/analyze")
async def analyze(
    video: UploadFile = File(...),
    venue: str        = Form(default="Main Stage"),
):
    """
    Analyze a crowd video file.

    Returns:
      peak_frame_b64  base64 PNG of the highest-anomaly pressure field frame
      summary         one-line text summary
      claude_briefing structured Claude situational awareness
      rl_explanation  Claude explanation of RL policy decision
      timeline        list[{time, status, score, probability}]
      peak_physics    full physics state dict at peak anomaly
    """
    # Save upload to temp file
    suffix = Path(video.filename).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await video.read())
        tmp_path = tmp.name

    try:
        cap = cv2.VideoCapture(tmp_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

        frames: list = []
        while len(frames) < 500:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        if len(frames) < 2:
            raise HTTPException(status_code=400, detail="Cannot read video")

        result = _analyze_frames(frames, fps=fps, venue=venue)

    finally:
        os.unlink(tmp_path)

    return JSONResponse(_numpy_clean(result))


# ── POST /api/monitor_url ─────────────────────────────────────────────────────

class MonitorURLRequest(BaseModel):
    url: str
    venue:    str = "Live Camera"
    n_frames: int = 45


@app.post("/api/monitor_url")
def monitor_url(req: MonitorURLRequest):
    """
    Monitor a live web camera / livestream page via Browserbase.

    Spins up a Browserbase cloud browser, navigates to the page, captures
    rendered frames, and runs them through the same crowd-physics pipeline
    as /api/analyze. Returns the Monitor-tab result plus a `source` block.
    """
    if not os.environ.get("BROWSERBASE_API_KEY") or \
            not os.environ.get("BROWSERBASE_PROJECT_ID"):
        raise HTTPException(
            status_code=400,
            detail="BROWSERBASE_API_KEY / BROWSERBASE_PROJECT_ID not set")

    try:
        from agents.browserbase_monitor import capture_frames
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Browserbase capture unavailable: {exc}")

    n_frames = max(2, min(req.n_frames, 120))
    try:
        frames, fps = capture_frames(req.url, n_frames=n_frames)
    except Exception as exc:
        raise HTTPException(
            status_code=502, detail=f"Browserbase capture failed: {exc}")

    if len(frames) < 2:
        raise HTTPException(
            status_code=502,
            detail=f"Captured only {len(frames)} frame(s) from {req.url}")

    result = _analyze_frames(frames, fps=fps, venue=req.venue)
    result["source"] = {
        "url":             req.url,
        "frames_captured": len(frames),
        "capture_fps":     round(fps, 2),
    }
    return JSONResponse(_numpy_clean(result))


# ── POST /api/simulate ────────────────────────────────────────────────────────

class SimulateRequest(BaseModel):
    venue_name: str = "Demo Arena"
    capacity:   int = 5000
    n_exits:    int = 2
    density:    float = 0.65


@app.post("/api/simulate")
def simulate(req: SimulateRequest):
    """
    Run pre-event crowd physics simulation.

    Returns:
      frame_b64       base64 PNG of pressure heatmap
      metrics         text summary
      safety_report   Claude go/no-go report
      danger_zones    list of high-pressure cells
      safe_capacity   recommended max attendance
      peak_pressure   scalar
    """
    n_exits = max(1, min(req.n_exits, 4))

    config = VenueConfig(
        name=req.venue_name,
        total_capacity=req.capacity,
        elements=[
            VenueElement("stage",   0.2,  0.05, 0.6,  0.22, label="STAGE"),
            VenueElement("wall",    0.0,  0.0,  0.04, 1.0),
            VenueElement("wall",    0.96, 0.0,  0.04, 1.0),
            VenueElement("wall",    0.0,  0.0,  1.0,  0.04),
            VenueElement("wall",    0.0,  0.96, 1.0,  0.04),
            VenueElement("entry",   0.38, 0.87, 0.24, 0.08, label="MAIN ENTRY"),
        ],
    )
    exit_positions = [
        (0.04, 0.45, "EXIT A"),
        (0.88, 0.45, "EXIT B"),
        (0.44, 0.88, "EXIT C"),
        (0.44, 0.04, "EXIT D"),
    ]
    for i in range(n_exits):
        x, y, label = exit_positions[i]
        config.elements.append(
            VenueElement("gate", x, y, 0.08, 0.10, label=label))

    sim = CrowdSimulator(grid_size=20)
    sim.configure_from_venue(config)
    sim.run_steps(n_steps=80, crowd_density=req.density)

    canvas     = sim.render_simulation(size=(480, 640))
    danger     = sim.get_danger_zones(threshold=3.0)
    safe_cap   = sim.estimate_safe_capacity(req.capacity)
    peak_p     = float(sim.pressure.max())

    metrics = (
        f"SIMULATION — {config.name}\n"
        f"Capacity requested : {req.capacity:,}\n"
        f"Safe capacity      : {safe_cap:,}\n"
        f"Peak pressure      : {peak_p:.1f} / 12.0\n"
        f"Danger zones       : {len(danger)}\n"
        f"Exits              : {n_exits}\n"
        + ("⚠  HIGH RISK" if danger else "✓  Layout safe")
    )

    sim_results = {
        "n_danger_zones": len(danger),
        "peak_pressure":  round(peak_p, 2),
        "safe_capacity":  safe_cap,
        "danger_zones":   danger[:5],
        "n_exits":        n_exits,
    }
    venue_info = {
        "name":     config.name,
        "capacity": req.capacity,
        "exits":    n_exits,
    }

    try:
        report = generate_safety_report(venue_info, sim_results)
    except Exception as exc:
        report = f"(Claude unavailable: {exc})\n\nSafe capacity: {safe_cap:,}."

    return JSONResponse(_numpy_clean({
        "frame_b64":     _frame_to_b64(canvas),
        "metrics":       metrics,
        "safety_report": report,
        "danger_zones":  danger[:10],
        "safe_capacity": safe_cap,
        "peak_pressure": peak_p,
    }))


# ── GET /api/discover ─────────────────────────────────────────────────────────

_PROBE_PATH = ROOT / "probe_results.json"

# Fallback used only if probe_latent.py hasn't been run (clearly marked).
_PROBE_FALLBACK = {
    "latent_dim": 64,
    "computed": False,
    "concepts": {
        "crowd_velocity":   {"r2": 0.0, "top_dimensions": [],
                             "description": "Mean crowd movement speed"},
        "turbulence":       {"r2": 0.0, "top_dimensions": [],
                             "description": "Chaotic motion intensity"},
        "backward_pressure": {"r2": 0.0, "top_dimensions": [],
                              "description": "Crowd moving against primary flow"},
        "boundary_stress":  {"r2": 0.0, "top_dimensions": [],
                             "description": "Compression at walls and barriers"},
    },
    "unknown": {"dimensions": [], "separation_z_score": 0.0,
                "verdict": "Run probe_latent.py to compute real values"},
    "table_md": ("| Concept | R² | Key Dimensions | Status |\n|---|---|---|---|\n"
                 "| _probe not yet computed_ | — | — | run `probe_latent.py` |"),
}


@app.get("/api/discover")
def discover():
    """
    Return the REAL linear-probe of the world model's latent space (computed
    by probe_latent.py → probe_results.json) and Claude's hypothesis for the
    unexplained dimensions. Falls back to a clearly-marked placeholder if the
    probe hasn't been run yet.
    """
    if _PROBE_PATH.exists():
        with open(_PROBE_PATH) as f:
            probe = json.load(f)
    else:
        probe = _PROBE_FALLBACK

    # Claude receives the computed concepts + unknown dims.
    claude_probe = {
        "latent_dim": probe.get("latent_dim", 64),
        **probe.get("concepts", {}),
        "unknown": probe.get("unknown", {}),
    }
    try:
        hypothesis = name_discovered_physics(claude_probe)
    except Exception as exc:
        hypothesis = (
            f"(Claude unavailable: {exc})\n\n"
            "The unexplained dimensions likely encode pre-turbulent pressure "
            "fluctuation — the transition from laminar to turbulent crowd "
            "flow that precedes catastrophic compression."
        )

    return {
        "table_md": probe.get("table_md", _PROBE_FALLBACK["table_md"]),
        "hypothesis": hypothesis,
        "computed": probe.get("computed", False),
        "latent_dim": probe.get("latent_dim", 64),
        "unknown": probe.get("unknown", {}),
    }


# ── GET /api/rl_metrics ───────────────────────────────────────────────────────

@app.get("/api/rl_metrics")
def rl_metrics():
    """
    Real RL policy artifacts for the RL tab:
      summary       latest training summary (reward/loss/episodes)
      curve_b64     base64 PNG of the training curves (if available)
      live_sample   live Q-values from the loaded policy on a sampled
                    elevated crowd state (demonstrates the trained network)
    """
    import glob

    summary, curve_b64 = None, None
    runs = sorted(glob.glob(str(ROOT / "logs" / "rl_policy_*")))
    if runs:
        latest = Path(runs[-1])
        s_path, p_path = latest / "summary.json", latest / "curves.png"
        if s_path.exists():
            with open(s_path) as f:
                summary = json.load(f)
        if p_path.exists():
            curve_b64 = base64.b64encode(p_path.read_bytes()).decode()

    # Live readout: ask the trained policy what it would do in an elevated state.
    live_sample = None
    try:
        torch.manual_seed(7)
        z = torch.randn(64) * 1.5
        live_sample = _trainer.get_intervention(z.numpy())
    except Exception as exc:
        live_sample = {"error": str(exc)}

    return JSONResponse(_numpy_clean({
        "summary": summary,
        "curve_b64": curve_b64,
        "live_sample": live_sample,
        "rl_policy_loaded": _rl_path.exists(),
    }))


# ── RUN LOCALLY ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
