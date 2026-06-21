# backend/main.py
"""
CrowdPhysics FastAPI backend.

Endpoints:
  POST /api/analyze       — video file → physics timeline + forecast + traces
  POST /api/monitor_url   — live web feed (Browserbase) → same as analyze
  POST /api/plan          — venue photo + purpose → layout + sim + agent plan
  POST /api/simulate      — preset venue config → pressure simulation + report
  GET  /api/discover      — probe world model latent space → Claude hypothesis
  GET  /api/health        — readiness check
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


def _load_env_file(path: Path) -> None:
    """
    Minimal .env loader so the API picks up keys (ANTHROPIC, BROWSERBASE, ARIZE)
    without needing the shell to `source .env` first. Supports lines like
    `export KEY="value"` and `KEY=value`; does not overwrite existing env vars.
    """
    try:
        for raw in path.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):]
            if "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val
    except FileNotFoundError:
        pass


_load_env_file(ROOT / ".env")

# Initialize Arize tracing BEFORE importing claude_interpreter (which builds the
# Anthropic client). Auto-instruments every Claude call.
from instrumentation import setup_tracing
setup_tracing()

from flow_extractor import (
    extract_flow,
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
    extract_venue_layout,
    plan_event_layout,
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
    # strict=False: the shipped baseline predates the feat_mean/feat_std buffers,
    # so they stay at identity (0/1) -> standardize() is a no-op (raw features),
    # which is exactly how this checkpoint was trained.
    _wm.load_state_dict(torch.load(_wm_path, map_location="cpu"), strict=False)
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


def _frames_to_gif_b64(frames_rgb: list[np.ndarray], fps: float = 12.0) -> str | None:
    """
    Encode a list of RGB uint8 frames as a looping animated GIF (base64).

    Used to animate the flow-statistics (pressure) field over time so the
    Monitor UI can show the crowd physics evolving rather than a single still.
    Best-effort: returns None on any failure so analysis never breaks.
    """
    if not frames_rgb:
        return None
    try:
        from PIL import Image
        imgs = [Image.fromarray(f) for f in frames_rgb]
        buf = io.BytesIO()
        imgs[0].save(
            buf, format="GIF", save_all=True, append_images=imgs[1:],
            duration=int(1000.0 / max(fps, 1.0)), loop=0, disposal=2,
            optimize=True,
        )
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None


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


def _forecast_future(feat_history: list[np.ndarray], current_prob: float,
                     fps: float, horizon_steps: int = 36) -> dict | None:
    """
    'Potential future of crowds' — roll the world model forward in latent space
    from the most recent observed state and project the crowd's risk trajectory.

    The world model was trained self-supervised to predict the next latent
    state. Here we prime it on the observed history, then autoregress with the
    deterministic mean prediction to imagine the next ~`horizon_steps` frames.
    Each imagined latent is decoded back to flow features; rising crowd
    intensity (speed + turbulence) relative to now scales the projected risk.

    Returns a dict the Monitor UI renders as a forecast (curve + projected
    pressure field + lead-time-to-danger), or None if there isn't enough data.
    """
    if len(feat_history) < 5:
        return None
    try:
        seq = np.asarray(feat_history[-60:], dtype=np.float32)
        x = torch.from_numpy(seq).unsqueeze(0)            # (1, T, 256)
        with torch.no_grad():
            z = _wm.encode_sequence(x)                    # (1, T, 64)
            _wm.transition.hidden = None
            _wm.transition(z, reset_hidden=True)          # prime hidden on history
            cur = z[:, -1:, :]                            # (1, 1, 64)
            future = []
            for _ in range(horizon_steps):
                mu, _lv = _wm.transition(cur, reset_hidden=False)
                cur = mu
                future.append(mu[0, 0])
            fut = torch.stack(future)                     # (H, 64)
            dec = _wm.decoder(fut).cpu().numpy()          # (H, 256)
            base_dec = _wm.decoder(z[:, -1, :]).cpu().numpy()[0]

        def _intensity(f: np.ndarray) -> float:
            mag  = np.abs(f[2::4])                        # mean flow magnitude
            turb = np.abs(f[3::4])                        # turbulence / variance
            return float(mag.mean() + turb.mean())

        p0 = max(_intensity(base_dec), 1e-6)
        base = max(float(current_prob), 0.02) * 100.0

        points, worst, worst_i = [], -1.0, 0
        for i, f in enumerate(dec):
            ratio = _intensity(f) / p0
            risk = float(np.clip(base * ratio, 1.0, 99.0))
            points.append({"t": round((i + 1) / max(fps, 1e-3), 1),
                           "risk": round(risk, 1)})
            if risk > worst:
                worst, worst_i = risk, i

        lead = next((pt["t"] for pt in points if pt["risk"] >= 66.0), None)
        proj_status = ("DANGER" if worst >= 66 else
                       "WARNING" if worst >= 40 else "SAFE")

        # Render the projected (imagined) pressure field at its worst moment by
        # reconstructing a coarse flow field from the decoded means.
        wf = dec[worst_i].reshape(8, 8, 4)
        flow_small = np.zeros((8, 8, 2), dtype=np.float32)
        flow_small[:, :, 0] = wf[:, :, 0]
        flow_small[:, :, 1] = wf[:, :, 1]
        flow_big = cv2.resize(flow_small, (640, 480),
                              interpolation=cv2.INTER_CUBIC) * 6.0
        proj_state = {"status": proj_status,
                      "score": round(worst / 40.0, 2),
                      "probability": worst / 100.0,
                      "turbulence": float(np.abs(wf[:, :, 3]).mean()),
                      "backward_flow": 0.0, "boundary_stress": 0.0,
                      "mean_speed": float(np.abs(wf[:, :, 2]).mean())}
        field, _ = render_pressure_field(flow_big, proj_state,
                                         frame_shape=(480, 640))

        return {
            "points":              points,
            "lead_time_s":         lead,
            "horizon_s":           round(horizon_steps / max(fps, 1e-3), 1),
            "projected_status":    proj_status,
            "projected_risk":      round(worst, 1),
            "projected_field_b64": _frame_to_b64(field),
        }
    except Exception as exc:  # forecast is best-effort; never break analysis
        return {"error": str(exc)}


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
    feat_history: list = []   # decoded flow features per frame (for forecasting)
    field_frames: list = []   # rendered flow-statistics fields (for the GIF)
    peak_flow = None          # cheapest source — render only the peak, once
    peak_score = -999.0
    peak_physics = None
    last_claude = "Calibrating..."
    last_rl = ""
    did_claude = False

    # Animate the flow-statistics field: render a subsampled sequence of
    # pressure-field frames and stitch them into a looping GIF for the UI.
    GIF_STRIDE = 2            # render every Nth frame
    GIF_MAX_FRAMES = 80       # cap GIF length so payload/size stay reasonable

    # ── DETECTION PASS ───────────────────────────────────────────────────────
    # Single optical flow per frame. The full-detail peak render is still done
    # once after the loop; here we also collect a subsampled, lower-res sequence
    # of field frames to animate.
    prev_frame = frames[0]
    for step, curr in enumerate(frames[1:]):
        sm_curr = cv2.resize(curr, (320, 240))
        sm_prev = cv2.resize(prev_frame, (320, 240))

        flow = extract_flow(sm_prev, sm_curr)
        features = flow_to_features(flow)
        feat_history.append(features)
        physics = _detector.process_frame(features)

        timeline.append({
            "time":        round(step / fps, 1),
            "status":      physics["status"],
            "score":       physics["score"],
            "probability": round(physics["probability"] * 100, 1),
        })

        if step % GIF_STRIDE == 0 and len(field_frames) < GIF_MAX_FRAMES:
            f_img, _ = render_pressure_field(
                flow, physics, frame_shape=(240, 320))
            field_frames.append(cv2.cvtColor(f_img, cv2.COLOR_BGR2RGB))

        if physics["score"] > peak_score:
            peak_score = physics["score"]
            peak_flow = flow      # new array each iteration — safe to keep ref
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

    # ── RENDER PEAK FRAME + ANIMATED FLOW FIELD ──────────────────────────────
    peak_frame = None
    if peak_flow is not None:
        peak_frame, _ = render_pressure_field(
            peak_flow, peak_physics, frame_shape=(480, 640))

    # ~12 fps loop regardless of source fps so short/long clips animate nicely.
    flow_gif_b64 = _frames_to_gif_b64(field_frames, fps=12.0)

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

    # ── FORECAST: imagine the next frames in latent space ────────────────────
    cur_prob = (peak_physics.get("probability", 0.0) if peak_physics else 0.0)
    forecast = _forecast_future(feat_history, cur_prob, fps)

    # ── AGENT TRACE: the sequence of agents that produced this result ─────────
    trace = [
        {"agent": "Calibration Agent", "icon": "calibrate",
         "action": "Established calm baseline",
         "detail": f"{len(cal_feats)} opening frames used as normal reference",
         "status": "ok"},
        {"agent": "World Model", "icon": "brain",
         "action": "Encoded crowd into latent physics",
         "detail": f"{total} frames → 64-D self-supervised state space",
         "status": "ok"},
        {"agent": "Anomaly Detector", "icon": "pulse",
         "action": "Scored crush risk per frame",
         "detail": (f"peak {peak_score:.2f}σ · "
                    f"{danger_n}/{total} frames flagged danger"),
         "status": "danger" if danger_n else "ok"},
    ]
    if forecast and not forecast.get("error"):
        lead = forecast.get("lead_time_s")
        trace.append({
            "agent": "Forecast Engine", "icon": "forecast",
            "action": "Projected the crowd's near future",
            "detail": (f"{forecast['horizon_s']}s horizon · "
                       + (f"danger in ~{lead}s" if lead
                          else "no crush projected")),
            "status": "danger" if lead else "ok"})
    if did_claude:
        first_line = (last_claude.split("\n", 1)[0]
                      if isinstance(last_claude, str) else "")
        trace.append({
            "agent": "Claude · Situational Awareness", "icon": "claude",
            "action": "Briefed the operator",
            "detail": first_line[:120],
            "status": "ok"})
    if last_rl:
        iv = (peak_physics or {}).get("intervention") or {}
        trace.append({
            "agent": "RL Policy", "icon": "shield",
            "action": "Recommended an intervention",
            "detail": iv.get("action_name", "intervention selected"),
            "status": "ok"})

    return {
        "peak_frame_b64": _frame_to_b64(peak_frame) if peak_frame is not None else None,
        "flow_gif_b64":    flow_gif_b64,
        "summary":         summary,
        "claude_briefing": last_claude,
        "rl_explanation":  last_rl,
        "timeline":        timeline[-60:],
        "peak_physics":    peak_physics,
        "forecast":        forecast,
        "agent_trace":     trace,
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


# ── LIVE-VIEW SESSIONS (Browserbase) ──────────────────────────────────────────

# session_id -> {"connect_url", "url"} for warm sessions backing the live view.
_live_sessions: dict[str, dict] = {}


def _bb_ready() -> bool:
    return bool(os.environ.get("BROWSERBASE_API_KEY")
                and os.environ.get("BROWSERBASE_PROJECT_ID"))


def _release_live(session_id: str) -> None:
    info = _live_sessions.pop(session_id, None)
    if info is None:
        return
    try:
        from agents.browserbase_monitor import end_session
        end_session(session_id)
    except Exception:
        pass


class LiveSessionRequest(BaseModel):
    url: str


@app.post("/api/live_session")
def live_session(req: LiveSessionRequest):
    """
    Open a Browserbase cloud browser on `url` and return an embeddable
    live-view URL so the frontend can show the feed before analysis.
    """
    if not _bb_ready():
        raise HTTPException(
            status_code=400,
            detail="BROWSERBASE_API_KEY / BROWSERBASE_PROJECT_ID not set")

    try:
        from agents.browserbase_monitor import start_live_session
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Browserbase unavailable: {exc}")

    # Only keep one live preview session at a time.
    for old_id in list(_live_sessions.keys()):
        _release_live(old_id)

    try:
        info = start_live_session(req.url)
    except Exception as exc:
        raise HTTPException(
            status_code=502, detail=f"Live session failed: {exc}")

    sid = info["session_id"]
    _live_sessions[sid] = {"connect_url": info["connect_url"], "url": req.url}
    return {"session_id": sid, "live_view_url": info["live_view_url"]}


class EndSessionRequest(BaseModel):
    session_id: str


@app.post("/api/end_live_session")
def end_live_session(req: EndSessionRequest):
    _release_live(req.session_id)
    return {"ok": True}


# ── POST /api/monitor_url ─────────────────────────────────────────────────────

class MonitorURLRequest(BaseModel):
    url: str
    venue:      str = "Live Camera"
    n_frames:   int = 45
    session_id: str | None = None


@app.post("/api/monitor_url")
def monitor_url(req: MonitorURLRequest):
    """
    Monitor a live web camera / livestream page via Browserbase.

    Captures rendered frames and runs them through the same crowd-physics
    pipeline as /api/analyze. If `session_id` references a warm live-view
    session, it reuses (and then releases) that session for a faster capture;
    otherwise it spins up a fresh session.
    """
    if not _bb_ready():
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
    warm = _live_sessions.get(req.session_id) if req.session_id else None

    try:
        if warm:
            frames, fps = capture_frames(
                req.url, n_frames=n_frames,
                connect_url=warm["connect_url"], navigate=False,
                release_session_id=req.session_id)
            _live_sessions.pop(req.session_id, None)
        else:
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


def _run_venue_simulation(config: VenueConfig, capacity: int,
                          density: float, layout: dict | None = None) -> dict:
    """
    Shared pre-event simulation: configure → run physics → Claude report.

    Used by both /api/simulate (preset layout) and /api/simulate_from_image
    (Claude-vision layout). `layout` is the raw vision extraction echoed back
    to the UI so it can show what was detected.
    """
    n_exits = max(1, len({(round(e.x, 3), round(e.y, 3))
                          for e in config.elements if e.type == "gate"}))

    sim = CrowdSimulator(grid_size=20)
    sim.configure_from_venue(config)
    sim.run_steps(n_steps=80, crowd_density=density)

    canvas   = sim.render_simulation(size=(480, 640))
    danger   = sim.get_danger_zones(threshold=3.0)
    safe_cap = sim.estimate_safe_capacity(capacity)
    peak_p   = float(sim.pressure.max())

    metrics = (
        f"SIMULATION — {config.name}\n"
        f"Capacity requested : {capacity:,}\n"
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
    venue_info = {"name": config.name, "capacity": capacity, "exits": n_exits}
    if layout:
        venue_info["layout_notes"] = layout.get("notes", "")
        venue_info["view"] = layout.get("view", "")

    try:
        report = generate_safety_report(venue_info, sim_results)
    except Exception as exc:
        report = f"(Claude unavailable: {exc})\n\nSafe capacity: {safe_cap:,}."

    result = {
        "frame_b64":     _frame_to_b64(canvas),
        "metrics":       metrics,
        "safety_report": report,
        "danger_zones":  danger[:10],
        "safe_capacity": safe_cap,
        "peak_pressure": peak_p,
        "n_exits":       n_exits,
        "venue_name":    config.name,
    }
    if layout is not None:
        result["layout"] = layout
    return result


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

    return JSONResponse(_numpy_clean(
        _run_venue_simulation(config, req.capacity, req.density)))


_VISION_MEDIA = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
    ".webp": "image/webp", ".gif": "image/gif",
}


@app.post("/api/simulate_from_image")
async def simulate_from_image(
    image: UploadFile = File(...),
    capacity: int = Form(default=0),
    density: float = Form(default=0.65),
):
    """
    Build a venue from a photo / satellite image / floor plan, then run the
    pre-event simulation on it.

    Claude vision extracts a top-down layout (stage / walls / barriers /
    entries / exits) which feeds the SAME CrowdSimulator as /api/simulate.

    Returns the simulate payload PLUS `layout` (the detected elements,
    inferred name, capacity, view and confidence) so the UI can show what
    was recognised.
    """
    suffix = Path(image.filename or "").suffix.lower()
    media_type = _VISION_MEDIA.get(suffix, "image/jpeg")

    raw = await image.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty image upload")
    image_b64 = base64.b64encode(raw).decode()

    cap_hint = capacity if capacity and capacity > 0 else None

    try:
        layout = extract_venue_layout(image_b64, media_type, capacity_hint=cap_hint)
    except Exception as exc:
        raise HTTPException(
            status_code=502, detail=f"Vision layout extraction failed: {exc}")

    elements = [
        VenueElement(
            type=e["type"], x=e["x"], y=e["y"], w=e["w"], h=e["h"],
            label=e.get("label", ""),
        )
        for e in layout["elements"]
    ]
    final_capacity = cap_hint or layout["capacity"]
    config = VenueConfig(
        name=layout["name"],
        total_capacity=final_capacity,
        elements=elements,
    )

    result = _run_venue_simulation(config, final_capacity, density, layout=layout)
    return JSONResponse(_numpy_clean(result))


# ── POST /api/plan ────────────────────────────────────────────────────────────

@app.post("/api/plan")
async def plan(
    image:    UploadFile = File(...),
    purpose:  str   = Form(default="general gathering"),
    capacity: int   = Form(default=0),
    density:  float = Form(default=0.65),
):
    """
    Plan mode: photo of a place → virtual simulation → agentic arrangement plan.

    1. Claude vision reconstructs a top-down venue layout from the photo.
    2. The crowd fluid-dynamics simulator finds the danger zones.
    3. A planning agent designs how to arrange people/flow/staff for the stated
       `purpose`, grounded in the simulation.

    Returns the simulate payload PLUS `layout`, `plan`, `purpose` and
    `agent_trace` so the Plan UI can show the full agent reasoning.
    """
    suffix = Path(image.filename or "").suffix.lower()
    media_type = _VISION_MEDIA.get(suffix, "image/jpeg")

    raw = await image.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty image upload")
    image_b64 = base64.b64encode(raw).decode()
    cap_hint = capacity if capacity and capacity > 0 else None

    try:
        layout = extract_venue_layout(image_b64, media_type, capacity_hint=cap_hint)
    except Exception as exc:
        raise HTTPException(
            status_code=502, detail=f"Vision layout extraction failed: {exc}")

    elements = [
        VenueElement(type=e["type"], x=e["x"], y=e["y"], w=e["w"], h=e["h"],
                     label=e.get("label", ""))
        for e in layout["elements"]
    ]
    final_capacity = cap_hint or layout["capacity"]
    config = VenueConfig(name=layout["name"],
                         total_capacity=final_capacity, elements=elements)

    result = _run_venue_simulation(config, final_capacity, density, layout=layout)

    sim_results = {
        "n_danger_zones": result["danger_zones"] and len(result["danger_zones"]),
        "peak_pressure":  result["peak_pressure"],
        "safe_capacity":  result["safe_capacity"],
        "n_exits":        result["n_exits"],
        "danger_zones":   result["danger_zones"][:5],
    }
    try:
        plan_text = plan_event_layout(layout, sim_results, purpose, final_capacity)
    except Exception as exc:
        plan_text = f"(Planning agent unavailable: {exc})"

    n_danger = len(result["danger_zones"])
    result["plan"] = plan_text
    result["purpose"] = purpose
    result["agent_trace"] = [
        {"agent": "Vision Surveyor", "icon": "eye",
         "action": "Reconstructed top-down layout",
         "detail": (f"{len(elements)} elements · "
                    f"{int(round(layout.get('confidence', 0) * 100))}% confidence"),
         "status": "ok"},
        {"agent": "Crowd Simulator", "icon": "pulse",
         "action": "Ran fluid-dynamics simulation",
         "detail": (f"peak pressure {result['peak_pressure']:.1f} · "
                    f"{n_danger} danger zones"),
         "status": "danger" if n_danger else "ok"},
        {"agent": "Safety Analyst · Claude", "icon": "claude",
         "action": "Generated pre-event safety report",
         "detail": f"safe capacity {result['safe_capacity']:,}",
         "status": "ok"},
        {"agent": "Event Planner · Claude", "icon": "plan",
         "action": "Designed arrangement for purpose",
         "detail": f"optimized for: {purpose}",
         "status": "ok"},
    ]
    return JSONResponse(_numpy_clean(result))


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
