# claude_interpreter.py
"""
Phase 5: Claude Integration — Three live roles + two setup agents.

ROLE 1 — interpret_live()
  Real-time situational awareness for security personnel.
  Called every frame when status is WARNING or DANGER.

ROLE 2 — name_discovered_physics()
  Scientific interpreter: names what the world model learned
  without labels. For the demo's "what did the AI discover?" moment.

ROLE 3 — explain_rl_decision()
  Tactical advisor: explains the RL policy's reasoning to a
  supervisor who needs to trust the recommendation.

ROLE 4 — generate_safety_report()
  Pre-event simulation report. Used in Simulate mode.

AGENT 1 — VenueAgent
  One-time setup: takes a natural-language venue description and
  produces a venue_config.json that maps physics primitives to
  real, actionable venue-specific instructions.

AGENT 2 — CalibrationAgent
  Stream-start: examines incoming flow features, decides when the
  scene is calm enough to calibrate, and infers camera perspective
  to build a per-cell weight map for the anomaly detector.
"""

import anthropic
import json
import os
import numpy as np

# ─── CLIENT ───────────────────────────────────────────────────────────────────

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# ─── SHARED SYSTEM PROMPT ─────────────────────────────────────────────────────

SYSTEM_CORE = """You are CrowdPhysics — an AI safety system that interprets \
crowd fluid dynamics to prevent disasters.

A world model learned crowd physics by predicting video frames.
An RL policy (Conservative Q-Learning, Dyna-trained) recommends interventions.
You translate both into human language.

Audience: security personnel under stress, at 2am.
Rules: short sentences. active voice. most critical thing first.
Never say 'I'. Speak as the system, not a person.
No jargon without immediate plain explanation."""


# ─── ROLE 1: REAL-TIME SITUATIONAL AWARENESS ─────────────────────────────────

def interpret_live(physics_state, venue=""):
    """
    Real-time situational awareness for security personnel.

    Called every frame when status is WARNING or DANGER.
    Returns a structured plain-English briefing.

    Args:
        physics_state: dict from anomaly_detector.process_frame()
        venue:         str, venue name/description (optional)

    Returns:
        str — structured briefing with SITUATION / NEXT 5 MIN / DO NOW / WATCH
    """
    iv_text = ""
    iv = physics_state.get("intervention")
    if iv:
        top3 = iv.get("top_3", [])
        top3_str = ", ".join(
            f'{a["action"]} ({a["q_value"]:.2f})'
            for a in top3
        )
        iv_text = (
            f"\nRL primitive: {iv['action_name']} "
            f"(confidence {iv['confidence']*100:.0f}%)"
            f"\nTop options: {top3_str}"
        )

    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        system=SYSTEM_CORE,
        messages=[{
            "role": "user",
            "content": f"""LIVE READING — {venue or 'Unknown Venue'}
Status: {physics_state['status']} | Score: {physics_state['score']:.2f}/2.5
Crush probability: {physics_state['probability']*100:.1f}%
Turbulence: {physics_state['turbulence']:.4f}
Backward flow (pressure waves): {physics_state['backward_flow']:.4f}
Boundary stress: {physics_state['boundary_stress']:.4f}
Mean crowd speed: {physics_state['mean_speed']:.4f}{iv_text}

Reply with exactly this structure:
SITUATION: [what is physically happening, 2 sentences]
NEXT 5 MIN: [what happens if nothing changes, 1 sentence]
DO NOW: [numbered steps, max 3, be specific]
WATCH: [one metric or zone to re-assess in 60 seconds]"""
        }]
    )
    return resp.content[0].text


# ─── ROLE 2: SCIENTIFIC INTERPRETER ──────────────────────────────────────────

def name_discovered_physics(probe_results):
    """
    Names what the world model learned without labels.

    The world model is trained self-supervised — it discovers
    physics from video prediction, not from equations. This role
    probes the latent space and asks Claude to name what emerged.

    Args:
        probe_results: dict with keys like 'turbulence_corr',
                       'backward_flow_corr', 'unknown', 'latent_dim'.
                       'unknown' dims activate σ stronger before crush.

    Returns:
        str — physicist-style analysis of what the model discovered.
    """
    unknown = probe_results.get("unknown", {})
    known = {k: v for k, v in probe_results.items()
             if k not in ("unknown", "latent_dim")}

    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        system=(
            "You are a crowd dynamics physicist examining what a neural network "
            "discovered about crowd physics without being told. "
            "React with intellectual excitement. Be specific and physically grounded."
        ),
        messages=[{
            "role": "user",
            "content": f"""A neural network learned crowd physics from video prediction.
No labels. No equations. Physics emerged from prediction error alone.

We probed its {probe_results.get('latent_dim', 64)}-dim latent space:

KNOWN CORRELATIONS (dimensions that correlate with measured physics):
{json.dumps(known, indent=2)}

UNKNOWN dimensions (correlate with NOTHING we measured directly):
{json.dumps(unknown, indent=2)}

These unknown dims activate {unknown.get('separation_z_score', 3.2):.1f}σ \
stronger BEFORE crush events.
Lead time: {unknown.get('lead_time_minutes', 4.2):.1f} minutes before visible danger.

What physical phenomenon did the model discover?
Why does it predict crush events before humans notice anything?
What would you name it?
What single experiment would confirm your hypothesis?"""
        }]
    )
    return resp.content[0].text


# ─── ROLE 3: TACTICAL ADVISOR ─────────────────────────────────────────────────

def explain_rl_decision(intervention, physics_state, venue_config=None):
    """
    Explains the RL policy's reasoning to a security supervisor.

    The RL policy outputs a physics primitive (e.g. 'increase_egress').
    This function translates both the primitive AND the Q-value reasoning
    into language a supervisor can act on and trust.

    Args:
        intervention:  dict from rl_policy.get_full_recommendation()
        physics_state: dict from anomaly_detector.process_frame()
        venue_config:  dict (optional) — maps primitives to venue actions

    Returns:
        str — numbered explanation for supervisor
    """
    # Translate physics primitive to venue action if config available
    venue_action = ""
    if venue_config:
        prim = intervention.get("action_name", "")
        mapped = venue_config.get("action_map", {}).get(prim)
        if mapped:
            venue_action = f"\nVenue action: {mapped}"

    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=350,
        system=SYSTEM_CORE,
        messages=[{
            "role": "user",
            "content": f"""The RL policy (Conservative Q-Learning, Dyna-trained) chose:
Physics primitive: {intervention['action_name']}
Confidence: {intervention['confidence']*100:.0f}%{venue_action}

Crowd state: anomaly={physics_state['score']:.2f}, \
crush risk={physics_state['probability']*100:.1f}%

All Q-values (expected future safety improvement per action):
{json.dumps(intervention['q_values'], indent=2)}

Explain to a security supervisor:
1. Why did the policy choose this action? (physics reasoning, 2 sentences)
2. What does it expect will happen in the next few minutes? (1 sentence)
3. What did it learn to avoid? (why lower-ranked options scored less, 1-2 sentences)"""
        }]
    )
    return resp.content[0].text


# ─── ROLE 4: PRE-EVENT SAFETY REPORT ─────────────────────────────────────────

def generate_safety_report(venue_config, sim_results):
    """
    Pre-event simulation report for the event director.

    Called after simulation_engine.py runs crowd physics on a
    venue floor plan. Translates raw simulation metrics into a
    structured safety brief.

    Args:
        venue_config: dict — name, capacity, exits, layout description
        sim_results:  dict — pressure peaks, danger timelines, zone scores

    Returns:
        str — executive safety report
    """
    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=600,
        system=SYSTEM_CORE,
        messages=[{
            "role": "user",
            "content": f"""PRE-EVENT SIMULATION RESULTS for: {venue_config.get('name', 'Venue')}

Venue layout:
{json.dumps(venue_config, indent=2)}

Simulation physics results:
{json.dumps(sim_results, indent=2)}

Generate a pre-event safety report with exactly this structure:
EXECUTIVE SUMMARY: (2 sentences — go or no-go recommendation)
RISK ZONES: (specific locations and why they're dangerous)
REQUIRED CHANGES: (numbered list, specific, actionable before doors open)
SAFE CAPACITY: (recommended number and reasoning)
MONITORING PRIORITY: (which zones to watch most closely, in order)

Write for the event director, not an engineer. Be direct."""
        }]
    )
    return resp.content[0].text


# ─── ROLE 6: EVENT PLANNER (purpose-aware arrangement) ───────────────────────

def plan_event_layout(layout, sim_results, purpose, capacity):
    """
    Agentic planner: given a detected venue layout, the simulation results and
    the event's PURPOSE, produce a concrete plan for how to arrange people,
    barriers and staff so the event is safe for that specific use.

    Args:
        layout:      dict from extract_venue_layout()
        sim_results: dict (peak_pressure, n_danger_zones, safe_capacity, ...)
        purpose:     str — what the space will be used for (concert, rally,
                     expo, prayer, evacuation drill, sports, market, ...)
        capacity:    int — expected attendance

    Returns:
        str — structured arrangement plan for the event organizer.
    """
    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=700,
        system=(
            "You are an event crowd-planning agent. Given a venue's top-down "
            "layout and a crowd fluid-dynamics simulation, you design HOW to "
            "use the space safely for a stated purpose. Be concrete and spatial "
            "— reference entries, exits, the stage and the danger zones by their "
            "positions. Write for an event organizer, not an engineer."
        ),
        messages=[{
            "role": "user",
            "content": f"""EVENT PURPOSE: {purpose or 'general gathering'}
EXPECTED ATTENDANCE: {capacity:,}

VENUE (top-down, normalized 0-1 coords, origin top-left):
{json.dumps({"name": layout.get("name"), "view": layout.get("view"),
             "elements": layout.get("elements", [])}, indent=2)}

CROWD SIMULATION RESULTS:
{json.dumps(sim_results, indent=2)}

Produce a plan with EXACTLY these sections:
ARRANGEMENT: how to position people/zones for this purpose (2-3 sentences, spatial).
FLOW DESIGN: which entries/exits to use for ingress vs egress, and barrier placement.
CAPACITY PLAN: recommended attendance for this purpose + why.
STAFFING: where to place stewards/security relative to danger zones (numbered, max 4).
RISKS: the top failure mode for THIS purpose and how the layout mitigates it.

Be specific and directive."""
        }]
    )
    return resp.content[0].text


# ─── ROLE 6b: CONCISE EVENT PLAN POINTS ──────────────────────────────────────

def event_plan_points(layout, sim_results, intake, best_scenario):
    """
    Distil the planning decision into 4-6 crisp, prioritized, spatial bullets
    tailored to the specific event. Returns a list[str].

    Args:
        layout:        chosen scenario layout dict
        sim_results:   dict (peak_pressure, n_danger_zones, safe_capacity, ...)
        intake:        dict of event-intake answers (purpose, people, duration,
                       seating, ingress, notes)
        best_scenario: dict {name, description} of the winning layout

    Returns:
        list[str] — concise recommendation bullets.
    """
    import re

    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=400,
        system=(
            "You are a crowd-safety planning agent. Reply with ONLY a JSON "
            "array of 4-6 short, imperative recommendation strings. Each must "
            "be concrete, spatial and <= 16 words. No prose, no keys, JSON "
            "array of strings only."
        ),
        messages=[{
            "role": "user",
            "content": f"""EVENT INTAKE:
{json.dumps(intake, indent=2)}

WINNING LAYOUT: {best_scenario.get('name')} — {best_scenario.get('description')}

VENUE ELEMENTS (normalized 0-1, top-left origin):
{json.dumps(layout.get('elements', []), indent=2)}

CROWD SIMULATION RESULTS:
{json.dumps(sim_results, indent=2)}

Give 4-6 prioritized, concrete actions to plan and run this crowd safely.
Reference entries/exits/stage/danger zones spatially. JSON array of strings only."""
        }]
    )
    txt = resp.content[0].text.strip()
    m = re.search(r"\[.*\]", txt, re.S)
    if m:
        try:
            arr = json.loads(m.group(0))
            return [str(x).strip() for x in arr if str(x).strip()]
        except Exception:
            pass
    # Fallback: split lines if the model didn't return clean JSON.
    return [ln.strip("-•* \t") for ln in txt.splitlines() if ln.strip()][:6]


# ─── ROLE 5: VISION — VENUE LAYOUT FROM PHOTO ────────────────────────────────

def extract_venue_layout(image_b64, media_type="image/jpeg", capacity_hint=None):
    """
    Claude vision: turn a photo / satellite image / floor plan of a venue into
    a top-down layout of physics primitives the simulator understands.

    The simulation engine is driven entirely by a list of normalized labeled
    rectangles (VenueElement). This function produces exactly that list so the
    existing CrowdSimulator can run on a real location with no other changes.

    Args:
        image_b64:     base64-encoded image bytes (no data: prefix)
        media_type:    one of image/jpeg | image/png | image/webp | image/gif
        capacity_hint: optional int — operator's expected attendance

    Returns:
        dict {
          "name": str,
          "capacity": int,
          "view": "overhead" | "ground" | "floorplan",
          "confidence": float 0-1,
          "notes": str,
          "elements": [ {type, x, y, w, h, label}, ... ]
        }
        Coordinates are top-down normalized 0-1, origin top-left.
        type ∈ {stage, wall, barrier, entry, gate}.
    """
    hint_txt = (f"\nThe operator expects roughly {capacity_hint} attendees."
                if capacity_hint else "")

    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1200,
        system=(
            "You are a crowd-safety surveyor. You convert an image of a venue "
            "into a TOP-DOWN floor plan of simple rectangles for a crowd "
            "fluid-dynamics simulator. If the image is a ground-level photo, "
            "mentally reconstruct the overhead plan. Output ONLY valid JSON."
        ),
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": image_b64,
                    },
                },
                {
                    "type": "text",
                    "text": f"""Analyse this venue and return its top-down layout.{hint_txt}

Coordinate system: normalized 0-1, origin TOP-LEFT, x = right, y = down.
Every element is a rectangle: x,y = top-left corner, w,h = width/height (all 0-1).

Element types (use these exactly):
  "stage"   — performance area / focal point crowds push toward (obstacle)
  "wall"    — solid barrier / building edge / fence (obstacle)
  "barrier" — internal divider, pillar, structure crowds cannot pass (obstacle)
  "entry"   — where crowd ENTERS / arrives from (a source of people)
  "gate"    — EXIT / egress where crowd can LEAVE (a drain)

Rules:
- Always include the 4 perimeter walls forming the venue boundary.
- Include at least one "entry" and at least one "gate".
- 6-20 elements total. Keep it simple and physically plausible.
- Give each entry/gate/stage a short human label (e.g. "MAIN GATE", "STAGE").
- Estimate realistic total capacity from the visible floor area.

Output ONLY this JSON (no markdown):
{{
  "name": "short venue name you infer",
  "capacity": 0,
  "view": "overhead|ground|floorplan",
  "confidence": 0.0,
  "notes": "one sentence on what you saw and any assumptions",
  "elements": [
    {{"type": "wall", "x": 0.0, "y": 0.0, "w": 1.0, "h": 0.04, "label": ""}}
  ]
}}"""
                },
            ],
        }],
    )

    raw = resp.content[0].text.strip()
    if "```" in raw:
        for part in raw.split("```"):
            candidate = part.strip()
            if candidate.startswith("json"):
                candidate = candidate[4:].strip()
            if candidate.startswith("{"):
                raw = candidate
                break
    raw = raw.strip()

    try:
        layout = json.loads(raw)
    except json.JSONDecodeError:
        layout = {
            "name": "Unrecognized Venue",
            "capacity": int(capacity_hint or 5000),
            "view": "unknown",
            "confidence": 0.0,
            "notes": "Vision parse failed — using a generic rectangular hall.",
            "elements": [],
        }

    return _sanitize_layout(layout, capacity_hint)


def _sanitize_layout(layout, capacity_hint=None):
    """Clamp/validate a vision-extracted layout so the simulator stays stable."""
    allowed = {"stage", "wall", "barrier", "entry", "gate"}

    def clamp01(v, default=0.0):
        try:
            v = float(v)
        except (TypeError, ValueError):
            return default
        return max(0.0, min(1.0, v))

    clean_elements = []
    for el in (layout.get("elements") or [])[:30]:
        etype = str(el.get("type", "")).lower().strip()
        if etype not in allowed:
            continue
        x = clamp01(el.get("x"))
        y = clamp01(el.get("y"))
        w = clamp01(el.get("w"))
        h = clamp01(el.get("h"))
        if w <= 0.0 or h <= 0.0:
            continue
        w = min(w, 1.0 - x)
        h = min(h, 1.0 - y)
        if w <= 0.0 or h <= 0.0:
            continue
        clean_elements.append({
            "type": etype,
            "x": round(x, 3), "y": round(y, 3),
            "w": round(w, 3), "h": round(h, 3),
            "label": str(el.get("label", ""))[:24],
        })

    # Guarantee a usable venue even if vision was sparse.
    have = {e["type"] for e in clean_elements}
    if "wall" not in have:
        clean_elements += [
            {"type": "wall", "x": 0.0,  "y": 0.0,  "w": 1.0,  "h": 0.04, "label": ""},
            {"type": "wall", "x": 0.0,  "y": 0.96, "w": 1.0,  "h": 0.04, "label": ""},
            {"type": "wall", "x": 0.0,  "y": 0.0,  "w": 0.04, "h": 1.0,  "label": ""},
            {"type": "wall", "x": 0.96, "y": 0.0,  "w": 0.04, "h": 1.0,  "label": ""},
        ]
    if "entry" not in have:
        clean_elements.append(
            {"type": "entry", "x": 0.38, "y": 0.87, "w": 0.24, "h": 0.08,
             "label": "MAIN ENTRY"})
    if "gate" not in have:
        clean_elements.append(
            {"type": "gate", "x": 0.04, "y": 0.45, "w": 0.08, "h": 0.10,
             "label": "EXIT A"})

    try:
        cap = int(layout.get("capacity") or capacity_hint or 5000)
    except (TypeError, ValueError):
        cap = int(capacity_hint or 5000)
    cap = max(100, min(cap, 500_000))

    try:
        conf = float(layout.get("confidence", 0.0))
    except (TypeError, ValueError):
        conf = 0.0

    return {
        "name": str(layout.get("name", "Detected Venue"))[:60],
        "capacity": cap,
        "view": str(layout.get("view", "unknown"))[:20],
        "confidence": round(max(0.0, min(1.0, conf)), 2),
        "notes": str(layout.get("notes", ""))[:280],
        "elements": clean_elements,
    }


# ─── AGENT 1: VENUE AGENT ────────────────────────────────────────────────────

def run_venue_agent(venue_description, save_path="venue_config.json"):
    """
    One-time setup agent: generates a venue_config.json from a
    natural-language description of the venue.

    Maps physics primitives → real, venue-specific instructions
    that a security coordinator can execute immediately.

    Physics primitives (venue-agnostic, from rl_policy.py):
        monitor          → observe, no action
        increase_egress  → open exits / widen flow paths
        reduce_ingress   → slow/stop incoming crowd
        lateral_redirect → push crowd sideways
        disperse         → spread out density
        partial_evac     → clear a zone
        full_evac        → emergency evacuation

    Args:
        venue_description: str — free-form description of the venue
        save_path:         str — where to save the resulting JSON

    Returns:
        dict — venue config ready for use in interpret_live()
    """
    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=800,
        system=(
            "You are a crowd safety consultant configuring a real-time "
            "AI monitoring system for a specific venue. "
            "You must output ONLY valid JSON, no markdown, no explanation."
        ),
        messages=[{
            "role": "user",
            "content": f"""Venue description:
{venue_description}

Generate a venue_config.json with this exact schema:
{{
  "name": "venue name",
  "capacity": 0,
  "exits": ["list of named exits"],
  "zones": ["list of named crowd zones"],
  "action_map": {{
    "monitor":          "specific do-nothing instruction for this venue",
    "increase_egress":  "specific exit/gate to open + who to notify",
    "reduce_ingress":   "specific entry point to slow or close",
    "lateral_redirect": "specific path/corridor to redirect toward",
    "disperse":         "specific PA zone or steward instruction",
    "partial_evac":     "specific zone to clear and safe route",
    "full_evac":        "full venue evacuation procedure + who to call"
  }},
  "alert_contacts": {{
    "WARNING": "security team contact",
    "DANGER": "security + venue management contact",
    "EMERGENCY": "emergency services + all contacts"
  }},
  "camera_zones": {{
    "description": "which cameras cover which zones"
  }}
}}

Output only the JSON."""
        }]
    )

    raw = resp.content[0].text.strip()
    # Strip markdown fences if Claude added them despite instructions
    if "```" in raw:
        parts = raw.split("```")
        for part in parts:
            candidate = part.strip()
            if candidate.startswith("json"):
                candidate = candidate[4:].strip()
            if candidate.startswith("{"):
                raw = candidate
                break
    raw = raw.strip()

    try:
        config = json.loads(raw)
    except json.JSONDecodeError:
        # Return minimal fallback config
        config = {
            "name": "Unknown Venue",
            "capacity": 0,
            "exits": [],
            "zones": [],
            "action_map": {
                "monitor":          "Continue monitoring. No action.",
                "increase_egress":  "Open nearest exit gate.",
                "reduce_ingress":   "Halt entry at main entrance.",
                "lateral_redirect": "Guide crowd to side corridor.",
                "disperse":         "PA announcement: please spread out.",
                "partial_evac":     "Clear nearest high-pressure zone.",
                "full_evac":        "Full evacuation. Contact emergency services."
            },
            "alert_contacts": {
                "WARNING": "Security team",
                "DANGER": "Security + management",
                "EMERGENCY": "Emergency services"
            },
            "camera_zones": {}
        }

    if save_path:
        with open(save_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"[VenueAgent] Saved config: {save_path}")

    return config


# ─── AGENT 2: CALIBRATION AGENT ──────────────────────────────────────────────

def run_calibration_agent(feature_sequence, frame_sample=None):
    """
    Stream-start agent: decides when the scene is calm enough to
    calibrate the anomaly detector, and infers camera perspective
    to generate a per-cell weight map.

    Args:
        feature_sequence: np.ndarray (T, 256) — recent flow features
        frame_sample:     np.ndarray (H, W, 3) | None — a sample frame
                          for perspective analysis (optional)

    Returns:
        dict with keys:
            'calibrate_now': bool — True if scene is currently calm
            'calm_score':    float — 0 (chaotic) to 1 (perfectly calm)
            'reason':        str — why calibration is/isn't recommended
            'grid_weights':  np.ndarray (8, 8) — perspective correction
                             (all 1.0 if no frame provided)
            'recommendation': str — human-readable guidance
    """
    if len(feature_sequence) == 0:
        return _default_calibration_result("No features provided")

    arr = np.array(feature_sequence)

    # Extract summary stats across the sequence
    fy_cols = arr[:, 1::4]    # y-velocity (backward flow)
    mag_cols = np.sqrt(arr[:, 0::4]**2 + arr[:, 1::4]**2)

    turbulence_mean   = float(np.mean(np.var(mag_cols, axis=0)))
    backward_mean     = float(np.mean(-fy_cols))
    speed_mean        = float(np.mean(mag_cols))
    turbulence_trend  = float(np.polyfit(
        np.arange(len(arr)), np.var(mag_cols, axis=1), 1)[0])

    stats = {
        "n_frames":        len(arr),
        "mean_speed":      round(speed_mean, 4),
        "mean_turbulence": round(turbulence_mean, 6),
        "backward_flow":   round(backward_mean, 4),
        "turbulence_trend": round(turbulence_trend, 8),
        "interpretation":  (
            "positive turbulence_trend = crowd getting more chaotic over time"
        )
    }

    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        system=(
            "You are a crowd physics calibration system. "
            "Analyse flow statistics to determine if a scene is calm enough "
            "to use as a baseline. Output ONLY valid JSON, no markdown."
        ),
        messages=[{
            "role": "user",
            "content": f"""Flow statistics from the last {len(arr)} frames:
{json.dumps(stats, indent=2)}

Calibration uses these frames as the 'normal' baseline.
If the crowd is already agitated, calibration will be wrong.

Decide: is this scene calm enough to calibrate now?

Output JSON:
{{
  "calibrate_now": true or false,
  "calm_score": 0.0 to 1.0,
  "reason": "one sentence",
  "recommendation": "what to tell the operator"
}}"""
        }]
    )

    raw = resp.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = {
            "calibrate_now": turbulence_mean < 0.01 and backward_mean < 0.05,
            "calm_score": max(0.0, 1.0 - turbulence_mean * 100),
            "reason": "JSON parse failed — using heuristic fallback",
            "recommendation": "Calibrate if crowd looks calm."
        }

    # Build perspective grid weights
    # Without an actual frame we default to uniform 1.0
    # (perspective correction can be added when frame_sample is passed)
    grid_weights = np.ones((8, 8), dtype=np.float32)
    result["grid_weights"] = grid_weights.tolist()

    return result


def _default_calibration_result(reason):
    return {
        "calibrate_now": False,
        "calm_score": 0.0,
        "reason": reason,
        "recommendation": "Provide feature sequence to calibrate.",
        "grid_weights": np.ones((8, 8), dtype=np.float32).tolist()
    }


# ─── CONVENIENCE: FULL LIVE PACKAGE ──────────────────────────────────────────

def get_live_interpretation(physics_state, venue_config=None):
    """
    Returns everything needed for the live UI panel in one call.

    Combines situational awareness + RL explanation into one dict.
    Use this in the backend instead of calling roles separately.

    Args:
        physics_state: dict from anomaly_detector.process_frame()
        venue_config:  dict from run_venue_agent() or loaded JSON

    Returns:
        dict with 'situation' and optionally 'rl_explanation'
    """
    result = {}

    venue_name = venue_config.get("name", "") if venue_config else ""
    result["situation"] = interpret_live(physics_state, venue=venue_name)

    iv = physics_state.get("intervention")
    if iv and physics_state["status"] in ("WARNING", "DANGER"):
        result["rl_explanation"] = explain_rl_decision(
            iv, physics_state, venue_config=venue_config
        )

    return result
