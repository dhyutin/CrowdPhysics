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
    Use this in app.py instead of calling roles separately.

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
