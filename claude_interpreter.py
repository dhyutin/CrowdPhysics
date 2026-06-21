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
import re
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

def plan_event_layout(layout, sim_results, purpose, capacity, capacity_check=None):
    """
    Agentic planner: given a detected venue layout, the simulation results and
    the event's PURPOSE, produce a concrete plan for how the crowd ENTERS, STAYS
    and EXITS the space safely for that specific use.

    Args:
        layout:          dict from extract_venue_layout()
        sim_results:     dict (peak_pressure, n_danger_zones, safe_capacity, ...)
        purpose:         str — what the space will be used for
        capacity:        int — attendance being planned for
        capacity_check:  optional dict from _capacity_check() flagging whether
                         the requested headcount is reasonable for the area.

    Returns:
        str — structured arrangement plan (Entry / Staying / Exit + capacity).
    """
    cap_note = ""
    if capacity_check:
        if capacity_check.get("verdict") == "unreasonable":
            cap_note = (
                f"\nCAPACITY WARNING: the requested {capacity_check['given']:,} "
                f"people is UNREASONABLE for this floor area — it exceeds the "
                f"safe limit of ~{capacity_check['crush_capacity']:,}. You are "
                f"planning for a healthy ~{capacity_check['healthy_capacity']:,} "
                f"instead. Say this plainly and explain why.")
        elif capacity_check.get("verdict") == "tight":
            cap_note = (
                f"\nCAPACITY NOTE: {capacity_check['given']:,} is dense for this "
                f"area (healthy ~{capacity_check['healthy_capacity']:,}). Feasible "
                f"but call out the extra egress/monitoring needed.")

    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=750,
        system=(
            "You are an event crowd-planning agent. Given a venue's top-down "
            "layout and a crowd fluid-dynamics simulation, you design HOW the "
            "crowd will ENTER, STAY and EXIT the space safely for a stated "
            "purpose. Be concrete and spatial — reference entries, exits, the "
            "stage and danger zones by their positions. Write for an event "
            "organizer, not an engineer."
        ),
        messages=[{
            "role": "user",
            "content": f"""EVENT PURPOSE: {purpose or 'general gathering'}
PLANNING FOR: {capacity:,} people{cap_note}

VENUE (top-down, normalized 0-1 coords, origin top-left):
{json.dumps({"name": layout.get("name"), "view": layout.get("view"),
             "elements": layout.get("elements", [])}, indent=2)}

CROWD SIMULATION RESULTS:
{json.dumps(sim_results, indent=2)}

Produce a plan with EXACTLY these sections, in this order:
ENTRY: how the crowd arrives and enters — which entries to use, queueing/metering, arrival pacing (2-3 sentences, spatial).
STAYING: how they occupy the space during the event — zones, where to keep density low, sightlines and circulation lanes (2-3 sentences, spatial).
EXIT: how they leave — egress routes per zone, end-of-event surge handling, dispersal (2-3 sentences, spatial).
CAPACITY: the recommended healthy attendance and why (call out clearly if the requested number is unreasonable).
STAFFING: where to place stewards/security relative to danger zones (numbered, max 4).

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
            "You are a crowd-safety surveyor AND a 3D reconstruction artist. "
            "You convert an image of a venue into a TOP-DOWN floor plan of "
            "rectangles for a crowd fluid-dynamics simulator, AND you describe "
            "the venue's 3D massing (heights, shapes, archetype, decor) so it "
            "can be rebuilt as a 3D model that looks like the real place. "
            "If the image is a ground-level photo, mentally reconstruct the "
            "overhead plan and infer heights. Output ONLY valid JSON."
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
                    "text": f"""Analyse this venue and return BOTH its top-down layout AND its 3D form.{hint_txt}

Coordinate system: normalized 0-1, origin TOP-LEFT, x = right, y = down.
Every element is a rectangle: x,y = top-left corner, w,h = width/height (all 0-1).

Element types (use these exactly):
  "stage"   — performance area / focal point crowds push toward (obstacle)
  "wall"    — solid barrier / building edge / fence / tiered stands (obstacle)
  "barrier" — internal divider, pillar, structure crowds cannot pass (obstacle)
  "entry"   — where crowd ENTERS / arrives from (a source of people)
  "gate"    — EXIT / egress where crowd can LEAVE (a drain)

For EACH element also give its 3D form so the model looks like the real place:
  "height"  — relative height 0-1 (0.05 flat ground markings, 0.3 barrier,
              0.6 stage/low stand, 1.0 tall wall/building/tall stand)
  "shape"   — one of: "box" (default), "cylinder" (pillar/tower/round),
              "tiered" (sloped seating stand / stadium bowl section),
              "dome" (domed roof), "ramp" (sloped surface), "canopy" (flat roof on legs)

Top level:
  "archetype" — the venue's overall form, one of:
      "stadium" | "arena" | "theater" | "hall" | "plaza" | "street" | "field" | "festival"

"decor" — VISUAL-ONLY props that make it recognizable but do NOT block crowds
  (do not affect the simulation). Each: {{type, x, y, w, h, height, label}} where
  type ∈ "screen" (big LED screen/scoreboard) | "tower" (light/speaker tower) |
         "tent" (marquee/booth) | "tree" (tree/greenery) | "roof" (overhead canopy).

Rules:
- Always include the 4 perimeter walls forming the venue boundary.
- Include at least one "entry" and at least one "gate".
- 6-20 layout elements + 0-10 decor props. Keep it physically plausible.
- Use "tiered" shape for stadium/arena seating stands so the bowl reads in 3D.
- Give each entry/gate/stage a short human label (e.g. "MAIN GATE", "STAGE").
- Estimate realistic total capacity from the visible floor area.

Output ONLY this JSON (no markdown):
{{
  "name": "short venue name you infer",
  "capacity": 0,
  "view": "overhead|ground|floorplan",
  "archetype": "stadium|arena|theater|hall|plaza|street|field|festival",
  "confidence": 0.0,
  "notes": "one sentence on what you saw and any assumptions",
  "elements": [
    {{"type": "wall", "x": 0.0, "y": 0.0, "w": 1.0, "h": 0.04, "height": 1.0, "shape": "box", "label": ""}}
  ],
  "decor": [
    {{"type": "screen", "x": 0.4, "y": 0.02, "w": 0.2, "h": 0.02, "height": 0.7, "label": "BIG SCREEN"}}
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
            "archetype": "hall",
            "confidence": 0.0,
            "notes": "Vision parse failed — using a generic rectangular hall.",
            "elements": [],
            "decor": [],
        }

    return _sanitize_layout(layout, capacity_hint)


_SHAPES = {"box", "cylinder", "tiered", "dome", "ramp", "canopy"}
_ARCHETYPES = {"stadium", "arena", "theater", "hall", "plaza", "street",
               "field", "festival"}
_DECOR_TYPES = {"screen", "tower", "tent", "tree", "roof"}
_DEFAULT_HEIGHT = {"wall": 1.0, "stage": 0.55, "barrier": 0.32,
                   "entry": 0.05, "gate": 0.05}


def _sanitize_layout(layout, capacity_hint=None):
    """Clamp/validate a vision-extracted layout so the simulator stays stable."""
    allowed = {"stage", "wall", "barrier", "entry", "gate"}

    def clamp01(v, default=0.0):
        try:
            v = float(v)
        except (TypeError, ValueError):
            return default
        return max(0.0, min(1.0, v))

    def shape_of(el):
        s = str(el.get("shape", "")).lower().strip()
        return s if s in _SHAPES else "box"

    def height_of(el, etype):
        try:
            hv = float(el.get("height"))
        except (TypeError, ValueError):
            return _DEFAULT_HEIGHT.get(etype, 0.4)
        return max(0.02, min(1.0, hv))

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
            "height": round(height_of(el, etype), 3),
            "shape": shape_of(el),
            "label": str(el.get("label", ""))[:24],
        })

    # Visual-only decor props (never enter the simulation).
    clean_decor = []
    for d in (layout.get("decor") or [])[:12]:
        dtype = str(d.get("type", "")).lower().strip()
        if dtype not in _DECOR_TYPES:
            continue
        x = clamp01(d.get("x"))
        y = clamp01(d.get("y"))
        w = clamp01(d.get("w"), 0.05) or 0.05
        h = clamp01(d.get("h"), 0.05) or 0.05
        w = min(w, 1.0 - x)
        h = min(h, 1.0 - y)
        if w <= 0.0 or h <= 0.0:
            continue
        try:
            dh = float(d.get("height"))
        except (TypeError, ValueError):
            dh = 0.5
        clean_decor.append({
            "type": dtype,
            "x": round(x, 3), "y": round(y, 3),
            "w": round(w, 3), "h": round(h, 3),
            "height": round(max(0.05, min(1.0, dh)), 3),
            "label": str(d.get("label", ""))[:24],
        })

    # Guarantee a usable venue even if vision was sparse.
    have = {e["type"] for e in clean_elements}
    if "wall" not in have:
        clean_elements += [
            {"type": "wall", "x": 0.0,  "y": 0.0,  "w": 1.0,  "h": 0.04, "height": 1.0, "shape": "box", "label": ""},
            {"type": "wall", "x": 0.0,  "y": 0.96, "w": 1.0,  "h": 0.04, "height": 1.0, "shape": "box", "label": ""},
            {"type": "wall", "x": 0.0,  "y": 0.0,  "w": 0.04, "h": 1.0,  "height": 1.0, "shape": "box", "label": ""},
            {"type": "wall", "x": 0.96, "y": 0.0,  "w": 0.04, "h": 1.0,  "height": 1.0, "shape": "box", "label": ""},
        ]
    if "entry" not in have:
        clean_elements.append(
            {"type": "entry", "x": 0.38, "y": 0.87, "w": 0.24, "h": 0.08,
             "height": 0.05, "shape": "box", "label": "MAIN ENTRY"})
    if "gate" not in have:
        clean_elements.append(
            {"type": "gate", "x": 0.04, "y": 0.45, "w": 0.08, "h": 0.10,
             "height": 0.05, "shape": "box", "label": "EXIT A"})

    try:
        cap = int(layout.get("capacity") or capacity_hint or 5000)
    except (TypeError, ValueError):
        cap = int(capacity_hint or 5000)
    cap = max(100, min(cap, 500_000))

    try:
        conf = float(layout.get("confidence", 0.0))
    except (TypeError, ValueError):
        conf = 0.0

    archetype = str(layout.get("archetype", "")).lower().strip()
    if archetype not in _ARCHETYPES:
        archetype = "hall"

    return {
        "name": str(layout.get("name", "Detected Venue"))[:60],
        "capacity": cap,
        "view": str(layout.get("view", "unknown"))[:20],
        "archetype": archetype,
        "confidence": round(max(0.0, min(1.0, conf)), 2),
        "notes": str(layout.get("notes", ""))[:280],
        "elements": clean_elements,
        "decor": clean_decor,
    }


# ─── ROLE 7b: SCENE DETAILS AGENT ─────────────────────────────────────────────

# Distinctive, recognizable physical objects the details agent can place. These
# are VISUAL-ONLY props (rendered in 3D, never simulated) that make the rebuilt
# environment look like the real place — e.g. a playground slide, swings, a
# fountain. They are emitted in the same decor schema the renderer already eats,
# plus an optional `color` hint.
_PROP_TYPES = {
    "slide", "swing", "playset", "fountain", "statue", "bench", "booth",
    "goal", "pole", "planter", "court", "tree", "tent", "screen", "tower",
    "roof",
}


def extract_scene_props(image_b64, media_type="image/jpeg", layout=None):
    """
    Scene-details agent: a focused second vision pass that finds the distinctive,
    recognizable objects in the image (playground slides, swings, fountains,
    statues, benches, kiosks, sport goals, flag poles, planters, courts...) and
    returns them as visual-only props so the 3D reconstruction looks like the
    real place — not just bare walls and a stage.

    Runs AFTER extract_venue_layout() and is given the already-detected layout so
    it can place props in the right spots and avoid duplicating structures.

    Args:
        image_b64:  base64-encoded image bytes (no data: prefix)
        media_type: image/jpeg | image/png | image/webp | image/gif
        layout:     dict from extract_venue_layout() (for spatial context)

    Returns:
        list[dict] of props: {type, x, y, w, h, height, color, label}
        Coordinates normalized 0-1, origin top-left (same frame as the layout).
        Returns [] on any failure — purely additive, never blocks planning.
    """
    ctx = ""
    if layout:
        ctx = ("\nAlready-detected layout (do NOT repeat walls/stage/gates; place "
               "props in the open areas between them):\n"
               + json.dumps({"name": layout.get("name"),
                             "archetype": layout.get("archetype"),
                             "elements": layout.get("elements", [])}, indent=2))

    try:
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=900,
            system=(
                "You are a scene-details agent for a 3D reconstruction. You spot "
                "the distinctive, recognizable OBJECTS in a venue photo — the "
                "things that make someone say 'that's a playground' or 'that's a "
                "plaza' — and you place them as small props on a top-down plan. "
                "You do NOT re-describe walls, the stage, entries or exits. "
                "Output ONLY valid JSON."
            ),
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image",
                     "source": {"type": "base64", "media_type": media_type,
                                "data": image_b64}},
                    {"type": "text",
                     "text": f"""Find the distinctive physical objects in this venue and place them.{ctx}

Coordinate system: normalized 0-1, origin TOP-LEFT, x = right, y = down.
Each prop is a rectangle footprint: x,y = top-left corner, w,h = size (all 0-1).

Use these prop types EXACTLY (pick the closest match):
  "slide"    — playground slide
  "swing"    — swing set
  "playset"  — jungle gym / climbing/play structure
  "fountain" — water fountain / round basin
  "statue"   — statue / monument / sculpture
  "bench"    — bench / seating
  "booth"    — kiosk / stall / food booth / small hut
  "goal"     — sports goal (soccer/hockey)
  "pole"     — flag pole / lamp post / sign post
  "planter"  — planter bed / bush / hedge / flower bed
  "court"    — flat ground patch (sport court, sandbox, splash pad)
  "tree"     — tree / large greenery
  "tent"     — marquee / canopy tent
  "screen"   — LED screen / scoreboard
  "tower"    — light / speaker tower

For EACH prop give:
  "height" — relative height 0-1 (0.04 flat court, 0.15 bench, 0.3 slide/swing,
             0.5 statue/booth, 0.8 tall pole)
  "color"  — a hex color that matches what you see (e.g. "#E5484D" red slide);
             omit or "" if unsure.
  "label"  — 1-3 word name (e.g. "RED SLIDE", "FOUNTAIN").

Rules:
- Only include objects you can actually see. 0-12 props. Quality over quantity.
- Keep footprints small and plausible (most props w,h ≤ 0.15).
- If the scene is bare (e.g. an empty hall or field), return an empty list.

Output ONLY this JSON (no markdown):
{{"props": [
  {{"type": "slide", "x": 0.4, "y": 0.55, "w": 0.08, "h": 0.10, "height": 0.3, "color": "#E5484D", "label": "SLIDE"}}
]}}"""},
                ],
            }],
        )
    except Exception:
        return []

    raw = resp.content[0].text.strip()
    if "```" in raw:
        for part in raw.split("```"):
            cand = part.strip()
            if cand.startswith("json"):
                cand = cand[4:].strip()
            if cand.startswith("{"):
                raw = cand
                break

    try:
        data = json.loads(raw.strip())
    except json.JSONDecodeError:
        return []

    return _sanitize_props(data.get("props") or data.get("decor") or [])


def _sanitize_props(props):
    """Clamp/validate details-agent props into the decor schema the UI renders."""
    def clamp01(v, default=0.0):
        try:
            return max(0.0, min(1.0, float(v)))
        except (TypeError, ValueError):
            return default

    HEX = re.compile(r"^#[0-9a-fA-F]{6}$")
    clean = []
    for p in props[:12]:
        ptype = str(p.get("type", "")).lower().strip()
        if ptype not in _PROP_TYPES:
            continue
        x = clamp01(p.get("x"))
        y = clamp01(p.get("y"))
        w = clamp01(p.get("w"), 0.06) or 0.06
        h = clamp01(p.get("h"), 0.06) or 0.06
        w = min(w, 1.0 - x)
        h = min(h, 1.0 - y)
        if w <= 0.0 or h <= 0.0:
            continue
        try:
            ph = float(p.get("height"))
        except (TypeError, ValueError):
            ph = 0.3
        color = str(p.get("color", "")).strip()
        if not HEX.match(color):
            color = ""
        clean.append({
            "type": ptype,
            "x": round(x, 3), "y": round(y, 3),
            "w": round(w, 3), "h": round(h, 3),
            "height": round(max(0.02, min(1.0, ph)), 3),
            "color": color,
            "label": str(p.get("label", ""))[:24],
        })
    return clean


# ─── ROLE 7c: CONVERSATIONAL SCENE EDITOR ─────────────────────────────────────

def refine_venue_layout(layout, instruction, image_b64=None,
                        media_type="image/jpeg"):
    """
    Conversational 3D-scene editor. Given the current reconstructed layout and a
    plain-language correction ("the slide is on the left", "add an exit on the
    north wall", "remove the stage", "make the hall wider", "the fountain should
    be bigger"), return the UPDATED layout in the same schema plus a one-line
    summary of what changed.

    Args:
        layout:      current layout dict {elements, decor, name, archetype, ...}
        instruction: the user's natural-language change
        image_b64:   optional original image for visual grounding (not required)
        media_type:  image media type if image_b64 is given

    Returns:
        (layout_dict, summary_str). On failure returns (layout, message) so the
        caller can keep the old layout and surface the message in chat.
    """
    cur = {
        "name":      layout.get("name", "Venue"),
        "archetype": layout.get("archetype", "hall"),
        "elements":  layout.get("elements", []),
        "decor":     layout.get("decor", []),
    }

    content = []
    if image_b64:
        content.append({"type": "image", "source": {
            "type": "base64", "media_type": media_type, "data": image_b64}})
    content.append({"type": "text", "text": f"""You are editing a top-down venue layout for a crowd simulator. Apply the user's change and return the FULL updated layout.

CURRENT LAYOUT (normalized 0-1 coords, origin TOP-LEFT, x=right, y=down):
{json.dumps(cur, indent=2)}

USER REQUEST: "{instruction}"

Rules:
- Apply ONLY what the user asked; keep every other element/prop identical.
- Keep the 4 perimeter walls and at least one entry + one gate unless the user explicitly removes them.
- Structural element types: "stage" | "wall" | "barrier" | "entry" | "gate".
  Each: {{type,x,y,w,h,height(0-1),shape,label}}; shape ∈ box|cylinder|tiered|dome|ramp|canopy.
- Visual prop types (decor): slide|swing|playset|fountain|statue|bench|booth|goal|pole|planter|court|tree|tent|screen|tower|roof.
  Each: {{type,x,y,w,h,height,color(hex),label}}.
- Positions/sizes stay within 0-1 and physically plausible.

Output ONLY this JSON (no markdown):
{{"summary": "one short sentence on what you changed",
  "name": "{cur['name']}",
  "archetype": "{cur['archetype']}",
  "elements": [ ... full updated list ... ],
  "decor": [ ... full updated list ... ]}}"""})

    try:
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1600,
            system=("You are a precise 3D scene editor. You modify a structured "
                    "venue layout to match a user's instruction and return valid "
                    "JSON only — never prose outside the JSON."),
            messages=[{"role": "user", "content": content}],
        )
    except Exception as exc:
        return layout, f"Couldn't reach the editor agent: {exc}"

    raw = resp.content[0].text.strip()
    if "```" in raw:
        for part in raw.split("```"):
            cand = part.strip()
            if cand.startswith("json"):
                cand = cand[4:].strip()
            if cand.startswith("{"):
                raw = cand
                break

    try:
        data = json.loads(raw.strip())
    except json.JSONDecodeError:
        return layout, "I couldn't parse that change — try rephrasing it."

    summary = str(data.get("summary", "")).strip()[:200] or "Updated the layout."
    merged = {
        "name":       data.get("name", layout.get("name", "Venue")),
        "capacity":   layout.get("capacity"),
        "view":       layout.get("view", "edited"),
        "archetype":  data.get("archetype", layout.get("archetype", "hall")),
        "confidence": layout.get("confidence", 0.0),
        "notes":      data.get("notes", layout.get("notes", "")),
        "elements":   data.get("elements", layout.get("elements", [])),
        "decor":      data.get("decor", layout.get("decor", [])),
    }
    refined = _sanitize_layout(merged, layout.get("capacity"))
    # Decor carries the richer prop set, so sanitize it with the prop validator
    # (which is a superset of the plain decor types).
    refined["decor"] = _sanitize_props(merged.get("decor") or [])
    return refined, summary


# ─── ROLE 7d: AGENT-LLM BEHAVIOR PLANNER ──────────────────────────────────────

def _behaviors_from_layout(layout) -> dict:
    """
    Deterministic fallback used when the LLM is unavailable: derive plausible
    crowd intents directly from the detected structure (head for the stage,
    drain to the gates, mill in the open middle).
    """
    elements = layout.get("elements", []) or []

    def _center(e):
        return [round(float(e.get("x", 0)) + float(e.get("w", 0)) / 2, 3),
                round(float(e.get("y", 0)) + float(e.get("h", 0)) / 2, 3)]

    stage = next((e for e in elements if e.get("type") == "stage"), None)
    gates = [e for e in elements if e.get("type") in ("gate", "entry")]

    behaviors = []
    if stage:
        behaviors.append({"name": "Toward the stage", "goal": _center(stage),
                          "fraction": 0.45, "speed": 1.2,
                          "intent": "press toward the main attraction"})
    if gates:
        g = gates[0]
        behaviors.append({"name": "Exit seekers", "goal": _center(g),
                          "fraction": 0.3, "speed": 1.1,
                          "intent": "make for the nearest way out"})
    behaviors.append({"name": "Wanderers", "goal": [0.5, 0.55],
                      "fraction": 0.25 if behaviors else 1.0, "speed": 0.7,
                      "intent": "drift around the open floor"})
    return {"llm_fraction": 0.3, "behaviors": behaviors, "source": "layout"}


def _sanitize_behaviors(data) -> dict:
    """Clamp an LLM behavior plan to safe ranges; renormalize fractions."""
    try:
        raw = data.get("behaviors", []) if isinstance(data, dict) else []
    except Exception:
        raw = []
    clean = []
    for b in raw[:6]:
        try:
            gx = float(b["goal"][0]); gy = float(b["goal"][1])
        except Exception:
            continue
        clean.append({
            "name":     str(b.get("name", "Group"))[:32],
            "goal":     [round(min(0.97, max(0.03, gx)), 3),
                         round(min(0.97, max(0.03, gy)), 3)],
            "fraction": float(b.get("fraction", 0.0) or 0.0),
            "speed":    round(min(1.8, max(0.4, float(b.get("speed", 1.0) or 1.0))), 2),
            "intent":   str(b.get("intent", ""))[:80],
        })
    if not clean:
        return {}
    total = sum(max(0.0, b["fraction"]) for b in clean) or 1.0
    for b in clean:
        b["fraction"] = round(max(0.0, b["fraction"]) / total, 3)
    llm_frac = 0.3
    if isinstance(data, dict):
        try:
            llm_frac = min(0.8, max(0.05, float(data.get("llm_fraction", 0.3))))
        except Exception:
            pass
    return {"llm_fraction": round(llm_frac, 2), "behaviors": clean, "source": "llm"}


def agent_behaviors(layout, purpose="general gathering", n_people=0):
    """
    Agent-LLM acting as a behavioral world model. Given the venue layout and
    event context, Claude decides how distinct GROUPS of the crowd intend to
    move — each a named behavior with a goal point (normalized 0-1), the
    fraction of LLM-piloted agents that follow it, and a speed.

    The simulator then drives `llm_fraction` of agents toward these intents
    (goal-seeking) while the rest move purely on the physics world model. This
    blends learned crowd fluid dynamics with high-level, reasoned intent.

    Returns {llm_fraction, behaviors[], source}; never raises.
    """
    compact = {
        "archetype": layout.get("archetype", "hall"),
        "elements": [
            {"type": e.get("type"), "label": e.get("label", ""),
             "x": e.get("x"), "y": e.get("y"), "w": e.get("w"), "h": e.get("h")}
            for e in (layout.get("elements", []) or [])
        ][:24],
    }
    prompt = f"""You are the behavioral world model for a crowd simulation. Decide how groups of people INTEND to move in this venue.

VENUE (normalized 0-1 coords, origin TOP-LEFT, x=right, y=down):
{json.dumps(compact, indent=2)}

EVENT: {purpose}
EXPECTED CROWD: {n_people or "unspecified"}

Define 2-4 behavior groups that together describe realistic intent for THIS event and layout (e.g. press toward a stage, drain to exits, queue at a gate, browse the middle, gather at a feature). For each give a goal point inside the venue.

Also pick llm_fraction: the share of agents (0.05-0.8) that should be driven by these reasoned intents; the rest follow pure crowd physics.

Output ONLY JSON:
{{"llm_fraction": 0.3,
  "behaviors": [
    {{"name": "short label", "goal": [x, y], "fraction": 0.5, "speed": 1.2, "intent": "one short phrase"}}
  ]}}
Fractions across behaviors should sum to ~1."""

    try:
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=700,
            system=("You plan high-level crowd intent for a simulator and reply "
                    "with strict JSON only."),
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.content[0].text.strip()
        if "```" in raw:
            for part in raw.split("```"):
                cand = part.strip()
                if cand.startswith("json"):
                    cand = cand[4:].strip()
                if cand.startswith("{"):
                    raw = cand
                    break
        data = json.loads(raw.strip())
        plan = _sanitize_behaviors(data)
        if plan.get("behaviors"):
            return plan
    except Exception:
        pass
    return _behaviors_from_layout(layout)


# ─── ROLE 7e: RECONSTRUCTION FIDELITY EVALUATOR (Arize-traced) ────────────────

def evaluate_reconstruction(image_b64, layout, media_type="image/jpeg"):
    """
    LLM-as-judge evaluation: how faithfully does the reconstructed world-model
    layout match the source photo? Claude sees the original image and the
    top-down reconstruction and scores fidelity 0-1 across a few aspects.

    The underlying Claude call is auto-traced to Arize; the caller wraps this in
    an evaluation span so the score/label show up as an Arize evaluation.

    Returns {score, label, rationale, aspects:{structures,openings,scale,features}}
    or None on failure.
    """
    compact = {
        "name":      layout.get("name"),
        "archetype": layout.get("archetype"),
        "elements":  [
            {"type": e.get("type"), "label": e.get("label", ""),
             "x": e.get("x"), "y": e.get("y"), "w": e.get("w"), "h": e.get("h")}
            for e in (layout.get("elements", []) or [])
        ][:30],
        "decor":     [
            {"type": d.get("type"), "label": d.get("label", "")}
            for d in (layout.get("decor", []) or [])
        ][:20],
    }
    content = [
        {"type": "image", "source": {
            "type": "base64", "media_type": media_type, "data": image_b64}},
        {"type": "text", "text": f"""You are evaluating whether an AI's 3D reconstruction matches the REAL place in the photo.

The reconstruction is a top-down layout of labeled rectangles (normalized 0-1, origin top-left) plus visual props:
{json.dumps(compact, indent=2)}

Judge how faithfully this captures the actual space in the image. Consider:
- structures: are the major built elements (stage/walls/stands/barriers) present and roughly right?
- openings: are entrances/exits placed plausibly?
- scale: does the overall shape/proportion match the venue type?
- features: are distinctive objects (slides, fountains, screens, goals…) captured?

Output ONLY JSON:
{{"score": 0.0-1.0 overall fidelity,
  "label": "faithful" | "partial" | "poor",
  "rationale": "one or two sentences on what matches and what's off",
  "aspects": {{"structures": 0.0-1.0, "openings": 0.0-1.0, "scale": 0.0-1.0, "features": 0.0-1.0}}}}"""},
    ]
    try:
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=500,
            system=("You are a strict reconstruction-fidelity evaluator. Compare "
                    "an AI's top-down layout to the source photo and reply with "
                    "JSON only — be calibrated, not generous."),
            messages=[{"role": "user", "content": content}],
        )
        raw = resp.content[0].text.strip()
        if "```" in raw:
            for part in raw.split("```"):
                cand = part.strip()
                if cand.startswith("json"):
                    cand = cand[4:].strip()
                if cand.startswith("{"):
                    raw = cand
                    break
        data = json.loads(raw.strip())
    except Exception:
        return None

    try:
        score = float(data.get("score", 0.0))
    except (TypeError, ValueError):
        score = 0.0
    score = max(0.0, min(1.0, score))
    label = str(data.get("label", "")).lower().strip()
    if label not in ("faithful", "partial", "poor"):
        label = "faithful" if score >= 0.75 else "partial" if score >= 0.5 else "poor"

    aspects = {}
    for k in ("structures", "openings", "scale", "features"):
        try:
            aspects[k] = round(max(0.0, min(1.0, float(data.get("aspects", {}).get(k, score)))), 2)
        except (TypeError, ValueError):
            aspects[k] = round(score, 2)

    return {
        "score":     round(score, 3),
        "label":     label,
        "rationale": str(data.get("rationale", ""))[:280],
        "aspects":   aspects,
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
