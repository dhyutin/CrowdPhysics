"""
Test suite for claude_interpreter.py
Runs all 4 roles + 2 agents with synthetic data.
Requires ANTHROPIC_API_KEY in environment.
"""

import sys, json, numpy as np
sys.path.insert(0, '.')

SEP = "\n" + "=" * 60 + "\n"

# ── Synthetic WARNING-level physics state ──────────────────────
PHYSICS_WARNING = {
    "status": "WARNING",
    "score": 1.9,
    "probability": 0.72,
    "error": 0.004,
    "turbulence": 0.0312,
    "backward_flow": 0.018,
    "boundary_stress": 0.041,
    "mean_speed": 0.089,
    "z_latent": np.random.randn(64).astype("float32"),
    "intervention": {
        "action_name": "increase_egress",
        "confidence": 0.81,
        "q_values": {
            "monitor": -1.2,
            "increase_egress": 3.4,
            "reduce_ingress": 2.1,
            "lateral_redirect": 1.8,
            "disperse": 0.9,
            "partial_evac": 0.2,
            "full_evac": -3.1,
        },
        "top_3": [
            {"rank": 1, "action": "increase_egress",
             "description": "Open exits", "q_value": 3.4},
            {"rank": 2, "action": "reduce_ingress",
             "description": "Slow entry", "q_value": 2.1},
            {"rank": 3, "action": "lateral_redirect",
             "description": "Redirect", "q_value": 1.8},
        ],
    },
}

PROBE_RESULTS = {
    "latent_dim": 64,
    "turbulence_corr": {"dims": [2, 7, 15], "r": 0.84},
    "backward_flow_corr": {"dims": [1, 9], "r": 0.77},
    "speed_corr": {"dims": [0, 3, 12], "r": 0.91},
    "unknown": {
        "dims": [22, 31, 45, 58],
        "separation_z_score": 3.7,
        "lead_time_minutes": 4.2,
        "description": "activate before any measured signal changes",
    },
}

SIM_RESULTS = {
    "peak_pressure_zone": "section_104_north_corner",
    "peak_score": 3.1,
    "danger_onset_minutes": 8,
    "affected_capacity_pct": 0.18,
    "recommended_actions": ["open north exit", "redirect to field B"],
}

VENUE_DESC = (
    "Madison Square Garden, NYC. Capacity 20,000. "
    "4 main exits: north, south, east, west. "
    "Main floor + 3 tiers. Event: sold-out concert."
)


def test_interpret_live():
    from claude_interpreter import interpret_live
    print("── ROLE 1: interpret_live (WARNING) ──────────────────")
    result = interpret_live(PHYSICS_WARNING, venue="Madison Square Garden")
    print(result)
    assert "SITUATION" in result, "Missing SITUATION block"
    assert "DO NOW" in result, "Missing DO NOW block"
    print("  PASS\n")


def test_explain_rl_decision():
    from claude_interpreter import explain_rl_decision
    print("── ROLE 3: explain_rl_decision ───────────────────────")
    result = explain_rl_decision(PHYSICS_WARNING["intervention"], PHYSICS_WARNING)
    print(result)
    assert len(result) > 100, "Response too short"
    print("  PASS\n")


def test_name_discovered_physics():
    from claude_interpreter import name_discovered_physics
    print("── ROLE 2: name_discovered_physics ───────────────────")
    result = name_discovered_physics(PROBE_RESULTS)
    print(result)
    assert len(result) > 100, "Response too short"
    print("  PASS\n")


def test_venue_agent():
    from claude_interpreter import run_venue_agent
    print("── AGENT 1: run_venue_agent ──────────────────────────")
    config = run_venue_agent(VENUE_DESC, save_path=None)
    print(json.dumps(config, indent=2))
    assert "action_map" in config, "Missing action_map"
    assert "increase_egress" in config["action_map"], "Missing increase_egress"
    print("  PASS\n")


def test_calibration_agent():
    from claude_interpreter import run_calibration_agent
    print("── AGENT 2: run_calibration_agent (calm) ─────────────")
    calm = np.random.randn(80, 256).astype("float32") * 0.01
    result = run_calibration_agent(calm)
    grid = result.pop("grid_weights")
    print(json.dumps(result, indent=2))
    print(f"grid_weights: {np.array(grid).shape}")
    assert "calibrate_now" in result
    assert "calm_score" in result
    print("  PASS\n")


def test_generate_safety_report():
    from claude_interpreter import generate_safety_report
    print("── ROLE 4: generate_safety_report ────────────────────")
    report = generate_safety_report(
        {"name": "MSG Concert", "capacity": 20000,
         "exits": ["north", "south", "east", "west"]},
        SIM_RESULTS,
    )
    print(report)
    assert "EXECUTIVE SUMMARY" in report, "Missing EXECUTIVE SUMMARY"
    assert "RISK ZONES" in report, "Missing RISK ZONES"
    print("  PASS\n")


if __name__ == "__main__":
    print(SEP + "CrowdPhysics — claude_interpreter.py test suite" + SEP)
    tests = [
        test_interpret_live,
        test_explain_rl_decision,
        test_name_discovered_physics,
        test_venue_agent,
        test_calibration_agent,
        test_generate_safety_report,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}\n")

    print(SEP + f"Results: {passed}/{len(tests)} passed" + SEP)
