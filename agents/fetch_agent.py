# agents/fetch_agent.py
"""
CrowdPhysics autonomous monitoring agent (Fetch.ai uAgents / Agentverse).

Runs as a standalone autonomous agent that periodically polls the
CrowdPhysics backend health endpoint and broadcasts a heartbeat. In a
full deployment this agent would subscribe to live pressure readings and
raise alerts to other agents on Agentverse when crush risk is detected.

Run:
    pip install uagents requests
    python agents/fetch_agent.py
"""

from __future__ import annotations

import os

import requests
from uagents import Agent, Context

API_URL = os.environ.get("CROWDPHYSICS_API_URL", "http://localhost:8000")

agent = Agent(
    name="crowdphysics-monitor",
    seed="crowdphysics_hackathon_ucb_2026",
    port=8001,
    endpoint=["http://localhost:8001/submit"],
)


@agent.on_event("startup")
async def start(ctx: Context):
    ctx.logger.info("CrowdPhysics Agent online")
    ctx.logger.info(f"Address: {agent.address}")
    ctx.logger.info(f"Backend: {API_URL}")


@agent.on_interval(period=10.0)
async def monitor(ctx: Context):
    """Poll backend health and report crowd-safety status."""
    try:
        r = requests.get(f"{API_URL}/api/health", timeout=5)
        if r.ok:
            data = r.json()
            wm = "loaded" if data.get("world_model") else "demo"
            rl = "loaded" if data.get("rl_policy") else "demo"
            ctx.logger.info(
                f"CrowdPhysics: monitoring active "
                f"(world_model={wm}, rl_policy={rl})"
            )
        else:
            ctx.logger.warning(f"Backend unhealthy: HTTP {r.status_code}")
    except Exception as exc:
        ctx.logger.warning(f"Backend unreachable ({exc}) — monitoring offline")


if __name__ == "__main__":
    agent.run()
