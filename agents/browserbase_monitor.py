# agents/browserbase_monitor.py
"""
Browserbase web-camera stream monitor (sponsor integration).

Spins up a headless cloud browser via Browserbase, which can be pointed
at any public web camera / livestream page to capture frames for the
CrowdPhysics pipeline. This module handles the session lifecycle; frame
capture would feed into flow_extractor in a full deployment.

Run:
    export BROWSERBASE_API_KEY=...        # required
    export BROWSERBASE_PROJECT_ID=...     # required
    pip install requests
    python agents/browserbase_monitor.py
"""

from __future__ import annotations

import os

import requests

BB_KEY = os.environ.get("BROWSERBASE_API_KEY", "")
BB_PROJECT = os.environ.get("BROWSERBASE_PROJECT_ID", "")
BB_BASE = "https://api.browserbase.com/v1"


def create_session() -> dict:
    """Create a new Browserbase browser session."""
    r = requests.post(
        f"{BB_BASE}/sessions",
        headers={
            "x-bb-api-key": BB_KEY,
            "Content-Type": "application/json",
        },
        json={"projectId": BB_PROJECT},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def end_session(session_id: str) -> None:
    """Release a Browserbase session."""
    requests.post(
        f"{BB_BASE}/sessions/{session_id}",
        headers={
            "x-bb-api-key": BB_KEY,
            "Content-Type": "application/json",
        },
        json={"projectId": BB_PROJECT, "status": "REQUEST_RELEASE"},
        timeout=30,
    )


if __name__ == "__main__":
    if not BB_KEY or not BB_PROJECT:
        print(
            "Set BROWSERBASE_API_KEY and BROWSERBASE_PROJECT_ID "
            "to use Browserbase monitoring."
        )
    else:
        try:
            s = create_session()
            sid = s.get("id")
            print(f"Browserbase session created: {sid}")
            print(f"Connect URL: {s.get('connectUrl', 'n/a')}")
            if sid:
                end_session(sid)
                print("Session released.")
        except Exception as exc:
            print(f"Browserbase error: {exc}")
