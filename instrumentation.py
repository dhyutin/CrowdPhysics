# instrumentation.py
"""
Arize AX tracing setup for CrowdPhysics.

Auto-instruments every Anthropic (Claude) call made through
claude_interpreter.py and ships OpenInference spans to Arize AX.

Call setup_tracing() ONCE at process startup, before the Anthropic
client issues any requests. Safe to call multiple times (no-ops after
the first successful registration). Fails gracefully if credentials are
missing so the app still runs without observability.

Env vars:
    ARIZE_SPACE_ID   — from app.arize.com space settings
    ARIZE_API_KEY    — from app.arize.com space settings
    ARIZE_PROJECT    — optional project name (default: crowdphysics)
"""

from __future__ import annotations

import os

_INITIALIZED = False


def setup_tracing() -> bool:
    """Register Arize tracing + Anthropic instrumentor. Returns True if active."""
    global _INITIALIZED
    if _INITIALIZED:
        return True

    space_id = os.environ.get("ARIZE_SPACE_ID")
    api_key = os.environ.get("ARIZE_API_KEY")

    if not space_id or not api_key:
        print(
            "[arize] ⚠  ARIZE_SPACE_ID / ARIZE_API_KEY not set — "
            "tracing disabled (app runs normally)."
        )
        return False

    try:
        from arize.otel import register
        from openinference.instrumentation.anthropic import AnthropicInstrumentor

        tracer_provider = register(
            space_id=space_id,
            api_key=api_key,
            project_name=os.environ.get("ARIZE_PROJECT", "crowdphysics"),
        )
        AnthropicInstrumentor().instrument(tracer_provider=tracer_provider)

        _INITIALIZED = True
        print("[arize] ✓ Tracing active — Claude calls streaming to Arize AX")
        return True
    except Exception as exc:
        print(f"[arize] ⚠  Tracing setup failed ({exc}) — continuing without it")
        return False
