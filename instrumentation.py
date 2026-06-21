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


def _get_tracer():
    """OTel tracer if Arize tracing is active, else None."""
    if not _INITIALIZED:
        return None
    try:
        from opentelemetry import trace
        return trace.get_tracer("crowdphysics.evals")
    except Exception:
        return None


def trace_evaluation(span_name: str, eval_name: str, fn):
    """
    Run `fn()` (an LLM-as-judge that returns {score, label, rationale, ...})
    inside a dedicated Arize span and attach the result as an evaluation.

    The score/label/explanation are written as `eval.<eval_name>.*` span
    attributes, which Arize AX ingests and displays as an evaluation on the
    trace. Fully best-effort: if tracing is off or anything fails, `fn()` still
    runs and its result is returned unchanged.

    Returns whatever `fn()` returns (or None on judge failure).
    """
    tracer = _get_tracer()
    if tracer is None:
        try:
            return fn()
        except Exception:
            return None

    with tracer.start_as_current_span(span_name) as span:
        try:
            span.set_attribute("openinference.span.kind", "EVALUATOR")
        except Exception:
            pass
        try:
            result = fn()
        except Exception as exc:
            try:
                span.set_attribute("error.message", str(exc))
            except Exception:
                pass
            return None

        if isinstance(result, dict):
            try:
                if result.get("score") is not None:
                    span.set_attribute(f"eval.{eval_name}.score", float(result["score"]))
                if result.get("label") is not None:
                    span.set_attribute(f"eval.{eval_name}.label", str(result["label"]))
                expl = result.get("rationale") or result.get("explanation")
                if expl:
                    span.set_attribute(f"eval.{eval_name}.explanation", str(expl)[:1000])
            except Exception:
                pass
        return result
