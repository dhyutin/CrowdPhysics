"""
External danger alerting for CrowdPhysics.

When the live Monitor confirms a DANGER state, we don't just paint the UI red —
we notify the people who can actually act. This module fans a single alert out
to whatever channels are configured via environment variables, entirely
best-effort: a missing dependency, a bad webhook, or a network hiccup must never
break the monitoring stream.

Channels (all optional, enabled by presence of the env var):
  - ALERT_WEBHOOK_URL    generic JSON POST   {"text": ...}
  - SLACK_WEBHOOK_URL    Slack incoming hook {"text": ...}
  - DISCORD_WEBHOOK_URL  Discord webhook     {"content": ...}
  - Twilio SMS, if ALL of TWILIO_ACCOUNT_SID / TWILIO_AUTH_TOKEN /
    TWILIO_FROM / ALERT_SMS_TO are set.

A module-level cooldown (ALERT_COOLDOWN_S, default 60s) per venue stops repeated
capture→analyze passes from spamming the same incident. Dispatch happens on a
background thread so the caller (the streaming generator) never blocks.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any, Optional

try:
    import requests  # already a backend dependency
except Exception:  # pragma: no cover — keep import-safe even if missing
    requests = None  # type: ignore


# ── cooldown bookkeeping ──────────────────────────────────────────────────────
_COOLDOWN_S = float(os.environ.get("ALERT_COOLDOWN_S", "60"))
_last_sent: dict[str, float] = {}
_lock = threading.Lock()


def _channels_configured() -> list[str]:
    """Names of channels that have the env vars needed to fire."""
    chans: list[str] = []
    if os.environ.get("SLACK_WEBHOOK_URL"):
        chans.append("slack")
    if os.environ.get("DISCORD_WEBHOOK_URL"):
        chans.append("discord")
    if os.environ.get("ALERT_WEBHOOK_URL"):
        chans.append("webhook")
    if (os.environ.get("TWILIO_ACCOUNT_SID")
            and os.environ.get("TWILIO_AUTH_TOKEN")
            and os.environ.get("TWILIO_FROM")
            and os.environ.get("ALERT_SMS_TO")):
        chans.append("sms")
    return chans


def _format_message(payload: dict[str, Any]) -> str:
    """Build a single human-readable alert line from the danger payload."""
    venue = payload.get("venue", "venue")
    prob = payload.get("probability")
    score = payload.get("score")
    action = payload.get("action_name") or payload.get("action") or "review crowd flow"
    cf = payload.get("counterfactual") or {}

    parts = [f"🚨 CROWD DANGER at {venue}"]
    if prob is not None:
        parts.append(f"crush probability {float(prob):.0f}%")
    elif score is not None:
        parts.append(f"risk score {float(score):.2f}")
    parts.append(f"recommended action: {action}")

    if cf and not cf.get("error"):
        dn = cf.get("do_nothing_risk")
        wa = cf.get("action_risk")
        if dn is not None and wa is not None:
            parts.append(
                f"world-model projection: {action} cuts projected risk "
                f"{float(dn):.0f}% → {float(wa):.0f}%")
    return " — ".join(parts)


# ── per-channel senders (each best-effort, return True/False) ─────────────────
def _post_json(url: str, body: dict[str, Any]) -> bool:
    if not requests:
        return False
    try:
        r = requests.post(url, json=body, timeout=6)
        return 200 <= r.status_code < 300
    except Exception:
        return False


def _send_slack(text: str) -> bool:
    url = os.environ.get("SLACK_WEBHOOK_URL", "")
    return _post_json(url, {"text": text}) if url else False


def _send_discord(text: str) -> bool:
    url = os.environ.get("DISCORD_WEBHOOK_URL", "")
    return _post_json(url, {"content": text}) if url else False


def _send_webhook(text: str, payload: dict[str, Any]) -> bool:
    url = os.environ.get("ALERT_WEBHOOK_URL", "")
    if not url:
        return False
    return _post_json(url, {"text": text, "event": "crowd_danger", **payload})


def _send_sms(text: str) -> bool:
    sid = os.environ.get("TWILIO_ACCOUNT_SID", "")
    token = os.environ.get("TWILIO_AUTH_TOKEN", "")
    frm = os.environ.get("TWILIO_FROM", "")
    to = os.environ.get("ALERT_SMS_TO", "")
    if not (sid and token and frm and to and requests):
        return False
    try:
        url = f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json"
        ok_any = False
        for dest in [t.strip() for t in to.split(",") if t.strip()]:
            r = requests.post(
                url,
                data={"From": frm, "To": dest, "Body": text[:1500]},
                auth=(sid, token),
                timeout=8,
            )
            ok_any = ok_any or (200 <= r.status_code < 300)
        return ok_any
    except Exception:
        return False


def _dispatch(text: str, payload: dict[str, Any], chans: list[str]) -> None:
    """Fire every configured channel. Runs on a background thread."""
    senders = {
        "slack":   lambda: _send_slack(text),
        "discord": lambda: _send_discord(text),
        "webhook": lambda: _send_webhook(text, payload),
        "sms":     lambda: _send_sms(text),
    }
    for name in chans:
        try:
            senders[name]()
        except Exception:
            pass


def send_danger_alert(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Notify staff of a confirmed danger. Respects a per-venue cooldown and
    dispatches on a background thread. Returns a small status dict the stream
    surfaces to the UI; never raises.

    payload keys (all optional except venue is nice to have):
      venue, probability, score, action_name, counterfactual
    """
    venue = str(payload.get("venue", "venue"))
    now = time.time()

    with _lock:
        last = _last_sent.get(venue, 0.0)
        on_cooldown = (now - last) < _COOLDOWN_S
        if not on_cooldown:
            _last_sent[venue] = now

    chans = _channels_configured()
    text = _format_message(payload)
    sent_at = time.strftime("%H:%M:%S", time.localtime(now))

    if on_cooldown:
        return {"sent": False, "reason": "cooldown", "channels": chans,
                "message": text, "sent_at": sent_at}

    if not chans:
        # Nothing configured — tell the UI so it can prompt to add a webhook.
        return {"sent": False, "reason": "not_configured", "channels": [],
                "message": text, "sent_at": sent_at}

    threading.Thread(
        target=_dispatch, args=(text, payload, chans), daemon=True).start()

    return {"sent": True, "channels": chans, "message": text, "sent_at": sent_at}
