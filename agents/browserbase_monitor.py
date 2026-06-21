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
import time

import numpy as np
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


# ── FRAME CAPTURE ─────────────────────────────────────────────────────────────

def _try_start_playback(page) -> None:
    """Best-effort: dismiss overlays and start video playback on the page."""
    selectors = [
        "button.ytp-large-play-button",   # YouTube
        ".ytp-play-button",
        "button[aria-label*='Play' i]",
        "button[title*='Play' i]",
        ".vjs-big-play-button",            # video.js players
        "video",
    ]
    for sel in selectors:
        try:
            el = page.query_selector(sel)
            if el:
                el.click(timeout=1500)
                break
        except Exception:
            continue
    # As a fallback, try to .play() every <video> element directly.
    try:
        page.evaluate(
            "() => document.querySelectorAll('video')"
            ".forEach(v => { v.muted = true; v.play().catch(() => {}); })"
        )
    except Exception:
        pass


def capture_frames(url: str, n_frames: int = 45, interval_s: float = 0.4,
                   viewport=(1280, 720), settle_s: float = 4.0,
                   nav_timeout_ms: int = 60000):
    """
    Drive a Browserbase cloud browser to a web page and capture rendered
    frames as BGR numpy arrays (the CrowdPhysics pipeline's frame format).

    Connects to the session's CDP `connectUrl` with Playwright, navigates to
    `url`, attempts to start any video playback, then screenshots the page at
    a fixed interval. The session is always released afterwards.

    Returns:
        (frames, fps)  — list[np.ndarray HxWx3 BGR], effective frames-per-sec
    """
    # cv2 imported lazily so importing this module never hard-requires OpenCV.
    import cv2
    from playwright.sync_api import sync_playwright

    if not BB_KEY or not BB_PROJECT:
        raise RuntimeError(
            "BROWSERBASE_API_KEY / BROWSERBASE_PROJECT_ID not set")

    session = create_session()
    sid = session.get("id")
    connect_url = session.get("connectUrl")
    if not connect_url:
        raise RuntimeError(f"Browserbase session has no connectUrl: {session}")

    frames: list[np.ndarray] = []
    try:
        with sync_playwright() as p:
            browser = p.chromium.connect_over_cdp(connect_url)
            try:
                context = (browser.contexts[0] if browser.contexts
                           else browser.new_context())
                page = context.pages[0] if context.pages else context.new_page()
                page.set_viewport_size(
                    {"width": viewport[0], "height": viewport[1]})
                page.goto(url, wait_until="domcontentloaded",
                          timeout=nav_timeout_ms)

                time.sleep(settle_s)
                _try_start_playback(page)
                time.sleep(1.0)

                for _ in range(n_frames):
                    png = page.screenshot(type="png")
                    arr = cv2.imdecode(
                        np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
                    if arr is not None:
                        frames.append(arr)
                    page.wait_for_timeout(int(interval_s * 1000))
            finally:
                browser.close()
    finally:
        if sid:
            try:
                end_session(sid)
            except Exception:
                pass

    fps = 1.0 / interval_s if interval_s > 0 else 25.0
    return frames, fps


if __name__ == "__main__":
    import sys

    if not BB_KEY or not BB_PROJECT:
        print(
            "Set BROWSERBASE_API_KEY and BROWSERBASE_PROJECT_ID "
            "to use Browserbase monitoring."
        )
    elif len(sys.argv) > 1:
        # Capture mode: python agents/browserbase_monitor.py <url> [n_frames]
        url = sys.argv[1]
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        print(f"Capturing {n} frames from {url} via Browserbase...")
        try:
            frames, fps = capture_frames(url, n_frames=n)
            shapes = {f.shape for f in frames}
            print(f"Captured {len(frames)} frames @ ~{fps:.1f} fps, "
                  f"resolution(s): {shapes}")
        except Exception as exc:
            print(f"Capture error: {exc}")
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
