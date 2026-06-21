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
import re
import time

import numpy as np
import requests

BB_BASE = "https://api.browserbase.com/v1"


def _creds() -> tuple[str, str]:
    """Read Browserbase credentials at call time (after .env is loaded)."""
    return (
        os.environ.get("BROWSERBASE_API_KEY", ""),
        os.environ.get("BROWSERBASE_PROJECT_ID", ""),
    )


# ── URL NORMALIZATION ─────────────────────────────────────────────────────────

# Extract an 11-char YouTube video id from the common URL shapes:
#   youtube.com/watch?v=ID   youtu.be/ID   /shorts/ID   /live/ID   /embed/ID
_YT_ID_PATTERNS = [
    re.compile(r"[?&]v=([A-Za-z0-9_-]{11})"),
    re.compile(r"youtu\.be/([A-Za-z0-9_-]{11})"),
    re.compile(r"/shorts/([A-Za-z0-9_-]{11})"),
    re.compile(r"/live/([A-Za-z0-9_-]{11})"),
    re.compile(r"/embed/([A-Za-z0-9_-]{11})"),
]


def youtube_video_id(url: str) -> str | None:
    """Return the 11-char YouTube video id from any common link shape, else None."""
    if not url or "youtu" not in url.lower():
        return None
    for pat in _YT_ID_PATTERNS:
        m = pat.search(url)
        if m:
            return m.group(1)
    return None


_EMBED_HOST = "https://example.com/"


def _load_youtube_embed(page, video_id: str, nav_timeout_ms: int = 60000) -> None:
    """
    Play a YouTube video inside a cloud browser the same way any normal website
    does: load a neutral third-party host page and inject the standard <iframe>
    embed into it.

    Why not navigate straight to the embed URL or frame it inside youtube.com?
      • Top-level navigation to the embed URL has no parent origin → Error 153.
      • Framing the embed inside youtube.com trips YouTube's own page CSP /
        Trusted Types and yields "video unavailable" config errors (152).
    A plain external origin (example.com) is exactly the context YouTube's
    embed player expects, so embeddable videos play normally.
    """
    page.goto(_EMBED_HOST, wait_until="domcontentloaded",
              timeout=nav_timeout_ms)
    time.sleep(0.5)
    page.evaluate(
        """([id, origin]) => {
            document.documentElement.style.margin = '0';
            document.body.replaceChildren();
            document.body.style.cssText = 'margin:0;background:#000;overflow:hidden';
            const f = document.createElement('iframe');
            f.src = 'https://www.youtube.com/embed/' + id +
                '?autoplay=1&mute=1&playsinline=1&rel=0&controls=0' +
                '&modestbranding=1&origin=' + encodeURIComponent(origin);
            f.allow = 'accelerometer; autoplay; clipboard-write; ' +
                'encrypted-media; gyroscope; picture-in-picture; web-share';
            f.setAttribute('frameborder', '0');
            f.setAttribute('allowfullscreen', '');
            f.style.cssText =
                'position:fixed;inset:0;width:100vw;height:100vh;border:0';
            document.body.appendChild(f);
        }""",
        [video_id, _EMBED_HOST.rstrip("/")],
    )
    # Give the embedded player time to negotiate and start buffering.
    time.sleep(3.0)


def _dismiss_consent(page) -> None:
    """Best-effort: click through any cookie / consent dialog."""
    selectors = [
        "button[aria-label*='Accept all' i]",
        "button[aria-label*='Accept the use' i]",
        "button[aria-label*='Reject all' i]",
        "form[action*='consent'] button",
        "button:has-text('Accept all')",
        "button:has-text('I agree')",
    ]
    for sel in selectors:
        try:
            el = page.query_selector(sel)
            if el:
                el.click(timeout=1500)
                time.sleep(0.5)
                return
        except Exception:
            continue


# Default session lifetime requested for live previews (seconds). Plans cap the
# max; if the requested value is rejected we retry without it (project default).
LIVE_SESSION_TIMEOUT_S = 1800


def create_session(keep_alive: bool = False, use_proxy: bool = False,
                   timeout_s: int | None = None) -> dict:
    """
    Create a new Browserbase browser session.

    keep_alive=True keeps the session running after the automation client
    disconnects — required so a live-view URL stays valid for embedding.

    use_proxy=True routes traffic through Browserbase's residential proxy pool.
    This is needed for YouTube, which blocks playback from datacenter IPs with
    a "Sign in to confirm you're not a bot" wall; a residential IP avoids it.

    timeout_s extends the session's auto-release window so a live preview does
    not stop after the (short) default. Falls back to the project default if the
    plan rejects the requested value.
    """
    key, project = _creds()

    def _post(body: dict) -> requests.Response:
        return requests.post(
            f"{BB_BASE}/sessions",
            headers={"x-bb-api-key": key, "Content-Type": "application/json"},
            json=body,
            timeout=30,
        )

    body: dict = {"projectId": project}
    if keep_alive:
        body["keepAlive"] = True
    if use_proxy:
        body["proxies"] = True
    if timeout_s:
        body["timeout"] = int(timeout_s)

    r = _post(body)
    # A too-large timeout for the plan returns 4xx — retry with project default.
    if r.status_code >= 400 and "timeout" in body:
        body.pop("timeout", None)
        r = _post(body)
    r.raise_for_status()
    return r.json()


def list_running_sessions() -> list[dict]:
    """Return all currently RUNNING sessions for the project."""
    key, _ = _creds()
    try:
        r = requests.get(
            f"{BB_BASE}/sessions",
            headers={"x-bb-api-key": key},
            params={"status": "RUNNING"},
            timeout=30,
        )
        r.raise_for_status()
        return r.json() or []
    except Exception:
        return []


def release_running_sessions() -> int:
    """
    Release every RUNNING session for the project. Used to clear orphaned
    keepAlive sessions (abandoned live previews) that otherwise exhaust the
    plan's concurrency cap and cause 429s. Returns the number released.
    """
    n = 0
    for s in list_running_sessions():
        sid = s.get("id")
        if sid:
            try:
                end_session(sid)
                n += 1
            except Exception:
                pass
    return n


def end_session(session_id: str) -> None:
    """Release a Browserbase session."""
    key, project = _creds()
    requests.post(
        f"{BB_BASE}/sessions/{session_id}",
        headers={
            "x-bb-api-key": key,
            "Content-Type": "application/json",
        },
        json={"projectId": project, "status": "REQUEST_RELEASE"},
        timeout=30,
    )


def session_debug(session_id: str) -> dict:
    """Fetch a session's live-view URLs (debuggerFullscreenUrl, pages, ...)."""
    key, _ = _creds()
    r = requests.get(
        f"{BB_BASE}/sessions/{session_id}/debug",
        headers={"x-bb-api-key": key},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def live_view_url(session_id: str) -> str:
    """Return an embeddable live-view URL for a running session."""
    dbg = session_debug(session_id)
    pages = dbg.get("pages") or []
    if pages and pages[0].get("debuggerFullscreenUrl"):
        return pages[0]["debuggerFullscreenUrl"]
    return dbg.get("debuggerFullscreenUrl", "")


def start_live_session(url: str, viewport=(1280, 720),
                       nav_timeout_ms: int = 60000) -> dict:
    """
    Create a Browserbase session, navigate it to `url`, start any video
    playback, and leave it running so its live view can be embedded.

    Returns: {"session_id", "connect_url", "live_view_url", "url"}
    The caller is responsible for releasing the session via end_session().
    """
    from playwright.sync_api import sync_playwright

    if not all(_creds()):
        raise RuntimeError(
            "BROWSERBASE_API_KEY / BROWSERBASE_PROJECT_ID not set")

    vid = youtube_video_id(url)

    # Clear any orphaned keepAlive sessions from abandoned previews so we don't
    # hit the plan's concurrency cap (Browserbase returns 429 otherwise).
    release_running_sessions()

    # keepAlive so the session survives the Playwright disconnect below and
    # the live-view URL stays valid for embedding in the frontend.
    # Use a residential proxy for YouTube to dodge its datacenter-IP bot wall.
    session = create_session(keep_alive=True, use_proxy=bool(vid),
                             timeout_s=LIVE_SESSION_TIMEOUT_S)
    sid = session.get("id")
    connect_url = session.get("connectUrl")
    if not connect_url:
        raise RuntimeError(f"Browserbase session has no connectUrl: {session}")

    try:
        with sync_playwright() as p:
            browser = p.chromium.connect_over_cdp(connect_url)
            # Navigate + start playback, then DISCONNECT (do not close the
            # remote browser — closing would end the session). Dropping the
            # CDP connection leaves the keepAlive session running on the page.
            context = (browser.contexts[0] if browser.contexts
                       else browser.new_context())
            page = context.pages[0] if context.pages else context.new_page()
            page.set_viewport_size(
                {"width": viewport[0], "height": viewport[1]})
            if vid:
                _load_youtube_embed(page, vid, nav_timeout_ms)
            else:
                page.goto(url, wait_until="domcontentloaded",
                          timeout=nav_timeout_ms)
                time.sleep(2.0)
                _dismiss_consent(page)
                _try_start_playback(page)
            # Intentionally NOT calling browser.close().
    except Exception:
        if sid:
            try:
                end_session(sid)
            except Exception:
                pass
        raise

    return {
        "session_id":    sid,
        "connect_url":   connect_url,
        "live_view_url": live_view_url(sid) if sid else "",
        "url":           url,
    }


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
                   nav_timeout_ms: int = 60000,
                   connect_url: str | None = None, navigate: bool = True,
                   release_session_id: str | None = "__own__"):
    """
    Drive a Browserbase cloud browser to a web page and capture rendered
    frames as BGR numpy arrays (the CrowdPhysics pipeline's frame format).

    Connects to a session's CDP `connectUrl` with Playwright, (optionally)
    navigates to `url`, starts any video playback, then screenshots the page
    at a fixed interval.

    By default it creates and releases its own session. To reuse a warm
    session (e.g. an existing live-view session), pass `connect_url` and set
    `navigate=False`; pass `release_session_id` to release it afterwards
    (or None to leave it running).

    Returns:
        (frames, fps)  — list[np.ndarray HxWx3 BGR], effective frames-per-sec
    """
    # cv2 imported lazily so importing this module never hard-requires OpenCV.
    import cv2
    from playwright.sync_api import sync_playwright

    if not all(_creds()):
        raise RuntimeError(
            "BROWSERBASE_API_KEY / BROWSERBASE_PROJECT_ID not set")

    vid = youtube_video_id(url)
    own_session = connect_url is None
    sid_to_release: str | None = None
    if own_session:
        session = create_session(use_proxy=bool(vid))
        sid_to_release = session.get("id")
        connect_url = session.get("connectUrl")
        if not connect_url:
            raise RuntimeError(
                f"Browserbase session has no connectUrl: {session}")
    elif release_session_id and release_session_id != "__own__":
        sid_to_release = release_session_id

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
                if navigate:
                    if vid:
                        _load_youtube_embed(page, vid, nav_timeout_ms)
                        time.sleep(settle_s)
                    else:
                        page.goto(url, wait_until="domcontentloaded",
                                  timeout=nav_timeout_ms)
                        time.sleep(settle_s)
                        _dismiss_consent(page)
                        _try_start_playback(page)
                        time.sleep(1.0)
                else:
                    # Warm session — page already loaded; just ensure playback.
                    _try_start_playback(page)
                    time.sleep(0.5)

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
        if sid_to_release:
            try:
                end_session(sid_to_release)
            except Exception:
                pass

    fps = 1.0 / interval_s if interval_s > 0 else 25.0
    return frames, fps


if __name__ == "__main__":
    import sys

    if not all(_creds()):
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
