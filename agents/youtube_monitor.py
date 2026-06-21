"""
Direct YouTube frame ingest — no Browserbase.

Browserbase drives a real cloud browser and screenshots it, which adds seconds
of session spin-up + navigation latency per capture pass. For YouTube links we
don't need a browser at all: yt-dlp resolves the underlying media/HLS URL and
OpenCV (built with ffmpeg in opencv-python-headless) decodes frames straight
from it. That's a much faster, lower-latency path for the live Monitor.

Returns frames in the exact format the CrowdPhysics pipeline expects:
    (frames, fps)  — list[np.ndarray HxWx3 BGR], effective frames-per-second
so callers can hand them straight to _analyze_frames / _analyze_frames_stream.
"""

from __future__ import annotations

import os

import numpy as np


_YT_FORMAT = "best[height<=720]/bestvideo[height<=720]/best"

# YouTube increasingly answers unauthenticated yt-dlp requests with a
# "Sign in to confirm you're not a bot" challenge. We work around it by trying
# a sequence of strategies until one resolves a stream URL:
#   1. alternate player clients (tv / ios / web_safari dodge the bot gate)
#   2. cookies pulled straight from a local browser (most reliable locally)
#   3. a cookies.txt file, if one is configured
#
# Override the order/values with env vars:
#   YT_PLAYER_CLIENTS      comma list, e.g. "android,tv_embedded,ios,default,web"
#   YT_COOKIES_FROM_BROWSER comma list, e.g. "chrome,brave,safari,firefox,edge"
#   YT_COOKIE_FILE         absolute path to a cookies.txt
#
# Order matters: `android` and `tv_embedded` still expose stream formats once
# the bot gate is cleared with cookies, whereas the web/ios clients now demand
# a PO token and return "No video formats found".
_PLAYER_CLIENTS = [
    c.strip() for c in os.environ.get(
        "YT_PLAYER_CLIENTS", "android,tv_embedded,ios,mweb,tv,default,web"
    ).split(",")
    if c.strip()
]
_COOKIE_BROWSERS = [
    b.strip() for b in os.environ.get(
        "YT_COOKIES_FROM_BROWSER", "chrome,brave,edge,firefox,safari").split(",")
    if b.strip()
]
_COOKIE_FILE = os.environ.get("YT_COOKIE_FILE", "").strip()


def _base_opts() -> dict:
    return {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "format": _YT_FORMAT,
        # Live: grab from the live edge so we analyse "now", not the start.
        "live_from_start": False,
    }


def _strategies() -> list[tuple[str, dict]]:
    """
    Ordered (label, extra-opts) attempts to get past YouTube's bot gate.

    The reliable combo is browser cookies (clears the "confirm you're not a
    bot" challenge) PLUS a player client that still serves formats (android /
    tv_embedded). We try those combos first, then cookie-only, then
    player-client-only for IPs/videos that aren't gated at all.
    """
    out: list[tuple[str, dict]] = []

    def _client_arg(client: str) -> dict:
        return {"extractor_args": {"youtube": {"player_client": [client]}}}

    cookie_sources: list[tuple[str, dict]] = []
    for browser in _COOKIE_BROWSERS:
        cookie_sources.append((f"cookies={browser}",
                               {"cookiesfrombrowser": (browser,)}))
    if _COOKIE_FILE:
        cookie_sources.append((f"cookiefile={_COOKIE_FILE}",
                               {"cookiefile": _COOKIE_FILE}))

    # 1. cookies × working player clients (most reliable when bot-gated).
    for cs_label, cs_opts in cookie_sources:
        for client in _PLAYER_CLIENTS:
            out.append((f"{cs_label}+client={client}",
                        {**cs_opts, **_client_arg(client)}))
    # 2. player-client only (no cookies) — for un-flagged IPs/videos.
    for client in _PLAYER_CLIENTS:
        out.append((f"client={client}", _client_arg(client)))
    return out


def _resolve_stream(url: str):
    """
    Use yt-dlp to turn a YouTube watch/live/short URL into a directly-decodable
    media URL (a progressive MP4 for VODs, an HLS .m3u8 manifest for live).

    Tries several strategies (player clients, then browser cookies) to survive
    YouTube's "confirm you're not a bot" challenge.

    Returns (stream_url, src_fps|None, is_live, title).
    """
    try:
        import yt_dlp
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "yt-dlp is required for YouTube ingest (pip install yt-dlp)"
        ) from exc

    errors: list[str] = []
    for label, extra in _strategies():
        opts = {**_base_opts(), **extra}
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=False)
        except Exception as exc:
            errors.append(f"{label}: {exc}")
            continue

        # Playlists / channels → take the first entry.
        if isinstance(info, dict) and info.get("entries"):
            entries = [e for e in info["entries"] if e]
            if not entries:
                errors.append(f"{label}: no playable entries")
                continue
            info = entries[0]

        stream_url = info.get("url")
        if not stream_url:
            # Fall back to the highest-listed format that carries a direct URL.
            for fmt in reversed(info.get("formats") or []):
                if fmt.get("url"):
                    stream_url = fmt["url"]
                    break
        if not stream_url:
            errors.append(f"{label}: no direct stream URL")
            continue

        try:
            src_fps = float(info.get("fps") or 0) or None
        except (TypeError, ValueError):
            src_fps = None

        return stream_url, src_fps, bool(info.get("is_live")), info.get("title")

    # Every strategy failed — surface a concise, actionable error.
    detail = " | ".join(errors[-3:]) if errors else "no strategies available"
    raise RuntimeError(
        "yt-dlp could not resolve the YouTube stream (likely YouTube's "
        "bot check). Tried player clients + browser cookies. "
        "Set YT_COOKIES_FROM_BROWSER to a browser you're signed into, or "
        f"YT_COOKIE_FILE to a cookies.txt. Last errors: {detail}"
    )


def _resize(frame: np.ndarray, max_width: int) -> np.ndarray:
    import cv2
    h, w = frame.shape[:2]
    if w > max_width:
        nh = max(1, int(round(h * max_width / w)))
        frame = cv2.resize(frame, (max_width, nh), interpolation=cv2.INTER_AREA)
    return frame


def capture_youtube_frames(url: str, n_frames: int = 40, read_stride: int = 2,
                           max_width: int = 640):
    """
    Decode a short buffer of consecutive frames directly from a YouTube stream.

    Args:
        url:         any YouTube watch / live / youtu.be / shorts / embed URL
        n_frames:    how many frames to keep
        read_stride: keep every Nth decoded frame (widens the temporal gap so
                     optical flow sees motion); effective fps = src_fps/stride
        max_width:   downscale wide frames to at most this width

    Returns:
        (frames, fps, meta) — list[np.ndarray BGR], float, dict(is_live,title)
    """
    import cv2

    stream_url, src_fps, is_live, title = _resolve_stream(url)

    cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(
            "OpenCV/ffmpeg could not open the resolved YouTube stream")

    read_stride = max(1, int(read_stride))
    frames: list[np.ndarray] = []
    decoded = 0
    misses = 0
    # Bound the loop so a flaky stream can't hang the request.
    hard_cap = n_frames * read_stride + 120
    try:
        while len(frames) < n_frames and decoded < hard_cap:
            ok, frame = cap.read()
            if not ok or frame is None:
                misses += 1
                if misses > 40:
                    break
                continue
            misses = 0
            if decoded % read_stride == 0:
                frames.append(_resize(frame, max_width))
            decoded += 1
    finally:
        cap.release()

    base_fps = src_fps if src_fps and src_fps > 0 else 25.0
    fps = max(1.0, base_fps / read_stride)
    return frames, fps, {"is_live": is_live, "title": title}


if __name__ == "__main__":
    import sys
    test_url = sys.argv[1] if len(sys.argv) > 1 else "https://youtu.be/jNQXAC9IVRw"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    fr, f, meta = capture_youtube_frames(test_url, n_frames=n)
    print(f"Captured {len(fr)} frames @ ~{f:.1f} fps from {test_url}")
    print(f"  live={meta['is_live']} title={meta['title']!r}")
    if fr:
        print(f"  frame shape: {fr[0].shape}")
