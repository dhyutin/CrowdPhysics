# flow_extractor.py
"""
Phase 1: Optical Flow Core
Everything downstream depends on this being correct.
"""

import cv2
import numpy as np


# ─── CORE FLOW EXTRACTION ────────────────────────────────────────────────────

def extract_farneback_flow(frame1, frame2):
    """
    Compute dense optical flow between two frames.

    Farneback is the right choice here:
    - Fast (no GPU needed for demo)
    - Dense (every pixel, not just corners)
    - Proven in 2026 Nature stampede detection paper

    Args:
        frame1, frame2: BGR images, same shape

    Returns:
        flow: np.ndarray shape (H, W, 2)
              flow[:,:,0] = horizontal velocity per pixel
              flow[:,:,1] = vertical velocity per pixel
              Negative y = crowd moving backward (pressure wave signal)
    """
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(
        gray1, gray2,
        flow=None,
        pyr_scale=0.5,    # how much to shrink each pyramid level
        levels=3,          # pyramid levels (more = detects larger motion)
        winsize=15,        # smoothing window (larger = smoother but slower)
        iterations=3,      # refinement passes per level
        poly_n=5,          # polynomial neighborhood size
        poly_sigma=1.2,    # Gaussian for polynomial expansion
        flags=0
    )
    return flow


def flow_to_features(flow, grid_size=8):
    """
    Compress full flow field (H×W×2) to 256-dim feature vector.

    Method: divide frame into 8×8 grid of cells.
    For each cell, compute 4 statistics:
    - mean x-velocity    (crowd drift direction)
    - mean y-velocity    (backward = pressure wave)
    - mean magnitude     (crowd speed)
    - magnitude variance (turbulence — chaotic motion)

    8 × 8 × 4 = 256 features.

    Why this compression?
    The 8×8 grid preserves spatial structure (where in the frame
    the danger is building) while being small enough for an LSTM
    to learn from efficiently.
    """
    H, W = flow.shape[:2]
    cell_h = H // grid_size
    cell_w = W // grid_size
    features = []

    for row in range(grid_size):
        for col in range(grid_size):
            # Extract this grid cell's flow
            y0, y1 = row * cell_h, (row + 1) * cell_h
            x0, x1 = col * cell_w, (col + 1) * cell_w
            cell = flow[y0:y1, x0:x1]

            fx = cell[:, :, 0]   # horizontal velocity
            fy = cell[:, :, 1]   # vertical velocity
            mag = np.sqrt(fx**2 + fy**2)

            features.extend([
                float(fx.mean()),    # x-drift: positive=right, neg=left
                float(fy.mean()),    # y-drift: NEGATIVE = backward pressure
                float(mag.mean()),   # speed
                float(mag.var())     # turbulence (key danger signal)
            ])

    return np.array(features, dtype=np.float32)  # shape: (256,)


# ─── VISUALIZATION ────────────────────────────────────────────────────────────

def render_pressure_field(flow, physics_state=None,
                          grid_size=8, frame_shape=None):
    """
    Render crowd dynamics as a CFD-style pressure field.

    This is the SIGNATURE VISUAL of CrowdPhysics.
    Instead of video with arrows, we show PURE PHYSICS —
    a heatmap of pressure intensity across the venue grid.

    The effect: the crowd becomes invisible.
    Only the physics remains.

    Returns:
        pressure_img: BGR image of the pressure field
        pressure_grid: (8,8) array of pressure values per cell
    """
    if frame_shape is None:
        frame_shape = (480, 640)
    H, W = frame_shape[:2]
    cell_h = H // grid_size
    cell_w = W // grid_size

    status = physics_state.get('status', 'SAFE') if physics_state else 'SAFE'
    score = physics_state.get('score', 0.0) if physics_state else 0.0

    # Pressure field canvas
    canvas = np.zeros((H, W, 3), dtype=np.uint8)

    # Draw background grid (blueprint aesthetic)
    grid_color = (22, 32, 53)  # subtle navy grid
    for y in range(0, H, cell_h):
        cv2.line(canvas, (0, y), (W, y), grid_color, 1)
    for x in range(0, W, cell_w):
        cv2.line(canvas, (x, 0), (x, H), grid_color, 1)

    pressure_grid = np.zeros((grid_size, grid_size))

    for row in range(grid_size):
        for col in range(grid_size):
            y0, y1 = row * cell_h, (row + 1) * cell_h
            x0, x1 = col * cell_w, (col + 1) * cell_w

            # Get this cell's physics from flow
            cell_flow = flow[
                int(row * flow.shape[0] / grid_size):
                int((row+1) * flow.shape[0] / grid_size),
                int(col * flow.shape[1] / grid_size):
                int((col+1) * flow.shape[1] / grid_size)
            ]
            fx = cell_flow[:, :, 0]
            fy = cell_flow[:, :, 1]
            mag = np.sqrt(fx**2 + fy**2)

            # Pressure = speed + turbulence + backward component
            turbulence = float(mag.var())
            backward = float(max(0, -fy.mean()))  # backward flow
            speed = float(mag.mean())
            pressure = speed * 0.3 + turbulence * 0.5 + backward * 0.2
            pressure_grid[row, col] = pressure

            # Map pressure to color (void → teal → amber → crimson)
            p_norm = min(1.0, pressure / 3.0)

            if p_norm < 0.25:
                # void → deep teal
                t = p_norm / 0.25
                color = (
                    int(6 + t * 14),    # R
                    int(10 + t * 85),   # G
                    int(18 + t * 110)   # B
                )
            elif p_norm < 0.5:
                # teal → amber
                t = (p_norm - 0.25) / 0.25
                color = (
                    int(20 + t * 225),  # R
                    int(95 + t * 63),   # G
                    int(128 - t * 117)  # B
                )
            elif p_norm < 0.75:
                # amber → dark red
                t = (p_norm - 0.5) / 0.25
                color = (
                    int(245 - t * 65),  # R
                    int(158 - t * 130), # G
                    int(11 - t * 5)     # B
                )
            else:
                # dark red → crimson
                t = (p_norm - 0.75) / 0.25
                color = (
                    int(180 + t * 42),  # R
                    int(28 - t * 10),   # G
                    int(6)              # B
                )

            # Fill cell with pressure color
            cv2.rectangle(canvas, (x0+1, y0+1), (x1-1, y1-1),
                         color, -1)

            # Draw flow arrow in center of cell
            cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
            mean_fx = float(cell_flow[:, :, 0].mean())
            mean_fy = float(cell_flow[:, :, 1].mean())
            cell_mag = np.sqrt(mean_fx**2 + mean_fy**2)

            if cell_mag > 0.3:
                scale = min(cell_w * 0.35, cell_mag * 8)
                ex = int(cx + mean_fx / (cell_mag + 1e-6) * scale)
                ey = int(cy + mean_fy / (cell_mag + 1e-6) * scale)
                ex = max(x0+2, min(x1-2, ex))
                ey = max(y0+2, min(y1-2, ey))
                # Arrow color: white if safe, yellow if warning, red if danger
                arrow_color = (220, 220, 220)
                if p_norm > 0.75:
                    arrow_color = (80, 80, 220)
                elif p_norm > 0.4:
                    arrow_color = (80, 180, 220)
                cv2.arrowedLine(canvas, (cx, cy), (ex, ey),
                               arrow_color, 1, tipLength=0.35)

    # ── HUD OVERLAY ───────────────────────────────────────────────────────────
    # Semi-transparent top bar
    hud_h = 56
    hud_overlay = canvas.copy()
    cv2.rectangle(hud_overlay, (0, 0), (W, hud_h), (6, 10, 18), -1)
    cv2.addWeighted(hud_overlay, 0.85, canvas, 0.15, 0, canvas)

    # Thin accent line under HUD
    status_line_color = {
        'SAFE': (16, 185, 129),      # emerald
        'WARNING': (245, 158, 11),   # amber
        'DANGER': (220, 38, 38),     # crimson
        'CALIBRATING': (100, 116, 139)
    }.get(status, (100, 116, 139))

    cv2.line(canvas, (0, hud_h), (W, hud_h), status_line_color, 2)

    # Logo text
    cv2.putText(canvas, "CROWDPHYSICS",
                (12, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (148, 163, 184), 1, cv2.LINE_AA)

    # Status
    cv2.putText(canvas, f"STATUS: {status}",
                (12, 42), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, status_line_color, 1, cv2.LINE_AA)

    # Score
    score_text = f"ANOMALY {score:.2f}"
    cv2.putText(canvas, score_text,
                (W - 150, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, (148, 163, 184), 1, cv2.LINE_AA)

    prob = physics_state.get('probability', 0) if physics_state else 0
    cv2.putText(canvas, f"CRUSH RISK {prob*100:.0f}%",
                (W - 150, 42), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, status_line_color, 1, cv2.LINE_AA)

    return canvas, pressure_grid


# ─── VIDEO PROCESSING ─────────────────────────────────────────────────────────

def process_video_to_features(video_path, max_frames=None):
    """
    Extract feature sequence from a video file.
    Returns: list of 256-dim feature vectors (one per frame pair)
    """
    cap = cv2.VideoCapture(video_path)
    features = []
    frames_read = 0

    ret, prev = cap.read()
    while True:
        ret, curr = cap.read()
        if not ret:
            break
        if max_frames and frames_read >= max_frames:
            break

        flow = extract_farneback_flow(prev, curr)
        feat = flow_to_features(flow)
        features.append(feat)

        prev = curr
        frames_read += 1

    cap.release()
    return np.array(features, dtype=np.float32)  # (N, 256)
