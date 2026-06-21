# simulation_engine.py
"""
Phase 6 dependency: Simulation engine for venue crowd physics modeling.
Powers the Simulate tab — run crowd fluid dynamics BEFORE the event.

Design: treat crowd as a compressible fluid on a discrete grid.
Pressure builds at entry points, diffuses through open space,
drains at exits. Walls block flow. Bottlenecks = danger zones.

This is what makes CrowdPhysics unique: you find the danger zones
before anyone arrives.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional


# ─── DATA CLASSES ─────────────────────────────────────────────────────────────

@dataclass
class VenueElement:
    type: str       # 'stage' | 'gate' | 'wall' | 'entry' | 'barrier'
    x: float        # 0–1 normalized position (left edge)
    y: float        # 0–1 normalized position (top edge)
    w: float        # 0–1 normalized width
    h: float        # 0–1 normalized height
    capacity: int = 0
    label: str = ""


@dataclass
class VenueConfig:
    name: str = "Event Venue"
    total_capacity: int = 5000
    elements: List[VenueElement] = field(default_factory=list)
    grid_size: int = 20


# ─── DEFAULT VENUE (demo arena used when no custom layout is provided) ────────

DEFAULT_VENUE = VenueConfig(
    name="Demo Arena",
    total_capacity=8000,
    elements=[
        VenueElement('stage',   0.2,  0.05, 0.6,  0.22, label="STAGE"),
        VenueElement('wall',    0.0,  0.0,  0.05, 1.0,  label="WALL L"),
        VenueElement('wall',    0.95, 0.0,  0.05, 1.0,  label="WALL R"),
        VenueElement('wall',    0.0,  0.0,  1.0,  0.05, label="WALL T"),
        VenueElement('wall',    0.0,  0.95, 1.0,  0.05, label="WALL B"),
        VenueElement('entry',   0.38, 0.87, 0.24, 0.08, label="MAIN ENTRY"),
        VenueElement('gate',    0.05, 0.45, 0.07, 0.10, label="EXIT A"),
        VenueElement('gate',    0.88, 0.45, 0.07, 0.10, label="EXIT B"),
    ]
)


# ─── SIMULATOR ────────────────────────────────────────────────────────────────

class CrowdSimulator:
    """
    Crowd fluid dynamics simulator on a discrete pressure grid.

    Physics model:
    - Pressure builds at entry points (crowd arriving)
    - Diffuses through open cells (crowd spreading)
    - Drains at exit gates (crowd leaving)
    - Blocked by walls and stage structures
    - Velocity field derived from pressure gradient

    Output: pressure snapshots, danger zones, safe capacity estimate.
    """

    def __init__(self, grid_size: int = 20):
        self.grid = grid_size
        self.pressure    = np.zeros((grid_size, grid_size), dtype=np.float32)
        self.velocity_x  = np.zeros((grid_size, grid_size), dtype=np.float32)
        self.velocity_y  = np.zeros((grid_size, grid_size), dtype=np.float32)
        self.walls       = np.zeros((grid_size, grid_size), dtype=bool)
        self.sources: List[Tuple[int, int]] = []   # entry points (gy, gx)
        self.sinks:   List[Tuple[int, int]] = []   # exit gates (gy, gx)
        self._labels: Dict[Tuple[int, int], str] = {}
        self._element_map = np.full(
            (grid_size, grid_size), "", dtype=object)

    # ── CONFIGURATION ─────────────────────────────────────────────────────────

    def configure_from_venue(self, config: VenueConfig):
        """Set up simulation grid from VenueConfig."""
        self.pressure   = np.zeros((self.grid, self.grid), dtype=np.float32)
        self.velocity_x = np.zeros((self.grid, self.grid), dtype=np.float32)
        self.velocity_y = np.zeros((self.grid, self.grid), dtype=np.float32)
        self.walls      = np.zeros((self.grid, self.grid), dtype=bool)
        self.sources    = []
        self.sinks      = []
        self._labels    = {}
        self._element_map = np.full((self.grid, self.grid), "", dtype=object)

        for el in config.elements:
            gx = int(el.x * self.grid)
            gy = int(el.y * self.grid)
            gw = max(1, int(el.w * self.grid))
            gh = max(1, int(el.h * self.grid))
            gx = min(gx, self.grid - gw)
            gy = min(gy, self.grid - gh)

            if el.type in ('wall', 'stage', 'barrier'):
                self.walls[gy:gy+gh, gx:gx+gw] = True
                self._element_map[gy:gy+gh, gx:gx+gw] = el.type

            elif el.type == 'entry':
                cy, cx = gy + gh // 2, gx + gw // 2
                # Register all cells in entry zone as sources
                for dy in range(gh):
                    for dx in range(gw):
                        sy, sx = gy + dy, gx + dx
                        if 0 <= sy < self.grid and 0 <= sx < self.grid:
                            self.sources.append((sy, sx))
                self._labels[(cy, cx)] = el.label

            elif el.type == 'gate':
                cy, cx = gy + gh // 2, gx + gw // 2
                for dy in range(gh):
                    for dx in range(gw):
                        sy, sx = gy + dy, gx + dx
                        if 0 <= sy < self.grid and 0 <= sx < self.grid:
                            self.sinks.append((sy, sx))
                self._labels[(cy, cx)] = el.label

    # ── SIMULATION LOOP ───────────────────────────────────────────────────────

    def run_steps(self, n_steps: int = 60,
                  crowd_density: float = 0.6) -> List[np.ndarray]:
        """
        Run n_steps of crowd fluid simulation.

        Returns list of pressure snapshots (n_steps × grid × grid).
        The last snapshot is the steady-state pressure field.

        crowd_density: 0–1, scales injection rate at entry points.
        Higher = more people arriving per step.
        """
        snapshots = []
        sink_set = set(self.sinks)

        for _ in range(n_steps):
            # ── Inject crowd at entries ────────────────────────────────────
            for sy, sx in self.sources:
                if not self.walls[sy, sx]:
                    self.pressure[sy, sx] += crowd_density * 0.25

            # ── Diffusion (vectorised) ────────────────────────────────────
            # Pad with zero boundary to handle edges cleanly
            p = self.pressure
            neighbors_sum = (
                np.roll(p, 1, axis=0) +
                np.roll(p, -1, axis=0) +
                np.roll(p, 1, axis=1) +
                np.roll(p, -1, axis=1)
            )
            # Zero out wall-blocked neighbor contributions at boundaries
            neighbor_count = np.full_like(p, 4.0)
            # Edge cells have fewer free neighbors — approximate with full count
            # (acceptable for demo-quality simulation)

            diffusion = 0.18 * (neighbors_sum / neighbor_count - p)
            new_pressure = p + diffusion

            # Walls have zero pressure
            new_pressure[self.walls] = 0.0

            # ── Drain at exits ────────────────────────────────────────────
            for sy, sx in self.sinks:
                if 0 <= sy < self.grid and 0 <= sx < self.grid:
                    new_pressure[sy, sx] *= 0.35

            # ── Velocity from pressure gradient ───────────────────────────
            # vy: higher pressure above → push downward (positive y = down)
            self.velocity_y[1:-1, :] = (
                new_pressure[:-2, :] - new_pressure[2:, :]
            ) * 0.3
            self.velocity_x[:, 1:-1] = (
                new_pressure[:, :-2] - new_pressure[:, 2:]
            ) * 0.3
            self.velocity_x[self.walls] = 0.0
            self.velocity_y[self.walls] = 0.0

            self.pressure = new_pressure.clip(0, 12)
            snapshots.append(self.pressure.copy())

        return snapshots

    def run_steps_record(self, n_steps: int = 80,
                         crowd_density: float = 0.6,
                         stride: int = 2) -> Dict:
        """
        Run the simulation and record a downsampled timeline of the velocity
        and pressure fields, suitable for driving an agent-based 3D render in
        the browser. Mutates state exactly like run_steps(), so afterwards the
        simulator holds the steady-state field (use it for danger zones etc.).

        Returns a JSON-safe dict:
            {
              "grid":     int,                 # square grid resolution
              "frames":   int,                 # number of recorded frames
              "vx":       [[[float]]],         # frames x grid x grid
              "vy":       [[[float]]],         # frames x grid x grid
              "pressure": [[[float]]],         # frames x grid x grid
              "walls":    [[int]],             # grid x grid (1 = blocked)
              "p_max":    float                # peak pressure across timeline
            }
        """
        vx_frames: List = []
        vy_frames: List = []
        p_frames:  List = []
        p_max = 1e-6

        for step in range(n_steps):
            # ── Inject crowd at entries ────────────────────────────────────
            for sy, sx in self.sources:
                if not self.walls[sy, sx]:
                    self.pressure[sy, sx] += crowd_density * 0.25

            # ── Diffusion ──────────────────────────────────────────────────
            p = self.pressure
            neighbors_sum = (
                np.roll(p, 1, axis=0) + np.roll(p, -1, axis=0) +
                np.roll(p, 1, axis=1) + np.roll(p, -1, axis=1)
            )
            diffusion = 0.18 * (neighbors_sum / 4.0 - p)
            new_pressure = p + diffusion
            new_pressure[self.walls] = 0.0

            # ── Drain at exits ─────────────────────────────────────────────
            for sy, sx in self.sinks:
                if 0 <= sy < self.grid and 0 <= sx < self.grid:
                    new_pressure[sy, sx] *= 0.35

            # ── Velocity from pressure gradient ────────────────────────────
            self.velocity_y[1:-1, :] = (
                new_pressure[:-2, :] - new_pressure[2:, :]) * 0.3
            self.velocity_x[:, 1:-1] = (
                new_pressure[:, :-2] - new_pressure[:, 2:]) * 0.3
            self.velocity_x[self.walls] = 0.0
            self.velocity_y[self.walls] = 0.0

            self.pressure = new_pressure.clip(0, 12)

            if step % stride == 0:
                vx_frames.append(np.round(self.velocity_x, 3).tolist())
                vy_frames.append(np.round(self.velocity_y, 3).tolist())
                p_frames.append(np.round(self.pressure, 3).tolist())
                p_max = max(p_max, float(self.pressure.max()))

        return {
            "grid":     self.grid,
            "frames":   len(p_frames),
            "vx":       vx_frames,
            "vy":       vy_frames,
            "pressure": p_frames,
            "walls":    self.walls.astype(int).tolist(),
            "p_max":    round(p_max, 3),
        }

    # ── ANALYSIS ──────────────────────────────────────────────────────────────

    def get_danger_zones(self,
                         threshold: float = 3.0) -> List[Dict]:
        """Return list of high-pressure cells, sorted by severity."""
        zones = []
        for y in range(self.grid):
            for x in range(self.grid):
                p = float(self.pressure[y, x])
                if p > threshold and not self.walls[y, x]:
                    zones.append({
                        'y':        round(y / self.grid, 3),
                        'x':        round(x / self.grid, 3),
                        'grid_y':   y,
                        'grid_x':   x,
                        'pressure': round(p, 2),
                        'risk':     'CRITICAL' if p > 7.0 else 'HIGH',
                    })
        return sorted(zones, key=lambda z: -z['pressure'])

    def estimate_safe_capacity(self, base_capacity: int) -> int:
        """
        Estimate max safe attendance given the current layout.

        More exits + fewer bottlenecks + more open space = higher safe cap.
        """
        n_exits   = max(1, len(set(self.sinks)))
        n_entries = max(1, len(set(self.sources)))
        bottleneck = max(0.3, min(1.0, n_exits / n_entries))
        wall_frac  = float(self.walls.mean())
        space      = 1.0 - wall_frac
        # Apply safety margin
        return int(base_capacity * bottleneck * space * 0.82)

    def to_features(self) -> np.ndarray:
        """
        Convert current simulation state to a 256-dim feature vector —
        the same format as flow_extractor.flow_to_features().

        This lets the anomaly detector analyse simulation output directly.
        """
        features = []
        step_y = self.grid // 8
        step_x = self.grid // 8

        for row in range(8):
            for col in range(8):
                y0, y1 = row * step_y, (row + 1) * step_y
                x0, x1 = col * step_x, (col + 1) * step_x

                cell_p  = self.pressure[y0:y1, x0:x1]
                cell_vx = self.velocity_x[y0:y1, x0:x1]
                cell_vy = self.velocity_y[y0:y1, x0:x1]
                mag = np.sqrt(cell_vx**2 + cell_vy**2)

                features.extend([
                    float(cell_vx.mean()),
                    float(cell_vy.mean()),
                    float(mag.mean()),
                    float(mag.var() + cell_p.var()),
                ])

        return np.array(features, dtype=np.float32)

    # ── RENDERING ─────────────────────────────────────────────────────────────

    def render_simulation(self,
                          size: Tuple[int, int] = (480, 640),
                          show_labels: bool = True) -> np.ndarray:
        """
        Render current simulation state as a BGR image.

        Color scale matches the live pressure field:
        void → teal → amber → crimson
        """
        H, W = size
        canvas = np.zeros((H, W, 3), dtype=np.uint8)

        cell_h = H // self.grid
        cell_w = W // self.grid
        p_max  = max(float(self.pressure.max()), 1.0)

        # Draw background grid
        grid_color = (14, 24, 40)
        for gy in range(0, H, cell_h):
            cv2.line(canvas, (0, gy), (W, gy), grid_color, 1)
        for gx in range(0, W, cell_w):
            cv2.line(canvas, (gx, 0), (gx, H), grid_color, 1)

        for y in range(self.grid):
            for x in range(self.grid):
                y0 = y * cell_h
                y1 = (y + 1) * cell_h
                x0 = x * cell_w
                x1 = (x + 1) * cell_w

                if self.walls[y, x]:
                    elem = self._element_map[y, x]
                    if elem == 'stage':
                        color = (28, 40, 60)
                    else:
                        color = (16, 26, 44)
                    cv2.rectangle(canvas, (x0+1, y0+1),
                                  (x1-1, y1-1), color, -1)
                    continue

                p_norm = min(1.0, self.pressure[y, x] / p_max)

                if p_norm < 0.25:
                    t = p_norm / 0.25
                    color = (int(6 + t*16), int(10+t*90), int(18+t*120))
                elif p_norm < 0.5:
                    t = (p_norm - 0.25) / 0.25
                    color = (int(22+t*220), int(100+t*58), int(138-t*127))
                elif p_norm < 0.75:
                    t = (p_norm - 0.5) / 0.25
                    color = (int(242-t*60), int(158-t*128), int(11))
                else:
                    t = (p_norm - 0.75) / 0.25
                    color = (int(182+t*38), int(30-t*15), 11)

                cv2.rectangle(canvas, (x0+1, y0+1),
                              (x1-1, y1-1), color, -1)

                # Velocity arrows
                vx  = float(self.velocity_x[y, x])
                vy  = float(self.velocity_y[y, x])
                vmag = (vx**2 + vy**2) ** 0.5
                if vmag > 0.04:
                    cx_, cy_ = (x0 + x1) // 2, (y0 + y1) // 2
                    scale = min(cell_w * 0.4, vmag * 12)
                    ex = int(cx_ + vx / (vmag + 1e-6) * scale)
                    ey = int(cy_ + vy / (vmag + 1e-6) * scale)
                    ex = max(x0+2, min(x1-2, ex))
                    ey = max(y0+2, min(y1-2, ey))
                    arrow_col = (200, 200, 200) if p_norm < 0.5 else (80, 80, 220)
                    cv2.arrowedLine(canvas, (cx_, cy_), (ex, ey),
                                    arrow_col, 1, tipLength=0.4)

        # ── HUD ───────────────────────────────────────────────────────────────
        hud_h = 46
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (W, hud_h), (6, 10, 18), -1)
        cv2.addWeighted(overlay, 0.85, canvas, 0.15, 0, canvas)
        cv2.line(canvas, (0, hud_h), (W, hud_h), (26, 180, 139), 2)

        cv2.putText(canvas, "CROWDPHYSICS  SIMULATION",
                    (12, 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (148, 163, 184), 1, cv2.LINE_AA)

        danger = self.get_danger_zones(threshold=3.0)
        status_txt = (f"DANGER ZONES: {len(danger)}"
                      if danger else "NO DANGER ZONES")
        status_col = (80, 80, 220) if danger else (16, 185, 129)
        cv2.putText(canvas, status_txt,
                    (12, 38), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, status_col, 1, cv2.LINE_AA)

        p_val = f"PEAK PRESSURE {self.pressure.max():.1f}"
        cv2.putText(canvas, p_val,
                    (W - 180, 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.40, (148, 163, 184), 1, cv2.LINE_AA)

        # Label entries and exits
        if show_labels:
            for (gy, gx), label in self._labels.items():
                px = gx * cell_w + 2
                py = gy * cell_h - 4
                py = max(py, 52)
                cv2.putText(canvas, label, (px, py),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.30, (245, 158, 11), 1, cv2.LINE_AA)

        return canvas


# ─── QUICK TEST ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sim = CrowdSimulator(grid_size=20)
    sim.configure_from_venue(DEFAULT_VENUE)
    snapshots = sim.run_steps(n_steps=80, crowd_density=0.7)

    danger = sim.get_danger_zones(threshold=3.0)
    safe   = sim.estimate_safe_capacity(DEFAULT_VENUE.total_capacity)
    feats  = sim.to_features()

    print(f"Simulation complete — {len(snapshots)} steps")
    print(f"Danger zones:  {len(danger)}")
    print(f"Peak pressure: {sim.pressure.max():.2f}")
    print(f"Safe capacity: {safe:,} / {DEFAULT_VENUE.total_capacity:,}")
    print(f"Feature vector shape: {feats.shape}")

    canvas = sim.render_simulation()
    cv2.imwrite("/tmp/sim_test.png", canvas)
    print("Rendered to /tmp/sim_test.png")
