# metrics_logger.py
"""
Lightweight training-metrics logger shared by every training job.

For each run it writes, under logs/<job>_<timestamp>/:
  - metrics.csv   crash-safe, one row per logged step (appended live)
  - summary.json  config + final/best values + run metadata
  - curves.png    loss/metric curves (one subplot per metric)

No external services or accounts. matplotlib is optional — if it's not
installed the CSV/JSON are still written and a warning is printed.

Usage:
    log = MetricsLogger("world_model", config={"epochs": 80})
    for epoch in range(epochs):
        ...
        log.log(epoch, loss=avg, best=best_loss)
    log.close(plot_keys=["loss", "best"])
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any


class MetricsLogger:
    def __init__(self, job_name: str, out_dir: str = "logs",
                 config: dict[str, Any] | None = None):
        self.job_name = job_name
        self.start = time.time()
        self.run_id = f"{job_name}_{time.strftime('%Y%m%d_%H%M%S')}"
        self.dir = Path(out_dir) / self.run_id
        self.dir.mkdir(parents=True, exist_ok=True)

        self.csv_path = self.dir / "metrics.csv"
        self.json_path = self.dir / "summary.json"
        self.png_path = self.dir / "curves.png"

        self.rows: list[dict[str, Any]] = []
        self.fieldnames: list[str] | None = None
        self.config = config or {}

        print(f"[metrics] run '{self.run_id}' → {self.dir}/")

    # ── logging ───────────────────────────────────────────────────────────────

    def log(self, step: int, **metrics: Any) -> dict[str, Any]:
        """Record one step. Numeric values are cast to float."""
        row: dict[str, Any] = {
            "step": int(step),
            "elapsed_s": round(time.time() - self.start, 1),
        }
        for k, v in metrics.items():
            try:
                row[k] = float(v)
            except (TypeError, ValueError):
                row[k] = v
        self.rows.append(row)
        self._write_csv(row)
        return row

    def _write_csv(self, row: dict[str, Any]) -> None:
        # If a new key appears, rewrite the whole file with a unified header.
        needs_rewrite = self.fieldnames is None or any(
            k not in self.fieldnames for k in row)
        if needs_rewrite:
            keys: list[str] = []
            for r in self.rows:
                for k in r:
                    if k not in keys:
                        keys.append(k)
            self.fieldnames = keys
            with open(self.csv_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=self.fieldnames)
                w.writeheader()
                w.writerows(self.rows)
        else:
            with open(self.csv_path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=self.fieldnames)
                w.writerow(row)

    # ── finalize ───────────────────────────────────────────────────────────────

    def close(self, plot_keys: list[str] | None = None) -> None:
        """Write summary.json and render curves.png."""
        numeric_keys = self._numeric_keys()
        summary = {
            "run_id": self.run_id,
            "job": self.job_name,
            "config": self.config,
            "n_steps": len(self.rows),
            "duration_s": round(time.time() - self.start, 1),
            "final": self.rows[-1] if self.rows else {},
            "best": self._best(numeric_keys),
            "files": {
                "csv": str(self.csv_path),
                "png": str(self.png_path),
            },
        }
        with open(self.json_path, "w") as f:
            json.dump(summary, f, indent=2)

        self._plot(plot_keys or numeric_keys)
        print(f"[metrics] saved summary → {self.json_path}")

    def _numeric_keys(self) -> list[str]:
        skip = {"step", "elapsed_s"}
        keys: list[str] = []
        for r in self.rows:
            for k, v in r.items():
                if k not in skip and k not in keys and isinstance(v, float):
                    keys.append(k)
        return keys

    def _best(self, keys: list[str]) -> dict[str, Any]:
        """Best (min for losses, max for rewards/accuracy) per metric."""
        best: dict[str, Any] = {}
        for k in keys:
            vals = [r[k] for r in self.rows if isinstance(r.get(k), float)]
            if not vals:
                continue
            is_loss = "loss" in k.lower() or k.lower() in ("best",)
            best[k] = round(min(vals) if is_loss else max(vals), 6)
        return best

    def _plot(self, keys: list[str]) -> None:
        keys = [k for k in keys if any(isinstance(r.get(k), float)
                                       for r in self.rows)]
        if not keys:
            return
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as exc:  # noqa: BLE001
            print(f"[metrics] matplotlib unavailable ({exc}); "
                  f"skipping PNG. Data is in {self.csv_path}")
            return

        steps = [r["step"] for r in self.rows]
        n = len(keys)
        fig, axes = plt.subplots(n, 1, figsize=(8, 2.6 * n), squeeze=False)
        for ax, k in zip(axes[:, 0], keys):
            ys = [r.get(k) for r in self.rows]
            xs = [s for s, y in zip(steps, ys) if isinstance(y, float)]
            yy = [y for y in ys if isinstance(y, float)]
            ax.plot(xs, yy, color="#0ea5e9", linewidth=1.6)
            ax.set_title(f"{self.job_name} — {k}", fontsize=10)
            ax.set_xlabel("step")
            ax.set_ylabel(k)
            ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(self.png_path, dpi=120)
        plt.close(fig)
        print(f"[metrics] saved curves → {self.png_path}")
