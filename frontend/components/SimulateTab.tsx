"use client";

import { useState } from "react";
import { runSimulation, type SimulateResult } from "@/lib/api";

export default function SimulateTab() {
  const [venueName, setVenueName] = useState("Demo Arena");
  const [capacity, setCapacity]   = useState("5000");
  const [exits, setExits]         = useState("2");
  const [loading, setLoad]        = useState(false);
  const [result, setResult]       = useState<SimulateResult | null>(null);
  const [error, setError]         = useState<string | null>(null);

  async function handleSim() {
    setLoad(true);
    setError(null);
    try {
      const r = await runSimulation({
        venue_name: venueName,
        capacity:   parseInt(capacity) || 5000,
        n_exits:    parseInt(exits) || 2,
      });
      setResult(r);
    } catch (e: unknown) {
      setError(String(e));
    } finally {
      setLoad(false);
    }
  }

  return (
    <div className="flex gap-4 p-4">
      {/* Controls */}
      <div className="w-56 flex-shrink-0 flex flex-col gap-3">
        <div className="card p-3 flex flex-col gap-3">
          <p className="card-label">Venue Config</p>

          {[
            { label: "Venue name",          val: venueName,  set: setVenueName },
            { label: "Expected attendance", val: capacity,   set: setCapacity },
            { label: "Exit gates (1–4)",    val: exits,      set: setExits },
          ].map(({ label, val, set }) => (
            <div key={label}>
              <p className="mono text-[10px] text-text3 uppercase tracking-wider mb-1">
                {label}
              </p>
              <input
                className="bg-void border border-border rounded px-2 py-1 text-xs mono text-text1 w-full"
                value={val}
                onChange={(e) => set(e.target.value)}
              />
            </div>
          ))}

          <button
            className="btn-primary w-full mt-1"
            onClick={handleSim}
            disabled={loading}
          >
            {loading ? "Simulating..." : "⚡  Run Simulation"}
          </button>
        </div>

        <div className="card p-3">
          <p className="card-label mb-2">Physics Model</p>
          {[
            "Crowd = compressible fluid",
            "Pressure builds at entry",
            "Diffuses through open space",
            "Drains at exit gates",
            "Walls block flow",
          ].map((s) => (
            <p key={s} className="mono text-[10px] text-text2 leading-7">
              {s}
            </p>
          ))}
        </div>

        {result && (
          <div className="card p-3">
            <p className="card-label mb-2">Results</p>
            <div className="flex flex-col gap-2">
              <Metric label="Safe capacity"   value={result.safe_capacity.toLocaleString()} color="text-emerald" />
              <Metric label="Peak pressure"   value={`${result.peak_pressure.toFixed(1)} / 12`} color={result.peak_pressure > 6 ? "text-crimson" : result.peak_pressure > 3 ? "text-amber" : "text-emerald"} />
              <Metric label="Danger zones"    value={String(result.danger_zones.length)} color={result.danger_zones.length > 0 ? "text-crimson" : "text-emerald"} />
            </div>
          </div>
        )}
      </div>

      {/* Simulation heatmap */}
      <div className="flex-1 flex flex-col gap-3">
        <div className="card flex-1 relative min-h-[380px] flex items-center justify-center">
          {result?.frame_b64 ? (
            <img
              src={`data:image/png;base64,${result.frame_b64}`}
              alt="Crowd simulation"
              className="w-full h-full object-contain"
            />
          ) : (
            <p className="mono text-xs text-text3">
              {loading ? "Running simulation..." : "Configure venue and run simulation"}
            </p>
          )}
        </div>

        {result && (
          <div className="card px-4 py-2">
            <pre className="mono text-[11px] text-text2 whitespace-pre-wrap">
              {result.metrics}
            </pre>
          </div>
        )}

        {error && (
          <div className="card border-crimson p-3 text-crimson text-xs mono">
            {error}
          </div>
        )}
      </div>

      {/* Safety report */}
      <div className="w-72 flex-shrink-0">
        <div className="card h-full flex flex-col">
          <div className="px-3 py-2 border-b border-border">
            <p className="card-label">Pre-Event Safety Report · Claude</p>
          </div>
          <div className="p-3 text-xs text-text2 leading-relaxed whitespace-pre-wrap overflow-y-auto flex-1">
            {result?.safety_report ?? "Run a simulation to generate a safety report."}
          </div>
        </div>
      </div>
    </div>
  );
}

function Metric({ label, value, color }: { label: string; value: string; color: string }) {
  return (
    <div className="flex justify-between items-baseline border-b border-border/50 pb-1 last:border-0">
      <span className="text-[11px] text-text2">{label}</span>
      <span className={`mono text-sm font-medium ${color}`}>{value}</span>
    </div>
  );
}
