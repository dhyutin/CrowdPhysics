"use client";

import { useState } from "react";
import { runSimulation, type SimulateResult, type DangerZone } from "@/lib/api";

function NumberField({
  label, value, setValue, min, max, step = 1, unit,
}: {
  label: string; value: string; setValue: (v: string) => void;
  min?: number; max?: number; step?: number; unit?: string;
}) {
  const num = parseInt(value) || 0;
  const dec = () => setValue(String(Math.max(min ?? 0, num - step)));
  const inc = () => setValue(String(max !== undefined ? Math.min(max, num + step) : num + step));
  return (
    <div>
      <label className="field-label">{label}</label>
      <div className="flex items-center gap-1">
        <button onClick={dec} className="btn-secondary px-2 py-1.5 text-xs rounded-md flex-shrink-0">−</button>
        <div className="flex-1 relative">
          <input
            className="input text-center text-xs pr-8"
            value={value}
            onChange={(e) => setValue(e.target.value)}
          />
          {unit && (
            <span className="absolute right-2.5 top-1/2 -translate-y-1/2 font-mono text-[9px] text-text3">{unit}</span>
          )}
        </div>
        <button onClick={inc} className="btn-secondary px-2 py-1.5 text-xs rounded-md flex-shrink-0">+</button>
      </div>
      {(min !== undefined || max !== undefined) && (
        <p className="font-mono text-[9px] text-text3 mt-1 text-right">
          {min !== undefined && max !== undefined ? `${min} – ${max}` : ""}
        </p>
      )}
    </div>
  );
}

function KPICard({ label, value, sub, color }: { label: string; value: string; sub?: string; color: string }) {
  return (
    <div className="card-inset p-3 flex flex-col gap-0.5">
      <p className="kpi-label">{label}</p>
      <p className={`kpi-value text-xl ${color}`}>{value}</p>
      {sub && <p className="font-mono text-[9px] text-text3 mt-0.5">{sub}</p>}
    </div>
  );
}

function DangerZoneRow({ z, i }: { z: DangerZone; i: number }) {
  const color = z.risk === "CRITICAL" ? "text-crimson" : "text-amber";
  const badge = z.risk === "CRITICAL" ? "badge-danger" : "badge-warning";
  return (
    <div className="flex items-center gap-2 py-2 border-b border-border/40 last:border-0">
      <span className="font-mono text-[10px] text-text3 w-4">{i + 1}</span>
      <div className="flex-1">
        <p className={`font-mono text-[10px] ${color} font-medium`}>
          ({z.x}, {z.y})
        </p>
        <p className="font-mono text-[9px] text-text3">P = {z.pressure.toFixed(1)}</p>
      </div>
      <span className={badge}>{z.risk}</span>
    </div>
  );
}

export default function SimulateTab() {
  const [venueName, setVenueName] = useState("Demo Arena");
  const [capacity,  setCapacity]  = useState("5000");
  const [exits,     setExits]     = useState("2");
  const [loading,   setLoad]      = useState(false);
  const [result,    setResult]    = useState<SimulateResult | null>(null);
  const [error,     setError]     = useState<string | null>(null);

  async function handleSim() {
    setLoad(true);
    setError(null);
    try {
      setResult(await runSimulation({
        venue_name: venueName,
        capacity:   parseInt(capacity) || 5000,
        n_exits:    parseInt(exits) || 2,
      }));
    } catch (e: unknown) {
      setError(String(e));
    } finally {
      setLoad(false);
    }
  }

  const pressureColor =
    result && result.peak_pressure > 6 ? "text-crimson" :
    result && result.peak_pressure > 3 ? "text-amber"   : "text-emerald";

  return (
    <div className="flex h-full gap-0">

      {/* ── Controls sidebar ──────────────────────── */}
      <div className="w-52 flex-shrink-0 flex flex-col gap-3 p-4 border-r border-border overflow-y-auto">

        <div className="card p-4 flex flex-col gap-3">
          <p className="panel-label">Venue Configuration</p>

          <div>
            <label className="field-label">Venue Name</label>
            <input className="input text-sm" value={venueName}
              onChange={(e) => setVenueName(e.target.value)} />
          </div>

          <NumberField label="Expected Attendance" value={capacity} setValue={setCapacity}
            min={100} step={500} unit="ppl" />

          <NumberField label="Exit Gates" value={exits} setValue={setExits}
            min={1} max={4} unit="gates" />

          <button className="btn-primary w-full mt-1" onClick={handleSim} disabled={loading}>
            {loading ? (
              <><span className="spinner-white" /> Simulating…</>
            ) : (
              <><svg className="w-3.5 h-3.5" fill="currentColor" viewBox="0 0 16 16"><path d="M9.5 1L3 9h5.5L7 15l7.5-9H9L9.5 1z" /></svg> Run Simulation</>
            )}
          </button>
        </div>

        {/* Physics info */}
        <div className="card p-4">
          <p className="panel-label mb-3">Physics Model</p>
          <div className="space-y-2">
            {[
              ["◆", "Crowd = compressible fluid"],
              ["→", "Pressure builds at entry"],
              ["↔", "Diffuses through space"],
              ["↓", "Drains at exit gates"],
              ["■", "Walls block flow"],
            ].map(([sym, txt]) => (
              <div key={txt} className="flex items-start gap-2">
                <span className="font-mono text-[10px] text-teal/50 mt-0.5 w-3 flex-shrink-0">{sym}</span>
                <span className="font-mono text-[10px] text-text2 leading-tight">{txt}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Danger zones */}
        {result && result.danger_zones.length > 0 && (
          <div className="card p-4 animate-fade-in">
            <div className="flex items-center justify-between mb-3">
              <p className="panel-label">Danger Zones</p>
              <span className="badge-danger">{result.danger_zones.length}</span>
            </div>
            <div>
              {result.danger_zones.slice(0, 5).map((z, i) => (
                <DangerZoneRow key={i} z={z} i={i} />
              ))}
              {result.danger_zones.length > 5 && (
                <p className="font-mono text-[9px] text-text3 pt-2">
                  +{result.danger_zones.length - 5} more zones
                </p>
              )}
            </div>
          </div>
        )}
      </div>

      {/* ── Centre heatmap ───────────────────────── */}
      <div className="flex-1 flex flex-col gap-3 p-4 overflow-y-auto min-w-0">

        {/* KPIs */}
        {result && (
          <div className="grid grid-cols-3 gap-3 animate-fade-in">
            <KPICard
              label="Safe Capacity"
              value={result.safe_capacity.toLocaleString()}
              sub="estimated max"
              color="text-emerald"
            />
            <KPICard
              label="Peak Pressure"
              value={`${result.peak_pressure.toFixed(1)}`}
              sub="out of 12 max"
              color={pressureColor}
            />
            <KPICard
              label="Danger Zones"
              value={String(result.danger_zones.length)}
              sub={result.danger_zones.length > 0 ? "require attention" : "all clear"}
              color={result.danger_zones.length > 0 ? "text-crimson" : "text-emerald"}
            />
          </div>
        )}

        {/* Heatmap */}
        <div className="card flex-1 relative min-h-80 flex items-center justify-center">
          {result?.frame_b64 ? (
            <>
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={`data:image/png;base64,${result.frame_b64}`}
                alt="Crowd simulation heatmap"
                className="w-full h-full object-contain animate-fade-in"
              />
              <div className="absolute bottom-3 left-3 font-mono text-[9px] text-text3 bg-void/80 px-2 py-1 rounded">
                CFD Pressure Heatmap · {venueName}
              </div>
              {/* colour legend */}
              <div className="absolute top-3 right-3 flex flex-col gap-1 bg-void/80 px-2 py-2 rounded">
                {[["void", "#0D1117"], ["low", "#4493F8"], ["med", "#D29922"], ["high", "#F85149"]].map(([l, c]) => (
                  <div key={l} className="flex items-center gap-1.5">
                    <span className="w-3 h-2 rounded-sm flex-shrink-0" style={{ background: c }} />
                    <span className="font-mono text-[8px] text-text3 capitalize">{l}</span>
                  </div>
                ))}
              </div>
            </>
          ) : (
            <div className="flex flex-col items-center gap-3 text-text3">
              {loading ? (
                <>
                  <span className="spinner" />
                  <p className="font-mono text-xs">Running fluid dynamics simulation…</p>
                </>
              ) : (
                <>
                  <svg className="w-10 h-10 opacity-20" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth="1">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M9 17.25v1.007a3 3 0 01-.879 2.122L7.5 21h9l-.621-.621A3 3 0 0115 18.257V17.25m6-12V15a2.25 2.25 0 01-2.25 2.25H5.25A2.25 2.25 0 013 15V5.25m18 0A2.25 2.25 0 0018.75 3H5.25A2.25 2.25 0 003 5.25m18 0H3" />
                  </svg>
                  <p className="font-mono text-xs">Configure venue and run simulation</p>
                </>
              )}
            </div>
          )}
        </div>

        {/* Metrics */}
        {result && (
          <div className="card px-4 py-3 animate-fade-in">
            <p className="panel-label mb-2">Simulation Metrics</p>
            <pre className="font-mono text-[11px] text-text2 whitespace-pre-wrap leading-relaxed">
              {result.metrics}
            </pre>
          </div>
        )}

        {error && (
          <div className="card border border-crimson/30 px-4 py-3 animate-fade-in"
            style={{ background: "rgba(248,81,73,0.05)" }}>
            <p className="font-mono text-[10px] text-crimson uppercase tracking-wider mb-1">Error</p>
            <p className="font-mono text-xs text-crimson/80">{error}</p>
          </div>
        )}
      </div>

      {/* ── Safety report ────────────────────────── */}
      <div className="w-64 flex-shrink-0 flex flex-col border-l border-border overflow-hidden">
        <div className="panel-header flex-shrink-0">
          <p className="panel-label">Pre-Event Safety Report</p>
          <span className="badge-teal text-[9px] px-1.5 py-0.5">Claude</span>
        </div>
        <div className="flex-1 overflow-y-auto p-4 text-xs text-text2 leading-relaxed">
          {loading ? (
            <div className="space-y-2">
              {[1,1,0.9,1,0.8,0.7,1,0.9].map((w, i) => (
                <div key={i} className="skeleton h-3" style={{ width: `${w * 100}%` }} />
              ))}
            </div>
          ) : (
            <p className="whitespace-pre-wrap">
              {result?.safety_report ?? "Run a simulation to generate a pre-event safety report."}
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
