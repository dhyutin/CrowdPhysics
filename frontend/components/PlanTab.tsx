"use client";

import { useMemo, useState } from "react";
import {
  planEvent,
  type PlanResult,
  type DangerZone,
  type VenueLayout,
} from "@/lib/api";
import AgentTrace from "@/components/AgentTrace";

const PURPOSES = [
  "Concert",
  "Sports match",
  "Rally / protest",
  "Expo / conference",
  "Religious gathering",
  "Night market",
  "Evacuation drill",
];

const EL_STYLE: Record<string, { fill: string; label: string }> = {
  stage:   { fill: "#2DD4BF", label: "Stage" },
  wall:    { fill: "#30363D", label: "Wall" },
  barrier: { fill: "#A371F7", label: "Barrier" },
  entry:   { fill: "#3FB950", label: "Entry" },
  gate:    { fill: "#4493F8", label: "Exit" },
};

function KPICard({ label, value, sub, color }: { label: string; value: string; sub?: string; color: string }) {
  return (
    <div className="card-inset p-3 flex flex-col gap-0.5">
      <p className="kpi-label">{label}</p>
      <p className={`kpi-value text-xl ${color}`}>{value}</p>
      {sub && <p className="font-mono text-[9px] text-text3 mt-0.5">{sub}</p>}
    </div>
  );
}

function LayoutDiagram({ layout, danger }: { layout: VenueLayout; danger: DangerZone[] }) {
  const S = 280;
  return (
    <svg viewBox={`0 0 ${S} ${S}`} className="w-full h-full rounded-md">
      <rect x={0} y={0} width={S} height={S} fill="#0D1117" />
      {/* danger heat dots underneath the structures */}
      {danger.map((z, i) => (
        <circle
          key={`d${i}`}
          cx={z.x * S}
          cy={z.y * S}
          r={z.risk === "CRITICAL" ? 12 : 8}
          fill={z.risk === "CRITICAL" ? "#F85149" : "#D29922"}
          fillOpacity={0.28}
        />
      ))}
      {layout.elements.map((e, i) => {
        const st = EL_STYLE[e.type] ?? EL_STYLE.wall;
        const isObstacle = e.type === "wall" || e.type === "stage" || e.type === "barrier";
        return (
          <g key={i}>
            <rect
              x={e.x * S} y={e.y * S}
              width={Math.max(2, e.w * S)} height={Math.max(2, e.h * S)}
              fill={st.fill} fillOpacity={isObstacle ? 0.85 : 0.55}
              stroke={st.fill} strokeWidth={1}
              rx={e.type === "gate" || e.type === "entry" ? 2 : 0}
            />
            {e.label && (e.type === "gate" || e.type === "entry" || e.type === "stage") && (
              <text
                x={(e.x + e.w / 2) * S} y={(e.y + e.h / 2) * S}
                fill="#fff" fontSize={7} textAnchor="middle" dominantBaseline="middle"
                style={{ fontFamily: "monospace" }}
              >
                {e.label.slice(0, 16)}
              </text>
            )}
          </g>
        );
      })}
    </svg>
  );
}

export default function PlanTab() {
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [purpose, setPurpose]     = useState("Concert");
  const [capacity, setCapacity]   = useState("");
  const [density, setDensity]     = useState("65");
  const [loading, setLoad]        = useState(false);
  const [result, setResult]       = useState<PlanResult | null>(null);
  const [error, setError]         = useState<string | null>(null);

  const imagePreview = useMemo(
    () => (imageFile ? URL.createObjectURL(imageFile) : null),
    [imageFile]
  );

  async function handlePlan() {
    if (!imageFile) { setError("Upload a photo of the location first."); return; }
    setLoad(true);
    setError(null);
    try {
      const res = await planEvent(
        imageFile,
        purpose,
        parseInt(capacity) || 0,
        (parseInt(density) || 65) / 100
      );
      setResult(res);
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

      {/* ── Controls ─────────────────────────────────── */}
      <div className="w-56 flex-shrink-0 flex flex-col gap-3 p-4 border-r border-border overflow-y-auto">
        <div className="card p-4 flex flex-col gap-3">
          <p className="panel-label">Plan a Space</p>
          <p className="font-mono text-[10px] text-text3 leading-tight">
            Upload a photo of a location. Agents reconstruct it, simulate the
            crowd, and design how to arrange people for your event.
          </p>

          {/* Photo */}
          <label className="card-inset rounded-md p-3 flex flex-col items-center gap-2 cursor-pointer hover:border-teal/40 border border-dashed border-border transition-colors">
            {imagePreview ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img src={imagePreview} alt="location" className="w-full h-24 object-cover rounded" />
            ) : (
              <>
                <svg className="w-6 h-6 text-text3" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth="1.2">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5l4.5-4.5 3 3 4.5-4.5 6 6M3 19.5h18a1.5 1.5 0 001.5-1.5V6A1.5 1.5 0 0021 4.5H3A1.5 1.5 0 001.5 6v12A1.5 1.5 0 003 19.5z" />
                </svg>
                <span className="font-mono text-[10px] text-text3">Click to choose photo</span>
                <span className="font-mono text-[8px] text-text3">overhead · ground · floor plan</span>
              </>
            )}
            <input
              type="file"
              accept="image/png,image/jpeg,image/webp"
              className="hidden"
              onChange={(e) => { setImageFile(e.target.files?.[0] ?? null); setResult(null); }}
            />
          </label>
          {imageFile && (
            <p className="font-mono text-[9px] text-text3 truncate">{imageFile.name}</p>
          )}

          {/* Purpose */}
          <div>
            <label className="field-label">Purpose</label>
            <input
              className="input text-sm"
              value={purpose}
              onChange={(e) => setPurpose(e.target.value)}
              placeholder="What is the space for?"
            />
            <div className="flex flex-wrap gap-1 mt-2">
              {PURPOSES.map((p) => (
                <button
                  key={p}
                  onClick={() => setPurpose(p)}
                  className={`font-mono text-[9px] px-1.5 py-0.5 rounded border transition-colors ${
                    purpose === p
                      ? "bg-teal/15 text-teal border-teal/30"
                      : "text-text3 border-border hover:text-text2"
                  }`}
                >
                  {p}
                </button>
              ))}
            </div>
          </div>

          <div>
            <label className="field-label">Expected Attendance</label>
            <input
              className="input text-sm"
              value={capacity}
              onChange={(e) => setCapacity(e.target.value)}
              placeholder="auto-estimate"
            />
          </div>

          <div>
            <label className="field-label">Crowd Density (%)</label>
            <input
              className="input text-sm"
              value={density}
              onChange={(e) => setDensity(e.target.value)}
            />
          </div>

          <button className="btn-primary w-full mt-1" onClick={handlePlan} disabled={loading || !imageFile}>
            {loading ? (
              <><span className="spinner-white" /> Agents planning…</>
            ) : (
              <><svg className="w-3.5 h-3.5" fill="currentColor" viewBox="0 0 16 16"><path d="M9.5 1L3 9h5.5L7 15l7.5-9H9L9.5 1z" /></svg> Plan with Agents</>
            )}
          </button>
        </div>

        {/* Detected layout meta */}
        {result?.layout && (
          <div className="card p-4 animate-fade-in flex flex-col gap-2">
            <div className="flex items-center justify-between">
              <p className="panel-label">Reconstruction</p>
              <span className="badge-teal text-[9px] px-1.5 py-0.5">
                {Math.round((result.layout.confidence ?? 0) * 100)}% conf
              </span>
            </div>
            <div className="flex flex-wrap gap-1.5">
              {Object.entries(EL_STYLE).map(([k, v]) => (
                <span key={k} className="flex items-center gap-1">
                  <span className="w-2.5 h-2.5 rounded-sm" style={{ background: v.fill }} />
                  <span className="font-mono text-[8px] text-text3">{v.label}</span>
                </span>
              ))}
            </div>
            {result.layout.notes && (
              <p className="font-mono text-[9px] text-text2 leading-tight italic">
                “{result.layout.notes}”
              </p>
            )}
          </div>
        )}
      </div>

      {/* ── Virtual simulation ───────────────────────── */}
      <div className="flex-1 flex flex-col gap-3 p-4 overflow-y-auto min-w-0">
        {/* KPIs */}
        {result && (
          <div className="grid grid-cols-3 gap-3 animate-fade-in">
            <KPICard label="Safe Capacity" value={result.safe_capacity.toLocaleString()} sub="for this purpose" color="text-emerald" />
            <KPICard label="Peak Pressure" value={result.peak_pressure.toFixed(1)} sub="out of 12 max" color={pressureColor} />
            <KPICard label="Danger Zones" value={String(result.danger_zones.length)} sub={result.danger_zones.length ? "require attention" : "all clear"} color={result.danger_zones.length ? "text-crimson" : "text-emerald"} />
          </div>
        )}

        {!result && !loading ? (
          <div className="card flex-1 min-h-80 flex flex-col items-center justify-center gap-3 text-text3 px-6 text-center">
            <svg className="w-12 h-12 opacity-20" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth="1">
              <path strokeLinecap="round" strokeLinejoin="round" d="M9 6.75V15m6-6v8.25m.503 3.498l4.875-2.437c.381-.19.622-.58.622-1.006V4.82c0-.836-.88-1.38-1.628-1.006l-3.869 1.934c-.317.159-.69.159-1.006 0L9.503 3.252a1.125 1.125 0 00-1.006 0L3.622 5.689C3.24 5.88 3 6.27 3 6.695V19.18c0 .836.88 1.38 1.628 1.006l3.869-1.934c.317-.159.69-.159 1.006 0l4.994 2.497c.317.158.69.158 1.006 0z" />
            </svg>
            <p className="font-mono text-xs">Upload a location photo and click “Plan with Agents”</p>
          </div>
        ) : loading && !result ? (
          <div className="card flex-1 min-h-80 flex flex-col items-center justify-center gap-3 text-text3">
            <span className="spinner" />
            <p className="font-mono text-xs">Vision → simulation → planning…</p>
          </div>
        ) : (
          <>
            {/* Source + reconstruction side by side */}
            <div className="grid grid-cols-2 gap-3 animate-fade-in">
              <div className="card flex flex-col">
                <div className="panel-header"><p className="panel-label">Source Photo</p></div>
                <div className="flex-1 flex items-center justify-center bg-black/40 min-h-[200px]">
                  {imagePreview && (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img src={imagePreview} alt="location" className="w-full h-full object-contain max-h-64" />
                  )}
                </div>
              </div>
              <div className="card flex flex-col">
                <div className="panel-header">
                  <p className="panel-label">Reconstructed Venue</p>
                  <span className="badge-teal text-[9px] px-1.5 py-0.5">top-down</span>
                </div>
                <div className="flex-1 flex items-center justify-center min-h-[200px] p-2">
                  {result?.layout && (
                    <LayoutDiagram layout={result.layout} danger={result.danger_zones} />
                  )}
                </div>
              </div>
            </div>

            {/* Crowd simulation heatmap */}
            <div className="card flex flex-col animate-fade-in">
              <div className="panel-header">
                <p className="panel-label">Crowd Simulation · {result?.venue_name}</p>
                <span className="badge-neutral text-[9px] px-1.5 py-0.5">CFD</span>
              </div>
              <div className="relative flex items-center justify-center min-h-72 bg-black/30">
                {result?.frame_b64 && (
                  <>
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img
                      src={`data:image/png;base64,${result.frame_b64}`}
                      alt="Crowd simulation"
                      className="w-full h-full object-contain"
                    />
                    <div className="absolute top-3 right-3 flex flex-col gap-1 bg-void/80 px-2 py-2 rounded">
                      {[["void", "#0D1117"], ["low", "#4493F8"], ["med", "#D29922"], ["high", "#F85149"]].map(([l, c]) => (
                        <div key={l} className="flex items-center gap-1.5">
                          <span className="w-3 h-2 rounded-sm flex-shrink-0" style={{ background: c }} />
                          <span className="font-mono text-[8px] text-text3 capitalize">{l}</span>
                        </div>
                      ))}
                    </div>
                  </>
                )}
              </div>
            </div>
          </>
        )}

        {error && (
          <div className="card border border-crimson/30 px-4 py-3 animate-fade-in"
            style={{ background: "rgba(248,81,73,0.05)" }}>
            <p className="font-mono text-[10px] text-crimson uppercase tracking-wider mb-1">Error</p>
            <p className="font-mono text-xs text-crimson/80">{error}</p>
          </div>
        )}
      </div>

      {/* ── Agent plan & report ──────────────────────── */}
      <div className="w-80 flex-shrink-0 flex flex-col border-l border-border overflow-y-auto">
        <div className="p-4 flex flex-col gap-3">
          {/* Agent trace */}
          {result?.agent_trace && (
            <AgentTrace steps={result.agent_trace} title="Planning Agents" />
          )}

          {/* Arrangement plan */}
          <div className="card flex flex-col">
            <div className="panel-header">
              <p className="panel-label">Arrangement Plan</p>
              <span className="badge-teal text-[9px] px-1.5 py-0.5">Planner</span>
            </div>
            <div className="p-4 text-xs text-text2 leading-relaxed">
              {loading ? (
                <div className="space-y-2">
                  {[1, 0.9, 1, 0.8, 0.7, 1].map((w, i) => (
                    <div key={i} className="skeleton h-3" style={{ width: `${w * 100}%` }} />
                  ))}
                </div>
              ) : (
                <p className="whitespace-pre-wrap">
                  {result?.plan ?? "Plan a space to generate an agent arrangement."}
                </p>
              )}
            </div>
          </div>

          {/* Safety report */}
          {result?.safety_report && (
            <div className="card flex flex-col">
              <div className="panel-header">
                <p className="panel-label">Safety Report</p>
                <span className="badge-teal text-[9px] px-1.5 py-0.5">Claude</span>
              </div>
              <div className="p-4 text-xs text-text2 leading-relaxed">
                <p className="whitespace-pre-wrap">{result.safety_report}</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
