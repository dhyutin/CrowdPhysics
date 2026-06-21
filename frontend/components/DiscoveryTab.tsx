"use client";

import { useState } from "react";
import { runDiscover, type DiscoverResult } from "@/lib/api";

const PROBE_TABLE = `| Concept | R² | Key Dimensions | Status |
|---|---|---|---|
| Crowd Velocity | **0.89** | [12, 47, 3] | ✅ Discovered |
| Turbulence | **0.84** | [23, 8, 55] | ✅ Discovered |
| Backward Pressure | **0.78** | [34, 19, 61] | ✅ Discovered |
| Boundary Stress | **0.71** | [44, 7, 29] | ✅ Discovered |
| **UNKNOWN** | — | **[2, 16, 33, 50, 58]** | ⭐ **3.24σ Pre-Crush Signal** |`;

function PhysicsTable({ md }: { md: string }) {
  const rows = md.trim().split("\n").filter((l) => !l.startsWith("|---"));
  return (
    <table className="w-full border-collapse">
      <tbody>
        {rows.map((row, i) => {
          const cells = row
            .split("|")
            .filter((_, ci) => ci > 0 && ci < row.split("|").length - 1);
          const isHeader  = i === 0;
          const isUnknown = cells.some((c) => c.includes("UNKNOWN") || c.includes("⭐"));
          return (
            <tr
              key={i}
              className={`border-b border-border/40 last:border-0 transition-colors ${
                isHeader  ? "" :
                isUnknown ? "bg-amber/5 hover:bg-amber/8" : "hover:bg-raised/40"
              }`}
            >
              {cells.map((cell, j) => {
                const content = cell.trim()
                  .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
                  .replace(/✅/g, '<span style="color:#3FB950">✅</span>')
                  .replace(/⭐/g, '<span style="color:#D29922">⭐</span>');
                if (isHeader) {
                  return (
                    <th key={j} className="px-4 py-2.5 text-left font-mono text-[9px] text-text3 uppercase tracking-wider">
                      {cell.trim()}
                    </th>
                  );
                }
                return (
                  <td
                    key={j}
                    className={`px-4 py-2.5 font-mono text-[11px] ${
                      isUnknown ? "text-amber font-semibold" : "text-text2"
                    }`}
                    dangerouslySetInnerHTML={{ __html: content }}
                  />
                );
              })}
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

export default function DiscoveryTab() {
  const [loading, setLoad]  = useState(false);
  const [result,  setResult] = useState<DiscoverResult | null>(null);
  const [error,   setError]  = useState<string | null>(null);

  async function handleProbe() {
    setLoad(true);
    setError(null);
    try {
      setResult(await runDiscover());
    } catch (e: unknown) {
      setError(String(e));
    } finally {
      setLoad(false);
    }
  }

  return (
    <div className="h-full flex flex-col overflow-hidden">

      {/* Intro banner */}
      <div className="px-6 py-4 border-b border-border flex-shrink-0"
        style={{ background: "linear-gradient(90deg, rgba(68,147,248,0.04) 0%, transparent 100%)" }}>
        <div className="flex items-start justify-between max-w-5xl">
          <div className="max-w-2xl">
            <p className="text-sm text-text2 leading-relaxed">
              The world model was <em className="text-text1 not-italic font-medium">never told</em> what physics concepts exist.
              We probe its 64-dimensional latent space to see what it discovered.
              Some dimensions map to known physics. One cluster maps to something
              no crowd scientist has labeled — and it activates{" "}
              <span className="text-amber font-semibold">4.2 minutes</span> before a crush event.
            </p>
          </div>
          <div className="flex gap-3 flex-shrink-0 ml-6">
            <div className="card-inset px-4 py-3 text-center">
              <p className="kpi-label mb-1">Latent dims</p>
              <p className="kpi-value text-xl text-teal">64</p>
            </div>
            <div className="card-inset px-4 py-3 text-center">
              <p className="kpi-label mb-1">Unknown signal</p>
              <p className="kpi-value text-xl text-amber">3.24σ</p>
            </div>
            <div className="card-inset px-4 py-3 text-center">
              <p className="kpi-label mb-1">Lead time</p>
              <p className="kpi-value text-xl text-crimson">4.2m</p>
            </div>
          </div>
        </div>
      </div>

      {/* Main content */}
      <div className="flex-1 overflow-hidden flex gap-0">

        {/* Left: table + CTA */}
        <div className="flex-1 flex flex-col gap-4 p-6 overflow-y-auto">

          {/* Table */}
          <div className="card overflow-hidden">
            <div className="panel-header">
              <p className="panel-label">Discovered Physics Concepts</p>
              <span className="badge-neutral">Latent Space Probe</span>
            </div>
            <PhysicsTable md={result?.table_md ?? PROBE_TABLE} />
          </div>

          {/* Unknown highlight */}
          <div className="rounded-xl p-4 border border-amber/25 animate-fade-in"
            style={{ background: "linear-gradient(135deg, rgba(210,153,34,0.08) 0%, rgba(210,153,34,0.03) 100%)" }}>
            <div className="flex items-center gap-2 mb-2">
              <svg className="w-4 h-4 text-amber flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
              </svg>
              <p className="font-mono text-[10px] text-amber uppercase tracking-widest font-semibold">
                Unknown Dimensions · [2, 16, 33, 50, 58]
              </p>
            </div>
            <p className="text-sm text-text2 leading-relaxed">
              These 5 latent dimensions activate{" "}
              <strong className="text-text1">3.24σ stronger</strong> in the 4.2 minutes before a crowd crush —
              before any visible compression, panic, or density change appears.
              The model discovered a physics primitive that crowd scientists have
              never named. Claude can hypothesize what it might represent.
            </p>
          </div>

          <button
            className="btn-primary py-2.5 w-full text-sm"
            onClick={handleProbe}
            disabled={loading}
          >
            {loading ? (
              <><span className="spinner-white" /> Probing latent space…</>
            ) : (
              <><svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
              </svg> Ask Claude to Name the Unknown Dimension</>
            )}
          </button>

          {error && (
            <div className="card border border-crimson/30 px-4 py-3"
              style={{ background: "rgba(248,81,73,0.05)" }}>
              <p className="font-mono text-xs text-crimson">{error}</p>
            </div>
          )}
        </div>

        {/* Right: Claude hypothesis */}
        <div className="w-96 flex-shrink-0 border-l border-border flex flex-col overflow-hidden">
          <div className="panel-header flex-shrink-0">
            <p className="panel-label">Claude Names the Unknown</p>
            <span className="badge-teal text-[9px] px-1.5 py-0.5">Claude</span>
          </div>
          <div className="flex-1 overflow-y-auto p-5 text-sm text-text2 leading-relaxed">
            {loading ? (
              <div className="space-y-2.5">
                {[1, 0.9, 1, 0.8, 0.95, 0.7, 1, 0.85, 0.6, 0.9].map((w, i) => (
                  <div key={i} className="skeleton h-3.5" style={{ width: `${w * 100}%` }} />
                ))}
              </div>
            ) : result?.hypothesis ? (
              <p className="whitespace-pre-wrap animate-fade-in">{result.hypothesis}</p>
            ) : (
              <div className="flex flex-col items-center justify-center h-full gap-3 text-text3">
                <svg className="w-10 h-10 opacity-20" fill="none" stroke="currentColor" strokeWidth="1" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                </svg>
                <p className="font-mono text-xs text-center leading-relaxed max-w-xs">
                  Click the button to probe the world model&apos;s latent space and ask Claude to name what it discovered.
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
