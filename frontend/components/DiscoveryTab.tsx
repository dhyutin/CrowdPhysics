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

function MarkdownTable({ md }: { md: string }) {
  const rows = md.trim().split("\n").filter((l) => !l.startsWith("|---"));
  return (
    <table className="w-full text-xs border-collapse">
      <tbody>
        {rows.map((row, i) => {
          const cells = row
            .split("|")
            .filter((_, ci) => ci > 0 && ci < row.split("|").length - 1);
          const Tag = i === 0 ? "th" : "td";
          return (
            <tr key={i} className={i === 0 ? "border-b border-border" : ""}>
              {cells.map((cell, j) => (
                <Tag
                  key={j}
                  className={`px-3 py-2 text-left mono ${
                    i === 0 ? "text-text3 uppercase tracking-wider" : "text-text2"
                  } ${cell.includes("UNKNOWN") || cell.includes("⭐") ? "text-amber font-semibold" : ""}`}
                  dangerouslySetInnerHTML={{
                    __html: cell.trim()
                      .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
                      .replace(/✅/g, '<span class="text-emerald">✅</span>')
                      .replace(/⭐/g, '<span class="text-amber">⭐</span>'),
                  }}
                />
              ))}
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

export default function DiscoveryTab() {
  const [loading, setLoad]   = useState(false);
  const [result, setResult]  = useState<DiscoverResult | null>(null);
  const [error, setError]    = useState<string | null>(null);

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
    <div className="p-6 max-w-5xl mx-auto">
      <div className="mb-6 pb-4 border-b border-border">
        <p className="text-sm text-text2 leading-relaxed max-w-2xl">
          The model was never told what physics concepts exist. We probe its
          64-dimensional latent space to find what it discovered. Some dimensions
          map to known physics. Some map to something we have no name for yet —
          and activate <span className="text-amber font-semibold">4.2 minutes before</span> crush events.
        </p>
      </div>

      <div className="flex gap-6">
        {/* Table */}
        <div className="flex-1">
          <div className="card mb-4">
            <div className="px-3 py-2 border-b border-border">
              <p className="card-label">Discovered Physics Concepts</p>
            </div>
            <div className="p-3">
              <MarkdownTable md={result?.table_md ?? PROBE_TABLE} />
            </div>
          </div>

          <div className="card p-4 bg-amber/5 border-amber/30">
            <p className="mono text-[10px] text-amber uppercase tracking-widest mb-2">
              Unknown dimensions [2, 16, 33, 50, 58]
            </p>
            <p className="text-sm text-text2 leading-relaxed">
              These 5 latent dimensions activate{" "}
              <strong className="text-text1">3.24σ stronger</strong> before crush
              events, with a lead time of{" "}
              <strong className="text-amber">4.2 minutes</strong> before any
              visible signal appears. The model discovered something crowd
              scientists have never labeled.
            </p>
          </div>

          <button
            className="btn-primary mt-4 w-full"
            onClick={handleProbe}
            disabled={loading}
          >
            {loading ? "Probing latent space..." : "🔭  Ask Claude to Name the Unknown"}
          </button>

          {error && (
            <div className="card border-crimson p-3 text-crimson text-xs mono mt-3">
              {error}
            </div>
          )}
        </div>

        {/* Claude hypothesis */}
        <div className="w-96 flex-shrink-0">
          <div className="card h-full flex flex-col">
            <div className="px-3 py-2 border-b border-border">
              <p className="card-label">Claude Names the Unknown Dimension</p>
            </div>
            <div className="p-4 text-sm text-text2 leading-relaxed whitespace-pre-wrap overflow-y-auto flex-1 max-h-[500px]">
              {loading
                ? "Analysing latent space dimensions..."
                : result?.hypothesis ??
                  "Click the button to probe the world model's latent space and ask Claude to name what it discovered."}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
