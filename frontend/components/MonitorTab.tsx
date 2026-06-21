"use client";

import { useState } from "react";
import { analyzeVideo, type AnalyzeResult, type TimelinePoint } from "@/lib/api";

const STATUS_COLOR: Record<string, string> = {
  SAFE:        "text-emerald border-emerald",
  WARNING:     "text-amber border-amber",
  DANGER:      "text-crimson border-crimson danger-pulse",
  CALIBRATING: "text-text2 border-border",
};

function Timeline({ points }: { points: TimelinePoint[] }) {
  if (!points.length) return null;
  const max = Math.max(...points.map((p) => p.score), 1);
  return (
    <div className="mt-4">
      <p className="mono text-[10px] text-text3 uppercase tracking-widest mb-2">
        Physics Timeline
      </p>
      <div className="flex items-end gap-[2px] h-10">
        {points.map((p, i) => {
          const h = Math.max(4, (p.score / max) * 40);
          const bg =
            p.status === "DANGER"
              ? "bg-crimson"
              : p.status === "WARNING"
              ? "bg-amber"
              : "bg-teal";
          return (
            <div
              key={i}
              title={`T+${p.time}s | ${p.status} | score ${p.score}`}
              className={`flex-1 min-w-[3px] rounded-t ${bg}`}
              style={{ height: h }}
            />
          );
        })}
      </div>
    </div>
  );
}

export default function MonitorTab() {
  const [file, setFile]     = useState<File | null>(null);
  const [venue, setVenue]   = useState("Main Stage");
  const [loading, setLoad]  = useState(false);
  const [result, setResult] = useState<AnalyzeResult | null>(null);
  const [error, setError]   = useState<string | null>(null);

  async function handleAnalyze() {
    if (!file) return;
    setLoad(true);
    setError(null);
    try {
      const r = await analyzeVideo(file, venue);
      setResult(r);
    } catch (e: unknown) {
      setError(String(e));
    } finally {
      setLoad(false);
    }
  }

  return (
    <div className="flex gap-4 p-4">
      {/* Left controls */}
      <div className="w-56 flex-shrink-0 flex flex-col gap-3">
        <div className="card p-3 flex flex-col gap-3">
          <p className="card-label">Video feed</p>
          <label className="flex flex-col items-center justify-center h-32 border border-dashed border-border rounded cursor-pointer hover:border-teal transition-colors text-text3 text-xs mono">
            {file ? (
              <span className="text-text2 text-center px-2 break-all">
                {file.name}
              </span>
            ) : (
              <>
                <span className="text-2xl mb-1">↑</span>
                <span>Upload .mp4</span>
              </>
            )}
            <input
              type="file"
              accept="video/*"
              className="hidden"
              onChange={(e) => setFile(e.target.files?.[0] ?? null)}
            />
          </label>
          <input
            className="bg-void border border-border rounded px-2 py-1 text-xs mono text-text1 w-full"
            placeholder="Venue (e.g. Gate A)"
            value={venue}
            onChange={(e) => setVenue(e.target.value)}
          />
          <button
            className="btn-primary w-full"
            disabled={!file || loading}
            onClick={handleAnalyze}
          >
            {loading ? "Analyzing..." : "▶  Analyze"}
          </button>
        </div>

        {/* Pipeline card */}
        <div className="card p-3">
          <p className="card-label mb-2">Pipeline</p>
          {[
            "RAFT / Farneback Flow",
            "8×8 Grid → 256-dim",
            "CNN Encoder → 64-dim z",
            "LSTM World Model",
            "CQL RL Policy",
            "Claude Sonnet 4.6",
          ].map((s) => (
            <p key={s} className="mono text-[11px] text-text2 leading-7">
              {s}
            </p>
          ))}
        </div>
      </div>

      {/* Centre — pressure field */}
      <div className="flex-1 flex flex-col gap-3">
        <div className="card flex-1 relative min-h-[380px] flex items-center justify-center">
          {result?.peak_frame_b64 ? (
            <img
              src={`data:image/png;base64,${result.peak_frame_b64}`}
              alt="Pressure field"
              className="w-full h-full object-contain"
            />
          ) : (
            <p className="mono text-xs text-text3">
              {loading ? "Processing frames..." : "Upload a video to begin"}
            </p>
          )}
        </div>

        {/* Status bar */}
        {result && (
          <div className="card px-4 py-2">
            <p className="mono text-xs text-text2">{result.summary}</p>
          </div>
        )}

        {result && <Timeline points={result.timeline} />}
      </div>

      {/* Right panel — Claude + RL */}
      <div className="w-64 flex-shrink-0 flex flex-col gap-3">
        {error && (
          <div className="card border-crimson p-3 text-crimson text-xs mono">
            {error}
          </div>
        )}

        <div className="card flex flex-col flex-1">
          <div className="px-3 py-2 border-b border-border">
            <p className="card-label">Claude · Situational Awareness</p>
          </div>
          <div className="p-3 text-xs text-text2 leading-relaxed whitespace-pre-wrap overflow-y-auto flex-1 max-h-64">
            {result?.claude_briefing ?? "Waiting for analysis..."}
          </div>
        </div>

        <div className="card flex flex-col">
          <div className="px-3 py-2 border-b border-border">
            <p className="card-label">RL Policy · Intervention</p>
          </div>
          <div className="p-3 text-xs text-text2 leading-relaxed whitespace-pre-wrap overflow-y-auto max-h-56">
            {result?.rl_explanation || "No anomaly detected yet."}
          </div>
        </div>
      </div>
    </div>
  );
}
