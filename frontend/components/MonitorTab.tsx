"use client";

import { useState } from "react";
import { analyzeVideo, type AnalyzeResult, type TimelinePoint } from "@/lib/api";

const STATUS_META: Record<string, { badge: string; dot: string; label: string }> = {
  SAFE:        { badge: "badge-safe",    dot: "dot-live",    label: "Safe" },
  WARNING:     { badge: "badge-warning", dot: "dot-warning", label: "Warning" },
  DANGER:      { badge: "badge-danger",  dot: "dot-danger",  label: "Danger" },
  CALIBRATING: { badge: "badge-neutral", dot: "",            label: "Calibrating" },
};

function Timeline({ points }: { points: TimelinePoint[] }) {
  if (!points.length) return null;
  const max = Math.max(...points.map((p) => p.score), 1);
  const counts = { SAFE: 0, WARNING: 0, DANGER: 0 };
  points.forEach((p) => { if (p.status in counts) counts[p.status as keyof typeof counts]++; });

  return (
    <div className="card p-4 animate-fade-in">
      <div className="flex items-center justify-between mb-3">
        <p className="panel-label">Physics Timeline</p>
        <div className="flex items-center gap-3">
          <span className="flex items-center gap-1 font-mono text-[9px] text-emerald">
            <span className="w-2 h-2 rounded-sm bg-emerald/70 inline-block" /> {counts.SAFE} safe
          </span>
          <span className="flex items-center gap-1 font-mono text-[9px] text-amber">
            <span className="w-2 h-2 rounded-sm bg-amber/70 inline-block" /> {counts.WARNING} warn
          </span>
          <span className="flex items-center gap-1 font-mono text-[9px] text-crimson">
            <span className="w-2 h-2 rounded-sm bg-crimson/70 inline-block" /> {counts.DANGER} danger
          </span>
        </div>
      </div>
      <div className="flex items-end gap-px h-12 rounded overflow-hidden">
        {points.map((p, i) => {
          const h = Math.max(6, (p.score / max) * 48);
          const bg =
            p.status === "DANGER"   ? "bg-crimson"         :
            p.status === "WARNING"  ? "bg-amber"           :
            p.status === "SAFE"     ? "bg-teal/70"         : "bg-text3/30";
          return (
            <div
              key={i}
              title={`T+${p.time}s · ${p.status} · σ=${p.score.toFixed(2)}`}
              className={`flex-1 min-w-[2px] rounded-t-sm transition-opacity hover:opacity-80 ${bg}`}
              style={{ height: h }}
            />
          );
        })}
      </div>
      <div className="flex justify-between mt-1.5">
        <span className="font-mono text-[9px] text-text3">T+0s</span>
        <span className="font-mono text-[9px] text-text3">T+{points[points.length - 1]?.time}s</span>
      </div>
    </div>
  );
}

function PipelineStep({ n, label }: { n: number; label: string }) {
  return (
    <div className="flex items-center gap-2.5">
      <div className="w-5 h-5 rounded-full border border-teal/30 bg-teal/10 flex items-center justify-center flex-shrink-0">
        <span className="font-mono text-[9px] text-teal font-medium">{n}</span>
      </div>
      <span className="font-mono text-[11px] text-text2">{label}</span>
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
      setResult(await analyzeVideo(file, venue));
    } catch (e: unknown) {
      setError(String(e));
    } finally {
      setLoad(false);
    }
  }

  const peakStatus = result?.timeline?.reduce((worst, p) => {
    const rank = { DANGER: 3, WARNING: 2, SAFE: 1, CALIBRATING: 0 };
    return (rank[p.status as keyof typeof rank] ?? 0) > (rank[worst as keyof typeof rank] ?? 0) ? p.status : worst;
  }, "SAFE") ?? "SAFE";

  return (
    <div className="flex h-full gap-0">

      {/* ── Left controls ──────────────────────────── */}
      <div className="w-52 flex-shrink-0 flex flex-col gap-3 p-4 border-r border-border overflow-y-auto">

        {/* Upload */}
        <div className="card p-4 flex flex-col gap-3">
          <p className="panel-label">Video Feed</p>

          <label className={`upload-zone h-28 gap-1.5 ${file ? "has-file" : ""}`}>
            {file ? (
              <div className="flex flex-col items-center gap-1 px-2">
                <svg className="w-5 h-5 text-teal" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth="1.5">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M15 10l4.553-2.069A1 1 0 0121 8.82v6.36a1 1 0 01-1.447.89L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                </svg>
                <span className="text-teal text-center text-[11px] break-all px-1 leading-tight">{file.name}</span>
                <span className="text-text3 text-[9px]">{(file.size / 1e6).toFixed(1)} MB</span>
              </div>
            ) : (
              <>
                <svg className="w-6 h-6 mb-1" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth="1.5">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
                </svg>
                <span className="font-medium">Upload video</span>
                <span className="text-[9px] text-text3">.mp4, .mov, .avi</span>
              </>
            )}
            <input type="file" accept="video/*" className="hidden"
              onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
          </label>

          <div>
            <label className="field-label">Venue / Location</label>
            <input
              className="input text-xs"
              placeholder="e.g. Gate A, Main Stage"
              value={venue}
              onChange={(e) => setVenue(e.target.value)}
            />
          </div>

          <button
            className="btn-primary w-full"
            disabled={!file || loading}
            onClick={handleAnalyze}
          >
            {loading ? (
              <><span className="spinner-white" /> Analyzing…</>
            ) : (
              <><svg className="w-3.5 h-3.5" fill="currentColor" viewBox="0 0 16 16"><path d="M6 3l8 5-8 5V3z"/></svg> Run Analysis</>
            )}
          </button>
        </div>

        {/* Pipeline */}
        <div className="card p-4">
          <p className="panel-label mb-3">Analysis Pipeline</p>
          <div className="flex flex-col gap-2.5">
            {[
              "Farneback Optical Flow",
              "8×8 Grid → 256-dim",
              "CNN Encoder → 64-dim",
              "LSTM World Model",
              "CQL RL Policy",
              "Claude Sonnet 4.6",
            ].map((s, i) => <PipelineStep key={s} n={i + 1} label={s} />)}
          </div>
        </div>
      </div>

      {/* ── Centre visualisation ────────────────────── */}
      <div className="flex-1 flex flex-col gap-3 p-4 overflow-y-auto min-w-0">

        {/* Pressure field */}
        <div className="card flex-1 relative min-h-80 flex items-center justify-center">
          {result?.peak_frame_b64 ? (
            <>
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={`data:image/png;base64,${result.peak_frame_b64}`}
                alt="Crowd pressure field"
                className="w-full h-full object-contain animate-fade-in"
              />
              <div className="absolute top-3 left-3">
                <div className={STATUS_META[peakStatus]?.badge ?? "badge-neutral"}>
                  <span className={STATUS_META[peakStatus]?.dot} />
                  Peak: {STATUS_META[peakStatus]?.label}
                </div>
              </div>
              <div className="absolute bottom-3 right-3 font-mono text-[9px] text-text3 bg-void/70 px-2 py-1 rounded">
                CFD Pressure Field
              </div>
            </>
          ) : (
            <div className="flex flex-col items-center gap-3 text-text3">
              {loading ? (
                <>
                  <span className="spinner" />
                  <p className="font-mono text-xs">Processing frames…</p>
                </>
              ) : (
                <>
                  <svg className="w-10 h-10 opacity-20" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth="1">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M15 10l4.553-2.069A1 1 0 0121 8.82v6.36a1 1 0 01-1.447.89L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                  </svg>
                  <p className="font-mono text-xs">Upload a video and run analysis</p>
                </>
              )}
            </div>
          )}
        </div>

        {/* Summary bar */}
        {result && (
          <div className="card px-4 py-3 animate-fade-in">
            <p className="panel-label mb-1">Summary</p>
            <p className="text-sm text-text2 leading-relaxed">{result.summary}</p>
          </div>
        )}

        {/* Timeline */}
        {result?.timeline && <Timeline points={result.timeline} />}

        {/* Error */}
        {error && (
          <div className="card border border-crimson/30 px-4 py-3 animate-fade-in"
            style={{ background: "rgba(248,81,73,0.05)" }}>
            <p className="font-mono text-[10px] text-crimson uppercase tracking-wider mb-1">Error</p>
            <p className="font-mono text-xs text-crimson/80">{error}</p>
          </div>
        )}
      </div>

      {/* ── Right panel ─────────────────────────────── */}
      <div className="w-64 flex-shrink-0 flex flex-col gap-3 p-4 border-l border-border overflow-y-auto">

        {/* Claude */}
        <div className="card flex flex-col flex-1 min-h-0">
          <div className="panel-header">
            <p className="panel-label">Situational Awareness</p>
            <span className="badge-teal text-[9px] px-1.5 py-0.5">Claude</span>
          </div>
          <div className="flex-1 overflow-y-auto p-4 text-xs text-text2 leading-relaxed min-h-0">
            {loading ? (
              <div className="space-y-2">
                <div className="skeleton h-3 w-full" />
                <div className="skeleton h-3 w-5/6" />
                <div className="skeleton h-3 w-4/5" />
                <div className="skeleton h-3 w-full" />
                <div className="skeleton h-3 w-3/4" />
              </div>
            ) : (
              <p className="whitespace-pre-wrap">
                {result?.claude_briefing ?? "Waiting for analysis…"}
              </p>
            )}
          </div>
        </div>

        {/* RL Intervention */}
        <div className="card flex flex-col">
          <div className="panel-header">
            <p className="panel-label">RL Intervention</p>
            <span className="badge-neutral text-[9px] px-1.5 py-0.5">Policy</span>
          </div>
          <div className="p-4 text-xs text-text2 leading-relaxed overflow-y-auto max-h-52">
            {loading ? (
              <div className="space-y-2">
                <div className="skeleton h-3 w-3/4" />
                <div className="skeleton h-3 w-full" />
                <div className="skeleton h-3 w-5/6" />
              </div>
            ) : (
              <p className="whitespace-pre-wrap">
                {result?.rl_explanation ?? "No anomaly detected yet."}
              </p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
