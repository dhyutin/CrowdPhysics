"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import {
  analyzeVideo,
  monitorUrl,
  startLiveSession,
  endLiveSession,
  type MonitorResult,
  type TimelinePoint,
} from "@/lib/api";
import AgentTrace from "@/components/AgentTrace";
import ForecastPanel from "@/components/ForecastPanel";

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
  const [mode, setMode]     = useState<"upload" | "live">("upload");
  const [file, setFile]     = useState<File | null>(null);
  const [liveUrl, setLiveUrl] = useState("https://www.abbeyroad.com/crossing");
  const [venue, setVenue]   = useState("Main Stage");
  const [loading, setLoad]  = useState(false);
  const [result, setResult] = useState<MonitorResult | null>(null);
  const [error, setError]   = useState<string | null>(null);

  // Browserbase live-view preview state.
  const [liveView, setLiveView]       = useState<string | null>(null);
  const [liveSession, setLiveSession] = useState<string | null>(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [previewErr, setPreviewErr]   = useState<string | null>(null);

  // Object URL for previewing the uploaded video locally.
  const objectUrl = useMemo(() => (file ? URL.createObjectURL(file) : null), [file]);
  useEffect(() => () => { if (objectUrl) URL.revokeObjectURL(objectUrl); }, [objectUrl]);

  // Release the live Browserbase session when the component unmounts.
  const sessionRef = useRef<string | null>(null);
  useEffect(() => { sessionRef.current = liveSession; }, [liveSession]);
  useEffect(() => () => { if (sessionRef.current) endLiveSession(sessionRef.current); }, []);

  const canRun = mode === "upload" ? !!file : liveUrl.trim().length > 0;

  async function startPreview() {
    const u = liveUrl.trim();
    if (!u) return;
    if (liveSession) endLiveSession(liveSession);
    setLiveView(null);
    setLiveSession(null);
    setResult(null);
    setPreviewErr(null);
    setPreviewLoading(true);
    try {
      const s = await startLiveSession(u);
      setLiveView(s.live_view_url);
      setLiveSession(s.session_id);
    } catch (e: unknown) {
      setPreviewErr(String(e));
    } finally {
      setPreviewLoading(false);
    }
  }

  function switchMode(next: "upload" | "live") {
    if (next === "upload" && liveSession) {
      endLiveSession(liveSession);
      setLiveSession(null);
      setLiveView(null);
    }
    setMode(next);
  }

  async function handleRun() {
    if (!canRun) return;
    setLoad(true);
    setError(null);
    try {
      if (mode === "upload") {
        setResult(await analyzeVideo(file!, venue));
      } else {
        const r = await monitorUrl(
          liveUrl.trim(), venue || "Live Camera", 35, liveSession ?? undefined);
        setResult(r);
        // The warm session was consumed by the capture.
        setLiveSession(null);
        setLiveView(null);
      }
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

        {/* Feed source */}
        <div className="card p-4 flex flex-col gap-3">
          <p className="panel-label">Video Feed</p>

          {/* Source toggle */}
          <div className="grid grid-cols-2 gap-1 p-1 rounded-lg border border-border bg-void/40">
            {([
              { id: "upload", label: "Upload" },
              { id: "live",   label: "Live URL" },
            ] as const).map(({ id, label }) => (
              <button
                key={id}
                onClick={() => switchMode(id)}
                className={`font-mono text-[10px] py-1.5 rounded-md transition-all ${
                  mode === id
                    ? "bg-teal/15 text-teal border border-teal/25"
                    : "text-text3 hover:text-text2 border border-transparent"
                }`}
              >
                {label}
              </button>
            ))}
          </div>

          {mode === "upload" ? (
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
                onChange={(e) => { setFile(e.target.files?.[0] ?? null); setResult(null); }} />
            </label>
          ) : (
            <div className="flex flex-col gap-1.5">
              <label className="field-label">Live stream / webcam URL</label>
              <input
                className="input text-xs"
                placeholder="https://… (any public camera page)"
                value={liveUrl}
                onChange={(e) => setLiveUrl(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") {
                    (e.target as HTMLInputElement).blur();
                    startPreview();
                  }
                }}
              />
              <button
                className="btn-secondary w-full text-[11px]"
                disabled={!liveUrl.trim() || previewLoading}
                onClick={startPreview}
              >
                {previewLoading ? (
                  <><span className="spinner-white" /> Opening browser…</>
                ) : (
                  <>Load live preview</>
                )}
              </button>
              <p className="font-mono text-[9px] text-text3 leading-snug">
                Opens a Browserbase cloud browser so you can watch the feed,
                then <span className="text-teal">Monitor Live</span> captures &amp;
                analyzes it (~30–60s).
              </p>
            </div>
          )}

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
            disabled={!canRun || loading}
            onClick={handleRun}
          >
            {loading ? (
              <><span className="spinner-white" /> {mode === "live" ? "Capturing…" : "Analyzing…"}</>
            ) : mode === "live" ? (
              <><svg className="w-3.5 h-3.5" fill="currentColor" viewBox="0 0 16 16"><circle cx="8" cy="8" r="3"/><path d="M8 1a7 7 0 100 14A7 7 0 008 1zm0 12.5A5.5 5.5 0 118 2.5a5.5 5.5 0 010 11z" opacity="0.5"/></svg> Monitor Live</>
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
              "Optical Flow (RAFT / Farneback)",
              "8×8 Grid → 256-dim",
              "CNN Encoder → 64-dim",
              "LSTM World Model",
              "CQL RL Policy",
              "Claude Sonnet 4.6",
            ].map((s, i) => <PipelineStep key={s} n={i + 1} label={s} />)}
          </div>
        </div>
      </div>

      {/* ── Source (left half) ──────────────────────── */}
      <div className="flex-1 flex flex-col gap-2 p-4 min-w-0 border-r border-border">
        <div className="flex items-center justify-between">
          <p className="panel-label">{mode === "live" ? "Live Source" : "Source Video"}</p>
          {mode === "live" && liveView && (
            <span className="font-mono text-[9px] text-teal flex items-center gap-1.5">
              <span className="dot-live" /> STREAMING
            </span>
          )}
        </div>

        <div className="card flex-1 relative flex items-center justify-center overflow-hidden min-h-80 bg-black/40">
          {mode === "upload" ? (
            objectUrl ? (
              <video
                key={objectUrl}
                src={objectUrl}
                controls autoPlay muted loop playsInline
                className="w-full h-full object-contain animate-fade-in"
              />
            ) : (
              <div className="flex flex-col items-center gap-3 text-text3">
                <svg className="w-10 h-10 opacity-20" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth="1">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M15 10l4.553-2.069A1 1 0 0121 8.82v6.36a1 1 0 01-1.447.89L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                </svg>
                <p className="font-mono text-xs">Upload a video to preview it here</p>
              </div>
            )
          ) : previewLoading ? (
            <div className="flex flex-col items-center gap-3 text-text3">
              <span className="spinner" />
              <p className="font-mono text-xs">Opening live cloud browser…</p>
            </div>
          ) : liveView ? (
            <iframe
              key={liveView}
              src={liveView}
              title="Live camera feed"
              className="w-full h-full border-0 animate-fade-in"
              sandbox="allow-same-origin allow-scripts"
              allow="clipboard-read; clipboard-write"
            />
          ) : result?.source ? (
            <div className="flex flex-col items-center gap-3 text-text3">
              <svg className="w-10 h-10 opacity-20 text-teal" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth="1">
                <path strokeLinecap="round" strokeLinejoin="round" d="M9 12.75L11.25 15 15 9.75M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <p className="font-mono text-xs text-center px-6 leading-relaxed">
                Captured {result.source.frames_captured} frames · session closed.<br />
                Load a new preview to watch again.
              </p>
            </div>
          ) : (
            <div className="flex flex-col items-center gap-3 text-text3 px-6 text-center">
              <svg className="w-10 h-10 opacity-20" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth="1">
                <path strokeLinecap="round" strokeLinejoin="round" d="M2.25 12.76c0 1.6 1.123 2.994 2.707 3.227 1.087.16 2.185.283 3.293.369V21l4.184-4.183a1.14 1.14 0 01.778-.332 48.294 48.294 0 005.83-.498c1.585-.233 2.708-1.626 2.708-3.228V6.741c0-1.602-1.123-2.995-2.707-3.228A48.394 48.394 0 0012 3c-2.392 0-4.744.175-7.043.513C3.373 3.746 2.25 5.14 2.25 6.741v6.018z" />
              </svg>
              <p className="font-mono text-xs">Click “Load live preview” to watch the feed</p>
              {previewErr && (
                <p className="font-mono text-[9px] text-crimson/80 break-all">{previewErr}</p>
              )}
            </div>
          )}
        </div>
        {mode === "live" && (
          <p className="font-mono text-[9px] text-text3 leading-snug">
            Live view streams the actual Browserbase cloud browser — the same
            session the physics pipeline analyzes.
          </p>
        )}
      </div>

      {/* ── Analysis (right half) ────────────────────── */}
      <div className="flex-1 flex flex-col gap-3 p-4 overflow-y-auto min-w-0">
        <div className="flex items-center justify-between">
          <p className="panel-label">Analysis</p>
          {result?.source && (
            <span className="font-mono text-[9px] text-teal flex items-center gap-1.5">
              <span className="dot-live" />
              {result.source.frames_captured} frames @ {result.source.capture_fps} fps
            </span>
          )}
        </div>

        {!result && !loading ? (
          /* Empty analysis state */
          <div className="card flex-1 min-h-80 flex flex-col items-center justify-center gap-3 text-text3">
            <svg className="w-10 h-10 opacity-20" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth="1">
              <path strokeLinecap="round" strokeLinejoin="round" d="M3 13.125C3 12.504 3.504 12 4.125 12h2.25c.621 0 1.125.504 1.125 1.125v6.75C7.5 20.496 6.996 21 6.375 21h-2.25A1.125 1.125 0 013 19.875v-6.75zM9.75 8.625c0-.621.504-1.125 1.125-1.125h2.25c.621 0 1.125.504 1.125 1.125v11.25c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V8.625zM16.5 4.125c0-.621.504-1.125 1.125-1.125h2.25C20.496 3 21 3.504 21 4.125v15.75c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V4.125z" />
            </svg>
            <p className="font-mono text-xs text-center px-6">
              {mode === "live"
                ? "Click “Monitor Live” to analyze the feed"
                : "Click “Run Analysis” to analyze this video"}
            </p>
          </div>
        ) : (
          <>
            {/* Pressure field */}
            <div className="card relative min-h-72 flex-1 flex items-center justify-center">
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
                  <span className="spinner" />
                  <p className="font-mono text-xs">
                    {mode === "live" ? "Pulling live frames via Browserbase…" : "Processing frames…"}
                  </p>
                </div>
              )}
            </div>

            {/* Summary bar */}
            {result && (
              <div className="card px-4 py-3 animate-fade-in">
                <p className="panel-label mb-1">Summary</p>
                <p className="text-sm text-text2 leading-relaxed">{result.summary}</p>
                {result.source && (
                  <p className="font-mono text-[9px] text-text3 mt-1 break-all">
                    Source: {result.source.url}
                  </p>
                )}
              </div>
            )}

            {/* Forecast — potential future of the crowd */}
            {result?.forecast && !result.forecast.error && (
              <ForecastPanel forecast={result.forecast} />
            )}

            {/* Timeline */}
            {result?.timeline && <Timeline points={result.timeline} />}

            {/* Agent trace */}
            {result?.agent_trace && result.agent_trace.length > 0 && (
              <AgentTrace steps={result.agent_trace} title="Agent Trace" />
            )}

            {/* Claude */}
            <div className="card flex flex-col">
              <div className="panel-header">
                <p className="panel-label">Situational Awareness</p>
                <span className="badge-teal text-[9px] px-1.5 py-0.5">Claude</span>
              </div>
              <div className="p-4 text-xs text-text2 leading-relaxed">
                {loading ? (
                  <div className="space-y-2">
                    <div className="skeleton h-3 w-full" />
                    <div className="skeleton h-3 w-5/6" />
                    <div className="skeleton h-3 w-4/5" />
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
              <div className="p-4 text-xs text-text2 leading-relaxed">
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
          </>
        )}

        {/* Error */}
        {error && (
          <div className="card border border-crimson/30 px-4 py-3 animate-fade-in"
            style={{ background: "rgba(248,81,73,0.05)" }}>
            <p className="font-mono text-[10px] text-crimson uppercase tracking-wider mb-1">Error</p>
            <p className="font-mono text-xs text-crimson/80">{error}</p>
          </div>
        )}
      </div>
    </div>
  );
}
