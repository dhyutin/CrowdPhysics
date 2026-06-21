"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import {
  streamAnalyze,
  streamMonitorUrl,
  startLiveSession,
  endLiveSession,
  type MonitorResult,
  type TimelinePoint,
  type Forecast,
  type Trend,
  type Hotspot,
  type CaptureSource,
  type LiveTick,
  type AgentTraceStep,
} from "@/lib/api";
import AgentTrace from "@/components/AgentTrace";
import ForecastPanel from "@/components/ForecastPanel";
import TrendPanel from "@/components/TrendPanel";
import FilmPlayer, { type FilmFrame } from "@/components/FilmPlayer";

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

const STATUS_TEXT: Record<string, string> = {
  SAFE: "#3FB950", WARNING: "#D29922", DANGER: "#F85149", CALIBRATING: "#6E7681",
};

function LiveStatusBar({
  now, status, score, processed, total, phase,
}: {
  now: number; status: string; score: number;
  processed: number; total: number; phase: string;
}) {
  const color = STATUS_TEXT[status] ?? "#6E7681";
  const pct = total > 1 ? Math.min(100, Math.round((processed / (total - 1)) * 100)) : 0;
  const meta = STATUS_META[status] ?? STATUS_META.CALIBRATING;
  return (
    <div className="card p-3 animate-fade-in">
      <div className="flex items-center justify-between mb-2">
        <span className="font-mono text-[9px] text-crimson flex items-center gap-1.5">
          <span className="dot-live" /> LIVE
        </span>
        <span className="font-mono text-[9px] text-text3">{phase}</span>
      </div>
      <div className="flex items-center gap-4">
        <div className="flex flex-col">
          <span className="kpi-label">Now</span>
          <span className="font-mono text-lg text-text1 tabular-nums">
            T+{now.toFixed(1)}s
          </span>
        </div>
        <div className="flex flex-col">
          <span className="kpi-label">Status</span>
          <div className={meta.badge}>
            <span className={meta.dot} /> {meta.label}
          </div>
        </div>
        <div className="flex flex-col">
          <span className="kpi-label">Anomaly</span>
          <span className="font-mono text-lg tabular-nums" style={{ color }}>
            {score.toFixed(2)}σ
          </span>
        </div>
      </div>
      <div className="mt-2.5 h-1 rounded-full bg-void/60 overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-200"
          style={{ width: `${pct}%`, background: color }}
        />
      </div>
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

  // Live streaming state (frame-by-frame ticks from the SSE-style stream).
  const [streaming, setStreaming]   = useState(false);
  const [liveTicks, setLiveTicks]   = useState<TimelinePoint[]>([]);
  const [liveForecast, setLiveForecast] = useState<Forecast | null>(null);
  const [liveTrend, setLiveTrend]   = useState<Trend | null>(null);
  const [liveHotspot, setLiveHotspot] = useState<Hotspot | null>(null);
  const [liveFrame, setLiveFrame]   = useState<string | null>(null);
  const [film, setFilm]             = useState<FilmFrame[]>([]);
  const [liveStatus, setLiveStatus] = useState("CALIBRATING");
  const [liveNow, setLiveNow]       = useState(0);
  const [liveScore, setLiveScore]   = useState(0);
  const [liveTotal, setLiveTotal]   = useState(0);
  const [livePhase, setLivePhase]   = useState("");
  const [liveSource, setLiveSource] = useState<CaptureSource | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  useEffect(() => () => abortRef.current?.abort(), []);

  // Browserbase live-view preview state.
  const [liveView, setLiveView]       = useState<string | null>(null);
  const [liveSession, setLiveSession] = useState<string | null>(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [previewErr, setPreviewErr]   = useState<string | null>(null);
  // Continuous live monitoring: once the preview session is warm we keep
  // capturing + analysing on a loop until the user stops.
  const [monitoring, setMonitoring]   = useState(false);

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
    setMonitoring(false);
    abortRef.current?.abort();
    if (liveSession) endLiveSession(liveSession);
    setLiveView(null);
    setLiveSession(null);
    setResult(null);
    setLiveTicks([]);
    setLiveForecast(null);
    setLiveTrend(null);
    setLiveHotspot(null);
    setLiveFrame(null);
    setFilm([]);
    setPreviewErr(null);
    setPreviewLoading(true);
    try {
      const s = await startLiveSession(u);
      setLiveView(s.live_view_url);
      setLiveSession(s.session_id);
      // As soon as the live session is warm, begin analysing automatically —
      // no Monitor click required. The loop effect picks this up.
      setMonitoring(true);
    } catch (e: unknown) {
      setPreviewErr(String(e));
    } finally {
      setPreviewLoading(false);
    }
  }

  function stopMonitoring() {
    setMonitoring(false);
    abortRef.current?.abort();
    setStreaming(false);
    setLoad(false);
    setLivePhase("Stopped");
  }

  function switchMode(next: "upload" | "live") {
    if (next === "upload") {
      setMonitoring(false);
      abortRef.current?.abort();
      if (liveSession) {
        endLiveSession(liveSession);
        setLiveSession(null);
        setLiveView(null);
      }
    }
    setMode(next);
  }

  async function handleRun() {
    if (!canRun || streaming) return;
    abortRef.current?.abort();
    const ac = new AbortController();
    abortRef.current = ac;

    setError(null);
    setLoad(true);
    setStreaming(true);
    // Upload starts each run from a clean slate. Live monitoring loops, so we
    // KEEP the last pass's forecast/trend/hotspot/result visible during the
    // capture gap — they refresh as new ticks arrive (feels continuous).
    if (mode === "upload") {
      setResult(null);
      setLiveTicks([]);
      setLiveForecast(null);
      setLiveTrend(null);
      setLiveHotspot(null);
      setLiveFrame(null);
      setFilm([]);
    }
    setLiveStatus("CALIBRATING");
    setLiveNow(0);
    setLiveScore(0);
    setLiveTotal(0);
    setLiveSource(null);
    setLivePhase(mode === "live"
      ? "Capturing buffer via Browserbase…" : "Calibrating…");

    let capturedSource: CaptureSource | undefined;
    const onEvent = (ev: LiveTick) => {
      switch (ev.type) {
        case "source":
          capturedSource = {
            url: ev.url ?? "",
            frames_captured: ev.frames_captured ?? 0,
            capture_fps: ev.capture_fps ?? 0,
          };
          setLiveSource(capturedSource);
          setLivePhase("Streaming live…");
          break;
        case "calibrating":
          setLiveTotal(ev.total_frames ?? 0);
          setLiveStatus("CALIBRATING");
          setLivePhase("Streaming live…");
          // New pass begins → reset only the per-frame timeline; keep the
          // rolling forecast/trend/marker from the prior pass until refreshed.
          setLiveTicks([]);
          setResult(null);
          // Start a fresh synchronized film for this pass.
          setFilm([]);
          break;
        case "tick": {
          const pt: TimelinePoint = {
            time: ev.time ?? 0,
            status: (ev.status ?? "SAFE") as TimelinePoint["status"],
            score: ev.score ?? 0,
            probability: ev.probability ?? 0,
          };
          setLiveTicks((prev) => {
            const next = [...prev, pt];
            return next.length > 180 ? next.slice(next.length - 180) : next;
          });
          setLiveNow(pt.time);
          setLiveStatus(pt.status);
          setLiveScore(pt.score);
          if (ev.forecast && !ev.forecast.error) setLiveForecast(ev.forecast);
          if (ev.trend) setLiveTrend(ev.trend);
          if (ev.hotspot) setLiveHotspot(ev.hotspot);
          if (ev.frame_b64) setLiveFrame(ev.frame_b64);
          // Pair the real frame with its pressure field for slow sync replay.
          if (ev.frame_b64 && ev.field_b64) {
            const ff: FilmFrame = {
              t: pt.time,
              status: pt.status,
              score: pt.score,
              frame: ev.frame_b64,
              field: ev.field_b64,
              hotspot: ev.hotspot ?? null,
            };
            setFilm((prev) => {
              const next = [...prev, ff];
              return next.length > 300 ? next.slice(next.length - 300) : next;
            });
          }
          break;
        }
        case "done":
          setResult({
            peak_frame_b64: ev.peak_frame_b64 ?? null,
            flow_gif_b64: ev.flow_gif_b64 ?? null,
            summary: ev.summary ?? "",
            claude_briefing: ev.claude_briefing ?? "",
            rl_explanation: ev.rl_explanation ?? "",
            timeline: ev.timeline ?? [],
            peak_physics: ev.peak_physics ?? null,
            forecast: ev.forecast ?? null,
            trend: ev.trend ?? null,
            hotspot: ev.hotspot ?? null,
            agent_trace: ev.agent_trace ?? [],
            source: capturedSource,
          });
          if (ev.forecast && !ev.forecast.error) setLiveForecast(ev.forecast);
          if (ev.trend) setLiveTrend(ev.trend);
          if (ev.hotspot) setLiveHotspot(ev.hotspot);
          setLivePhase("Complete");
          break;
      }
    };

    try {
      if (mode === "upload") {
        await streamAnalyze(file!, venue, onEvent, ac.signal);
      } else {
        // keepSession=true keeps the warm Browserbase session alive so the
        // live-view iframe never goes dark and the next loop reuses it.
        await streamMonitorUrl(
          liveUrl.trim(), venue || "Live Camera", onEvent, 35,
          liveSession ?? undefined, ac.signal, true);
      }
    } catch (e: unknown) {
      if (!ac.signal.aborted) setError(String(e));
    } finally {
      setStreaming(false);
      setLoad(false);
    }
  }

  // Continuous live loop: while monitoring is on and we have a warm session,
  // kick off an analysis pass whenever none is running. When a pass ends
  // (streaming flips false) this re-fires and starts the next one, so live
  // monitoring never stops until the user hits Stop. A short gap between
  // passes keeps the session healthy and avoids a tight error spin.
  const handleRunRef = useRef(handleRun);
  handleRunRef.current = handleRun;
  useEffect(() => {
    if (mode !== "live" || !monitoring || streaming || !liveSession) return;
    // Back off if the previous pass errored so we don't hammer the API.
    const delay = error ? 5000 : 1200;
    const id = setTimeout(() => {
      if (!streaming) handleRunRef.current();
    }, delay);
    return () => clearTimeout(id);
  }, [mode, monitoring, streaming, liveSession, error]);

  const peakStatus = result?.timeline?.reduce((worst, p) => {
    const rank = { DANGER: 3, WARNING: 2, SAFE: 1, CALIBRATING: 0 };
    return (rank[p.status as keyof typeof rank] ?? 0) > (rank[worst as keyof typeof rank] ?? 0) ? p.status : worst;
  }, "SAFE") ?? "SAFE";

  // A live agent trace so the pipeline is visible WHILE streaming (not only
  // after a pass finishes). Falls back to the final trace once `result` lands.
  const liveTrace = useMemo<AgentTraceStep[]>(() => {
    if (!streaming && !monitoring) return [];
    const danger = liveStatus === "DANGER";
    const warn = liveStatus === "WARNING";
    const steps: AgentTraceStep[] = [
      {
        agent: "Calibration Agent", icon: "calibrate",
        action: liveStatus === "CALIBRATING"
          ? "Establishing calm baseline…" : "Calm baseline locked",
        detail: `${liveTicks.length} frames processed`, status: "ok",
      },
      {
        agent: "World Model", icon: "brain",
        action: "Encoding crowd state → 64-dim latent",
        detail: `LSTM roll-forward · surprise ${liveScore.toFixed(2)}σ`,
        status: "ok",
      },
      {
        agent: "Anomaly Detector", icon: "pulse",
        action: danger ? "Crush pattern forming"
          : warn ? "Elevated crowd pressure" : "Flow within normal bounds",
        detail: `status ${liveStatus} · ${liveScore.toFixed(2)}σ`,
        status: danger ? "danger" : "ok",
      },
    ];
    if (liveForecast?.points?.length) {
      steps.push({
        agent: "Forecast Agent", icon: "forecast",
        action: `Imagined next ${liveForecast.horizon_s ?? ""}s`,
        detail: `peak risk ${Math.round(liveForecast.projected_risk ?? 0)}%`,
        status: liveForecast.projected_status === "DANGER" ? "danger" : "ok",
      });
    }
    if (liveTrend?.points?.length) {
      const sl = liveTrend.slope_per_min ?? 0;
      steps.push({
        agent: "Trend Projection", icon: "eye",
        action: "Minutes-ahead risk outlook",
        detail: `${sl > 0 ? "+" : ""}${sl}%/min trend`,
        status: liveTrend.projected_status === "DANGER" ? "danger" : "ok",
      });
    }
    steps.push({
      agent: "Claude", icon: "claude",
      action: "Operator briefing on completion",
      detail: "Sonnet 4.6", status: "ok",
    });
    return steps;
  }, [streaming, monitoring, liveStatus, liveScore, liveTicks.length,
      liveForecast, liveTrend]);

  return (
    <div className="flex h-full gap-0 min-h-0">

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
              {monitoring ? (
                <button
                  className="btn-secondary w-full text-[11px] !border-crimson/40 !text-crimson"
                  onClick={stopMonitoring}
                >
                  <span className="w-2.5 h-2.5 rounded-[2px] bg-crimson inline-block" />
                  Stop monitoring
                </button>
              ) : (
                <button
                  className="btn-primary w-full text-[11px]"
                  disabled={!liveUrl.trim() || previewLoading}
                  onClick={startPreview}
                >
                  {previewLoading ? (
                    <><span className="spinner-white" /> Connecting…</>
                  ) : (
                    <><span className="dot-live" /> Start live monitor</>
                  )}
                </button>
              )}
              <p className="font-mono text-[9px] text-text3 leading-snug">
                Opens a Browserbase cloud browser and{" "}
                <span className="text-teal">analyzes automatically</span> as soon
                as the feed is live — then keeps monitoring on a loop. No clicks
                needed.
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

          {mode === "upload" && (
            <button
              className="btn-primary w-full"
              disabled={!canRun || loading}
              onClick={handleRun}
            >
              {loading ? (
                <><span className="spinner-white" /> Analyzing…</>
              ) : (
                <><svg className="w-3.5 h-3.5" fill="currentColor" viewBox="0 0 16 16"><path d="M6 3l8 5-8 5V3z"/></svg> Run Analysis</>
              )}
            </button>
          )}
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
            monitoring ? (
              <span className="font-mono text-[9px] text-teal flex items-center gap-1.5">
                <span className="dot-live" />
                {streaming ? "STREAMING · ANALYZING" : "STREAMING"}
              </span>
            ) : (
              <span className="font-mono text-[9px] text-text3 flex items-center gap-1.5">
                PAUSED
              </span>
            )
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
              <p className="font-mono text-xs">Click “Start live monitor” to open and analyze the feed</p>
              {previewErr && (
                <p className="font-mono text-[9px] text-crimson/80 break-all">{previewErr}</p>
              )}
            </div>
          )}

          {/* Live analyzing pill */}
          {mode === "live" && liveView && streaming && (
            <div className="pointer-events-none absolute bottom-3 left-3 z-10 font-mono text-[9px] text-text2 bg-void/75 px-2 py-1 rounded flex items-center gap-1.5">
              <span className="dot-live" /> Analyzing · T+{liveNow.toFixed(1)}s
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
      <div className="flex-1 flex flex-col gap-3 p-4 overflow-y-auto min-w-0 min-h-0">
        <div className="flex items-center justify-between">
          <p className="panel-label">Analysis</p>
          {streaming ? (
            <span className="font-mono text-[9px] text-crimson flex items-center gap-1.5">
              <span className="dot-live" />
              {liveTicks.length}{liveTotal ? `/${liveTotal}` : ""} frames
            </span>
          ) : result?.source ? (
            <span className="font-mono text-[9px] text-teal flex items-center gap-1.5">
              <span className="dot-live" />
              {result.source.frames_captured} frames @ {result.source.capture_fps} fps
            </span>
          ) : null}
        </div>

        {!result && !loading && !streaming && !monitoring ? (
          /* Empty analysis state */
          <div className="card flex-1 min-h-80 flex flex-col items-center justify-center gap-3 text-text3">
            <svg className="w-10 h-10 opacity-20" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth="1">
              <path strokeLinecap="round" strokeLinejoin="round" d="M3 13.125C3 12.504 3.504 12 4.125 12h2.25c.621 0 1.125.504 1.125 1.125v6.75C7.5 20.496 6.996 21 6.375 21h-2.25A1.125 1.125 0 013 19.875v-6.75zM9.75 8.625c0-.621.504-1.125 1.125-1.125h2.25c.621 0 1.125.504 1.125 1.125v11.25c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V8.625zM16.5 4.125c0-.621.504-1.125 1.125-1.125h2.25C20.496 3 21 3.504 21 4.125v15.75c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V4.125z" />
            </svg>
            <p className="font-mono text-xs text-center px-6">
              {mode === "live"
                ? "Click “Start live monitor” — analysis begins automatically"
                : "Click “Run Analysis” to analyze this video"}
            </p>
          </div>
        ) : (
          <>
            {/* Live status bar — only while streaming */}
            {streaming && (
              <LiveStatusBar
                now={liveNow}
                status={liveStatus}
                score={liveScore}
                processed={liveTicks.length}
                total={liveTotal}
                phase={livePhase}
              />
            )}

            {/* Synchronized replay — real frames ↔ pressure field, slow + scrubbable */}
            {film.length > 0 && <FilmPlayer film={film} live={streaming} />}

            {/* Flow-statistics field (animated) — fallback when no film yet */}
            {film.length === 0 && (
            <div className="card relative min-h-72 flex-1 flex items-center justify-center overflow-hidden">
              {(streaming || (monitoring && !result)) && liveFrame ? (
                /* The actual analyzed frame with the danger marker drawn on the
                   SAME pixels the hotspot came from → guaranteed alignment. */
                <div className="relative inline-block animate-fade-in">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={`data:image/jpeg;base64,${liveFrame}`}
                    alt="Analyzed live frame"
                    className="block max-h-[440px] max-w-full w-auto rounded"
                  />

                  {liveHotspot &&
                    (liveStatus === "DANGER" || liveStatus === "WARNING") && (() => {
                    const danger = liveStatus === "DANGER";
                    const ring = danger ? "#F85149" : "#D29922";
                    const op = 0.5 + 0.45 * Math.min(1, liveHotspot.intensity);
                    return (
                      <div
                        className="pointer-events-none absolute z-10 transition-all duration-700 ease-out"
                        style={{
                          left: `${liveHotspot.x * 100}%`,
                          top: `${liveHotspot.y * 100}%`,
                          width: `${Math.round(liveHotspot.r * 100)}%`,
                          transform: "translate(-50%, -50%)",
                          opacity: op,
                        }}
                      >
                        <div
                          className={`aspect-square rounded-full ${danger ? "animate-pulse" : ""}`}
                          style={{
                            border: `2px solid ${ring}`,
                            boxShadow: `0 0 22px ${ring}, inset 0 0 28px ${danger ? "rgba(248,81,73,0.4)" : "rgba(210,153,34,0.28)"}`,
                            background: `radial-gradient(circle, ${danger ? "rgba(248,81,73,0.22)" : "rgba(210,153,34,0.16)"} 0%, transparent 70%)`,
                          }}
                        />
                        <div
                          className="absolute top-1/2 left-1/2 w-1.5 h-1.5 rounded-full -translate-x-1/2 -translate-y-1/2"
                          style={{ background: ring }}
                        />
                        <div className="absolute left-1/2 -translate-x-1/2 -top-6 whitespace-nowrap">
                          <div className={danger ? "badge-danger" : "badge-warning"}>
                            <span className={danger ? "dot-danger" : "dot-warning"} />
                            {danger ? "DANGER" : "RISK"} · {liveScore.toFixed(1)}σ
                          </div>
                        </div>
                      </div>
                    );
                  })()}

                  <div className="absolute top-2 left-2">
                    <div className={STATUS_META[liveStatus]?.badge ?? "badge-neutral"}>
                      <span className={STATUS_META[liveStatus]?.dot} />
                      {STATUS_META[liveStatus]?.label ?? "Live"}
                    </div>
                  </div>
                  <div className="absolute bottom-2 right-2 font-mono text-[9px] text-text3 bg-void/70 px-2 py-1 rounded flex items-center gap-1.5">
                    <span className="dot-live" /> Analyzed frame · region marked
                  </div>
                </div>
              ) : result?.flow_gif_b64 || result?.peak_frame_b64 ? (
                <>
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={
                      result.flow_gif_b64
                        ? `data:image/gif;base64,${result.flow_gif_b64}`
                        : `data:image/png;base64,${result.peak_frame_b64}`
                    }
                    alt="Crowd flow-statistics field"
                    className="w-full h-full object-contain animate-fade-in"
                  />
                  <div className="absolute top-3 left-3">
                    <div className={STATUS_META[peakStatus]?.badge ?? "badge-neutral"}>
                      <span className={STATUS_META[peakStatus]?.dot} />
                      Peak: {STATUS_META[peakStatus]?.label}
                    </div>
                  </div>
                  <div className="absolute bottom-3 right-3 font-mono text-[9px] text-text3 bg-void/70 px-2 py-1 rounded flex items-center gap-1.5">
                    {result.flow_gif_b64 ? (
                      <><span className="dot-live" /> Flow Field · animated</>
                    ) : (
                      <>CFD Pressure Field · peak</>
                    )}
                  </div>
                </>
              ) : streaming && liveForecast?.projected_field_b64 ? (
                <>
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={`data:image/png;base64,${liveForecast.projected_field_b64}`}
                    alt="Imagined crowd pressure field"
                    className="w-full h-full object-contain animate-fade-in"
                  />
                  <div className="absolute top-3 left-3">
                    <div className={STATUS_META[liveStatus]?.badge ?? "badge-neutral"}>
                      <span className={STATUS_META[liveStatus]?.dot} />
                      {STATUS_META[liveStatus]?.label ?? "Live"}
                    </div>
                  </div>
                  <div className="absolute bottom-3 right-3 font-mono text-[9px] text-text3 bg-void/70 px-2 py-1 rounded flex items-center gap-1.5">
                    <span className="dot-live" /> Imagined field · +{liveForecast.horizon_s}s
                  </div>
                </>
              ) : (
                <div className="flex flex-col items-center gap-3 text-text3">
                  <span className="spinner" />
                  <p className="font-mono text-xs">
                    {streaming
                      ? (livePhase || "Streaming live…")
                      : mode === "live"
                        ? "Pulling live frames via Browserbase…"
                        : "Processing frames…"}
                  </p>
                </div>
              )}
            </div>
            )}

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

            {/* Forecast — potential future of the crowd (rolling while live) */}
            {(() => {
              const fc = result?.forecast ?? liveForecast;
              return fc && !fc.error ? <ForecastPanel forecast={fc} /> : null;
            })()}

            {/* Minutes-ahead risk outlook (statistical trend projection) */}
            {(() => {
              const tr = result?.trend ?? liveTrend;
              return tr && tr.points && tr.points.length > 1 ? (
                <TrendPanel trend={tr} />
              ) : null;
            })()}

            {/* Timeline — grows bar-by-bar while streaming */}
            {(() => {
              const pts = streaming && !result ? liveTicks : result?.timeline;
              return pts && pts.length ? <Timeline points={pts} /> : null;
            })()}

            {/* Agent trace — live while streaming, final once result lands */}
            {(() => {
              const finalTrace = result?.agent_trace ?? [];
              const trace = finalTrace.length > 0 ? finalTrace : liveTrace;
              return trace.length > 0 ? (
                <AgentTrace
                  steps={trace}
                  title="Agent Trace"
                  live={finalTrace.length === 0 && (streaming || monitoring)}
                />
              ) : null;
            })()}

            {/* Claude */}
            <div className="card flex flex-col">
              <div className="panel-header">
                <p className="panel-label">Situational Awareness</p>
                <span className="badge-teal text-[9px] px-1.5 py-0.5">Claude</span>
              </div>
              <div className="p-4 text-xs text-text2 leading-relaxed max-h-56 overflow-y-auto">
                {result ? (
                  <p className="whitespace-pre-wrap">{result.claude_briefing}</p>
                ) : streaming ? (
                  <p className="font-mono text-[11px] text-text3 italic">
                    Live analysis in progress — Claude briefs the operator on completion…
                  </p>
                ) : loading ? (
                  <div className="space-y-2">
                    <div className="skeleton h-3 w-full" />
                    <div className="skeleton h-3 w-5/6" />
                    <div className="skeleton h-3 w-4/5" />
                    <div className="skeleton h-3 w-3/4" />
                  </div>
                ) : (
                  <p className="whitespace-pre-wrap">Waiting for analysis…</p>
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
                {result ? (
                  <p className="whitespace-pre-wrap">
                    {result.rl_explanation || "No anomaly detected yet."}
                  </p>
                ) : streaming ? (
                  <p className="font-mono text-[11px] text-text3 italic">
                    Policy recommendation resolves on completion…
                  </p>
                ) : loading ? (
                  <div className="space-y-2">
                    <div className="skeleton h-3 w-3/4" />
                    <div className="skeleton h-3 w-full" />
                    <div className="skeleton h-3 w-5/6" />
                  </div>
                ) : (
                  <p className="whitespace-pre-wrap">No anomaly detected yet.</p>
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
