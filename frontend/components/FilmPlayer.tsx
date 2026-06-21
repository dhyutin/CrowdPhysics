"use client";

import { useEffect, useState } from "react";
import { hotspotSeverity, type DangerSeverity, type Hotspot } from "@/lib/api";

// Severity → marker styling. "elevated" = slightly risky, "critical" = very dangerous.
const SEV_META: Record<
  Exclude<DangerSeverity, "calm">,
  { color: string; label: string; pulse: boolean; glow: string; fill: string }
> = {
  critical: {
    color: "#F85149", label: "VERY DANGEROUS", pulse: true,
    glow: "rgba(248,81,73,0.4)", fill: "rgba(248,81,73,0.22)",
  },
  elevated: {
    color: "#D29922", label: "SLIGHTLY RISKY", pulse: false,
    glow: "rgba(210,153,34,0.28)", fill: "rgba(210,153,34,0.16)",
  },
};

export interface FilmFrame {
  t: number;
  status: string;
  score: number;
  frame: string; // base64 jpeg of the real analyzed frame
  field: string; // base64 jpeg of the pressure field
  hotspot: Hotspot | null;
}

const META: Record<string, { c: string; l: string }> = {
  SAFE:        { c: "#3FB950", l: "Safe" },
  WARNING:     { c: "#D29922", l: "Warning" },
  DANGER:      { c: "#F85149", l: "Danger" },
  CALIBRATING: { c: "#6E7681", l: "Calibrating" },
};

const SPEEDS = [2, 4, 8]; // fps options; 4 = "slow" default

export default function FilmPlayer({
  film,
  live,
  markRegions = true,
  onToggleMarkRegions,
}: {
  film: FilmFrame[];
  live: boolean;
  markRegions?: boolean;
  onToggleMarkRegions?: () => void;
}) {
  const [idx, setIdx] = useState(0);
  const [playing, setPlaying] = useState(true);
  const [fps, setFps] = useState(4);
  const n = film.length;

  // While live, always follow the newest frame.
  useEffect(() => {
    if (live && n > 0) setIdx(n - 1);
  }, [live, n]);

  // When a pass finishes, restart the replay from the top.
  useEffect(() => {
    if (!live && n > 0) {
      setIdx(0);
      setPlaying(true);
    }
  }, [live, n]);

  // Slow playback timer (replay mode only).
  useEffect(() => {
    if (live || !playing || n === 0) return;
    const id = setInterval(
      () => setIdx((i) => (i + 1) % n),
      1000 / fps
    );
    return () => clearInterval(id);
  }, [live, playing, fps, n]);

  if (n === 0) return null;
  const cur = film[Math.min(idx, n - 1)];
  const meta = META[cur.status] ?? META.CALIBRATING;
  const sev = hotspotSeverity(cur.hotspot, cur.status);
  const sevMeta = sev === "calm" ? null : SEV_META[sev];
  const showMark = markRegions && cur.hotspot != null && sevMeta != null;

  return (
    <div className="card flex flex-col animate-fade-in">
      <div className="panel-header">
        <p className="panel-label flex items-center gap-1.5">
          {live && <span className="dot-live" />}
          Synchronized Replay · Frame ↔ Physics
        </p>
        <div className="flex items-center gap-1.5">
          {onToggleMarkRegions && (
            <button
              type="button"
              onClick={onToggleMarkRegions}
              title="Mark the regions in the frame where danger is building"
              className={`font-mono text-[9px] px-1.5 py-0.5 rounded border transition-colors ${
                markRegions
                  ? "bg-crimson/15 text-crimson border-crimson/30"
                  : "text-text3 border-border hover:text-text2"
              }`}
            >
              {markRegions ? "◉ Danger regions" : "○ Danger regions"}
            </button>
          )}
          <span
            className={`badge text-[9px] px-1.5 py-0.5 ${
              live ? "badge-danger" : "badge-teal"
            }`}
          >
            {live ? "LIVE" : "REPLAY"}
          </span>
        </div>
      </div>

      <div className="p-3 flex flex-col gap-3">
        <div className="grid grid-cols-2 gap-2">
          {/* Real analyzed frame + aligned danger marker */}
          <div className="relative card-inset overflow-hidden flex items-center justify-center min-h-[150px]">
            <div className="relative inline-block">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={`data:image/jpeg;base64,${cur.frame}`}
                alt="Analyzed frame"
                className="block max-h-[260px] max-w-full w-auto"
              />
              {showMark && cur.hotspot && sevMeta && (
                <div
                  className="pointer-events-none absolute z-10 transition-all duration-300 ease-out"
                  style={{
                    left: `${cur.hotspot.x * 100}%`,
                    top: `${cur.hotspot.y * 100}%`,
                    width: `${Math.round(cur.hotspot.r * 100)}%`,
                    transform: "translate(-50%, -50%)",
                    opacity: 0.5 + 0.45 * Math.min(1, cur.hotspot.intensity),
                  }}
                >
                  <div
                    className={`aspect-square rounded-full ${sevMeta.pulse ? "animate-pulse" : ""}`}
                    style={{
                      border: `2px solid ${sevMeta.color}`,
                      boxShadow: `0 0 18px ${sevMeta.color}, inset 0 0 22px ${sevMeta.glow}`,
                      background: `radial-gradient(circle, ${sevMeta.fill} 0%, transparent 70%)`,
                    }}
                  />
                  <div
                    className="absolute top-1/2 left-1/2 w-1.5 h-1.5 rounded-full -translate-x-1/2 -translate-y-1/2"
                    style={{ background: sevMeta.color }}
                  />
                  <div className="absolute left-1/2 -translate-x-1/2 -top-5 whitespace-nowrap">
                    <span
                      className="font-mono text-[8px] font-bold px-1.5 py-0.5 rounded"
                      style={{ background: `${sevMeta.color}22`, color: sevMeta.color, border: `1px solid ${sevMeta.color}55` }}
                    >
                      {sevMeta.label}
                    </span>
                  </div>
                </div>
              )}
            </div>
            <div className="absolute top-2 left-2 font-mono text-[8px] text-text3 bg-void/70 px-1.5 py-0.5 rounded">
              LIVE FEED
            </div>
          </div>

          {/* Corresponding pressure field */}
          <div className="relative card-inset overflow-hidden flex items-center justify-center min-h-[150px]">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={`data:image/jpeg;base64,${cur.field}`}
              alt="Pressure field"
              className="block max-h-[260px] max-w-full w-auto"
            />
            <div className="absolute top-2 left-2 font-mono text-[8px] text-text3 bg-void/70 px-1.5 py-0.5 rounded">
              PHYSICS FIELD
            </div>
          </div>
        </div>

        {/* Readout */}
        <div className="flex items-center justify-between font-mono text-[10px]">
          <span className="flex items-center gap-1.5" style={{ color: meta.c }}>
            <span
              className="w-1.5 h-1.5 rounded-full"
              style={{ background: meta.c }}
            />
            {meta.l} · {cur.score.toFixed(2)}σ
          </span>
          <span className="text-text3">
            T+{cur.t.toFixed(1)}s · frame {Math.min(idx, n - 1) + 1}/{n}
          </span>
        </div>

        {/* Transport (replay only — live just follows the stream) */}
        {!live ? (
          <div className="flex items-center gap-3">
            <button
              type="button"
              onClick={() => setPlaying((p) => !p)}
              className="btn-secondary px-2.5 py-1 text-[11px] flex-shrink-0"
            >
              {playing ? "❚❚ Pause" : "▶ Play"}
            </button>
            <input
              type="range"
              min={0}
              max={n - 1}
              value={Math.min(idx, n - 1)}
              onChange={(e) => {
                setPlaying(false);
                setIdx(Number(e.target.value));
              }}
              className="flex-1 accent-teal h-1"
            />
            <div className="flex gap-1 flex-shrink-0">
              {SPEEDS.map((s) => (
                <button
                  key={s}
                  type="button"
                  onClick={() => setFps(s)}
                  className={`font-mono text-[10px] px-1.5 py-0.5 rounded border transition-colors ${
                    fps === s
                      ? "bg-teal/15 text-teal border-teal/30"
                      : "text-text3 border-border hover:text-text2"
                  }`}
                >
                  {s}fps
                </button>
              ))}
            </div>
          </div>
        ) : (
          <p className="font-mono text-[9px] text-text3 text-center">
            Streaming live — slow synchronized replay unlocks when the pass
            completes.
          </p>
        )}
      </div>
    </div>
  );
}
