"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import dynamic from "next/dynamic";
import { plan3d, refinePlan3d, type EventIntake as EventIntakePayload, type Plan3DResult } from "@/lib/api";
import AgentTrace from "@/components/AgentTrace";
import EventIntake, { type IntakeValue } from "@/components/EventIntake";
import ScenarioCompare from "@/components/ScenarioCompare";
import PlaybackBar from "@/components/PlaybackBar";
import PlanPoints from "@/components/PlanPoints";

import type { CrowdPhase } from "@/components/Venue3D";

// three.js touches `window`, so the scene is client-only.
const Venue3D = dynamic(() => import("@/components/Venue3D"), {
  ssr: false,
  loading: () => (
    <div className="w-full h-full flex items-center justify-center">
      <span className="spinner" />
    </div>
  ),
});

const INITIAL_INTAKE: IntakeValue = {
  purpose: "Concert",
  nPeople: "",
  density: "65",
  durationMin: "120",
  seating: "standing",
  ingress: "gradual",
  notes: "",
  areaM2: "",
};

// Grab one representative frame from a video so the vision agent gets a still.
async function fileToImage(file: File): Promise<File> {
  if (!file.type.startsWith("video")) return file;
  return new Promise((resolve, reject) => {
    const video = document.createElement("video");
    video.preload = "auto";
    video.muted = true;
    video.src = URL.createObjectURL(file);
    video.onloadeddata = () => {
      video.currentTime = Math.min(1.2, (video.duration || 2) * 0.25);
    };
    video.onseeked = () => {
      const canvas = document.createElement("canvas");
      canvas.width = video.videoWidth || 1280;
      canvas.height = video.videoHeight || 720;
      const ctx = canvas.getContext("2d");
      if (!ctx) return reject(new Error("Canvas unavailable"));
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      canvas.toBlob(
        (blob) => {
          URL.revokeObjectURL(video.src);
          if (!blob) return reject(new Error("Frame extraction failed"));
          resolve(new File([blob], "frame.jpg", { type: "image/jpeg" }));
        },
        "image/jpeg",
        0.9
      );
    };
    video.onerror = () => reject(new Error("Could not read the video file"));
  });
}

const PHASE_META: { label: string; color: string; desc: string }[] = [
  { label: "Entry", color: "#3FB950", desc: "crowd streams in through the entrances" },
  { label: "Staying", color: "#4493F8", desc: "crowd occupies the space" },
  { label: "Exit", color: "#D29922", desc: "crowd disperses through the gates" },
];

function KPICard({ label, value, sub, color }: { label: string; value: string; sub?: string; color: string }) {
  return (
    <div className="card-inset p-3 flex flex-col gap-0.5">
      <p className="kpi-label">{label}</p>
      <p className={`kpi-value text-xl ${color}`}>{value}</p>
      {sub && <p className="font-mono text-[9px] text-text3 mt-0.5">{sub}</p>}
    </div>
  );
}

export default function PlanTab() {
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [intake, setIntake] = useState<IntakeValue>(INITIAL_INTAKE);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<Plan3DResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [selectedId, setSelectedId] = useState<string>("");
  const [frame, setFrame] = useState(0);
  const [playing, setPlaying] = useState(true);
  const [phase, setPhase] = useState<CrowdPhase>(0);

  // "Fix the scene" chat: correct the reconstruction in plain language.
  type ChatMsg = { role: "user" | "agent"; text: string };
  const [chat, setChat] = useState<ChatMsg[]>([]);
  const [chatInput, setChatInput] = useState("");
  const [refining, setRefining] = useState(false);

  // Refs the 3D render loop reads each animation frame (no re-render needed).
  const frameRef = useRef(0);
  const playingRef = useRef(true);

  const isVideo = imageFile?.type.startsWith("video") ?? false;
  const filePreview = useMemo(
    () => (imageFile ? URL.createObjectURL(imageFile) : null),
    [imageFile]
  );

  const selected = useMemo(() => {
    if (!result) return null;
    return result.scenarios.find((s) => s.id === selectedId) ?? result.scenarios[0] ?? null;
  }, [result, selectedId]);

  const frames = selected?.field.frames ?? 0;

  // Advance the playback head while playing.
  useEffect(() => {
    if (!selected || !playing || frames <= 1) return;
    const id = setInterval(() => {
      setFrame((prev) => {
        const next = (prev + 1) % frames;
        frameRef.current = next;
        return next;
      });
    }, 110);
    return () => clearInterval(id);
  }, [selected, playing, frames]);

  function patchIntake(patch: Partial<IntakeValue>) {
    setIntake((v) => ({ ...v, ...patch }));
  }

  function intakePayload(): EventIntakePayload {
    return {
      purpose: intake.purpose,
      nPeople: parseInt(intake.nPeople) || 0,
      density: (parseInt(intake.density) || 65) / 100,
      durationMin: parseInt(intake.durationMin) || 0,
      seating: intake.seating,
      ingress: intake.ingress,
      notes: intake.notes,
      areaM2: parseFloat(intake.areaM2) || 0,
    };
  }

  // Reset playback to the recommended scenario of a fresh result.
  function applyResult(res: Plan3DResult) {
    setResult(res);
    setSelectedId(res.best_scenario_id);
    frameRef.current = 0;
    setFrame(0);
    playingRef.current = true;
    setPlaying(true);
  }

  async function handleRun() {
    if (!imageFile) {
      setError("Upload a photo or video of the location first.");
      return;
    }
    setLoading(true);
    setError(null);
    setChat([]);
    try {
      const img = await fileToImage(imageFile);
      const res = await plan3d(img, intakePayload());
      applyResult(res);
    } catch (e: unknown) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }

  async function handleRefine() {
    const msg = chatInput.trim();
    if (!msg || !result?.layout || refining) return;
    setChatInput("");
    setChat((c) => [...c, { role: "user", text: msg }]);
    setRefining(true);
    setError(null);
    try {
      const res = await refinePlan3d(result.layout, msg, intakePayload());
      applyResult(res);
      setChat((c) => [...c, { role: "agent", text: res.chat_reply || "Updated the scene." }]);
    } catch (e: unknown) {
      setChat((c) => [...c, { role: "agent", text: `Couldn't apply that: ${String(e)}` }]);
    } finally {
      setRefining(false);
    }
  }

  function togglePlay() {
    setPlaying((p) => {
      playingRef.current = !p;
      return !p;
    });
  }
  function onScrub(f: number) {
    setFrame(f);
    frameRef.current = f;
  }
  function selectScenario(id: string) {
    setSelectedId(id);
    setFrame(0);
    frameRef.current = 0;
  }

  const m = selected?.metrics;
  const pressureColor = m && m.peak_pressure > 6 ? "text-crimson" : m && m.peak_pressure > 3 ? "text-amber" : "text-emerald";

  return (
    <div className="flex h-full gap-0">
      {/* ── Controls ─────────────────────────────────── */}
      <div className="w-64 flex-shrink-0 flex flex-col gap-3 p-4 border-r border-border overflow-y-auto">
        <div className="card p-4 flex flex-col gap-3">
          <p className="panel-label">Simulate a Space</p>
          <p className="font-mono text-[10px] text-text3 leading-tight">
            Drop a photo or video of a location. Agents rebuild it in 3D, fill it
            with a simulated crowd, test layouts, and design the safest plan.
          </p>

          <label
            className={`card-inset rounded-md p-3 flex flex-col items-center gap-2 cursor-pointer border border-dashed transition-colors ${
              imageFile ? "border-lavender/40" : "border-border hover:border-lavender/40"
            }`}
          >
            {filePreview ? (
              isVideo ? (
                <video src={filePreview} className="w-full h-24 object-cover rounded" muted playsInline />
              ) : (
                // eslint-disable-next-line @next/next/no-img-element
                <img src={filePreview} alt="location" className="w-full h-24 object-cover rounded" />
              )
            ) : (
              <>
                <svg className="w-6 h-6 text-text3" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth="1.2">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5l4.5-4.5 3 3 4.5-4.5 6 6M3 19.5h18a1.5 1.5 0 001.5-1.5V6A1.5 1.5 0 0021 4.5H3A1.5 1.5 0 001.5 6v12A1.5 1.5 0 003 19.5z" />
                </svg>
                <span className="font-mono text-[10px] text-text3">Click to choose photo or video</span>
                <span className="font-mono text-[8px] text-text3">overhead · ground · floor plan</span>
              </>
            )}
            <input
              type="file"
              accept="image/png,image/jpeg,image/webp,video/mp4,video/quicktime,video/webm"
              className="hidden"
              onChange={(e) => {
                setImageFile(e.target.files?.[0] ?? null);
                setResult(null);
                setError(null);
              }}
            />
          </label>
          {imageFile && <p className="font-mono text-[9px] text-text3 truncate">{imageFile.name}</p>}

          <EventIntake value={intake} onChange={patchIntake} disabled={loading} />

          <button className="btn-primary w-full mt-1" onClick={handleRun} disabled={loading || !imageFile}>
            {loading ? (
              <>
                <span className="spinner-white" /> Agents simulating…
              </>
            ) : (
              <>
                <svg className="w-3.5 h-3.5" fill="currentColor" viewBox="0 0 16 16">
                  <path d="M9.5 1L3 9h5.5L7 15l7.5-9H9L9.5 1z" />
                </svg>{" "}
                Simulate with Agents
              </>
            )}
          </button>
        </div>

        {result?.layout && (
          <div className="card p-4 animate-fade-in flex flex-col gap-2">
            <div className="flex items-center justify-between gap-2">
              <p className="panel-label">Reconstruction</p>
              <div className="flex items-center gap-1.5">
                {result.layout.archetype && (
                  <span className="badge-neutral text-[9px] px-1.5 py-0.5 uppercase tracking-wide">
                    {result.layout.archetype}
                  </span>
                )}
                <span className="badge-teal text-[9px] px-1.5 py-0.5">
                  {Math.round((result.layout.confidence ?? 0) * 100)}% conf
                </span>
              </div>
            </div>
            <div className="flex flex-wrap gap-1.5">
              {[
                ["Stage", "#2DD4BF"],
                ["Wall", "#30363D"],
                ["Barrier", "#A371F7"],
                ["Entry", "#3FB950"],
                ["Exit", "#4493F8"],
              ].map(([label, c]) => (
                <span key={label} className="flex items-center gap-1">
                  <span className="w-2.5 h-2.5 rounded-sm" style={{ background: c }} />
                  <span className="font-mono text-[8px] text-text3">{label}</span>
                </span>
              ))}
            </div>
            <div className="card-inset p-2.5 flex items-center justify-between">
              <div>
                <p className="kpi-label">Max Capacity</p>
                <p className="font-mono text-[8px] text-text3 mt-0.5">
                  {result.capacity_estimate
                    ? `${result.capacity_estimate.area_m2.toLocaleString()} m² · ${result.capacity_estimate.people_per_m2}/m²`
                    : "from reconstruction"}
                </p>
              </div>
              <p className="kpi-value text-lg text-lavender">
                {(result.venue_max_capacity ?? result.layout.capacity ?? 0).toLocaleString()}
              </p>
            </div>
            {result.layout.notes && (
              <p className="font-mono text-[9px] text-text2 leading-tight italic">“{result.layout.notes}”</p>
            )}
          </div>
        )}

        {result?.reconstruction_eval && (() => {
          const ev = result.reconstruction_eval!;
          const pct = Math.round(ev.score * 100);
          const tone =
            ev.label === "faithful"
              ? { bar: "#3FB950", text: "text-emerald", badge: "badge-safe" }
              : ev.label === "partial"
              ? { bar: "#D29922", text: "text-amber", badge: "badge-warning" }
              : { bar: "#F85149", text: "text-crimson", badge: "badge-danger" };
          return (
            <div className="card p-4 animate-fade-in flex flex-col gap-2.5">
              <div className="flex items-center justify-between gap-2">
                <p className="panel-label">Reconstruction Fidelity</p>
                <span className="badge-neutral text-[8px] px-1.5 py-0.5 uppercase tracking-wide">
                  Arize eval
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className={`${tone.badge} text-[9px] px-1.5 py-0.5 uppercase tracking-wide`}>
                  {ev.label}
                </span>
                <p className={`kpi-value text-2xl ${tone.text}`}>{pct}%</p>
              </div>
              <div className="card-inset p-1 h-2 rounded-full overflow-hidden">
                <div
                  className="h-full rounded-full transition-all"
                  style={{ width: `${pct}%`, background: tone.bar }}
                />
              </div>
              <div className="grid grid-cols-2 gap-1.5">
                {([
                  ["structures", "Structures"],
                  ["openings", "Openings"],
                  ["scale", "Scale"],
                  ["features", "Features"],
                ] as const).map(([k, label]) => (
                  <div key={k} className="card-inset px-2 py-1 flex items-center justify-between">
                    <span className="font-mono text-[8px] text-text3">{label}</span>
                    <span className="font-mono text-[9px] text-text2">
                      {Math.round((ev.aspects[k] ?? 0) * 100)}%
                    </span>
                  </div>
                ))}
              </div>
              {ev.rationale && (
                <p className="font-mono text-[9px] text-text2 leading-tight italic">
                  “{ev.rationale}”
                </p>
              )}
              <p className="font-mono text-[8px] text-text3">
                Claude judges the rebuilt world model against your photo · traced to Arize AX
              </p>
            </div>
          );
        })()}
      </div>

      {/* ── 3D simulation ────────────────────────────── */}
      <div className="flex-1 flex flex-col gap-3 p-4 overflow-y-auto min-w-0 min-h-0 [&>*]:shrink-0">
        {selected && m && (
          <div className="grid grid-cols-4 gap-3 animate-fade-in">
            <KPICard label="Crowd" value={result!.n_people.toLocaleString()} sub="people simulated" color="text-text1" />
            <KPICard label="Safe Capacity" value={m.safe_capacity.toLocaleString()} sub="recommended" color="text-emerald" />
            <KPICard label="Peak Pressure" value={m.peak_pressure.toFixed(1)} sub="out of 12 max" color={pressureColor} />
            <KPICard
              label="Danger Zones"
              value={String(m.n_danger_zones)}
              sub={m.n_danger_zones ? "need attention" : "all clear"}
              color={m.n_danger_zones ? "text-crimson" : "text-emerald"}
            />
          </div>
        )}

        {result?.capacity_check && result.capacity_check.verdict !== "ok" && (
          <div
            className="card border px-4 py-3 animate-fade-in flex items-start gap-3"
            style={
              result.capacity_check.verdict === "unreasonable"
                ? { background: "rgba(248,81,73,0.06)", borderColor: "rgba(248,81,73,0.35)" }
                : { background: "rgba(210,153,34,0.06)", borderColor: "rgba(210,153,34,0.35)" }
            }
          >
            <svg
              className={`w-5 h-5 flex-shrink-0 mt-0.5 ${result.capacity_check.verdict === "unreasonable" ? "text-crimson" : "text-amber"}`}
              fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth="1.8"
            >
              <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z" />
            </svg>
            <div className="flex-1 min-w-0">
              <p
                className={`font-mono text-[10px] uppercase tracking-wider mb-1 ${result.capacity_check.verdict === "unreasonable" ? "text-crimson" : "text-amber"}`}
              >
                {result.capacity_check.verdict === "unreasonable"
                  ? "Unreasonable capacity — planning for a healthy range"
                  : "Dense for this area"}
              </p>
              <p className="text-xs text-text2 leading-relaxed">{result.capacity_check.message}</p>
              <div className="flex flex-wrap gap-3 mt-2">
                <span className="font-mono text-[9px] text-text3">
                  Requested <span className="text-text1">{result.capacity_check.given.toLocaleString()}</span>
                </span>
                <span className="font-mono text-[9px] text-text3">
                  Healthy <span className="text-emerald">{result.capacity_check.healthy_capacity.toLocaleString()}</span>
                </span>
                <span className="font-mono text-[9px] text-text3">
                  Crush limit <span className="text-crimson">{result.capacity_check.crush_capacity.toLocaleString()}</span>
                </span>
                <span className="font-mono text-[9px] text-text3">
                  Simulating <span className="text-lavender">{result.n_people.toLocaleString()}</span>
                </span>
              </div>
            </div>
          </div>
        )}

        {!result && !loading ? (
          <div className="card flex-1 min-h-80 flex flex-col items-center justify-center gap-3 text-text3 px-6 text-center">
            <svg className="w-12 h-12 opacity-20" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth="1">
              <path strokeLinecap="round" strokeLinejoin="round" d="M21 7.5l-9-5.25L3 7.5m18 0l-9 5.25m9-5.25v9l-9 5.25M3 7.5l9 5.25M3 7.5v9l9 5.25m0-9v9" />
            </svg>
            <p className="font-mono text-xs">Upload a location, answer a few questions, and click “Simulate with Agents”</p>
          </div>
        ) : loading && !result ? (
          <div className="card flex-1 min-h-80 flex flex-col items-center justify-center gap-3 text-text3">
            <span className="spinner" />
            <p className="font-mono text-xs">Vision → 3D reconstruction → multi-scenario simulation…</p>
          </div>
        ) : (
          selected && (
            <>
              <div className="card flex flex-col animate-fade-in">
                <div className="panel-header">
                  <p className="panel-label">
                    3D Crowd Simulation · {selected.name}
                  </p>
                  <div className="flex items-center gap-2">
                    {selected.is_best && <span className="badge-safe text-[9px] px-1.5 py-0.5">Recommended</span>}
                    <span className="badge-neutral text-[9px] px-1.5 py-0.5">drag to orbit</span>
                  </div>
                </div>
                <div className="relative w-full" style={{ height: "440px", background: "#0D1117" }}>
                  <Venue3D
                    scenario={selected}
                    nPeople={result!.n_people}
                    frameRef={frameRef}
                    playingRef={playingRef}
                    agentPlan={result!.agent_plan}
                    onPhase={setPhase}
                  />

                  {/* Phase choreography: entry → staying → exit */}
                  <div className="absolute top-3 left-3 flex flex-col gap-1.5 pointer-events-none">
                    <div className="flex items-center gap-1">
                      {PHASE_META.map((p, i) => (
                        <div
                          key={p.label}
                          className="flex items-center gap-1 px-2 py-1 rounded transition-all"
                          style={{
                            background: i === phase ? `${p.color}26` : "rgba(13,17,23,0.7)",
                            border: `1px solid ${i === phase ? p.color : "transparent"}`,
                          }}
                        >
                          <span
                            className="w-1.5 h-1.5 rounded-full"
                            style={{ background: p.color, opacity: i === phase ? 1 : 0.4 }}
                          />
                          <span
                            className="font-mono text-[9px]"
                            style={{ color: i === phase ? p.color : "#6E7681" }}
                          >
                            {p.label}
                          </span>
                        </div>
                      ))}
                    </div>
                    <span className="font-mono text-[8px] text-text3 bg-void/70 px-2 py-0.5 rounded self-start">
                      {PHASE_META[phase].desc}
                    </span>
                  </div>
                  <div className="absolute top-3 right-3 flex flex-col gap-1 bg-void/80 px-2 py-2 rounded pointer-events-none">
                    {[
                      ["low", "#4493F8"],
                      ["building", "#D29922"],
                      ["crush", "#F85149"],
                    ].map(([l, c]) => (
                      <div key={l} className="flex items-center gap-1.5">
                        <span className="w-3 h-2 rounded-sm flex-shrink-0" style={{ background: c }} />
                        <span className="font-mono text-[8px] text-text3 capitalize">{l}</span>
                      </div>
                    ))}
                    {result!.agent_plan?.behaviors?.length ? (
                      <div className="flex items-center gap-1.5 mt-1 pt-1 border-t border-border">
                        <span className="w-2.5 h-2.5 rounded-full flex-shrink-0" style={{ background: "#A371F7" }} />
                        <span className="font-mono text-[8px] text-text3">LLM-piloted</span>
                      </div>
                    ) : null}
                  </div>
                  {result!.n_people > 1400 && (
                    <div className="absolute bottom-3 left-3 bg-void/80 px-2 py-1 rounded pointer-events-none">
                      <span className="font-mono text-[8px] text-text3">
                        showing 1,400 of {result!.n_people.toLocaleString()} agents
                      </span>
                    </div>
                  )}
                </div>
                <div className="p-3 border-t border-border">
                  <PlaybackBar
                    frame={frame}
                    frames={frames}
                    playing={playing}
                    onToggle={togglePlay}
                    onScrub={onScrub}
                  />
                </div>
              </div>

              {/* Fix the 3D scene — conversational layout editing */}
              <div className="card flex flex-col animate-fade-in">
                <div className="panel-header">
                  <p className="panel-label">Fix the 3D Scene</p>
                  <span className="badge-teal text-[9px] px-1.5 py-0.5">Scene Editor</span>
                </div>
                <div className="p-3 flex flex-col gap-2">
                  {chat.length === 0 ? (
                    <p className="font-mono text-[10px] text-text3 leading-snug">
                      Reconstruction wrong? Tell the agent in plain words — e.g.{" "}
                      <span className="text-text2">“move the slide to the left”</span>,{" "}
                      <span className="text-text2">“add an exit on the north wall”</span>,{" "}
                      <span className="text-text2">“remove the stage”</span>. It edits the
                      layout and re-runs the simulation.
                    </p>
                  ) : (
                    <div className="flex flex-col gap-1.5 max-h-44 overflow-y-auto">
                      {chat.map((mmsg, i) => (
                        <div
                          key={i}
                          className={`text-xs leading-snug ${mmsg.role === "user" ? "text-text1" : "text-text2"}`}
                        >
                          <span
                            className={`font-mono text-[9px] mr-1.5 ${mmsg.role === "user" ? "text-lavender" : "text-teal"}`}
                          >
                            {mmsg.role === "user" ? "you" : "agent"}
                          </span>
                          {mmsg.text}
                        </div>
                      ))}
                    </div>
                  )}
                  <div className="flex gap-2">
                    <input
                      className="input text-sm flex-1"
                      value={chatInput}
                      onChange={(e) => setChatInput(e.target.value)}
                      onKeyDown={(e) => { if (e.key === "Enter") handleRefine(); }}
                      placeholder="Describe what to change…"
                      disabled={refining}
                    />
                    <button
                      className="btn-secondary text-[11px] px-3"
                      onClick={handleRefine}
                      disabled={refining || !chatInput.trim()}
                    >
                      {refining ? <span className="spinner" /> : "Apply"}
                    </button>
                  </div>
                  {refining && (
                    <p className="font-mono text-[9px] text-text3">
                      Editing scene &amp; re-simulating…
                    </p>
                  )}
                </div>
              </div>

              <div className="animate-fade-in">
                <p className="panel-label mb-2">Scenarios · ranked safest first</p>
                <ScenarioCompare scenarios={result!.scenarios} selectedId={selected.id} onSelect={selectScenario} />
              </div>
            </>
          )
        )}

        {error && (
          <div
            className="card border border-crimson/30 px-4 py-3 animate-fade-in"
            style={{ background: "rgba(248,81,73,0.05)" }}
          >
            <p className="font-mono text-[10px] text-crimson uppercase tracking-wider mb-1">Error</p>
            <p className="font-mono text-xs text-crimson/80">{error}</p>
          </div>
        )}
      </div>

      {/* ── Agent plan & report ──────────────────────── */}
      <div className="w-80 flex-shrink-0 flex flex-col border-l border-border overflow-y-auto">
        <div className="p-4 flex flex-col gap-3">
          {result?.agent_trace && <AgentTrace steps={result.agent_trace} title="Planning Agents" />}

          {result?.agent_plan?.behaviors?.length ? (
            <div className="card flex flex-col animate-fade-in">
              <div className="panel-header">
                <p className="panel-label">Crowd Behavior</p>
                <span className="badge-teal text-[9px] px-1.5 py-0.5">Agent LLM</span>
              </div>
              <div className="p-3 flex flex-col gap-2">
                <p className="font-mono text-[9px] text-text3 leading-snug">
                  <span className="text-lavender">
                    {Math.round((result.agent_plan.llm_fraction ?? 0) * 100)}%
                  </span>{" "}
                  of agents follow these reasoned intents (LLM world model); the
                  rest move on crowd physics.
                </p>
                {result.agent_plan.behaviors.map((b, i) => (
                  <div key={i} className="flex items-start gap-2">
                    <span className="w-2 h-2 rounded-full bg-lavender/70 mt-1 flex-shrink-0" />
                    <div className="min-w-0">
                      <p className="text-xs text-text1 leading-tight">
                        {b.name}{" "}
                        <span className="font-mono text-[9px] text-text3">
                          · {Math.round(b.fraction * 100)}%
                        </span>
                      </p>
                      {b.intent && (
                        <p className="font-mono text-[9px] text-text3 leading-snug">{b.intent}</p>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ) : null}

          {result?.plan_points && <PlanPoints points={result.plan_points} />}

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
                  {result?.plan ?? "Simulate a space to generate an agent arrangement."}
                </p>
              )}
            </div>
          </div>

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
