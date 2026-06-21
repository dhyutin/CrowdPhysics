"use client";

import { useState } from "react";
import MonitorTab from "@/components/MonitorTab";
import PlanTab from "@/components/PlanTab";

type Mode = "monitor" | "plan";

const MODES: Record<Mode, { label: string; tagline: string; desc: string; icon: React.ReactNode }> = {
  monitor: {
    label: "Monitor",
    tagline: "Real-time crowd safety",
    desc: "Pull a live camera feed or upload video. See the crowd flow, the world model's forecast of what happens next, and the agents reasoning about what's safe — live.",
    icon: (
      <svg viewBox="0 0 24 24" className="w-7 h-7" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
        <rect x="2" y="4" width="15" height="12" rx="2" />
        <path d="M17 9l5-2.5v11L17 15" />
        <circle cx="9.5" cy="10" r="2.5" />
      </svg>
    ),
  },
  plan: {
    label: "Plan",
    tagline: "Design a space before the crowd",
    desc: "Upload a photo of a location. Agents reconstruct it, simulate the crowd, and design how to arrange people, flow and staff for your event — safely.",
    icon: (
      <svg viewBox="0 0 24 24" className="w-7 h-7" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
        <path d="M9 3v18m6-18v18M3 9h18M3 15h18" />
        <rect x="3" y="3" width="18" height="18" rx="2" />
      </svg>
    ),
  },
};

const SPONSORS = ["Anthropic Claude", "Browserbase", "Fetch.ai Agentverse", "Arize"];

function Landing({ onEnter }: { onEnter: (m: Mode) => void }) {
  return (
    <div className="min-h-screen flex flex-col bg-void overflow-y-auto">
      {/* ambient glow */}
      <div
        className="pointer-events-none fixed inset-0 opacity-60"
        style={{
          background:
            "radial-gradient(900px 500px at 50% -10%, rgba(68,147,248,0.10), transparent 70%), radial-gradient(700px 400px at 100% 100%, rgba(45,212,191,0.06), transparent 70%)",
        }}
      />
      <div className="relative flex-1 flex flex-col items-center justify-center px-6 py-16 max-w-5xl mx-auto w-full">
        {/* brand logo */}
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src="/crowd_physics_logo.png"
          alt="CrowdPhysics — Plan safe. Monitor live. Never react."
          className="w-full max-w-md mb-8 select-none rounded-xl"
          style={{ mixBlendMode: "lighten" }}
        />

        <h1 className="display text-2xl sm:text-3xl font-bold text-center text-text1 leading-tight max-w-3xl">
          See the crowd&apos;s future <span style={{ color: "#4493F8" }}>before</span> it becomes a crisis.
        </h1>
        <p className="text-text3 text-center mt-4 max-w-xl text-sm leading-relaxed">
          A world model learns crowd fluid dynamics from raw video. Agents read the
          flow, forecast the danger, and tell you what to do — in plain language.
        </p>

        {/* two entry cards */}
        <div className="grid sm:grid-cols-2 gap-4 mt-12 w-full max-w-3xl">
          {(Object.keys(MODES) as Mode[]).map((m) => {
            const meta = MODES[m];
            return (
              <button
                key={m}
                onClick={() => onEnter(m)}
                className="card group text-left p-6 transition-all duration-200 hover:-translate-y-1"
                style={{ borderColor: "#21262D" }}
              >
                <div className="w-12 h-12 rounded-xl flex items-center justify-center text-teal mb-4 border border-teal/20"
                  style={{ background: "rgba(68,147,248,0.08)" }}>
                  {meta.icon}
                </div>
                <div className="flex items-center gap-2">
                  <h2 className="display text-xl font-bold text-text1">{meta.label}</h2>
                  <svg className="w-4 h-4 text-text3 group-hover:text-teal group-hover:translate-x-0.5 transition-all" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth="2">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
                  </svg>
                </div>
                <p className="font-mono text-[10px] text-teal/80 uppercase tracking-wider mt-1">{meta.tagline}</p>
                <p className="text-text3 text-[13px] leading-relaxed mt-3">{meta.desc}</p>
              </button>
            );
          })}
        </div>
      </div>

      {/* footer */}
      <div className="relative border-t border-border px-6 py-4 flex flex-col sm:flex-row items-center justify-between gap-2 max-w-5xl mx-auto w-full">
        <p className="font-mono text-[9px] text-text3/60">UC Berkeley AI Hackathon 2026</p>
        <div className="flex items-center gap-3 flex-wrap justify-center">
          <span className="font-mono text-[9px] text-text3/50">Built with</span>
          {SPONSORS.map((s) => (
            <span key={s} className="font-mono text-[9px] text-text3/70">{s}</span>
          ))}
        </div>
      </div>
    </div>
  );
}

export default function Home() {
  const [entered, setEntered] = useState(false);
  const [mode, setMode] = useState<Mode>("monitor");

  if (!entered) {
    return <Landing onEnter={(m) => { setMode(m); setEntered(true); }} />;
  }

  return (
    <div className="flex flex-col h-screen bg-void overflow-hidden">
      {/* ── Top bar ──────────────────────────────────── */}
      <header className="flex items-center justify-between px-5 py-2.5 border-b border-border flex-shrink-0"
        style={{ background: "linear-gradient(90deg, #10151D 0%, #0D1117 100%)" }}>

        {/* brand → home */}
        <button onClick={() => setEntered(false)} className="flex items-center group" title="Back to home">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src="/crowd_physics_logo.png"
            alt="CrowdPhysics"
            className="h-11 w-auto object-contain group-hover:opacity-80 transition-opacity select-none"
            style={{ mixBlendMode: "lighten" }}
          />
        </button>

        {/* mode switch */}
        <div className="flex items-center gap-1 p-1 rounded-lg border border-border bg-void/50">
          {(Object.keys(MODES) as Mode[]).map((m) => (
            <button
              key={m}
              onClick={() => setMode(m)}
              className={`flex items-center gap-1.5 px-4 py-1.5 rounded-md text-sm font-medium transition-all ${
                mode === m ? "text-teal" : "text-text3 hover:text-text2"
              }`}
              style={mode === m ? { background: "rgba(68,147,248,0.10)", border: "1px solid rgba(68,147,248,0.25)" } : { border: "1px solid transparent" }}
            >
              <span className="w-3.5 h-3.5">{MODES[m].icon && (
                <svg viewBox="0 0 24 24" className="w-3.5 h-3.5" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
                  {m === "monitor"
                    ? <><rect x="2" y="5" width="14" height="11" rx="2" /><path d="M16 9l6-2v10l-6-2" /></>
                    : <><path d="M9 4v16m6-16v16M4 9h16M4 15h16" /></>}
                </svg>
              )}</span>
              {MODES[m].label}
            </button>
          ))}
        </div>

        {/* status */}
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5">
            <span className="dot-live" />
            <span className="font-mono text-[10px] text-text3">System Ready</span>
          </div>
          <div className="h-4 w-px bg-border" />
          <span className="font-mono text-[10px] text-text3 border border-border px-2 py-0.5 rounded-md">v1.0</span>
        </div>
      </header>

      {/* ── Content ──────────────────────────────────── */}
      <main className="flex-1 overflow-hidden min-h-0 animate-fade-in">
        {mode === "monitor" ? <MonitorTab /> : <PlanTab />}
      </main>
    </div>
  );
}
