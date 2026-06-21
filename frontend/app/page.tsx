"use client";

import { useState } from "react";
import MonitorTab from "@/components/MonitorTab";
import SimulateTab from "@/components/SimulateTab";
import DiscoveryTab from "@/components/DiscoveryTab";
import RLTab from "@/components/RLTab";

const NAV = [
  {
    id: "monitor",
    label: "Monitor",
    desc: "Real-time anomaly detection",
    icon: (
      <svg viewBox="0 0 16 16" className="w-3.5 h-3.5" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
        <rect x="1" y="3" width="11" height="8" rx="1.5" />
        <path d="M12 7l3-1.5v5L12 9" />
      </svg>
    ),
  },
  {
    id: "simulate",
    label: "Simulate",
    desc: "Pre-event fluid dynamics",
    icon: (
      <svg viewBox="0 0 16 16" className="w-3.5 h-3.5" fill="currentColor" opacity="0.85">
        <rect x="1"   y="1"   width="6" height="6" rx="1" />
        <rect x="9"   y="1"   width="6" height="6" rx="1" opacity="0.5" />
        <rect x="1"   y="9"   width="6" height="6" rx="1" opacity="0.5" />
        <rect x="9"   y="9"   width="6" height="6" rx="1" />
      </svg>
    ),
  },
  {
    id: "discovery",
    label: "Discovery",
    desc: "Latent space physics probe",
    icon: (
      <svg viewBox="0 0 16 16" className="w-3.5 h-3.5" fill="none" stroke="currentColor" strokeWidth="1.25">
        <circle cx="8" cy="8" r="2" fill="currentColor" opacity="0.6" />
        <ellipse cx="8" cy="8" rx="7" ry="3" />
        <ellipse cx="8" cy="8" rx="7" ry="3" transform="rotate(60 8 8)" />
        <ellipse cx="8" cy="8" rx="7" ry="3" transform="rotate(-60 8 8)" />
      </svg>
    ),
  },
  {
    id: "rl",
    label: "RL Policy",
    desc: "Intervention engine",
    icon: (
      <svg viewBox="0 0 16 16" className="w-3.5 h-3.5" fill="currentColor">
        <path d="M9.5 1L3 9h5.5L7 15l7.5-9H9L9.5 1z" />
      </svg>
    ),
  },
];

const STATS = [
  { label: "Avg. Lead Time",  value: "4.2 min", color: "text-crimson" },
  { label: "Lives at Risk",   value: "200+ / yr", color: "text-amber" },
  { label: "Infrastructure",  value: "Any camera", color: "text-teal" },
];

const SPONSORS = [
  "Anthropic Claude",
  "Fetch.ai Agentverse",
  "Browserbase",
  "Simular Agent S",
];

export default function Home() {
  const [active, setActive] = useState("monitor");
  const current = NAV.find((n) => n.id === active)!;

  return (
    <div className="flex h-screen bg-void overflow-hidden">

      {/* ── Sidebar ─────────────────────────────────── */}
      <aside className="w-52 flex-shrink-0 flex flex-col border-r border-border"
        style={{ background: "linear-gradient(180deg, #080e1a 0%, #060a12 100%)" }}>

        {/* Logo */}
        <div className="px-4 py-4 border-b border-border">
          <div className="flex items-center gap-2.5 mb-1">
            <div className="w-7 h-7 rounded-lg flex items-center justify-center flex-shrink-0"
              style={{ background: "linear-gradient(135deg, #0ea5e9 0%, #0369a1 100%)", boxShadow: "0 0 12px rgba(14,165,233,0.3)" }}>
              <span className="font-mono text-[10px] font-bold text-white">CP</span>
            </div>
            <div>
              <p className="display text-sm font-bold text-text1 leading-tight tracking-tight">CrowdPhysics</p>
              <p className="font-mono text-[9px] text-text3 leading-none tracking-wider mt-0.5">AI Safety Platform</p>
            </div>
          </div>
        </div>

        {/* Nav */}
        <nav className="flex-1 p-3 overflow-y-auto">
          <p className="font-mono text-[9px] text-text3 uppercase tracking-[0.15em] px-2 mb-2 mt-1">Modules</p>
          <div className="space-y-0.5">
            {NAV.map(({ id, label, icon }) => {
              const isActive = id === active;
              return (
                <button
                  key={id}
                  onClick={() => setActive(id)}
                  className={`w-full flex items-center gap-2.5 px-2.5 py-2 rounded-lg text-left transition-all duration-150 text-sm group ${
                    isActive
                      ? "text-teal border border-teal/20"
                      : "text-text2 hover:text-text1 border border-transparent hover:border-border"
                  }`}
                  style={isActive ? { background: "rgba(14,165,233,0.08)" } : {}}
                >
                  <span className={`flex-shrink-0 transition-colors ${isActive ? "text-teal" : "text-text3 group-hover:text-text2"}`}>
                    {icon}
                  </span>
                  <span className={`font-medium ${isActive ? "" : ""}`}>{label}</span>
                  {isActive && (
                    <span className="ml-auto w-1 h-1 rounded-full bg-teal flex-shrink-0" />
                  )}
                </button>
              );
            })}
          </div>

          {/* Key metrics */}
          <div className="mt-5 mb-2">
            <p className="font-mono text-[9px] text-text3 uppercase tracking-[0.15em] px-2 mb-3">Key Metrics</p>
            <div className="space-y-3 px-2">
              {STATS.map(({ label, value, color }) => (
                <div key={label}>
                  <p className="kpi-label mb-0.5">{label}</p>
                  <p className={`kpi-value text-base ${color}`}>{value}</p>
                </div>
              ))}
            </div>
          </div>
        </nav>

        {/* Sponsors */}
        <div className="px-4 py-3 border-t border-border">
          <p className="font-mono text-[9px] text-text3 uppercase tracking-[0.15em] mb-2">Built with</p>
          <div className="flex flex-col gap-1">
            {SPONSORS.map((s) => (
              <p key={s} className="font-mono text-[9px] text-text3/70 leading-4">{s}</p>
            ))}
          </div>
          <p className="font-mono text-[9px] text-text3/40 mt-3 leading-tight">
            UC Berkeley AI Hackathon 2026
          </p>
        </div>
      </aside>

      {/* ── Main area ───────────────────────────────── */}
      <div className="flex-1 flex flex-col overflow-hidden min-w-0">

        {/* Topbar */}
        <header className="flex items-center justify-between px-5 py-3 border-b border-border flex-shrink-0"
          style={{ background: "linear-gradient(90deg, #080e1a 0%, #060a12 100%)" }}>
          <div>
            <h1 className="display font-semibold text-text1 text-base leading-tight">{current.label}</h1>
            <p className="text-xs text-text3 mt-0.5">{current.desc}</p>
          </div>
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-1.5">
              <span className="dot-live" />
              <span className="font-mono text-[10px] text-text3">System Ready</span>
            </div>
            <div className="h-4 w-px bg-border" />
            <span className="font-mono text-[10px] text-text3 border border-border px-2 py-0.5 rounded-md">
              v1.0.0
            </span>
          </div>
        </header>

        {/* Content */}
        <main className="flex-1 overflow-auto animate-fade-in">
          {active === "monitor"   && <MonitorTab />}
          {active === "simulate"  && <SimulateTab />}
          {active === "discovery" && <DiscoveryTab />}
          {active === "rl"        && <RLTab />}
        </main>
      </div>
    </div>
  );
}
