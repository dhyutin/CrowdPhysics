"use client";

import { useState } from "react";
import Hero from "@/components/Hero";
import MonitorTab from "@/components/MonitorTab";
import SimulateTab from "@/components/SimulateTab";
import DiscoveryTab from "@/components/DiscoveryTab";
import RLTab from "@/components/RLTab";

const TABS = [
  { id: "monitor",   label: "📹  Monitor" },
  { id: "simulate",  label: "🏟️  Simulate" },
  { id: "discovery", label: "🔬  Discovery" },
  { id: "rl",        label: "⚡  RL Policy" },
];

export default function Home() {
  const [active, setActive] = useState("monitor");

  return (
    <div className="min-h-screen flex flex-col bg-void">
      <Hero />

      {/* Tab bar */}
      <div className="flex border-b border-border bg-ground sticky top-0 z-50">
        {TABS.map(({ id, label }) => (
          <button
            key={id}
            onClick={() => setActive(id)}
            className={`px-5 py-3 mono text-[11px] uppercase tracking-widest transition-colors
              ${
                active === id
                  ? "text-teal border-b-2 border-teal bg-surface/40"
                  : "text-text3 hover:text-text2"
              }`}
          >
            {label}
          </button>
        ))}
      </div>

      {/* Tab content */}
      <div className="flex-1 overflow-auto">
        {active === "monitor"   && <MonitorTab />}
        {active === "simulate"  && <SimulateTab />}
        {active === "discovery" && <DiscoveryTab />}
        {active === "rl"        && <RLTab />}
      </div>

      {/* Sponsor strip */}
      <div className="flex items-center gap-3 px-6 py-2 bg-ground border-t border-border flex-wrap">
        <span className="mono text-[10px] text-text3 uppercase tracking-widest mr-2">
          Built with
        </span>
        {[
          "Anthropic Claude",
          "Fetch.ai Agentverse",
          "Browserbase",
          "Simular Agent S",
        ].map((s) => (
          <span
            key={s}
            className="mono text-[10px] text-text3 border border-border px-2 py-0.5 rounded"
          >
            {s}
          </span>
        ))}
        <span className="ml-auto mono text-[10px] text-text3">
          UC Berkeley AI Hackathon 2026
        </span>
      </div>
    </div>
  );
}
