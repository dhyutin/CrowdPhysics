"use client";

import type { Scenario } from "@/lib/api";

function pressureColor(p: number) {
  return p > 6 ? "text-crimson" : p > 3 ? "text-amber" : "text-emerald";
}

export default function ScenarioCompare({
  scenarios,
  selectedId,
  onSelect,
}: {
  scenarios: Scenario[];
  selectedId: string;
  onSelect: (id: string) => void;
}) {
  const ordered = [...scenarios].sort((a, b) => a.rank - b.rank);
  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-2">
      {ordered.map((s) => {
        const active = s.id === selectedId;
        const m = s.metrics;
        return (
          <button
            key={s.id}
            onClick={() => onSelect(s.id)}
            className={`text-left card-inset p-3 flex flex-col gap-2 transition-all duration-150 ${
              active
                ? "border-lavender/60 ring-1 ring-lavender/30"
                : "hover:border-text3/40"
            }`}
            style={active ? { background: "rgba(226,169,241,0.06)" } : undefined}
          >
            <div className="flex items-center justify-between gap-1">
              <span className="font-semibold text-xs text-text1 truncate">{s.name}</span>
              {s.is_best ? (
                <span className="badge-safe text-[8px] px-1.5 py-0.5 flex-shrink-0">Best</span>
              ) : (
                <span className="font-mono text-[9px] text-text3 flex-shrink-0">#{s.rank}</span>
              )}
            </div>

            <div className="grid grid-cols-2 gap-x-2 gap-y-1">
              <div>
                <p className="kpi-label">Peak</p>
                <p className={`font-mono text-sm font-bold ${pressureColor(m.peak_pressure)}`}>
                  {m.peak_pressure.toFixed(1)}
                </p>
              </div>
              <div>
                <p className="kpi-label">Danger</p>
                <p className={`font-mono text-sm font-bold ${m.n_danger_zones ? "text-crimson" : "text-emerald"}`}>
                  {m.n_danger_zones}
                </p>
              </div>
              <div>
                <p className="kpi-label">Safe cap</p>
                <p className="font-mono text-[11px] text-text2">{m.safe_capacity.toLocaleString()}</p>
              </div>
              <div>
                <p className="kpi-label">Exits</p>
                <p className="font-mono text-[11px] text-text2">{m.n_exits}</p>
              </div>
            </div>

            <p className="font-mono text-[9px] text-text3 leading-tight line-clamp-2">
              {s.description}
            </p>
          </button>
        );
      })}
    </div>
  );
}
