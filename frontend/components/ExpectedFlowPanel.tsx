"use client";

import { useState } from "react";
import {
  simulateExpectedFlow,
  type ExpectedFlowResult,
  type PortMode,
  type VenueLayout,
} from "@/lib/api";

const MODE_META: Record<PortMode, { color: string; label: string }> = {
  inflow: { color: "#3FB950", label: "inflow" },
  outflow: { color: "#4493F8", label: "outflow" },
  mixed: { color: "#8B949E", label: "mixed" },
};

export default function ExpectedFlowPanel({
  layout,
  density,
}: {
  layout: VenueLayout;
  density: number;
}) {
  const [loading, setLoading] = useState(false);
  const [res, setRes] = useState<ExpectedFlowResult | null>(null);
  const [err, setErr] = useState<string | null>(null);

  async function run() {
    setLoading(true);
    setErr(null);
    try {
      const r = await simulateExpectedFlow(layout.elements, density, layout.name);
      if (r.error) {
        setErr(r.error);
        setRes(null);
      } else {
        setRes(r);
      }
    } catch (e: unknown) {
      setErr(String(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="card flex flex-col animate-fade-in">
      <div className="panel-header">
        <p className="panel-label">Expected Entry / Exit Flow</p>
        <span className="badge-teal text-[9px] px-1.5 py-0.5">Sim → RAFT</span>
      </div>

      <div className="p-3 flex flex-col gap-3">
        <p className="font-mono text-[10px] text-text3 leading-snug">
          Renders this layout as a synthetic-crowd video and runs it through the{" "}
          <span className="text-text2">same RAFT optical-flow extractor</span>{" "}
          used on a live camera — so you can preview the flow each door should
          show before the event.
        </p>

        {!res && (
          <button
            className="btn-secondary text-[11px] w-full"
            onClick={run}
            disabled={loading}
          >
            {loading ? (
              <>
                <span className="spinner" /> Running optical flow…
              </>
            ) : (
              "Run optical-flow check"
            )}
          </button>
        )}

        {err && (
          <p className="font-mono text-[10px] text-crimson/80 leading-snug">{err}</p>
        )}

        {res && (
          <>
            <div className="grid grid-cols-2 gap-2">
              <div className="card-inset overflow-hidden flex flex-col">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={`data:image/png;base64,${res.annotated_b64}`}
                  alt="Synthetic crowd with recovered port flow"
                  className="w-full object-contain"
                />
                <p className="font-mono text-[8px] text-text3 px-1.5 py-1">
                  Synthetic crowd · recovered flow
                </p>
              </div>
              <div className="card-inset overflow-hidden flex flex-col">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={`data:image/png;base64,${res.field_b64}`}
                  alt="RAFT-derived pressure field"
                  className="w-full object-contain"
                />
                <p className="font-mono text-[8px] text-text3 px-1.5 py-1">
                  RAFT pressure field
                </p>
              </div>
            </div>

            <div className="flex items-center justify-between gap-2">
              <p className="font-mono text-[9px] text-text2 leading-snug flex-1">
                {res.summary}
              </p>
              <span className="badge-neutral text-[8px] px-1.5 py-0.5 uppercase flex-shrink-0">
                {res.flow_backend}
              </span>
            </div>

            <div className="flex flex-col gap-1.5">
              {res.ports.map((p, i) => {
                const meta = MODE_META[p.mode];
                const portColor = p.type === "entry" ? "#3FB950" : "#4493F8";
                return (
                  <div
                    key={`${p.label}-${i}`}
                    className="card-inset px-2.5 py-1.5 flex items-center gap-2"
                  >
                    <span
                      className="w-2.5 h-2.5 rounded-full flex-shrink-0"
                      style={{ background: portColor }}
                    />
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-1.5">
                        <span className="text-xs text-text1 truncate">{p.label}</span>
                        <span
                          className="font-mono text-[8px] px-1 py-0.5 rounded uppercase tracking-wide"
                          style={{ color: meta.color, background: `${meta.color}1a` }}
                        >
                          {meta.label}
                        </span>
                      </div>
                      <div className="card-inset h-1 rounded-full overflow-hidden mt-1">
                        <div
                          className="h-full rounded-full transition-all"
                          style={{
                            width: `${Math.round(p.intensity * 100)}%`,
                            background: meta.color,
                          }}
                        />
                      </div>
                    </div>
                    <div className="text-right flex-shrink-0">
                      <p className="font-mono text-[10px] text-text2">
                        {Math.round(p.share * 100)}%
                      </p>
                      <p className="font-mono text-[8px] text-text3">share</p>
                    </div>
                  </div>
                );
              })}
            </div>

            <button
              className="btn-secondary text-[10px] w-full"
              onClick={run}
              disabled={loading}
            >
              {loading ? <span className="spinner" /> : "Re-run"}
            </button>
          </>
        )}
      </div>
    </div>
  );
}
