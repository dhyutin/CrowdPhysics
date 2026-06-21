"use client";

import {
  Area,
  AreaChart,
  CartesianGrid,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { Forecast } from "@/lib/api";

// Render a horizon in seconds as a friendly label (e.g. 120 → "2 min").
function fmtHorizon(s?: number): string {
  if (!s || s <= 0) return "—";
  if (s < 90) return `${Math.round(s)}s`;
  const m = s / 60;
  return Number.isInteger(m) ? `${m} min` : `${m.toFixed(1)} min`;
}

function fmtLead(s: number | null | undefined): string {
  if (s == null) return "—";
  if (s < 60) return `${Math.round(s)}s`;
  const m = Math.floor(s / 60);
  const rem = Math.round(s % 60);
  return rem ? `${m}m ${rem}s` : `${m}m`;
}

// "Near future danger needs immediate action" → escalate when a crush is
// projected within the next ~30s. One projection, green normally, red urgent.
const IMMEDIATE_LEAD_S = 30;

type Urgency = {
  key: "IMMEDIATE" | "SOON" | "WATCH" | "SAFE";
  color: string;
  headline: string;
  sub: string;
  pulse: boolean;
};

function getUrgency(
  status: string,
  lead: number | null | undefined,
  horizonLabel: string
): Urgency {
  if (lead != null && lead <= IMMEDIATE_LEAD_S) {
    return {
      key: "IMMEDIATE",
      color: "#F85149",
      headline: "Immediate action needed",
      sub: `Crush risk projected in ${fmtLead(lead)} — intervene now`,
      pulse: true,
    };
  }
  if (status === "DANGER") {
    return {
      key: "SOON",
      color: "#F85149",
      headline: "Danger building",
      sub: lead != null
        ? `Crush risk in ${fmtLead(lead)} if the trend holds`
        : "Act before it escalates",
      pulse: true,
    };
  }
  if (status === "WARNING") {
    return {
      key: "WATCH",
      color: "#D29922",
      headline: "Elevated risk",
      sub: "Pressure rising — monitor closely",
      pulse: false,
    };
  }
  return {
    key: "SAFE",
    color: "#3FB950",
    headline: "Projected safe",
    sub: `No crush risk in the next ${horizonLabel}`,
    pulse: false,
  };
}

export default function ForecastPanel({ forecast }: { forecast: Forecast }) {
  if (!forecast || forecast.error) return null;

  // When the agent (Claude) has decided the risk, it is authoritative: it
  // drives the headline number, lead time, and banner. Otherwise fall back to
  // the (de-saturated) world-model projection.
  const hasAgent =
    forecast.agent_source === "claude" && forecast.agent_risk != null;
  const riskNum = hasAgent
    ? (forecast.agent_risk as number)
    : Math.round(forecast.projected_risk ?? 0);
  const lead = hasAgent ? forecast.agent_lead_s : forecast.lead_time_s;
  const status = hasAgent
    ? riskNum >= 66 ? "DANGER" : riskNum >= 40 ? "WARNING" : "SAFE"
    : forecast.projected_status ?? "SAFE";

  const points = forecast.points ?? [];
  const horizon = fmtHorizon(forecast.horizon_s);
  const u = getUrgency(status, lead, horizon);
  const color = u.color;
  const danger = u.key === "IMMEDIATE" || u.key === "SOON";

  // Prefer the agent's own reasoning + recommendation for the sub-line.
  const subText =
    hasAgent && forecast.agent_reason
      ? `${forecast.agent_reason}${
          forecast.agent_recommendation
            ? ` — ${forecast.agent_recommendation}`
            : ""
        }`
      : u.sub;

  // Single source of truth for the risk verdict shown everywhere in this panel:
  // the headline number, the imagined-field label, and the risk curve all use
  // the SAME number + status so their severity categories always agree.
  const STATUS_HEX: Record<string, string> = {
    SAFE: "#3FB950", WARNING: "#D29922", DANGER: "#F85149",
  };
  const verdictColor = STATUS_HEX[status] ?? color;
  const rawPeak = points.reduce((m, p) => Math.max(m, p.risk), 0);
  const curveScale = hasAgent && rawPeak > 0 ? riskNum / rawPeak : 1;
  const shownPoints =
    curveScale === 1
      ? points
      : points.map((p) => ({
          ...p,
          risk: Math.round(Math.min(100, p.risk * curveScale)),
        }));

  return (
    <div
      className="card flex flex-col animate-fade-in transition-colors"
      style={{ borderColor: danger ? "rgba(248,81,73,0.45)" : undefined }}
    >
      <div className="panel-header">
        <p className="panel-label">Crowd Projection · Next {horizon}</p>
        <span
          className={`text-[9px] px-1.5 py-0.5 ${
            hasAgent
              ? "badge bg-lavender/15 text-lavender border-lavender/40"
              : "badge-teal"
          }`}
          title={
            hasAgent
              ? "Crush risk decided by the Claude agent reasoning over physics, trend and the world-model forecast"
              : "Risk projected by the world model"
          }
        >
          {hasAgent ? "Agent · Claude" : "World Model"}
        </span>
      </div>

      <div className="p-4 flex flex-col gap-3">
        {/* Single status banner — green when safe, red when near-future danger */}
        <div
          className="rounded-lg border px-3 py-2.5 flex items-center gap-3 transition-colors"
          style={{
            background: `${color}14`,
            borderColor: `${color}59`,
          }}
        >
          <span
            className={`relative flex h-3 w-3 flex-shrink-0`}
            aria-hidden
          >
            {u.pulse && (
              <span
                className="absolute inline-flex h-full w-full rounded-full opacity-60 animate-ping"
                style={{ background: color }}
              />
            )}
            <span
              className="relative inline-flex h-3 w-3 rounded-full"
              style={{ background: color }}
            />
          </span>
          <div className="flex-1 min-w-0">
            <p
              className="font-mono text-[12px] font-medium uppercase tracking-wide leading-tight"
              style={{ color }}
            >
              {u.headline}
            </p>
            <p className="font-mono text-[10px] text-text3 mt-0.5 leading-snug">
              {subText}
            </p>
          </div>
          <div className="text-right flex-shrink-0">
            <p className="kpi-value text-2xl leading-none" style={{ color }}>
              {riskNum}%
            </p>
            <p className="font-mono text-[9px] text-text3 mt-0.5">
              {hasAgent ? "agent risk" : "peak risk"}
            </p>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-3">
          {/* Imagined field at the worst projected moment */}
          <div className="card-inset overflow-hidden relative flex items-center justify-center min-h-[140px]">
            {forecast.projected_field_b64 ? (
              <>
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={`data:image/png;base64,${forecast.projected_field_b64}`}
                  alt="Projected crowd pressure field"
                  className="w-full h-full object-contain"
                />
                <div className="absolute top-2 left-2 font-mono text-[8px] text-text3 bg-void/70 px-1.5 py-0.5 rounded">
                  IMAGINED · +{horizon}
                </div>
                {/* Unified risk label — same number/status as the headline. */}
                <div
                  className="absolute top-2 right-2 font-mono text-[8px] px-1.5 py-0.5 rounded bg-void/70"
                  style={{ color: verdictColor }}
                >
                  {status} · {riskNum}%
                </div>
              </>
            ) : (
              <p className="font-mono text-[10px] text-text3">No projection</p>
            )}
          </div>

          {/* Risk curve over the projection window */}
          <div className="flex flex-col justify-center">
            {shownPoints.length > 1 ? (
              <div className="h-[140px] -mx-1">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={shownPoints} margin={{ top: 6, right: 8, left: -18, bottom: 0 }}>
                    <defs>
                      <linearGradient id="riskFill" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor={color} stopOpacity={0.4} />
                        <stop offset="100%" stopColor={color} stopOpacity={0.02} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#21262D" vertical={false} />
                    <XAxis
                      dataKey="t"
                      tick={{ fill: "#6E7681", fontSize: 9, fontFamily: "monospace" }}
                      tickLine={false}
                      axisLine={{ stroke: "#21262D" }}
                      unit="s"
                    />
                    <YAxis
                      domain={[0, 100]}
                      tick={{ fill: "#6E7681", fontSize: 9, fontFamily: "monospace" }}
                      tickLine={false}
                      axisLine={false}
                      width={28}
                    />
                    <Tooltip
                      contentStyle={{
                        background: "#0D1117",
                        border: "1px solid #21262D",
                        borderRadius: 8,
                        fontSize: 11,
                        fontFamily: "monospace",
                      }}
                      labelStyle={{ color: "#8B949E" }}
                      formatter={(v: number) => [`${v}%`, "risk"]}
                      labelFormatter={(l) => `+${l}s`}
                    />
                    <ReferenceLine y={66} stroke="#F85149" strokeDasharray="4 4" strokeOpacity={0.6} />
                    {lead != null && (
                      <ReferenceLine x={lead} stroke="#F85149" strokeDasharray="2 2" strokeOpacity={0.5} />
                    )}
                    <Area
                      type="monotone"
                      dataKey="risk"
                      stroke={color}
                      strokeWidth={2}
                      fill="url(#riskFill)"
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            ) : (
              <div className="flex items-center justify-center min-h-[140px]">
                <p className="font-mono text-[10px] text-text3">Building projection…</p>
              </div>
            )}
          </div>
        </div>

        <p className="font-mono text-[9px] text-text3 leading-snug">
          The world model rolls the crowd forward in latent space — simulating
          the next {horizon} and turning red the moment a crush is projected
          soon enough to need action.
        </p>
      </div>
    </div>
  );
}
