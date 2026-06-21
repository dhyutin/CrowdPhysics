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
import type { Trend } from "@/lib/api";

const STATUS_COLOR: Record<string, string> = {
  SAFE: "#3FB950",
  WARNING: "#D29922",
  DANGER: "#F85149",
};

function fmtLead(s: number | null | undefined): string {
  if (s == null) return "—";
  if (s < 60) return `${Math.round(s)}s`;
  const m = Math.floor(s / 60);
  const rem = Math.round(s % 60);
  return rem ? `${m}m ${rem}s` : `${m}m`;
}

function fmtTick(s: number): string {
  return s >= 60 ? `${(s / 60).toFixed(s % 60 === 0 ? 0 : 1)}m` : `${s}s`;
}

export default function TrendPanel({ trend }: { trend: Trend }) {
  if (!trend || !trend.points || trend.points.length < 2) return null;
  const status = trend.projected_status ?? "SAFE";
  const color = STATUS_COLOR[status] ?? "#3FB950";
  const lead = trend.lead_time_s;
  const points = trend.points;
  const slope = trend.slope_per_min ?? 0;
  const horizonMin = ((trend.horizon_s ?? 0) / 60).toFixed(0);
  const rising = slope > 0.5;
  const trendLabel =
    slope > 0.5 ? "rising" : slope < -0.5 ? "easing" : "flat";

  return (
    <div className="card flex flex-col animate-fade-in">
      <div className="panel-header">
        <p className="panel-label">Risk Outlook · Minutes Ahead</p>
        <span className="badge-warning text-[9px] px-1.5 py-0.5">
          Trend Projection
        </span>
      </div>

      <div className="p-4 flex flex-col gap-3">
        <div className="grid grid-cols-3 gap-2">
          <div className="card-inset p-2.5">
            <p className="kpi-label">Risk in {horizonMin}m</p>
            <p className="kpi-value text-xl" style={{ color }}>
              {Math.round(trend.projected_risk ?? 0)}%
            </p>
            <p className="font-mono text-[9px] text-text3 mt-0.5">{status}</p>
          </div>
          <div className="card-inset p-2.5">
            <p className="kpi-label">Trend</p>
            <p
              className="kpi-value text-xl"
              style={{ color: rising ? "#F85149" : "#3FB950" }}
            >
              {slope > 0 ? "+" : ""}
              {slope.toFixed(1)}
            </p>
            <p className="font-mono text-[9px] text-text3 mt-0.5">
              %/min · {trendLabel}
            </p>
          </div>
          <div className="card-inset p-2.5">
            <p className="kpi-label">Time to Danger</p>
            <p
              className="kpi-value text-xl"
              style={{ color: lead != null ? "#F85149" : "#3FB950" }}
            >
              {fmtLead(lead)}
            </p>
            <p className="font-mono text-[9px] text-text3 mt-0.5">
              {lead != null ? "if trend holds" : "no crush projected"}
            </p>
          </div>
        </div>

        <div className="h-28 -mx-1">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart
              data={points}
              margin={{ top: 6, right: 8, left: -18, bottom: 0 }}
            >
              <defs>
                <linearGradient id="trendFill" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor={color} stopOpacity={0.35} />
                  <stop offset="100%" stopColor={color} stopOpacity={0.02} />
                </linearGradient>
              </defs>
              <CartesianGrid
                strokeDasharray="3 3"
                stroke="#21262D"
                vertical={false}
              />
              <XAxis
                dataKey="t"
                tick={{ fill: "#6E7681", fontSize: 9, fontFamily: "monospace" }}
                tickLine={false}
                axisLine={{ stroke: "#21262D" }}
                tickFormatter={fmtTick}
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
                labelFormatter={(l) => `+${fmtTick(Number(l))}`}
              />
              <ReferenceLine
                y={66}
                stroke="#F85149"
                strokeDasharray="4 4"
                strokeOpacity={0.6}
              />
              {lead != null && (
                <ReferenceLine
                  x={lead}
                  stroke="#F85149"
                  strokeDasharray="2 2"
                  strokeOpacity={0.5}
                />
              )}
              <Area
                type="monotone"
                dataKey="risk"
                stroke={color}
                strokeWidth={2}
                strokeDasharray="5 3"
                fill="url(#trendFill)"
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        <p className="font-mono text-[9px] text-text3 leading-snug">
          Statistical extrapolation of the recent risk trend over the next{" "}
          {horizonMin} minutes — captures slow density build-up beyond the
          world model&apos;s seconds-ahead horizon. Not a physics rollout;
          confidence decays the further out it projects.
        </p>
      </div>
    </div>
  );
}
