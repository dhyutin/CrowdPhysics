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

const STATUS_COLOR: Record<string, string> = {
  SAFE: "#3FB950",
  WARNING: "#D29922",
  DANGER: "#F85149",
};

export default function ForecastPanel({ forecast }: { forecast: Forecast }) {
  if (!forecast || forecast.error) return null;
  const status = forecast.projected_status ?? "SAFE";
  const color = STATUS_COLOR[status] ?? "#3FB950";
  const lead = forecast.lead_time_s;
  const points = forecast.points ?? [];

  return (
    <div className="card flex flex-col animate-fade-in">
      <div className="panel-header">
        <p className="panel-label">Crowd Forecast · Potential Future</p>
        <span className="badge-teal text-[9px] px-1.5 py-0.5">World Model</span>
      </div>

      <div className="p-4 flex flex-col gap-3">
        <div className="grid grid-cols-2 gap-3">
          {/* Projected field */}
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
                  IMAGINED · +{forecast.horizon_s}s
                </div>
              </>
            ) : (
              <p className="font-mono text-[10px] text-text3">No projection</p>
            )}
          </div>

          {/* Stats */}
          <div className="flex flex-col gap-2 justify-center">
            <div className="card-inset p-2.5">
              <p className="kpi-label">Projected Risk</p>
              <p className="kpi-value text-xl" style={{ color }}>
                {Math.round(forecast.projected_risk ?? 0)}%
              </p>
              <p className="font-mono text-[9px] text-text3 mt-0.5">{status}</p>
            </div>
            <div className="card-inset p-2.5">
              <p className="kpi-label">Lead Time to Danger</p>
              <p
                className="kpi-value text-xl"
                style={{ color: lead ? "#F85149" : "#3FB950" }}
              >
                {lead ? `${lead}s` : "—"}
              </p>
              <p className="font-mono text-[9px] text-text3 mt-0.5">
                {lead ? "before crush projected" : "no crush projected"}
              </p>
            </div>
          </div>
        </div>

        {/* Risk curve */}
        {points.length > 1 && (
          <div className="h-28 -mx-1">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={points} margin={{ top: 6, right: 8, left: -18, bottom: 0 }}>
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
        )}
        <p className="font-mono text-[9px] text-text3 leading-snug">
          The world model rolls the crowd forward in latent space — projecting
          the next {forecast.horizon_s}s and the crush-risk trajectory before it
          is visible.
        </p>
      </div>
    </div>
  );
}
