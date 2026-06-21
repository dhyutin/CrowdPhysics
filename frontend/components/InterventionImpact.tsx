"use client";

import {
  Area,
  AreaChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { Counterfactual } from "@/lib/api";

const DO_NOTHING = "#F85149"; // crimson
const WITH_ACTION = "#3FB950"; // emerald

export default function InterventionImpact({
  cf,
}: {
  cf: Counterfactual;
}) {
  if (!cf || cf.error) return null;

  const dn = cf.do_nothing_risk;
  const wa = cf.action_risk;
  const reduction = cf.reduction_pct;
  const helps = reduction > 0.5;

  // Merge the two trajectories on t for a single overlaid chart.
  const len = Math.max(cf.points_do_nothing.length, cf.points_action.length);
  const data = Array.from({ length: len }, (_, i) => ({
    t: cf.points_do_nothing[i]?.t ?? cf.points_action[i]?.t ?? i,
    do_nothing: cf.points_do_nothing[i]?.risk ?? null,
    with_action: cf.points_action[i]?.risk ?? null,
  }));

  return (
    <div className="card flex flex-col animate-fade-in">
      <div className="panel-header">
        <p className="panel-label">Intervention Impact · Counterfactual</p>
        <span className="badge-teal text-[9px] px-1.5 py-0.5">World Model</span>
      </div>

      <div className="p-4 flex flex-col gap-3">
        {/* Recommended action */}
        <div className="card-inset p-2.5">
          <p className="kpi-label">Recommended Action</p>
          <p className="text-sm text-text1 font-medium capitalize">
            {(cf.action_name || "intervention").replace(/_/g, " ")}
          </p>
          {cf.action_description && (
            <p className="font-mono text-[9px] text-text3 mt-0.5 leading-snug">
              {cf.action_description}
            </p>
          )}
        </div>

        {/* Do nothing vs with action — the money numbers */}
        <div className="grid grid-cols-[1fr_auto_1fr] items-center gap-2">
          <div className="card-inset p-2.5 text-center">
            <p className="kpi-label">Do Nothing</p>
            <p className="kpi-value text-2xl" style={{ color: DO_NOTHING }}>
              {Math.round(dn)}%
            </p>
            <p className="font-mono text-[9px] text-text3 mt-0.5">projected risk</p>
          </div>

          <div className="flex flex-col items-center gap-1 px-1">
            <svg className="w-5 h-5 text-text3" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth="1.5">
              <path strokeLinecap="round" strokeLinejoin="round" d="M13.5 4.5L21 12m0 0l-7.5 7.5M21 12H3" />
            </svg>
            <span
              className="font-mono text-[10px] font-medium px-1.5 py-0.5 rounded"
              style={{
                color: helps ? WITH_ACTION : "#6E7681",
                background: helps ? "rgba(63,185,80,0.12)" : "transparent",
              }}
            >
              {helps ? `−${Math.round(reduction)}%` : "≈0%"}
            </span>
          </div>

          <div className="card-inset p-2.5 text-center" style={{ borderColor: helps ? "rgba(63,185,80,0.3)" : undefined }}>
            <p className="kpi-label">With Action</p>
            <p className="kpi-value text-2xl" style={{ color: WITH_ACTION }}>
              {Math.round(wa)}%
            </p>
            <p className="font-mono text-[9px] text-text3 mt-0.5">projected risk</p>
          </div>
        </div>

        {/* Overlaid risk curves */}
        {data.length > 1 && (
          <div className="h-32 -mx-1">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={data} margin={{ top: 6, right: 8, left: -18, bottom: 0 }}>
                <defs>
                  <linearGradient id="dnFill" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor={DO_NOTHING} stopOpacity={0.35} />
                    <stop offset="100%" stopColor={DO_NOTHING} stopOpacity={0.02} />
                  </linearGradient>
                  <linearGradient id="waFill" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor={WITH_ACTION} stopOpacity={0.3} />
                    <stop offset="100%" stopColor={WITH_ACTION} stopOpacity={0.02} />
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
                  formatter={(v: number, name: string) => [
                    `${Math.round(v)}%`,
                    name === "do_nothing" ? "do nothing" : "with action",
                  ]}
                  labelFormatter={(l) => `+${l}s`}
                />
                <Legend
                  iconType="plainline"
                  wrapperStyle={{ fontSize: 9, fontFamily: "monospace" }}
                  formatter={(value) =>
                    value === "do_nothing" ? "do nothing" : "with action"
                  }
                />
                <Area
                  type="monotone"
                  dataKey="do_nothing"
                  stroke={DO_NOTHING}
                  strokeWidth={2}
                  fill="url(#dnFill)"
                  connectNulls
                />
                <Area
                  type="monotone"
                  dataKey="with_action"
                  stroke={WITH_ACTION}
                  strokeWidth={2}
                  fill="url(#waFill)"
                  connectNulls
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        )}

        <p className="font-mono text-[9px] text-text3 leading-snug">
          {helps ? (
            <>
              The world model projects the next {cf.horizon_s}s under both
              futures: acting now cuts peak crush risk from{" "}
              <span style={{ color: DO_NOTHING }}>{Math.round(dn)}%</span> to{" "}
              <span style={{ color: WITH_ACTION }}>{Math.round(wa)}%</span>.
            </>
          ) : (
            <>
              The world model expects this intervention to hold risk near{" "}
              <span style={{ color: WITH_ACTION }}>{Math.round(wa)}%</span> over
              the next {cf.horizon_s}s.
            </>
          )}
        </p>
      </div>
    </div>
  );
}
