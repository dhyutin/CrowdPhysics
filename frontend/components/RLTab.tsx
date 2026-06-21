"use client";

import { useEffect, useState } from "react";
import { runRLMetrics, type RLMetricsResult } from "@/lib/api";

const ACTIONS = [
  { id: "A0", name: "monitor",          effect: "No change to crowd dynamics",              cost: 0.0, risk: "low"  },
  { id: "A1", name: "increase_egress",  effect: "Dampen y-axis compression",                cost: 0.1, risk: "low"  },
  { id: "A2", name: "reduce_ingress",   effect: "Damp incoming flow energy",                cost: 0.1, risk: "low"  },
  { id: "A3", name: "lateral_redirect", effect: "Increase x-dims, reduce y-dims",           cost: 0.3, risk: "med"  },
  { id: "A4", name: "disperse",         effect: "Inject noise → global damping",            cost: 0.2, risk: "med"  },
  { id: "A5", name: "partial_evac",     effect: "Strong targeted damping (0.6×)",           cost: 0.2, risk: "med"  },
  { id: "A6", name: "full_evac",        effect: "Global damping (0.3×) — emergency only",  cost: 1.0, risk: "high" },
];

const REWARD_CODE = `reward = (danger_before - danger_after) × 3.0   # safety improvement
if danger_after > 3.5:  reward -= 15.0           # crush threshold crossed
if danger_after < 0.8:  reward +=  1.5           # staying safe bonus
if action==monitor and danger>2: reward -= 2.0   # inaction penalty
if action==full_evac and danger<1: reward -= 3.0 # overreaction penalty
reward -= action_cost[action]   # [0, .1, .1, .3, .2, .2, 1.0]`;

function SectionHeader({ title, sub }: { title: string; sub: string }) {
  return (
    <div className="mb-4">
      <h2 className="display text-base font-semibold text-text1 leading-tight">{title}</h2>
      <p className="text-xs text-text3 mt-0.5">{sub}</p>
    </div>
  );
}

function ConceptCard({ title, body, tag }: { title: string; body: string; tag: string }) {
  return (
    <div className="card p-5">
      <div className="flex items-center justify-between mb-3">
        <p className="panel-label">{title}</p>
        <span className="badge-teal">{tag}</span>
      </div>
      <p className="text-sm text-text2 leading-relaxed">{body}</p>
    </div>
  );
}

function LiveResults() {
  const [data, setData] = useState<RLMetricsResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    runRLMetrics().then(setData).catch((e) => setError(String(e)));
  }, []);

  if (error) {
    return (
      <div className="card border border-amber/30 px-4 py-3"
        style={{ background: "rgba(210,153,34,0.05)" }}>
        <p className="font-mono text-[11px] text-amber">
          Live RL metrics unavailable — start the backend to load them. ({error})
        </p>
      </div>
    );
  }
  if (!data) {
    return <div className="card p-5"><div className="skeleton h-40 w-full" /></div>;
  }

  const best = data.summary?.best ?? {};
  const final = data.summary?.final ?? {};
  const sample = data.live_sample;
  const maxAbsQ = sample
    ? Math.max(...Object.values(sample.q_values).map((q) => Math.abs(q)), 1)
    : 1;

  return (
    <div>
      <SectionHeader title="Live Training Results"
        sub="Real metrics from the trained policy — not illustrative" />
      <div className="grid grid-cols-2 gap-4">

        {/* Curve */}
        <div className="card p-4">
          <div className="flex items-center justify-between mb-3">
            <p className="panel-label">Dyna-CQL Training Curves</p>
            <span className="badge-teal text-[9px] px-1.5 py-0.5">
              {data.rl_policy_loaded ? "rl_policy.pt" : "no checkpoint"}
            </span>
          </div>
          {data.curve_b64 ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img src={`data:image/png;base64,${data.curve_b64}`}
              alt="RL training curves"
              className="w-full rounded border border-border" />
          ) : (
            <p className="font-mono text-[11px] text-text3">
              No curve yet — run train_rl.py to generate logs/.
            </p>
          )}
          <div className="grid grid-cols-3 gap-2 mt-3">
            {[
              { l: "Best reward", v: best.avg_reward_50?.toFixed(1) ?? "—" },
              { l: "Final loss", v: final.loss?.toFixed(2) ?? "—" },
              { l: "Episodes", v: data.summary?.n_steps ?? "—" },
            ].map(({ l, v }) => (
              <div key={l} className="card-inset px-2 py-2 text-center">
                <p className="kpi-label mb-0.5">{l}</p>
                <p className="kpi-value text-sm text-teal">{v}</p>
              </div>
            ))}
          </div>
        </div>

        {/* Live Q-values */}
        <div className="card p-4">
          <div className="flex items-center justify-between mb-3">
            <p className="panel-label">Live Policy Readout</p>
            <span className="badge-neutral text-[9px] px-1.5 py-0.5">sampled state</span>
          </div>
          {sample && !sample.error ? (
            <>
              <p className="text-xs text-text2 mb-3">
                On a sampled elevated crowd state, the policy recommends{" "}
                <span className="font-mono text-teal">{sample.action_name}</span>{" "}
                ({(sample.confidence * 100).toFixed(0)}% confidence).
              </p>
              <div className="space-y-1.5">
                {Object.entries(sample.q_values)
                  .sort((a, b) => b[1] - a[1])
                  .map(([name, q]) => {
                    const w = (Math.abs(q) / maxAbsQ) * 100;
                    const best = name === sample.action_name;
                    return (
                      <div key={name} className="flex items-center gap-2">
                        <span className={`font-mono text-[10px] w-28 truncate ${best ? "text-teal" : "text-text3"}`}>
                          {name}
                        </span>
                        <div className="flex-1 h-3 rounded bg-raised/40 overflow-hidden">
                          <div className={`h-full ${q >= 0 ? "bg-teal/60" : "bg-crimson/50"}`}
                            style={{ width: `${w}%` }} />
                        </div>
                        <span className="font-mono text-[10px] text-text3 w-12 text-right">
                          {q.toFixed(2)}
                        </span>
                      </div>
                    );
                  })}
              </div>
            </>
          ) : (
            <p className="font-mono text-[11px] text-text3">
              Policy readout unavailable{sample?.error ? ` (${sample.error})` : ""}.
            </p>
          )}
        </div>
      </div>
    </div>
  );
}

export default function RLTab() {
  return (
    <div className="h-full overflow-y-auto">
      <div className="max-w-4xl mx-auto p-6 space-y-6">

        {/* Architecture header */}
        <div className="rounded-xl border border-border p-5"
          style={{ background: "linear-gradient(135deg, rgba(68,147,248,0.06) 0%, rgba(68,147,248,0.02) 100%)" }}>
          <div className="flex items-start justify-between">
            <div>
              <p className="font-mono text-[10px] text-teal uppercase tracking-widest mb-1">Architecture</p>
              <h1 className="display text-2xl font-bold text-text1 mb-2">Dyna-CQL</h1>
              <p className="text-sm text-text2 max-w-xl leading-relaxed">
                Conservative Q-Learning trained inside the world model via Dyna-style planning.
                The world model generates synthetic crowd scenarios in latent space. The policy
                learns safe interventions without a single real crush event — same family as DreamerV3.
              </p>
            </div>
            <div className="flex gap-3 flex-shrink-0 ml-4">
              <div className="card-inset px-4 py-3 text-center">
                <p className="kpi-label mb-1">Actions</p>
                <p className="kpi-value text-xl text-teal">7</p>
              </div>
              <div className="card-inset px-4 py-3 text-center">
                <p className="kpi-label mb-1">Episodes</p>
                <p className="kpi-value text-xl text-amber">300</p>
              </div>
              <div className="card-inset px-4 py-3 text-center">
                <p className="kpi-label mb-1">State dim</p>
                <p className="kpi-value text-xl text-text1">64</p>
              </div>
            </div>
          </div>
        </div>

        {/* Live training results (real) */}
        <LiveResults />

        {/* Why CQL + Why Dyna */}
        <div>
          <SectionHeader title="Design Rationale" sub="Why these two algorithms for crowd safety" />
          <div className="grid grid-cols-2 gap-4">
            <ConceptCard
              title="Why CQL?"
              tag="Conservative"
              body="Conservative Q-Learning penalizes overestimating Q-values on unseen state-action pairs. The policy only recommends interventions it has reliably seen work. For a safety-critical system, conservatism is not a limitation — it's the entire point."
            />
            <ConceptCard
              title="Why Dyna?"
              tag="Model-based"
              body="You cannot run live experiments on real crowds. The world model becomes the simulator. Generate synthetic crowd scenarios inside its latent space, train the policy on those imagined events. No real disasters needed to learn safe behavior."
            />
          </div>
        </div>

        {/* Action space */}
        <div>
          <SectionHeader title="Physics-Primitive Action Space" sub="Venue-agnostic interventions that map to fluid dynamics operations" />
          <div className="card overflow-hidden">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-border" style={{ background: "rgba(13,17,23,0.4)" }}>
                  {["ID", "Primitive", "Physics Effect", "Cost"].map((h) => (
                    <th key={h} className="px-4 py-2.5 text-left font-mono text-[9px] text-text3 uppercase tracking-wider">
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {ACTIONS.map(({ id, name, effect, cost, risk }) => {
                  const costColor =
                    cost === 0 ? "text-emerald" :
                    cost <= 0.2 ? "text-teal"  :
                    cost <= 0.3 ? "text-amber" : "text-crimson";
                  const nameColor =
                    risk === "high" ? "text-crimson" :
                    risk === "med"  ? "text-amber"   : "text-teal";
                  return (
                    <tr key={id} className="trow">
                      <td className="px-4 py-2.5 font-mono text-text3 text-[11px]">{id}</td>
                      <td className="px-4 py-2.5">
                        <span className={`font-mono font-medium ${nameColor}`}>{name}</span>
                      </td>
                      <td className="px-4 py-2.5 text-text2">{effect}</td>
                      <td className="px-4 py-2.5">
                        <span className={`font-mono font-semibold ${costColor}`}>{cost.toFixed(1)}</span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>

        {/* Reward function */}
        <div>
          <SectionHeader title="Reward Function" sub="Shaped to prevent both under- and over-reaction" />
          <div className="card p-5">
            <div className="flex items-center justify-between mb-3">
              <p className="panel-label">Reward Shaping</p>
              <span className="badge-neutral">Python pseudocode</span>
            </div>
            <div className="code-block">{REWARD_CODE}</div>
            <div className="mt-4 grid grid-cols-3 gap-3">
              {[
                { label: "Safety reward",  desc: "3× bonus for reducing danger", color: "text-emerald" },
                { label: "Crush penalty",  desc: "−15 if threshold crossed",      color: "text-crimson" },
                { label: "Cost penalty",   desc: "Deducted per action cost",      color: "text-amber" },
              ].map(({ label, desc, color }) => (
                <div key={label} className="card-inset px-3 py-2.5">
                  <p className={`font-mono text-[10px] font-semibold ${color} mb-0.5`}>{label}</p>
                  <p className="font-mono text-[9px] text-text3 leading-tight">{desc}</p>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Training pipeline */}
        <div>
          <SectionHeader title="Training Pipeline" sub="Two-stage: world model first, then Dyna RL" />
          <div className="card overflow-hidden">
            <div className="flex">
              {[
                {
                  step: "1",
                  label: "World Model",
                  color: "border-teal/40 bg-teal/5",
                  dot: "bg-teal",
                  items: [
                    "CNN Encoder: 256 → 64-dim z",
                    "LSTM transition: z_t → z_{t+1}",
                    "Trained on real crowd videos",
                    "Next-state prediction loss",
                  ],
                },
                {
                  step: "2",
                  label: "Dyna RL",
                  color: "border-amber/40 bg-amber/5",
                  dot: "bg-amber",
                  items: [
                    "Dueling DQN with CQL penalty",
                    "300 episodes in latent sim",
                    "7-action intervention space",
                    "Shaped reward function",
                  ],
                },
              ].map(({ step, label, color, dot, items }) => (
                <div key={step} className={`flex-1 p-5 border-r border-border last:border-0 ${color}`}>
                  <div className="flex items-center gap-2 mb-3">
                    <span className={`w-2 h-2 rounded-full ${dot}`} />
                    <p className="font-mono text-[10px] text-text3 uppercase tracking-wider">Stage {step}</p>
                  </div>
                  <p className="display text-sm font-semibold text-text1 mb-3">{label}</p>
                  <ul className="space-y-1.5">
                    {items.map((item) => (
                      <li key={item} className="flex items-start gap-2">
                        <span className="font-mono text-[10px] text-text3 mt-0.5">›</span>
                        <span className="font-mono text-[11px] text-text2 leading-snug">{item}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              ))}
            </div>
          </div>
        </div>

      </div>
    </div>
  );
}
