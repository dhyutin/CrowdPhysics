export default function RLTab() {
  return (
    <div className="p-6 max-w-4xl mx-auto text-sm text-text2 leading-relaxed">
      <h2 className="display text-xl font-bold text-text1 mb-4">
        Architecture: Dyna-CQL
      </h2>
      <p className="mb-4 max-w-2xl">
        Conservative Q-Learning trained inside the world model via Dyna. The
        world model generates synthetic crowd scenarios in latent space. The RL
        policy learns which interventions prevent disasters without a single real
        crush event in the training set — same architecture family as DreamerV3.
      </p>

      <div className="grid grid-cols-2 gap-4 mb-6">
        <div className="card p-4">
          <p className="card-label mb-3">Why CQL?</p>
          <p>
            Conservative Q-Learning adds a penalty for overestimating Q-values
            on unseen state-action pairs. The policy becomes conservative — it
            only recommends what it has seen work. For a safety-critical system,
            this is not a limitation. <strong className="text-text1">It&apos;s the entire point.</strong>
          </p>
        </div>
        <div className="card p-4">
          <p className="card-label mb-3">Why Dyna?</p>
          <p>
            You cannot run experiments on real crowds. The world model IS the
            simulator. Generate synthetic scenarios in latent space, train the
            policy inside those imagined scenarios.{" "}
            <strong className="text-text1">No real disasters needed.</strong>
          </p>
        </div>
      </div>

      <div className="card mb-4">
        <div className="px-3 py-2 border-b border-border">
          <p className="card-label">Physics-Primitive Action Space (Venue-Agnostic)</p>
        </div>
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-border">
              {["ID", "Primitive", "Physics Effect", "Cost"].map((h) => (
                <th key={h} className="px-3 py-2 text-left mono text-text3 uppercase tracking-wider">
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {[
              ["A0", "monitor",          "No change",                         "0.0"],
              ["A1", "increase_egress",  "Dampen y-compression",              "0.1"],
              ["A2", "reduce_ingress",   "Damp incoming flow energy",         "0.1"],
              ["A3", "lateral_redirect", "Increase x-dims, reduce y-dims",    "0.3"],
              ["A4", "disperse",         "Noise → global damping",            "0.2"],
              ["A5", "partial_evac",     "Strong targeted damping (0.6×)",    "0.2"],
              ["A6", "full_evac",        "Global damping (0.3×) — emergency", "1.0"],
            ].map(([id, name, effect, cost]) => (
              <tr key={id} className="border-b border-border/40 last:border-0">
                <td className="px-3 py-2 mono text-text3">{id}</td>
                <td className="px-3 py-2 mono text-amber">{name}</td>
                <td className="px-3 py-2 text-text2">{effect}</td>
                <td className="px-3 py-2 mono text-text3">{cost}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="card p-4">
        <p className="card-label mb-3">Reward Function</p>
        <pre className="mono text-xs text-text2 leading-7 whitespace-pre-wrap">{`reward = (danger_before - danger_after) × 3.0   // safety improvement
if danger_after > 3.5:  reward -= 15.0           // crush threshold crossed
if danger_after < 0.8:  reward += 1.5            // staying safe bonus
if action==monitor && danger>2: reward -= 2.0    // inaction penalty
if action==full_evac && danger<1: reward -= 3.0  // overreaction penalty
reward -= action_cost[action]                    // [0,.1,.1,.3,.2,.2,1.0]`}</pre>
      </div>
    </div>
  );
}
