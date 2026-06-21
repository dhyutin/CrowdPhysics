export default function Hero() {
  return (
    <div
      className="border-b border-border px-10 py-8"
      style={{
        background:
          "linear-gradient(135deg, #060a12 0%, #0a1628 60%, #060a12 100%)",
      }}
    >
      <p className="mono text-[10px] tracking-[0.3em] text-amber uppercase mb-2">
        CrowdPhysics · AI Safety Platform
      </p>
      <h1 className="display text-[34px] font-bold tracking-tight text-text1 leading-tight mb-3">
        Plan safe. Monitor live.
        <br />
        Never react.
      </h1>
      <p className="text-sm text-text2 max-w-xl leading-relaxed mb-6">
        A world model that discovered crowd fluid dynamics from video alone. An
        RL policy trained inside that model — no real disasters needed. Claude
        translates physics into language security personnel can act on at 2am.
      </p>
      <div className="flex gap-5 flex-wrap">
        {[
          { label: "LEAD TIME", value: "4.2 min", color: "text-crimson" },
          { label: "LIVES LOST 2024–25", value: "200+", color: "text-amber" },
          { label: "HARDWARE NEEDED", value: "Any cam", color: "text-emerald" },
        ].map(({ label, value, color }) => (
          <div
            key={label}
            className="bg-surface border border-border rounded-md px-5 py-3 min-w-[120px]"
          >
            <p className="mono text-[10px] tracking-widest text-text3 uppercase mb-1">
              {label}
            </p>
            <p className={`display text-3xl font-bold leading-none ${color}`}>
              {value}
            </p>
          </div>
        ))}
      </div>
    </div>
  );
}
