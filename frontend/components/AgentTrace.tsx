"use client";

import type { AgentTraceStep } from "@/lib/api";

const ICONS: Record<string, React.ReactNode> = {
  calibrate: (
    <path strokeLinecap="round" strokeLinejoin="round" d="M12 6v6l4 2M12 21a9 9 0 110-18 9 9 0 010 18z" />
  ),
  brain: (
    <path strokeLinecap="round" strokeLinejoin="round" d="M9 3a3 3 0 00-3 3v.5A2.5 2.5 0 004.5 9 2.5 2.5 0 006 11.5V12a3 3 0 003 3m0-12a3 3 0 013 3v9m-3-12v12m0 0a3 3 0 003 3m6-6a3 3 0 01-3 3" />
  ),
  pulse: (
    <path strokeLinecap="round" strokeLinejoin="round" d="M3 12h3l2-6 4 12 2-6h4" />
  ),
  forecast: (
    <path strokeLinecap="round" strokeLinejoin="round" d="M3 17l5-5 3 3 7-7m0 0v4m0-4h-4" />
  ),
  claude: (
    <path strokeLinecap="round" strokeLinejoin="round" d="M8 9h8M8 13h5m-9 7l3-3h9a2 2 0 002-2V6a2 2 0 00-2-2H5a2 2 0 00-2 2v12z" />
  ),
  shield: (
    <path strokeLinecap="round" strokeLinejoin="round" d="M12 3l7 3v6c0 4.5-3 7.5-7 9-4-1.5-7-4.5-7-9V6l7-3z" />
  ),
  eye: (
    <path strokeLinecap="round" strokeLinejoin="round" d="M2.5 12S5.5 5.5 12 5.5 21.5 12 21.5 12 18.5 18.5 12 18.5 2.5 12 2.5 12zM12 15a3 3 0 100-6 3 3 0 000 6z" />
  ),
  plan: (
    <path strokeLinecap="round" strokeLinejoin="round" d="M9 3v18m6-18v18M3 9h18M3 15h18M5 3h14a2 2 0 012 2v14a2 2 0 01-2 2H5a2 2 0 01-2-2V5a2 2 0 012-2z" />
  ),
};

function StepIcon({ icon, danger }: { icon: string; danger: boolean }) {
  return (
    <div
      className="w-7 h-7 rounded-lg flex items-center justify-center flex-shrink-0 border"
      style={{
        background: danger ? "rgba(248,81,73,0.1)" : "rgba(94,23,235,0.10)",
        borderColor: danger ? "rgba(248,81,73,0.3)" : "rgba(94,23,235,0.28)",
      }}
    >
      <svg
        viewBox="0 0 24 24"
        className={`w-3.5 h-3.5 ${danger ? "text-crimson" : "text-teal"}`}
        fill="none"
        stroke="currentColor"
        strokeWidth="1.4"
      >
        {ICONS[icon] ?? ICONS.pulse}
      </svg>
    </div>
  );
}

export default function AgentTrace({
  steps,
  title = "Agent Trace",
}: {
  steps: AgentTraceStep[];
  title?: string;
}) {
  if (!steps?.length) return null;
  return (
    <div className="card flex flex-col animate-fade-in">
      <div className="panel-header">
        <p className="panel-label">{title}</p>
        <span className="badge-teal text-[9px] px-1.5 py-0.5">{steps.length} agents</span>
      </div>
      <div className="p-4">
        <div className="relative flex flex-col gap-3">
          {/* connecting spine */}
          <div className="absolute left-[13px] top-3 bottom-3 w-px bg-border" />
          {steps.map((s, i) => {
            const danger = s.status === "danger";
            return (
              <div key={i} className="relative flex gap-3 items-start">
                <StepIcon icon={s.icon} danger={danger} />
                <div className="flex-1 min-w-0 pt-0.5">
                  <div className="flex items-center gap-2 flex-wrap">
                    <span className="font-mono text-[10px] text-text1 font-medium tracking-wide">
                      {s.agent}
                    </span>
                    <span
                      className={`w-1.5 h-1.5 rounded-full ${
                        danger ? "bg-crimson" : "bg-emerald"
                      }`}
                    />
                  </div>
                  <p className="text-[12px] text-text2 leading-tight mt-0.5">{s.action}</p>
                  {s.detail && (
                    <p className="font-mono text-[9px] text-text3 leading-snug mt-0.5 truncate">
                      {s.detail}
                    </p>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
