"use client";

export default function PlanPoints({ points }: { points: string[] }) {
  if (!points?.length) return null;
  return (
    <div className="card flex flex-col">
      <div className="panel-header">
        <p className="panel-label">Action Plan</p>
        <span className="badge-teal text-[9px] px-1.5 py-0.5">Agent</span>
      </div>
      <ul className="p-4 flex flex-col gap-2.5">
        {points.map((pt, i) => (
          <li key={i} className="flex gap-2.5 items-start">
            <span
              className="mt-0.5 w-4 h-4 rounded-full flex-shrink-0 flex items-center justify-center font-mono text-[9px] font-bold text-void"
              style={{ background: "linear-gradient(135deg, #e2a9f1 0%, #5e17eb 100%)" }}
            >
              {i + 1}
            </span>
            <span className="text-xs text-text2 leading-snug">{pt}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}
