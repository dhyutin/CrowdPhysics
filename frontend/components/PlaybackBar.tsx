"use client";

export default function PlaybackBar({
  frame,
  frames,
  playing,
  onToggle,
  onScrub,
}: {
  frame: number;
  frames: number;
  playing: boolean;
  onToggle: () => void;
  onScrub: (f: number) => void;
}) {
  const pct = frames > 1 ? Math.round((frame / (frames - 1)) * 100) : 0;
  return (
    <div className="card-inset flex items-center gap-3 px-3 py-2">
      <button
        onClick={onToggle}
        className="w-8 h-8 rounded-md flex items-center justify-center text-void flex-shrink-0 transition-transform hover:scale-105"
        style={{ background: "linear-gradient(135deg, #e2a9f1 0%, #5e17eb 100%)" }}
        aria-label={playing ? "Pause" : "Play"}
      >
        {playing ? (
          <svg className="w-3.5 h-3.5" fill="#fff" viewBox="0 0 16 16">
            <rect x="3" y="2" width="3.5" height="12" rx="1" />
            <rect x="9.5" y="2" width="3.5" height="12" rx="1" />
          </svg>
        ) : (
          <svg className="w-3.5 h-3.5" fill="#fff" viewBox="0 0 16 16">
            <path d="M4 2.5v11a.5.5 0 00.77.42l8.5-5.5a.5.5 0 000-.84l-8.5-5.5A.5.5 0 004 2.5z" />
          </svg>
        )}
      </button>

      <input
        type="range"
        min={0}
        max={Math.max(0, frames - 1)}
        value={frame}
        onChange={(e) => onScrub(Number(e.target.value))}
        className="flex-1 accent-lavender cursor-pointer"
        style={{ accentColor: "#e2a9f1" }}
      />

      <span className="font-mono text-[10px] text-text3 w-16 text-right tabular-nums">
        t {pct}% · {frame + 1}/{frames}
      </span>
    </div>
  );
}
