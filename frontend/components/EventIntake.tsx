"use client";

export interface IntakeValue {
  purpose: string;
  nPeople: string;
  density: string;
  durationMin: string;
  seating: string;
  ingress: string;
  notes: string;
  areaM2: string;
}

// Mirrors backend _capacity_from_area() so the UI can preview the estimate.
export function estimateCapacity(
  areaM2: number,
  seating: string,
  densityPct: number
): { perM2: number; capacity: number } | null {
  if (!areaM2 || areaM2 <= 0) return null;
  const d = Math.max(0, Math.min(1, (densityPct || 0) / 100));
  const usable = 0.78;
  const perM2 =
    seating === "seated" ? 1.0 + d * 0.5 :
    seating === "mixed"  ? 1.5 + d * 1.5 :
                           2.0 + d * 2.0;
  return { perM2: Math.round(perM2 * 100) / 100, capacity: Math.max(1, Math.round(areaM2 * usable * perM2)) };
}

const PURPOSES = [
  "Concert",
  "Sports match",
  "Rally / protest",
  "Expo / conference",
  "Religious gathering",
  "Night market",
  "Evacuation drill",
];

function Segmented({
  options,
  value,
  onChange,
}: {
  options: { id: string; label: string }[];
  value: string;
  onChange: (v: string) => void;
}) {
  return (
    <div className="flex gap-1">
      {options.map((o) => (
        <button
          key={o.id}
          onClick={() => onChange(o.id)}
          className={`flex-1 font-mono text-[9px] px-1 py-1 rounded border transition-colors ${
            value === o.id
              ? "bg-lavender/15 text-lavender border-lavender/40"
              : "text-text3 border-border hover:text-text2"
          }`}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

export default function EventIntake({
  value,
  onChange,
  disabled,
}: {
  value: IntakeValue;
  onChange: (patch: Partial<IntakeValue>) => void;
  disabled?: boolean;
}) {
  return (
    <fieldset disabled={disabled} className="flex flex-col gap-3">
      <div>
        <label className="field-label">What is the event?</label>
        <input
          className="input text-sm"
          value={value.purpose}
          onChange={(e) => onChange({ purpose: e.target.value })}
          placeholder="What is the space for?"
        />
        <div className="flex flex-wrap gap-1 mt-2">
          {PURPOSES.map((p) => (
            <button
              key={p}
              onClick={() => onChange({ purpose: p })}
              className={`font-mono text-[9px] px-1.5 py-0.5 rounded border transition-colors ${
                value.purpose === p
                  ? "bg-lavender/15 text-lavender border-lavender/30"
                  : "text-text3 border-border hover:text-text2"
              }`}
            >
              {p}
            </button>
          ))}
        </div>
      </div>

      <div>
        <label className="field-label">How many people?</label>
        <input
          className="input text-sm"
          value={value.nPeople}
          onChange={(e) => onChange({ nPeople: e.target.value })}
          placeholder="auto-estimate"
          inputMode="numeric"
        />
      </div>

      <div>
        <label className="field-label">Usable area (m²)</label>
        <input
          className="input text-sm"
          value={value.areaM2}
          onChange={(e) => onChange({ areaM2: e.target.value })}
          placeholder="e.g. 1200"
          inputMode="numeric"
        />
        {(() => {
          const est = estimateCapacity(
            parseFloat(value.areaM2) || 0,
            value.seating,
            parseInt(value.density) || 65
          );
          if (!est) {
            return (
              <p className="font-mono text-[9px] text-text3 mt-1 leading-snug">
                Enter the floor area and we’ll estimate max capacity (used when
                “people” is left blank).
              </p>
            );
          }
          return (
            <p className="font-mono text-[9px] text-text2 mt-1 leading-snug">
              ≈ <span className="text-lavender">{est.capacity.toLocaleString()}</span> max
              {" "}({est.perM2}/m² · {value.seating})
              {!value.nPeople && (
                <span className="text-text3"> — used as the crowd size</span>
              )}
            </p>
          );
        })()}
      </div>

      <div>
        <label className="field-label">How do they arrive?</label>
        <Segmented
          value={value.ingress}
          onChange={(v) => onChange({ ingress: v })}
          options={[
            { id: "gradual", label: "Gradual" },
            { id: "steady", label: "Steady" },
            { id: "burst", label: "All at once" },
          ]}
        />
      </div>

      <div>
        <label className="field-label">Crowd setup</label>
        <Segmented
          value={value.seating}
          onChange={(v) => onChange({ seating: v })}
          options={[
            { id: "standing", label: "Standing" },
            { id: "seated", label: "Seated" },
            { id: "mixed", label: "Mixed" },
          ]}
        />
      </div>

      <div className="grid grid-cols-2 gap-2">
        <div>
          <label className="field-label">Duration (min)</label>
          <input
            className="input text-sm"
            value={value.durationMin}
            onChange={(e) => onChange({ durationMin: e.target.value })}
            placeholder="120"
            inputMode="numeric"
          />
        </div>
        <div>
          <label className="field-label">Density (%)</label>
          <input
            className="input text-sm"
            value={value.density}
            onChange={(e) => onChange({ density: e.target.value })}
            inputMode="numeric"
          />
        </div>
      </div>

      <div>
        <label className="field-label">Anything else?</label>
        <textarea
          className="input text-sm resize-none"
          rows={2}
          value={value.notes}
          onChange={(e) => onChange({ notes: e.target.value })}
          placeholder="alcohol served, VIP area, kids, weather…"
        />
      </div>
    </fieldset>
  );
}
