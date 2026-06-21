import type { VenueLayout, VenueLayoutElement } from "@/lib/api";

// A numbered marker on the 3D venue: an access point (entry / exit) the
// crowd uses. The number ties the pin in the 3D scene to the legend.
export interface VenuePin {
  n: number; // 1-based pin number shown in the scene + legend
  type: "entry" | "gate";
  role: "Entry" | "Exit";
  label: string; // human-readable name, e.g. "Entry · N" or the agent's label
  x: number; // normalized centre (0-1)
  y: number;
  w: number;
  h: number;
  color: string;
}

const PIN_COLOR: Record<string, string> = {
  entry: "#3FB950", // green
  gate: "#4493F8", // blue
};

function compass(x: number, y: number): string {
  const v = y < 0.34 ? "N" : y > 0.66 ? "S" : "";
  const h = x < 0.34 ? "W" : x > 0.66 ? "E" : "";
  return v + h || "Center";
}

// Does the layout already specify any entrance / exit?
export function hasAccessPoints(layout?: VenueLayout | null): boolean {
  return (layout?.elements ?? []).some(
    (e) => e.type === "entry" || e.type === "gate"
  );
}

// When the photo / user gave no entrances or exits, assume sensible defaults
// (one entry on the north edge, exits on the south + east edges) sitting right
// on the boundary so the crowd can stream in and out. These are injected into
// the layout and clearly flagged so the user can correct them via the editor.
export function assumedAccessElements(): VenueLayoutElement[] {
  return [
    { type: "entry", x: 0.4, y: 0.0, w: 0.2, h: 0.04, label: "Assumed entry" },
    { type: "gate", x: 0.4, y: 0.96, w: 0.2, h: 0.04, label: "Assumed exit" },
    { type: "gate", x: 0.96, y: 0.4, w: 0.04, h: 0.2, label: "Assumed exit" },
  ];
}

// Ordered list of access-point pins for a layout. Entries first, then exits,
// so the legend reads "how they enter → how they leave".
export function buildVenuePins(layout?: VenueLayout | null): VenuePin[] {
  const els = (layout?.elements ?? []) as VenueLayoutElement[];
  const access = els.filter((e) => e.type === "entry" || e.type === "gate");
  access.sort((a, b) => (a.type === b.type ? 0 : a.type === "entry" ? -1 : 1));

  return access.map((e, i) => {
    const cx = e.x + e.w / 2;
    const cy = e.y + e.h / 2;
    const role = e.type === "entry" ? "Entry" : "Exit";
    const named = (e.label ?? "").trim();
    return {
      n: i + 1,
      type: e.type as "entry" | "gate",
      role,
      label: named || `${role} · ${compass(cx, cy)}`,
      x: cx,
      y: cy,
      w: e.w,
      h: e.h,
      color: PIN_COLOR[e.type] ?? "#9aa3b2",
    };
  });
}
