"use client";

import { useMemo, useRef, useEffect, type MutableRefObject } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import type {
  Scenario,
  VenueLayout,
  VenueLayoutElement,
  DangerZone,
  DecorProp,
  VenueArchetype,
  AgentPlan,
} from "@/lib/api";

// World is a SIZE x SIZE square centered at the origin, floor on the XZ plane,
// Y is up. Normalized layout coords (0-1, top-left origin) map onto it.
const SIZE = 24;
// World height of a full-height (relative 1.0) element.
const H_MAX = 7;
const wx = (nx: number) => (nx - 0.5) * SIZE;
const wz = (ny: number) => (ny - 0.5) * SIZE;

// Fallback relative heights when the vision agent didn't supply one.
const DEFAULT_REL_H: Record<string, number> = {
  wall: 1.0,
  stage: 0.55,
  barrier: 0.3,
  entry: 0.05,
  gate: 0.05,
};
const EL_COLOR: Record<string, string> = {
  wall: "#2b3340",
  stage: "#2DD4BF",
  barrier: "#A371F7",
};

// Default palette for scene-detail props when the details agent gives no color.
const PROP_COLOR: Record<string, string> = {
  slide: "#E5484D",
  swing: "#4493F8",
  playset: "#F2A93B",
  fountain: "#5BC8E0",
  statue: "#9aa3b2",
  bench: "#8a6a45",
  booth: "#D6409F",
  goal: "#e7eaf0",
  pole: "#9aa3b2",
  planter: "#3b7a47",
  court: "#2f6f5e",
};

const relH = (e: VenueLayoutElement) =>
  e.height && e.height > 0 ? e.height : DEFAULT_REL_H[e.type] ?? 0.4;
const worldH = (e: VenueLayoutElement) => Math.max(0.2, relH(e) * H_MAX);

// ── helpers ──────────────────────────────────────────────────────────────────

function isWall(walls: number[][], G: number, nx: number, ny: number): boolean {
  const gx = Math.min(G - 1, Math.max(0, Math.floor(nx * G)));
  const gy = Math.min(G - 1, Math.max(0, Math.floor(ny * G)));
  return walls?.[gy]?.[gx] === 1;
}

const C_LOW = new THREE.Color("#4493F8");
const C_MID = new THREE.Color("#D29922");
const C_HIGH = new THREE.Color("#F85149");

function pressureColor(out: THREE.Color, p: number, pMax: number) {
  const t = Math.max(0, Math.min(1, p / (pMax || 1)));
  if (t < 0.5) out.copy(C_LOW).lerp(C_MID, t / 0.5);
  else out.copy(C_MID).lerp(C_HIGH, (t - 0.5) / 0.5);
}

// ── Static venue geometry ─────────────────────────────────────────────────────

// One structural element rendered according to its 3D `shape`.
function StructureMesh({ e }: { e: VenueLayoutElement }) {
  const h = worldH(e);
  const w = Math.max(0.2, e.w * SIZE);
  const d = Math.max(0.2, e.h * SIZE);
  const cx = wx(e.x + e.w / 2);
  const cz = wz(e.y + e.h / 2);
  const color = EL_COLOR[e.type] ?? "#2b3340";
  const isStage = e.type === "stage";

  const mat = (
    <meshStandardMaterial
      color={color}
      roughness={0.55}
      metalness={0.15}
      emissive={isStage ? "#0d6b60" : "#000000"}
      emissiveIntensity={isStage ? 0.5 : 0}
    />
  );

  const shape = e.shape ?? "box";

  // Tiered seating / stadium stand: stacked, inward-shrinking steps.
  if (shape === "tiered") {
    const steps = 5;
    return (
      <group position={[cx, 0, cz]}>
        {Array.from({ length: steps }).map((_, s) => {
          const sh = h / steps;
          const shrink = 1 - (s / steps) * 0.55;
          return (
            <mesh key={s} position={[0, sh * (s + 0.5), 0]} castShadow receiveShadow>
              <boxGeometry args={[w * shrink, sh, d * shrink]} />
              {mat}
            </mesh>
          );
        })}
      </group>
    );
  }

  // Round pillar / tower.
  if (shape === "cylinder") {
    const r = Math.max(0.15, Math.min(w, d) / 2);
    return (
      <mesh position={[cx, h / 2, cz]} castShadow receiveShadow>
        <cylinderGeometry args={[r, r, h, 20]} />
        {mat}
      </mesh>
    );
  }

  // Domed roof / rotunda (squashed hemisphere over the footprint).
  if (shape === "dome") {
    const r = Math.max(0.3, Math.min(w, d) / 2);
    return (
      <group position={[cx, 0, cz]} scale={[1, h / r, 1]}>
        <mesh castShadow receiveShadow>
          <sphereGeometry args={[r, 24, 16, 0, Math.PI * 2, 0, Math.PI / 2]} />
          {mat}
        </mesh>
      </group>
    );
  }

  // Sloped ramp surface.
  if (shape === "ramp") {
    return (
      <mesh
        position={[cx, h / 2, cz]}
        rotation={[Math.atan2(h, d) * 0.5, 0, 0]}
        castShadow
        receiveShadow
      >
        <boxGeometry args={[w, Math.max(0.15, h * 0.4), Math.hypot(d, h)]} />
        {mat}
      </mesh>
    );
  }

  // Flat canopy roof raised on thin legs.
  if (shape === "canopy") {
    const legR = 0.12;
    const legs: [number, number][] = [
      [-w / 2 + legR, -d / 2 + legR],
      [w / 2 - legR, -d / 2 + legR],
      [-w / 2 + legR, d / 2 - legR],
      [w / 2 - legR, d / 2 - legR],
    ];
    return (
      <group position={[cx, 0, cz]}>
        {legs.map(([lx, lz], i) => (
          <mesh key={i} position={[lx, h / 2, lz]} castShadow>
            <cylinderGeometry args={[legR, legR, h, 8]} />
            <meshStandardMaterial color="#3a4250" roughness={0.6} />
          </mesh>
        ))}
        <mesh position={[0, h, 0]} castShadow receiveShadow>
          <boxGeometry args={[w, 0.25, d]} />
          {mat}
        </mesh>
      </group>
    );
  }

  // Default solid box (walls, plain stages/barriers).
  return (
    <mesh position={[cx, h / 2, cz]} castShadow receiveShadow>
      <boxGeometry args={[w, h, d]} />
      {mat}
    </mesh>
  );
}

function Structures({ layout }: { layout: VenueLayout }) {
  return (
    <group>
      {layout.elements.map((e, i) =>
        e.type === "wall" || e.type === "stage" || e.type === "barrier" ? (
          <StructureMesh key={`s${i}`} e={e} />
        ) : null
      )}
    </group>
  );
}

// ── Visual-only decor (screens, towers, tents, trees, roofs) ──────────────────

function Decor({ layout }: { layout: VenueLayout }) {
  const props = layout.decor ?? [];
  return (
    <group>
      {props.map((d: DecorProp, i: number) => {
        const w = Math.max(0.2, d.w * SIZE);
        const depth = Math.max(0.2, d.h * SIZE);
        const h = Math.max(0.4, (d.height ?? 0.5) * H_MAX);
        const cx = wx(d.x + d.w / 2);
        const cz = wz(d.y + d.h / 2);

        if (d.type === "screen") {
          // Big emissive LED panel on a stand, facing the venue centre.
          const yaw = Math.atan2(-cx, -cz);
          const panelW = Math.max(w, depth);
          return (
            <group key={`d${i}`} position={[cx, 0, cz]} rotation={[0, yaw, 0]}>
              <mesh position={[0, h * 0.62, 0]} castShadow>
                <boxGeometry args={[panelW, h * 0.5, 0.2]} />
                <meshStandardMaterial
                  color="#0b0f16"
                  emissive="#2DD4BF"
                  emissiveIntensity={0.7}
                  roughness={0.3}
                />
              </mesh>
              <mesh position={[0, h * 0.3, 0]} castShadow>
                <boxGeometry args={[panelW * 0.12, h * 0.6, panelW * 0.12]} />
                <meshStandardMaterial color="#39414f" roughness={0.6} />
              </mesh>
            </group>
          );
        }

        if (d.type === "tower") {
          const r = Math.max(0.12, Math.min(w, depth) / 2);
          return (
            <group key={`d${i}`} position={[cx, 0, cz]}>
              <mesh position={[0, h / 2, 0]} castShadow>
                <cylinderGeometry args={[r * 0.6, r, h, 10]} />
                <meshStandardMaterial color="#39414f" roughness={0.6} />
              </mesh>
              <mesh position={[0, h, 0]}>
                <boxGeometry args={[r * 3, r * 1.4, r * 1.4]} />
                <meshBasicMaterial color="#FFE9A8" toneMapped={false} />
              </mesh>
            </group>
          );
        }

        if (d.type === "tent") {
          return (
            <group key={`d${i}`} position={[cx, 0, cz]}>
              <mesh position={[0, h * 0.25, 0]} castShadow>
                <boxGeometry args={[w, h * 0.5, depth]} />
                <meshStandardMaterial color="#e7eaf0" roughness={0.7} />
              </mesh>
              <mesh position={[0, h * 0.72, 0]} castShadow>
                <coneGeometry args={[Math.max(w, depth) * 0.72, h * 0.45, 4]} />
                <meshStandardMaterial color="#A371F7" roughness={0.6} />
              </mesh>
            </group>
          );
        }

        if (d.type === "tree") {
          const r = Math.max(0.5, Math.min(w, depth) * 0.5 + 0.4);
          return (
            <group key={`d${i}`} position={[cx, 0, cz]}>
              <mesh position={[0, h * 0.25, 0]} castShadow>
                <cylinderGeometry args={[r * 0.18, r * 0.22, h * 0.5, 8]} />
                <meshStandardMaterial color="#6b4f2a" roughness={0.9} />
              </mesh>
              <mesh position={[0, h * 0.65, 0]} castShadow>
                <sphereGeometry args={[r, 12, 10]} />
                <meshStandardMaterial color="#2f8f4e" roughness={0.85} />
              </mesh>
            </group>
          );
        }

        // ── Scene-detail props (from the details agent) ───────────────────
        const col = d.color || PROP_COLOR[d.type] || "#9aa3b2";
        const span = Math.max(w, depth);

        if (d.type === "slide") {
          // Platform + ladder on one side, inclined chute down the other.
          const ph = Math.max(0.8, h);
          const plat = Math.max(0.6, span * 0.5);
          return (
            <group key={`d${i}`} position={[cx, 0, cz]}>
              {/* top platform */}
              <mesh position={[-plat * 0.5, ph, 0]} castShadow>
                <boxGeometry args={[plat, 0.16, plat]} />
                <meshStandardMaterial color={col} roughness={0.5} />
              </mesh>
              {/* ladder posts */}
              {[-1, 1].map((s) => (
                <mesh key={s} position={[-plat, ph / 2, s * plat * 0.4]} castShadow>
                  <cylinderGeometry args={[0.07, 0.07, ph, 6]} />
                  <meshStandardMaterial color="#cfd6e2" roughness={0.6} />
                </mesh>
              ))}
              {/* inclined chute */}
              <mesh
                position={[plat * 0.45, ph * 0.5, 0]}
                rotation={[0, 0, Math.atan2(ph, plat * 1.6)]}
                castShadow
              >
                <boxGeometry args={[Math.hypot(plat * 1.6, ph), 0.12, plat * 0.7]} />
                <meshStandardMaterial color={col} roughness={0.4} metalness={0.1} />
              </mesh>
            </group>
          );
        }

        if (d.type === "swing") {
          const ph = Math.max(0.9, h);
          const halfW = Math.max(0.7, span * 0.6);
          return (
            <group key={`d${i}`} position={[cx, 0, cz]}>
              {/* two A-frames */}
              {[-halfW, halfW].map((sx) =>
                [-1, 1].map((s) => (
                  <mesh
                    key={`${sx}-${s}`}
                    position={[sx, ph / 2, s * 0.3]}
                    rotation={[s * 0.18, 0, 0]}
                    castShadow
                  >
                    <cylinderGeometry args={[0.06, 0.06, ph * 1.05, 6]} />
                    <meshStandardMaterial color={col} roughness={0.5} />
                  </mesh>
                ))
              )}
              {/* top bar */}
              <mesh position={[0, ph, 0]} rotation={[0, 0, Math.PI / 2]} castShadow>
                <cylinderGeometry args={[0.06, 0.06, halfW * 2, 6]} />
                <meshStandardMaterial color={col} roughness={0.5} />
              </mesh>
              {/* hanging seats */}
              {[-halfW * 0.45, halfW * 0.45].map((sx) => (
                <group key={sx}>
                  <mesh position={[sx, ph * 0.45, 0]}>
                    <boxGeometry args={[0.02, ph * 0.5, 0.02]} />
                    <meshStandardMaterial color="#cfd6e2" />
                  </mesh>
                  <mesh position={[sx, ph * 0.22, 0]} castShadow>
                    <boxGeometry args={[0.28, 0.05, 0.18]} />
                    <meshStandardMaterial color="#1f2630" />
                  </mesh>
                </group>
              ))}
            </group>
          );
        }

        if (d.type === "playset") {
          const ph = Math.max(0.9, h);
          const base = Math.max(0.7, span * 0.55);
          const legs: [number, number][] = [
            [-base / 2, -base / 2], [base / 2, -base / 2],
            [-base / 2, base / 2], [base / 2, base / 2],
          ];
          return (
            <group key={`d${i}`} position={[cx, 0, cz]}>
              {legs.map(([lx, lz], k) => (
                <mesh key={k} position={[lx, ph * 0.4, lz]} castShadow>
                  <cylinderGeometry args={[0.07, 0.07, ph * 0.8, 6]} />
                  <meshStandardMaterial color="#cfd6e2" roughness={0.6} />
                </mesh>
              ))}
              {/* deck */}
              <mesh position={[0, ph * 0.8, 0]} castShadow>
                <boxGeometry args={[base, 0.16, base]} />
                <meshStandardMaterial color={col} roughness={0.5} />
              </mesh>
              {/* pitched roof */}
              <mesh position={[0, ph * 1.05, 0]} castShadow>
                <coneGeometry args={[base * 0.8, ph * 0.4, 4]} />
                <meshStandardMaterial color="#E5484D" roughness={0.6} />
              </mesh>
              {/* little slide off the deck */}
              <mesh
                position={[base * 0.6, ph * 0.42, 0]}
                rotation={[0, 0, Math.atan2(ph * 0.8, base)]}
                castShadow
              >
                <boxGeometry args={[Math.hypot(base, ph * 0.8), 0.1, base * 0.5]} />
                <meshStandardMaterial color="#F2A93B" roughness={0.45} />
              </mesh>
            </group>
          );
        }

        if (d.type === "fountain") {
          const r = Math.max(0.6, span * 0.55);
          const ph = Math.max(0.35, h * 0.5);
          return (
            <group key={`d${i}`} position={[cx, 0, cz]}>
              {/* basin rim */}
              <mesh position={[0, ph * 0.5, 0]} castShadow receiveShadow>
                <cylinderGeometry args={[r, r, ph, 24]} />
                <meshStandardMaterial color={col} roughness={0.6} />
              </mesh>
              {/* water surface */}
              <mesh position={[0, ph * 0.92, 0]}>
                <cylinderGeometry args={[r * 0.86, r * 0.86, 0.06, 24]} />
                <meshStandardMaterial
                  color="#7fd8ec"
                  emissive="#2DD4BF"
                  emissiveIntensity={0.35}
                  roughness={0.2}
                />
              </mesh>
              {/* central jet */}
              <mesh position={[0, ph * 1.4, 0]}>
                <cylinderGeometry args={[0.06, 0.1, ph * 1.0, 10]} />
                <meshBasicMaterial color="#bdeefb" toneMapped={false} />
              </mesh>
            </group>
          );
        }

        if (d.type === "statue") {
          const ph = Math.max(0.7, h);
          const r = Math.max(0.3, span * 0.3);
          return (
            <group key={`d${i}`} position={[cx, 0, cz]}>
              {/* pedestal */}
              <mesh position={[0, ph * 0.2, 0]} castShadow>
                <boxGeometry args={[r * 2.2, ph * 0.4, r * 2.2]} />
                <meshStandardMaterial color="#6b7280" roughness={0.7} />
              </mesh>
              {/* figure (body + head) */}
              <mesh position={[0, ph * 0.7, 0]} castShadow>
                <capsuleGeometry args={[r * 0.7, ph * 0.5, 6, 12]} />
                <meshStandardMaterial color={col} roughness={0.5} metalness={0.3} />
              </mesh>
              <mesh position={[0, ph * 1.05, 0]} castShadow>
                <sphereGeometry args={[r * 0.55, 14, 12]} />
                <meshStandardMaterial color={col} roughness={0.5} metalness={0.3} />
              </mesh>
            </group>
          );
        }

        if (d.type === "bench") {
          const ph = Math.max(0.3, h);
          const len = Math.max(0.8, span);
          return (
            <group key={`d${i}`} position={[cx, 0, cz]}>
              {/* seat */}
              <mesh position={[0, ph * 0.5, 0]} castShadow>
                <boxGeometry args={[len, 0.1, len * 0.32]} />
                <meshStandardMaterial color={col} roughness={0.7} />
              </mesh>
              {/* backrest */}
              <mesh position={[0, ph * 0.85, -len * 0.14]} castShadow>
                <boxGeometry args={[len, ph * 0.6, 0.08]} />
                <meshStandardMaterial color={col} roughness={0.7} />
              </mesh>
            </group>
          );
        }

        if (d.type === "booth") {
          const ph = Math.max(0.7, h);
          return (
            <group key={`d${i}`} position={[cx, 0, cz]}>
              <mesh position={[0, ph * 0.45, 0]} castShadow receiveShadow>
                <boxGeometry args={[w, ph * 0.9, depth]} />
                <meshStandardMaterial color="#e7eaf0" roughness={0.7} />
              </mesh>
              {/* striped canopy */}
              <mesh position={[0, ph * 0.96, depth * 0.18]} rotation={[0.18, 0, 0]} castShadow>
                <boxGeometry args={[w * 1.18, 0.08, depth * 0.75]} />
                <meshStandardMaterial color={col} roughness={0.5} />
              </mesh>
            </group>
          );
        }

        if (d.type === "goal") {
          const ph = Math.max(0.7, h);
          const halfW = Math.max(0.6, span * 0.7);
          return (
            <group key={`d${i}`} position={[cx, 0, cz]}>
              {[-halfW, halfW].map((sx) => (
                <mesh key={sx} position={[sx, ph / 2, 0]} castShadow>
                  <cylinderGeometry args={[0.06, 0.06, ph, 8]} />
                  <meshStandardMaterial color={col} roughness={0.4} />
                </mesh>
              ))}
              <mesh position={[0, ph, 0]} rotation={[0, 0, Math.PI / 2]} castShadow>
                <cylinderGeometry args={[0.06, 0.06, halfW * 2, 8]} />
                <meshStandardMaterial color={col} roughness={0.4} />
              </mesh>
            </group>
          );
        }

        if (d.type === "pole") {
          const ph = Math.max(1.0, h);
          return (
            <group key={`d${i}`} position={[cx, 0, cz]}>
              <mesh position={[0, ph / 2, 0]} castShadow>
                <cylinderGeometry args={[0.05, 0.07, ph, 8]} />
                <meshStandardMaterial color={col} roughness={0.5} metalness={0.3} />
              </mesh>
              {/* flag */}
              <mesh position={[0.28, ph * 0.85, 0]}>
                <boxGeometry args={[0.5, ph * 0.22, 0.02]} />
                <meshStandardMaterial color="#E5484D" roughness={0.6} side={THREE.DoubleSide} />
              </mesh>
            </group>
          );
        }

        if (d.type === "planter") {
          const ph = Math.max(0.25, h * 0.5);
          return (
            <group key={`d${i}`} position={[cx, 0, cz]}>
              <mesh position={[0, ph * 0.5, 0]} castShadow receiveShadow>
                <boxGeometry args={[w, ph, depth]} />
                <meshStandardMaterial color="#7a5a3a" roughness={0.85} />
              </mesh>
              {/* bushes */}
              {[-0.25, 0.05, 0.3].map((ox, k) => (
                <mesh key={k} position={[ox * w, ph + 0.18, 0]} castShadow>
                  <sphereGeometry args={[Math.max(0.18, w * 0.22), 10, 8]} />
                  <meshStandardMaterial color={col} roughness={0.85} />
                </mesh>
              ))}
            </group>
          );
        }

        if (d.type === "court") {
          // Flat colored ground patch (sport court, sandbox, splash pad).
          return (
            <mesh
              key={`d${i}`}
              position={[cx, 0.03, cz]}
              rotation={[-Math.PI / 2, 0, 0]}
              receiveShadow
            >
              <planeGeometry args={[w, depth]} />
              <meshStandardMaterial color={col} roughness={0.9} transparent opacity={0.92} />
            </mesh>
          );
        }

        // roof — large translucent overhead canopy.
        return (
          <mesh key={`d${i}`} position={[cx, h, cz]} castShadow>
            <boxGeometry args={[w, 0.3, depth]} />
            <meshStandardMaterial
              color="#2b3340"
              transparent
              opacity={0.5}
              roughness={0.6}
            />
          </mesh>
        );
      })}
    </group>
  );
}

function Markers({ layout }: { layout: VenueLayout }) {
  return (
    <group>
      {layout.elements.map((e, i) => {
        if (e.type !== "entry" && e.type !== "gate") return null;
        const color = e.type === "entry" ? "#3FB950" : "#4493F8";
        const w = Math.max(0.3, e.w * SIZE);
        const d = Math.max(0.3, e.h * SIZE);
        return (
          <group key={`m${i}`} position={[wx(e.x + e.w / 2), 0, wz(e.y + e.h / 2)]}>
            <mesh position={[0, 0.04, 0]} rotation={[-Math.PI / 2, 0, 0]}>
              <planeGeometry args={[w, d]} />
              <meshBasicMaterial color={color} transparent opacity={0.85} toneMapped={false} />
            </mesh>
            {/* soft beam so entries/exits read at a glance */}
            <mesh position={[0, 1.6, 0]}>
              <boxGeometry args={[Math.min(w, d) * 0.5, 3.2, Math.min(w, d) * 0.5]} />
              <meshBasicMaterial color={color} transparent opacity={0.12} toneMapped={false} />
            </mesh>
          </group>
        );
      })}
    </group>
  );
}

function DangerMarkers({ zones }: { zones: DangerZone[] }) {
  return (
    <group>
      {zones.map((z, i) => {
        const r = (z.risk === "CRITICAL" ? 1.7 : 1.1);
        return (
          <mesh key={`d${i}`} position={[wx(z.x), 0.06, wz(z.y)]} rotation={[-Math.PI / 2, 0, 0]}>
            <circleGeometry args={[r, 24]} />
            <meshBasicMaterial
              color={z.risk === "CRITICAL" ? "#F85149" : "#D29922"}
              transparent
              opacity={0.32}
              toneMapped={false}
            />
          </mesh>
        );
      })}
    </group>
  );
}

// ── Crowd agents — hybrid: physics world-model + agent-LLM goal-seeking ───────
//
// `world-model` agents advect through the simulated velocity/pressure field.
// `llm` agents (a fraction set by the agent-LLM behavior plan) steer toward a
// reasoned goal point for their group, lightly nudged by the same field so they
// still jam up in crushes. Two InstancedMeshes so the two kinds read distinctly.

const C_LLM = new THREE.Color("#A371F7"); // agent-LLM piloted (violet)

// Phase choreography (seconds): the crowd ENTERS, STAYS, then EXITS, looping.
const PH_ENTER = 7;
const PH_STAY = 12;
const PH_EXIT = 7;
const PH_TOTAL = PH_ENTER + PH_STAY + PH_EXIT;
export type CrowdPhase = 0 | 1 | 2; // 0 entry · 1 staying · 2 exit

function elCenter(e: VenueLayoutElement): [number, number] {
  return [e.x + e.w / 2, e.y + e.h / 2];
}

function Agents({
  scenario,
  count,
  frameRef,
  playingRef,
  agentPlan,
  onPhase,
}: {
  scenario: Scenario;
  count: number;
  frameRef: MutableRefObject<number>;
  playingRef: MutableRefObject<boolean>;
  agentPlan?: AgentPlan | null;
  onPhase?: (phase: CrowdPhase) => void;
}) {
  const wmRef = useRef<THREE.InstancedMesh>(null);
  const llmRef = useRef<THREE.InstancedMesh>(null);
  const G = scenario.field.grid;
  const walls = scenario.field.walls;
  const pMax = scenario.field.p_max || 1;

  // Phase clock + bookkeeping (advances only while playing).
  const phaseClock = useRef(0);
  const cycleRef = useRef(-1);
  const reportedPhase = useRef<CrowdPhase | -1>(-1);

  // Build per-agent state: stay position (home), entry point, nearest exit,
  // kind (llm/world), goal + speed, and the index within its own mesh.
  const sim = useMemo(() => {
    const behaviors = agentPlan?.behaviors ?? [];
    const llmFrac = behaviors.length
      ? Math.max(0, Math.min(0.8, agentPlan?.llm_fraction ?? 0))
      : 0;
    const nLLM = Math.round(count * llmFrac);

    // Entry / exit points from the layout (fall back to the perimeter).
    const els = scenario.layout.elements ?? [];
    const entryEls = els.filter((e) => e.type === "entry");
    const gateEls = els.filter((e) => e.type === "gate");
    const enterPts = (entryEls.length ? entryEls : gateEls).map(elCenter);
    const exitPts = (gateEls.length ? gateEls : entryEls).map(elCenter);
    const ENTER: [number, number][] = enterPts.length
      ? enterPts
      : [[0.5, 0.03], [0.03, 0.5], [0.97, 0.5], [0.5, 0.97]];
    const EXIT: [number, number][] = exitPts.length ? exitPts : ENTER;

    const pos = new Float32Array(count * 2);
    const home = new Float32Array(count * 2);
    const entryFrom = new Float32Array(count * 2);
    const exitTo = new Float32Array(count * 2);
    const goal = new Float32Array(count * 2);
    const speed = new Float32Array(count);
    const isLLM = new Uint8Array(count);
    const local = new Int32Array(count);

    const cum: number[] = [];
    let acc = 0;
    for (const b of behaviors) { acc += Math.max(0, b.fraction || 0); cum.push(acc); }
    const totalFrac = acc || 1;
    const jit = () => (Math.random() - 0.5) * 0.06;
    const clamp01 = (v: number) => Math.min(0.97, Math.max(0.03, v));

    let wmLocal = 0, llmLocal = 0;
    for (let i = 0; i < count; i++) {
      let nx = 0.5, ny = 0.5;
      for (let tries = 0; tries < 12; tries++) {
        nx = 0.04 + Math.random() * 0.92;
        ny = 0.04 + Math.random() * 0.92;
        if (!isWall(walls, G, nx, ny)) break;
      }
      home[i * 2] = nx;
      home[i * 2 + 1] = ny;
      pos[i * 2] = nx;
      pos[i * 2 + 1] = ny;

      // Enter from one of the entry points (round-robin + jitter).
      const ep = ENTER[i % ENTER.length];
      entryFrom[i * 2] = clamp01(ep[0] + jit());
      entryFrom[i * 2 + 1] = clamp01(ep[1] + jit());

      // Exit via the nearest gate to this agent's home spot.
      let best = 0, bd = Infinity;
      for (let k = 0; k < EXIT.length; k++) {
        const d = (EXIT[k][0] - nx) ** 2 + (EXIT[k][1] - ny) ** 2;
        if (d < bd) { bd = d; best = k; }
      }
      exitTo[i * 2] = clamp01(EXIT[best][0] + jit());
      exitTo[i * 2 + 1] = clamp01(EXIT[best][1] + jit());

      if (i < nLLM && behaviors.length) {
        isLLM[i] = 1;
        const t = ((i + 0.5) / Math.max(1, nLLM)) * totalFrac;
        let bi = cum.findIndex((c) => t <= c);
        if (bi < 0) bi = behaviors.length - 1;
        const b = behaviors[bi];
        goal[i * 2] = b.goal?.[0] ?? 0.5;
        goal[i * 2 + 1] = b.goal?.[1] ?? 0.5;
        speed[i] = b.speed || 1.0;
        local[i] = llmLocal++;
      } else {
        isLLM[i] = 0;
        speed[i] = 1.0;
        local[i] = wmLocal++;
      }
    }
    return { pos, home, entryFrom, exitTo, goal, speed, isLLM, local,
             nLLM, wmCount: count - nLLM };
  }, [count, G, walls, agentPlan, scenario.layout.elements]);

  const dummy = useMemo(() => new THREE.Object3D(), []);
  const color = useMemo(() => new THREE.Color(), []);

  useFrame((_, delta) => {
    const wm = wmRef.current;
    const llm = llmRef.current;
    const frames = scenario.field.frames;
    const f = Math.min(frames - 1, Math.max(0, Math.floor(frameRef.current)));
    const vx = scenario.field.vx[f];
    const vy = scenario.field.vy[f];
    const pr = scenario.field.pressure[f];
    if (!vx || !vy || !pr) return;

    const { pos, home, entryFrom, exitTo, goal, speed, isLLM, local } = sim;
    const moving = playingRef.current;
    const dt = Math.min(delta, 0.05);

    // Advance the phase clock and resolve the current phase.
    if (moving) phaseClock.current += dt;
    const tcyc = phaseClock.current % PH_TOTAL;
    const cycle = Math.floor(phaseClock.current / PH_TOTAL);
    const phase: CrowdPhase = tcyc < PH_ENTER ? 0 : tcyc < PH_ENTER + PH_STAY ? 1 : 2;

    // New cycle → everyone starts back at the entry points (loop restart).
    if (cycle !== cycleRef.current) {
      cycleRef.current = cycle;
      for (let i = 0; i < count; i++) {
        pos[i * 2] = entryFrom[i * 2];
        pos[i * 2 + 1] = entryFrom[i * 2 + 1];
      }
    }
    if (phase !== reportedPhase.current) {
      reportedPhase.current = phase;
      onPhase?.(phase);
    }

    const KV = 0.9;        // follow the flow field (world-model agents, stay)
    const KP = 0.05;       // gather toward higher pressure (crowd build-up)
    const NOISE = 0.18;
    const SPEED = 1.6;
    const GOAL_K = 1.15;   // goal pull (llm agents, stay)
    const LLM_FIELD = 0.35;
    const MOVE_K = 1.4;    // entry / exit steering strength

    const cl = (v: number) => Math.min(G - 1, Math.max(0, v));

    for (let i = 0; i < count; i++) {
      let nx = pos[i * 2];
      let ny = pos[i * 2 + 1];
      const gx = cl(Math.floor(nx * G));
      const gy = cl(Math.floor(ny * G));
      const llmAgent = isLLM[i] === 1;

      if (moving) {
        const ux = vx[gy][gx];
        const uy = vy[gy][gx];
        let dx: number, dy: number;

        if (phase === 0 || phase === 2) {
          // ENTRY: stream from the entrance to the stay spot.
          // EXIT:  stream from wherever they are to the nearest gate.
          const tgtX = phase === 0 ? home[i * 2] : exitTo[i * 2];
          const tgtY = phase === 0 ? home[i * 2 + 1] : exitTo[i * 2 + 1];
          let tx = tgtX - nx;
          let ty = tgtY - ny;
          const dist = Math.hypot(tx, ty) || 1e-3;
          if (dist < 0.02) { tx = Math.random() - 0.5; ty = Math.random() - 0.5; }
          else { tx /= dist; ty /= dist; }
          dx = tx * MOVE_K * speed[i] + (Math.random() - 0.5) * NOISE;
          dy = ty * MOVE_K * speed[i] + (Math.random() - 0.5) * NOISE;
        } else if (llmAgent) {
          // STAY (LLM): goal-seeking intent + a little field so crushes impede.
          let tx = goal[i * 2] - nx;
          let ty = goal[i * 2 + 1] - ny;
          const dist = Math.hypot(tx, ty) || 1e-3;
          if (dist < 0.06) { tx = Math.random() - 0.5; ty = Math.random() - 0.5; }
          else { tx /= dist; ty /= dist; }
          dx = tx * GOAL_K * speed[i] + ux * LLM_FIELD + (Math.random() - 0.5) * NOISE;
          dy = ty * GOAL_K * speed[i] + uy * LLM_FIELD + (Math.random() - 0.5) * NOISE;
        } else {
          // STAY (world model): advect through the simulated flow field.
          const gpx = pr[gy][cl(gx + 1)] - pr[gy][cl(gx - 1)];
          const gpy = pr[cl(gy + 1)][gx] - pr[cl(gy - 1)][gx];
          dx = ux * KV + gpx * KP + (Math.random() - 0.5) * NOISE;
          dy = uy * KV + gpy * KP + (Math.random() - 0.5) * NOISE;
        }

        let cand_x = nx + dx * dt * SPEED;
        let cand_y = ny + dy * dt * SPEED;
        if (isWall(walls, G, cand_x, ny)) cand_x = nx;
        if (isWall(walls, G, nx, cand_y)) cand_y = ny;

        nx = Math.min(0.985, Math.max(0.015, cand_x));
        ny = Math.min(0.985, Math.max(0.015, cand_y));
        pos[i * 2] = nx;
        pos[i * 2 + 1] = ny;
      }

      const p = pr[gy][gx];
      dummy.position.set(wx(nx), llmAgent ? 0.42 : 0.35, wz(ny));
      dummy.updateMatrix();

      if (llmAgent) {
        if (!llm) continue;
        llm.setMatrixAt(local[i], dummy.matrix);
        const t = Math.max(0, Math.min(1, p / (pMax || 1)));
        color.copy(C_LLM).lerp(C_HIGH, t * 0.8);
        llm.setColorAt(local[i], color);
      } else {
        if (!wm) continue;
        wm.setMatrixAt(local[i], dummy.matrix);
        pressureColor(color, p, pMax);
        wm.setColorAt(local[i], color);
      }
    }
    if (wm) {
      wm.instanceMatrix.needsUpdate = true;
      if (wm.instanceColor) wm.instanceColor.needsUpdate = true;
    }
    if (llm) {
      llm.instanceMatrix.needsUpdate = true;
      if (llm.instanceColor) llm.instanceColor.needsUpdate = true;
    }
  });

  return (
    <group>
      {sim.wmCount > 0 && (
        <instancedMesh
          key={`wm-${scenario.id}-${sim.wmCount}`}
          ref={wmRef}
          args={[undefined, undefined, sim.wmCount]}
          frustumCulled={false}
        >
          <cylinderGeometry args={[0.12, 0.12, 0.65, 6]} />
          <meshBasicMaterial toneMapped={false} />
        </instancedMesh>
      )}
      {sim.nLLM > 0 && (
        <instancedMesh
          key={`llm-${scenario.id}-${sim.nLLM}`}
          ref={llmRef}
          args={[undefined, undefined, sim.nLLM]}
          frustumCulled={false}
        >
          <coneGeometry args={[0.17, 0.85, 6]} />
          <meshBasicMaterial toneMapped={false} />
        </instancedMesh>
      )}
    </group>
  );
}

const FLOOR_COLOR: Record<string, string> = {
  stadium: "#13201a",
  arena: "#141a24",
  theater: "#16121c",
  hall: "#10151D",
  plaza: "#1b1f26",
  street: "#181b20",
  field: "#13201a",
  festival: "#15201a",
};

function Floor({ archetype }: { archetype?: VenueArchetype }) {
  const base = (archetype && FLOOR_COLOR[archetype]) || "#10151D";
  // Stadium/arena/field get a central green pitch so the bowl reads instantly.
  const pitch =
    archetype === "stadium" ||
    archetype === "arena" ||
    archetype === "field";
  return (
    <group>
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0, 0]} receiveShadow>
        <planeGeometry args={[SIZE, SIZE]} />
        <meshStandardMaterial color={base} roughness={0.92} metalness={0.04} />
      </mesh>
      {pitch && (
        <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.015, 0]} receiveShadow>
          <planeGeometry args={[SIZE * 0.62, SIZE * 0.46]} />
          <meshStandardMaterial color="#1f6b3a" roughness={0.95} metalness={0.02} />
        </mesh>
      )}
    </group>
  );
}

// ── Public component ──────────────────────────────────────────────────────────

export default function Venue3D({
  scenario,
  nPeople,
  frameRef,
  playingRef,
  agentPlan,
  onPhase,
  maxAgents = 1400,
}: {
  scenario: Scenario;
  nPeople: number;
  frameRef: MutableRefObject<number>;
  playingRef: MutableRefObject<boolean>;
  agentPlan?: AgentPlan | null;
  onPhase?: (phase: CrowdPhase) => void;
  maxAgents?: number;
}) {
  const count = Math.max(40, Math.min(maxAgents, nPeople || 600));
  const controlsRef = useRef<any>(null);

  // Stop the cinematic auto-rotate as soon as the user grabs the scene.
  useEffect(() => {
    const c = controlsRef.current;
    if (!c) return;
    const stop = () => {
      c.autoRotate = false;
    };
    c.addEventListener("start", stop);
    return () => c.removeEventListener("start", stop);
  }, []);

  return (
    <Canvas
      shadows
      dpr={[1, 2]}
      camera={{ position: [SIZE * 0.62, SIZE * 0.72, SIZE * 0.92], fov: 45 }}
      gl={{ antialias: true }}
    >
      <color attach="background" args={["#0D1117"]} />
      <fog attach="fog" args={["#0D1117", SIZE * 1.1, SIZE * 2.8]} />

      <ambientLight intensity={0.75} />
      <directionalLight
        position={[14, 24, 12]}
        intensity={0.9}
        castShadow
        shadow-mapSize-width={1024}
        shadow-mapSize-height={1024}
      />
      <pointLight position={[-12, 9, -8]} intensity={0.45} color="#e2a9f1" />
      <pointLight position={[12, 6, 10]} intensity={0.3} color="#2DD4BF" />

      <Floor archetype={scenario.layout.archetype} />
      <gridHelper args={[SIZE, 24, "#21262D", "#161B22"]} position={[0, 0.02, 0]} />

      <Structures layout={scenario.layout} />
      <Decor layout={scenario.layout} />
      <Markers layout={scenario.layout} />
      <DangerMarkers zones={scenario.danger_zones} />
      <Agents
        scenario={scenario}
        count={count}
        frameRef={frameRef}
        playingRef={playingRef}
        agentPlan={agentPlan}
        onPhase={onPhase}
      />

      <OrbitControls
        ref={controlsRef}
        makeDefault
        autoRotate
        autoRotateSpeed={0.45}
        enablePan
        minDistance={SIZE * 0.4}
        maxDistance={SIZE * 2}
        maxPolarAngle={Math.PI * 0.49}
        target={[0, 0.5, 0]}
      />
    </Canvas>
  );
}
