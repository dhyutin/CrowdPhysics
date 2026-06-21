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

// ── Crowd agents (one InstancedMesh, advected through the field) ──────────────

function Agents({
  scenario,
  count,
  frameRef,
  playingRef,
}: {
  scenario: Scenario;
  count: number;
  frameRef: MutableRefObject<number>;
  playingRef: MutableRefObject<boolean>;
}) {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const G = scenario.field.grid;
  const walls = scenario.field.walls;
  const pMax = scenario.field.p_max || 1;

  // Per-agent normalized positions, seeded in open (non-wall) space.
  const pos = useMemo(() => {
    const arr = new Float32Array(count * 2);
    for (let i = 0; i < count; i++) {
      let nx = 0.5,
        ny = 0.5;
      for (let tries = 0; tries < 12; tries++) {
        nx = 0.04 + Math.random() * 0.92;
        ny = 0.04 + Math.random() * 0.92;
        if (!isWall(walls, G, nx, ny)) break;
      }
      arr[i * 2] = nx;
      arr[i * 2 + 1] = ny;
    }
    return arr;
    // reseed when the scenario (its walls) or agent count changes
  }, [count, G, walls]);

  const dummy = useMemo(() => new THREE.Object3D(), []);
  const color = useMemo(() => new THREE.Color(), []);

  useFrame((_, delta) => {
    const mesh = meshRef.current;
    if (!mesh) return;
    const frames = scenario.field.frames;
    const f = Math.min(frames - 1, Math.max(0, Math.floor(frameRef.current)));
    const vx = scenario.field.vx[f];
    const vy = scenario.field.vy[f];
    const pr = scenario.field.pressure[f];
    if (!vx || !vy || !pr) return;

    const moving = playingRef.current;
    const dt = Math.min(delta, 0.05);
    const KV = 0.9; // follow the flow field
    const KP = 0.05; // gather toward higher pressure (crowd build-up)
    const NOISE = 0.18;
    const SPEED = 1.6;

    const cl = (v: number) => Math.min(G - 1, Math.max(0, v));

    for (let i = 0; i < count; i++) {
      let nx = pos[i * 2];
      let ny = pos[i * 2 + 1];
      const gx = cl(Math.floor(nx * G));
      const gy = cl(Math.floor(ny * G));

      if (moving) {
        const ux = vx[gy][gx];
        const uy = vy[gy][gx];
        const gpx = pr[gy][cl(gx + 1)] - pr[gy][cl(gx - 1)];
        const gpy = pr[cl(gy + 1)][gx] - pr[cl(gy - 1)][gx];

        const dx = ux * KV + gpx * KP + (Math.random() - 0.5) * NOISE;
        const dy = uy * KV + gpy * KP + (Math.random() - 0.5) * NOISE;

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
      dummy.position.set(wx(nx), 0.35, wz(ny));
      dummy.updateMatrix();
      mesh.setMatrixAt(i, dummy.matrix);
      pressureColor(color, p, pMax);
      mesh.setColorAt(i, color);
    }
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  });

  return (
    <instancedMesh
      key={`${scenario.id}-${count}`}
      ref={meshRef}
      args={[undefined, undefined, count]}
      frustumCulled={false}
    >
      <cylinderGeometry args={[0.12, 0.12, 0.65, 6]} />
      <meshBasicMaterial toneMapped={false} />
    </instancedMesh>
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
  maxAgents = 1400,
}: {
  scenario: Scenario;
  nPeople: number;
  frameRef: MutableRefObject<number>;
  playingRef: MutableRefObject<boolean>;
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
      <Agents scenario={scenario} count={count} frameRef={frameRef} playingRef={playingRef} />

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
