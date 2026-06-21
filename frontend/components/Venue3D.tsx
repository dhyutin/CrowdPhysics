"use client";

import { useMemo, useRef, useEffect, type MutableRefObject } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import type { Scenario, VenueLayout, DangerZone } from "@/lib/api";

// World is a SIZE x SIZE square centered at the origin, floor on the XZ plane,
// Y is up. Normalized layout coords (0-1, top-left origin) map onto it.
const SIZE = 24;
const wx = (nx: number) => (nx - 0.5) * SIZE;
const wz = (ny: number) => (ny - 0.5) * SIZE;

const EL_HEIGHT: Record<string, number> = {
  wall: 2.4,
  stage: 1.3,
  barrier: 0.9,
};
const EL_COLOR: Record<string, string> = {
  wall: "#2b3340",
  stage: "#2DD4BF",
  barrier: "#A371F7",
};

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

function Structures({ layout }: { layout: VenueLayout }) {
  return (
    <group>
      {layout.elements.map((e, i) => {
        if (e.type !== "wall" && e.type !== "stage" && e.type !== "barrier") return null;
        const h = EL_HEIGHT[e.type];
        const w = Math.max(0.2, e.w * SIZE);
        const d = Math.max(0.2, e.h * SIZE);
        return (
          <mesh
            key={`s${i}`}
            position={[wx(e.x + e.w / 2), h / 2, wz(e.y + e.h / 2)]}
            castShadow
            receiveShadow
          >
            <boxGeometry args={[w, h, d]} />
            <meshStandardMaterial
              color={EL_COLOR[e.type]}
              roughness={0.55}
              metalness={0.15}
              emissive={e.type === "stage" ? "#0d6b60" : "#000000"}
              emissiveIntensity={e.type === "stage" ? 0.5 : 0}
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

function Floor() {
  return (
    <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0, 0]} receiveShadow>
      <planeGeometry args={[SIZE, SIZE]} />
      <meshStandardMaterial color="#10151D" roughness={0.9} metalness={0.05} />
    </mesh>
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

      <Floor />
      <gridHelper args={[SIZE, 24, "#21262D", "#161B22"]} position={[0, 0.02, 0]} />

      <Structures layout={scenario.layout} />
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
