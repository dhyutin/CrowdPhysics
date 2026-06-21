const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export async function analyzeVideo(
  file: File,
  venue: string
): Promise<AnalyzeResult> {
  const form = new FormData();
  form.append("video", file);
  form.append("venue", venue);
  const res = await fetch(`${BASE}/api/analyze`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export interface LiveSession {
  session_id: string;
  live_view_url: string;
}

export async function startLiveSession(url: string): Promise<LiveSession> {
  const res = await fetch(`${BASE}/api/live_session`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function endLiveSession(sessionId: string): Promise<void> {
  try {
    await fetch(`${BASE}/api/end_live_session`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId }),
      keepalive: true,
    });
  } catch {
    /* best-effort cleanup */
  }
}

export async function monitorUrl(
  url: string,
  venue: string,
  nFrames = 35,
  sessionId?: string
): Promise<MonitorResult> {
  const res = await fetch(`${BASE}/api/monitor_url`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url, venue, n_frames: nFrames, session_id: sessionId ?? null }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

// ── Live streaming (newline-delimited JSON over a fetch ReadableStream) ──────
//
// EventSource can't POST a file or JSON body, so we stream the response body
// of a normal POST and parse one JSON event per line. Each event is a LiveTick.

export type LiveEventType =
  | "source" | "calibrating" | "tick" | "done" | "alert";

export interface LiveTick {
  type: LiveEventType;
  // tick
  step?: number;
  time?: number;
  status?: TimelinePoint["status"];
  score?: number;
  probability?: number;
  forecast?: Forecast;
  trend?: Trend;
  hotspot?: Hotspot;
  frame_b64?: string;
  field_b64?: string;
  // calibrating
  venue?: string;
  fps?: number;
  calibration_frames?: number;
  total_frames?: number;
  // source (live capture)
  url?: string;
  frames_captured?: number;
  capture_fps?: number;
  // done
  summary?: string;
  claude_briefing?: string;
  rl_explanation?: string;
  timeline?: TimelinePoint[];
  peak_physics?: Record<string, unknown> | null;
  peak_frame_b64?: string | null;
  flow_gif_b64?: string | null;
  agent_trace?: AgentTraceStep[];
  counterfactual?: Counterfactual | null;
  // alert event (type === "alert")
  sent?: boolean;
  channels?: string[];
  message?: string;
  sent_at?: string;
  reason?: string;
}

async function consumeStream(
  res: Response,
  onEvent: (ev: LiveTick) => void
): Promise<void> {
  if (!res.ok || !res.body) {
    const msg = await res.text().catch(() => res.statusText);
    throw new Error(msg || `stream failed (${res.status})`);
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  const flushLines = (final = false) => {
    let nl: number;
    while ((nl = buf.indexOf("\n")) >= 0) {
      const line = buf.slice(0, nl).trim();
      buf = buf.slice(nl + 1);
      if (line) {
        try { onEvent(JSON.parse(line) as LiveTick); } catch { /* skip */ }
      }
    }
    if (final) {
      const tail = buf.trim();
      if (tail) {
        try { onEvent(JSON.parse(tail) as LiveTick); } catch { /* skip */ }
      }
    }
  };
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    flushLines();
  }
  buf += decoder.decode();
  flushLines(true);
}

export async function streamAnalyze(
  file: File,
  venue: string,
  onEvent: (ev: LiveTick) => void,
  signal?: AbortSignal
): Promise<void> {
  const form = new FormData();
  form.append("video", file);
  form.append("venue", venue);
  const res = await fetch(`${BASE}/api/analyze_stream`, {
    method: "POST",
    body: form,
    signal,
  });
  await consumeStream(res, onEvent);
}

export async function streamMonitorUrl(
  url: string,
  venue: string,
  onEvent: (ev: LiveTick) => void,
  nFrames = 35,
  sessionId?: string,
  signal?: AbortSignal,
  keepSession = false
): Promise<void> {
  const res = await fetch(`${BASE}/api/monitor_url_stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      url,
      venue,
      n_frames: nFrames,
      session_id: sessionId ?? null,
      keep_session: keepSession,
    }),
    signal,
  });
  await consumeStream(res, onEvent);
}

// Direct YouTube ingest (yt-dlp + ffmpeg on the backend) — no Browserbase,
// much lower latency. Same NDJSON event stream as the other live monitors.
export async function streamMonitorYouTube(
  url: string,
  venue: string,
  onEvent: (ev: LiveTick) => void,
  nFrames = 40,
  signal?: AbortSignal
): Promise<void> {
  const res = await fetch(`${BASE}/api/monitor_youtube_stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url, venue, n_frames: nFrames }),
    signal,
  });
  await consumeStream(res, onEvent);
}

export async function runSimulation(
  payload: SimulatePayload
): Promise<SimulateResult> {
  const res = await fetch(`${BASE}/api/simulate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function simulateFromImage(
  file: File,
  capacity: number,
  density: number
): Promise<SimulateResult> {
  const form = new FormData();
  form.append("image", file);
  form.append("capacity", String(capacity));
  form.append("density", String(density));
  const res = await fetch(`${BASE}/api/simulate_from_image`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function planEvent(
  file: File,
  purpose: string,
  capacity: number,
  density: number
): Promise<PlanResult> {
  const form = new FormData();
  form.append("image", file);
  form.append("purpose", purpose);
  form.append("capacity", String(capacity));
  form.append("density", String(density));
  const res = await fetch(`${BASE}/api/plan`, { method: "POST", body: form });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export interface EventIntake {
  purpose: string;
  nPeople: number;
  density: number;
  durationMin: number;
  seating: string;
  ingress: string;
  notes: string;
  areaM2?: number;
}

export async function plan3d(
  file: File,
  intake: EventIntake
): Promise<Plan3DResult> {
  const form = new FormData();
  form.append("image", file);
  form.append("purpose", intake.purpose);
  form.append("n_people", String(intake.nPeople));
  form.append("density", String(intake.density));
  form.append("duration_min", String(intake.durationMin));
  form.append("seating", intake.seating);
  form.append("ingress", intake.ingress);
  form.append("notes", intake.notes);
  form.append("area_m2", String(intake.areaM2 ?? 0));
  const res = await fetch(`${BASE}/api/plan3d`, { method: "POST", body: form });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

// Conversationally edit the reconstructed scene, then re-simulate.
export async function refinePlan3d(
  layout: VenueLayout,
  instruction: string,
  intake: EventIntake
): Promise<Plan3DResult> {
  const res = await fetch(`${BASE}/api/plan3d/refine`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      layout,
      instruction,
      purpose: intake.purpose,
      n_people: intake.nPeople,
      density: intake.density,
      duration_min: intake.durationMin,
      seating: intake.seating,
      ingress: intake.ingress,
      notes: intake.notes,
      area_m2: intake.areaM2 ?? 0,
    }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function runDiscover(): Promise<DiscoverResult> {
  const res = await fetch(`${BASE}/api/discover`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function runRLMetrics(): Promise<RLMetricsResult> {
  const res = await fetch(`${BASE}/api/rl_metrics`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

// ── Types ──────────────────────────────────────────────────────────────────

export interface TimelinePoint {
  time: number;
  status: "SAFE" | "WARNING" | "DANGER" | "CALIBRATING";
  score: number;
  probability: number;
}

export interface ForecastPoint {
  t: number;
  risk: number;
}

export interface Forecast {
  points?: ForecastPoint[];
  lead_time_s?: number | null;
  horizon_s?: number;
  projected_status?: "SAFE" | "WARNING" | "DANGER";
  projected_risk?: number;
  projected_field_b64?: string;
  error?: string;
  // Agent-decided crush risk (Claude reasons over physics + trend + forecast).
  // When present the UI shows this calibrated number instead of the raw
  // world-model projection.
  agent_risk?: number;
  agent_lead_s?: number | null;
  agent_reason?: string;
  agent_recommendation?: string;
  agent_source?: "claude";
}

// Normalized region of danger on the frame (x,y,r in [0,1]).
// Severity grade for a localized danger region.
export type DangerSeverity = "calm" | "elevated" | "critical";

export interface Hotspot {
  x: number;
  y: number;
  r: number;
  intensity: number;
  severity?: DangerSeverity; // "elevated" = slightly risky, "critical" = very dangerous
}

// Resolve a region's severity, falling back to the frame status for older
// payloads that don't carry an explicit grade.
export function hotspotSeverity(
  h?: Hotspot | null,
  status?: string
): DangerSeverity {
  if (!h) return "calm";
  if (h.severity) return h.severity;
  if (status === "DANGER") return "critical";
  if (status === "WARNING") return "elevated";
  return "calm";
}

// Minutes-ahead statistical trend projection (NOT the world-model rollout).
export interface Trend {
  points?: ForecastPoint[];
  lead_time_s?: number | null;
  horizon_s?: number;
  projected_status?: "SAFE" | "WARNING" | "DANGER";
  projected_risk?: number;
  slope_per_min?: number;
  method?: string;
}

export interface AgentTraceStep {
  agent: string;
  icon: string;
  action: string;
  detail: string;
  status: "ok" | "danger";
}

// World-model counterfactual: projected risk doing nothing vs taking the
// recommended intervention, proving the fix lowers risk.
export interface CounterfactualPoint {
  t: number;
  risk: number;
}

export interface Counterfactual {
  action_idx: number;
  action_name: string;
  action_description: string;
  do_nothing_risk: number;
  action_risk: number;
  reduction_pct: number;
  points_do_nothing: CounterfactualPoint[];
  points_action: CounterfactualPoint[];
  horizon_s: number;
  error?: string;
}

// Status of a dispatched (or skipped) external danger alert.
export interface AlertStatus {
  sent: boolean;
  channels: string[];
  message?: string;
  sent_at?: string;
  reason?: string;
}

export interface AnalyzeResult {
  peak_frame_b64: string | null;
  flow_gif_b64?: string | null;
  summary: string;
  claude_briefing: string;
  rl_explanation: string;
  timeline: TimelinePoint[];
  peak_physics: Record<string, unknown> | null;
  forecast?: Forecast | null;
  trend?: Trend | null;
  hotspot?: Hotspot | null;
  counterfactual?: Counterfactual | null;
  alert?: AlertStatus | null;
  agent_trace?: AgentTraceStep[];
}

export interface CaptureSource {
  url: string;
  frames_captured: number;
  capture_fps: number;
}

export interface MonitorResult extends AnalyzeResult {
  source?: CaptureSource;
}

export interface SimulatePayload {
  venue_name: string;
  capacity: number;
  n_exits: number;
  density?: number;
}

export interface DangerZone {
  y: number;
  x: number;
  pressure: number;
  risk: "HIGH" | "CRITICAL";
}

export type VenueShape = "box" | "cylinder" | "tiered" | "dome" | "ramp" | "canopy";

export interface VenueLayoutElement {
  type: "stage" | "wall" | "barrier" | "entry" | "gate";
  x: number;
  y: number;
  w: number;
  h: number;
  height?: number; // relative 3D extrusion height 0-1
  shape?: VenueShape;
  label?: string;
}

export type VenueArchetype =
  | "stadium" | "arena" | "theater" | "hall"
  | "plaza" | "street" | "field" | "festival";

export type DecorType =
  // event-venue decor
  | "screen" | "tower" | "tent" | "tree" | "roof"
  // distinctive scene-detail props (from the details agent)
  | "slide" | "swing" | "playset" | "fountain" | "statue"
  | "bench" | "booth" | "goal" | "pole" | "planter" | "court";

export interface DecorProp {
  type: DecorType;
  x: number;
  y: number;
  w: number;
  h: number;
  height?: number;
  color?: string; // optional hex hint from the details agent
  label?: string;
}

export interface VenueLayout {
  name: string;
  capacity: number;
  view: string;
  archetype?: VenueArchetype;
  confidence: number;
  notes: string;
  elements: VenueLayoutElement[];
  decor?: DecorProp[];
}

// Fruin Level-of-Service breakdown of the settled crowd (A = free, F = crush).
export type LosGrade = "A" | "B" | "C" | "D" | "E" | "F";
export interface LevelOfService {
  max_density: number;   // peak people/m²
  mean_density: number;  // over occupied floor area
  worst_los: LosGrade;
  distribution: Record<LosGrade, number>; // fraction of occupied area per grade
  occupied_cells: number;
}

export interface SimulateResult {
  frame_b64: string;
  metrics: string;
  safety_report: string;
  danger_zones: DangerZone[];
  safe_capacity: number;
  peak_pressure: number;
  n_exits?: number;
  level_of_service?: LevelOfService;
  venue_name?: string;
  layout?: VenueLayout;
}

// Expected entry/exit flow recovered by running the simulation as synthetic
// crowd video through the SAME RAFT optical-flow extractor used on live cameras.
export type PortMode = "inflow" | "outflow" | "mixed";
export interface PortFlow {
  label: string;
  type: "entry" | "exit";
  x: number;          // normalized centre
  y: number;
  dir_x: number;      // recovered flow direction (image space, unit-ish)
  dir_y: number;
  speed_px: number;   // mean moving-pixel speed at the door
  intensity: number;  // 0-1 relative to the busiest sustained flow
  flux: number;       // signed: + into venue, - out of venue
  mode: PortMode;
  share: number;      // fraction of total throughput across ports
}
export interface ExpectedFlowResult {
  ports: PortFlow[];
  annotated_b64: string; // synthetic frame with recovered per-port flow drawn on
  field_b64: string;     // RAFT-derived pressure field (live-monitor viz)
  sample_b64: string;    // raw synthetic-crowd frame RAFT actually sees
  flow_backend: string;  // "raft" | "farneback"
  pairs: number;
  summary: string;
  error?: string;
}

export async function simulateExpectedFlow(
  elements: VenueLayoutElement[],
  density: number,
  venueName = "Venue"
): Promise<ExpectedFlowResult> {
  const res = await fetch(`${BASE}/api/simulate_flow`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      elements: elements.map((e) => ({
        type: e.type, x: e.x, y: e.y, w: e.w, h: e.h, label: e.label ?? "",
      })),
      density,
      venue_name: venueName,
    }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export interface PlanResult extends SimulateResult {
  plan: string;
  purpose: string;
  agent_trace: AgentTraceStep[];
}

// ── 3D Plan / Simulate ───────────────────────────────────────────────────────

// Downsampled crowd field timeline used to advect agents in three.js.
// vx/vy/pressure are [frame][gridY][gridX]; walls is [gridY][gridX] (1 = blocked).
export interface FieldTimeline {
  grid: number;
  frames: number;
  vx: number[][][];
  vy: number[][][];
  pressure: number[][][];
  walls: number[][];
  p_max: number;
}

export interface ScenarioMetrics {
  peak_pressure: number;
  n_danger_zones: number;
  safe_capacity: number;
  crush_prob: number;
  n_exits: number;
  peak_density?: number;
  worst_los?: LosGrade;
  los_distribution?: Record<LosGrade, number>;
}

export interface Scenario {
  id: string;
  name: string;
  description: string;
  layout: VenueLayout;
  metrics: ScenarioMetrics;
  danger_zones: DangerZone[];
  field: FieldTimeline;
  rank: number;
  is_best: boolean;
}

export interface CapacityEstimate {
  area_m2: number;
  people_per_m2: number;
  usable_fraction: number;
  seating: string;
  max_capacity: number;
}

// Agent-LLM behavioral world model: groups of agents move toward reasoned
// goal points; `llm_fraction` of the crowd follows these, the rest move on
// the physics world model.
// How a group behaves WHILE INSIDE the venue (occasion-dependent).
export type AgentMode = "seated" | "roam" | "press" | "queue" | "browse";

export interface AgentBehavior {
  name: string;
  goal: [number, number]; // normalized 0-1 (x,y)
  fraction: number;
  speed: number;
  mode?: AgentMode;     // in-venue conduct (default "roam")
  excursion?: number;   // 0-1 share that steps out an exit and returns via entry
  intent: string;
}

export interface AgentPlan {
  llm_fraction: number;
  behaviors: AgentBehavior[];
  source?: string; // "llm" | "layout"
}

export interface CapacityCheck {
  given: number;
  healthy_capacity: number;
  crush_capacity: number;
  planned_capacity: number;
  verdict: "ok" | "tight" | "unreasonable";
  message: string;
}

// Arize-traced LLM-as-judge: does the 3D reconstruction match the photo?
export interface ReconstructionEval {
  score: number; // 0-1 overall fidelity
  label: "faithful" | "partial" | "poor";
  rationale: string;
  aspects: { structures: number; openings: number; scale: number; features: number };
}

export interface Plan3DResult {
  layout: VenueLayout;
  n_people: number;
  venue_max_capacity?: number;
  purpose: string;
  scenarios: Scenario[];
  best_scenario_id: string;
  plan_points: string[];
  plan: string;
  safety_report: string;
  agent_trace: AgentTraceStep[];
  capacity_estimate?: CapacityEstimate | null;
  capacity_check?: CapacityCheck | null;
  agent_plan?: AgentPlan | null;
  reconstruction_eval?: ReconstructionEval | null;
  chat_reply?: string; // present on /api/plan3d/refine responses
}

export interface DiscoverUnknown {
  dimensions?: number[];
  separation_z_score?: number;
  verdict?: string;
}

export interface DiscoverResult {
  table_md: string;
  hypothesis: string;
  computed?: boolean;
  latent_dim?: number;
  unknown?: DiscoverUnknown;
}

export interface RLTopAction {
  rank: number;
  action: string;
  description: string;
  q_value: number;
}

export interface RLLiveSample {
  action_name: string;
  action_description: string;
  confidence: number;
  q_values: Record<string, number>;
  top_3: RLTopAction[];
  error?: string;
}

export interface RLSummary {
  config?: Record<string, number>;
  n_steps?: number;
  duration_s?: number;
  final?: Record<string, number>;
  best?: Record<string, number>;
}

export interface RLMetricsResult {
  summary: RLSummary | null;
  curve_b64: string | null;
  live_sample: RLLiveSample | null;
  rl_policy_loaded: boolean;
}
