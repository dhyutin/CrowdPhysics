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

export type LiveEventType = "source" | "calibrating" | "tick" | "done";

export interface LiveTick {
  type: LiveEventType;
  // tick
  step?: number;
  time?: number;
  status?: TimelinePoint["status"];
  score?: number;
  probability?: number;
  forecast?: Forecast;
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
  signal?: AbortSignal
): Promise<void> {
  const res = await fetch(`${BASE}/api/monitor_url_stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url, venue, n_frames: nFrames, session_id: sessionId ?? null }),
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
}

export interface AgentTraceStep {
  agent: string;
  icon: string;
  action: string;
  detail: string;
  status: "ok" | "danger";
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

export interface VenueLayoutElement {
  type: "stage" | "wall" | "barrier" | "entry" | "gate";
  x: number;
  y: number;
  w: number;
  h: number;
  label?: string;
}

export interface VenueLayout {
  name: string;
  capacity: number;
  view: string;
  confidence: number;
  notes: string;
  elements: VenueLayoutElement[];
}

export interface SimulateResult {
  frame_b64: string;
  metrics: string;
  safety_report: string;
  danger_zones: DangerZone[];
  safe_capacity: number;
  peak_pressure: number;
  n_exits?: number;
  venue_name?: string;
  layout?: VenueLayout;
}

export interface PlanResult extends SimulateResult {
  plan: string;
  purpose: string;
  agent_trace: AgentTraceStep[];
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
