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

export async function monitorUrl(
  url: string,
  venue: string,
  nFrames = 35
): Promise<MonitorResult> {
  const res = await fetch(`${BASE}/api/monitor_url`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url, venue, n_frames: nFrames }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
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

export interface AnalyzeResult {
  peak_frame_b64: string | null;
  summary: string;
  claude_briefing: string;
  rl_explanation: string;
  timeline: TimelinePoint[];
  peak_physics: Record<string, unknown> | null;
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

export interface SimulateResult {
  frame_b64: string;
  metrics: string;
  safety_report: string;
  danger_zones: DangerZone[];
  safe_capacity: number;
  peak_pressure: number;
}

export interface DiscoverResult {
  table_md: string;
  hypothesis: string;
  computed?: boolean;
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
