import type { Detection, Session, Plate } from "../types";

const API_BASE = "/api";

interface BackendSession {
  id: string;
  source_filename: string;
  status: string;
  fps_target: number | null;
  frames_processed: number;
  total_frames: number | null;
  error_message: string | null;
  created_at: string;
  ended_at: string | null;
}

interface BackendDetection {
  id: string;
  ocr_normalized_text: string | null;
  ocr_raw_text: string | null;
  is_valid_ro_plate: boolean;
  detection_confidence: number;
  crop_image_url: string | null;
  ticket_status: Detection["ticket_status"] | null;
  ticket_expires_at: string | null;
  created_at: string;
}

interface BackendPlate {
  id: string;
  normalized_text: string;
  county_code: string | null;
  county_name: string | null;
  plate_type: Plate["plate_type"] | null;
  first_seen_at: string;
  last_seen_at: string;
  seen_count: number;
  last_ticket_status: Plate["last_ticket_status"] | null;
}

async function fetchApi<T>(
  url: string,
  options?: RequestInit
): Promise<T> {
  const response = await fetch(url, options);
  if (!response.ok) {
    let detail = `HTTP ${response.status}`;
    try {
      const data = await response.json();
      if (typeof data?.detail === "string" && data.detail.trim()) {
        detail = data.detail;
      }
    } catch {
      // Keep default error detail.
    }
    throw new Error(detail);
  }

  if (response.status === 204) {
    return undefined as T;
  }

  return response.json() as Promise<T>;
}

function mapSessionStatus(status: string): Session["status"] {
  switch (status) {
    case "processing":
      return "running";
    case "completed":
      return "done";
    case "pending":
    case "running":
    case "done":
    case "failed":
      return status;
    default:
      return "failed";
  }
}

function mapSession(session: BackendSession): Session {
  return {
    id: String(session.id),
    source_filename: session.source_filename,
    status: mapSessionStatus(session.status),
    fps_target: session.fps_target ?? 0,
    frames_processed: session.frames_processed ?? 0,
    frames_total: session.total_frames ?? 0,
    error_message: session.error_message,
    created_at: session.created_at,
    ended_at: session.ended_at,
  };
}

function mapDetection(detection: BackendDetection): Detection {
  return {
    id: String(detection.id),
    ocr_normalized_text: detection.ocr_normalized_text ?? "",
    ocr_raw_text: detection.ocr_raw_text ?? "",
    is_valid_ro_plate: detection.is_valid_ro_plate,
    detection_confidence: detection.detection_confidence,
    crop_image_url: detection.crop_image_url ?? "",
    ticket_status: detection.ticket_status ?? "unknown",
    ticket_expires_at: detection.ticket_expires_at,
    created_at: detection.created_at,
  };
}

function mapPlate(plate: BackendPlate): Plate {
  return {
    id: String(plate.id),
    normalized_text: plate.normalized_text,
    county_code: plate.county_code ?? "",
    county_name: plate.county_name ?? "",
    plate_type: plate.plate_type ?? "unknown",
    first_seen_at: plate.first_seen_at,
    last_seen_at: plate.last_seen_at,
    seen_count: plate.seen_count,
    last_ticket_status: plate.last_ticket_status ?? "unknown",
  };
}

export async function createSession(file: File, fpsTarget: number): Promise<Session> {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("fps_target", String(fpsTarget));

  const session = await fetchApi<BackendSession>(`${API_BASE}/sessions`, {
    method: "POST",
    body: formData,
  });
  return mapSession(session);
}

export async function getHardcodedTestFiles(): Promise<string[]> {
  return fetchApi<string[]>(`${API_BASE}/sessions/test-files`);
}

export async function createSessionFromHardcodedTest(filename: string): Promise<Session> {
  const session = await fetchApi<BackendSession>(
    `${API_BASE}/sessions/test-files/${encodeURIComponent(filename)}`,
    {
      method: "POST",
    }
  );
  return mapSession(session);
}

export async function startSessionProcessing(id: string): Promise<Session> {
  const session = await fetchApi<BackendSession>(`${API_BASE}/sessions/${id}/process`, {
    method: "POST",
  });
  return mapSession(session);
}

export async function getSession(id: string): Promise<Session> {
  const session = await fetchApi<BackendSession>(`${API_BASE}/sessions/${id}`);
  return mapSession(session);
}

export async function getSessions(): Promise<Session[]> {
  const sessions = await fetchApi<BackendSession[]>(`${API_BASE}/sessions`);
  return sessions.map(mapSession);
}

export async function deleteSession(id: string): Promise<void> {
  await fetchApi<void>(`${API_BASE}/sessions/${id}`, {
    method: "DELETE",
  });
}

export async function cancelSession(id: string): Promise<Session> {
  const session = await fetchApi<BackendSession>(`${API_BASE}/sessions/${id}/cancel`, {
    method: "POST",
  });
  return mapSession(session);
}

export async function getDetections(sessionId: string): Promise<Detection[]> {
  const detections = await fetchApi<BackendDetection[]>(
    `${API_BASE}/sessions/${sessionId}/detections`
  );
  return detections.map(mapDetection);
}

export async function searchPlates(query: string, county?: string): Promise<Plate[]> {
  const params = new URLSearchParams();
  if (query) params.append("q", query);
  if (county) params.append("county", county);

  const plates = await fetchApi<BackendPlate[]>(`${API_BASE}/plates?${params.toString()}`);
  const mapped = plates.map(mapPlate);

  let filtered = mapped;
  if (query) {
    filtered = filtered.filter((p) =>
      p.normalized_text.toLowerCase().includes(query.toLowerCase())
    );
  }
  if (county) {
    filtered = filtered.filter(
      (p) => p.county_code.toLowerCase() === county.toLowerCase()
    );
  }

  return filtered;
}
