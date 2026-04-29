import type {
  Camera,
  Detection,
  MobileLprPose,
  ParkingSpot,
  ParkingZone,
  Plate,
  Session,
  SpotMatchStatus,
  SuspiciousOccupancy,
} from "../types";

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
  gps_latitude: number | null;
  gps_longitude: number | null;
  gps_heading_deg: number | null;
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
  voting_tag?: Detection["voting_tag"] | null;
  plate_annotation?: string | null;
  target_latitude?: number | null;
  target_longitude?: number | null;
  spot_match_status?: SpotMatchStatus | null;
  target_distance_m?: number | null;
  matched_spot_id?: number | null;
}

interface BackendParkingSpot {
  id: number;
  spot_label: string;
  parking_lot_id: number;
  spot_sequence: number | null;
  latitude: number | null;
  longitude: number | null;
  assigned_plate: string | null;
  allowed_plates: string[] | null;
  is_occupied: boolean;
  updated_at: string;
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
    gps_latitude: session.gps_latitude ?? null,
    gps_longitude: session.gps_longitude ?? null,
    gps_heading_deg: session.gps_heading_deg ?? null,
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
    voting_tag: detection.voting_tag ?? "not_final",
    plate_annotation:
      detection.plate_annotation ??
      (detection.ocr_normalized_text ?? detection.ocr_raw_text ?? detection.id),
    occurrences: 1,
    target_latitude: detection.target_latitude ?? null,
    target_longitude: detection.target_longitude ?? null,
    spot_match_status: detection.spot_match_status ?? null,
    target_distance_m: detection.target_distance_m ?? null,
    matched_spot_id: detection.matched_spot_id ?? null,
  };
}

function mapParkingSpot(spot: BackendParkingSpot): ParkingSpot {
  return {
    id: spot.id,
    spot_label: spot.spot_label,
    parking_lot_id: spot.parking_lot_id,
    spot_sequence: spot.spot_sequence ?? null,
    latitude: spot.latitude,
    longitude: spot.longitude,
    assigned_plate: spot.assigned_plate,
    allowed_plates: spot.allowed_plates ?? null,
    is_occupied: spot.is_occupied,
    updated_at: spot.updated_at,
  };
}

function isInvalidText(text: string): boolean {
  return (text || "").toUpperCase().startsWith("INVALID:");
}

function shouldPreferDetection(candidate: Detection, current: Detection): boolean {
  if (candidate.voting_tag !== current.voting_tag) {
    return candidate.voting_tag === "final";
  }

  if (candidate.is_valid_ro_plate !== current.is_valid_ro_plate) {
    return candidate.is_valid_ro_plate;
  }

  const candidateKnownTicket = candidate.ticket_status !== "unknown";
  const currentKnownTicket = current.ticket_status !== "unknown";
  if (candidateKnownTicket !== currentKnownTicket) {
    return candidateKnownTicket;
  }

  if (candidate.detection_confidence !== current.detection_confidence) {
    return candidate.detection_confidence > current.detection_confidence;
  }

  return new Date(candidate.created_at).getTime() > new Date(current.created_at).getTime();
}

function dedupeDetectionsByAnnotation(detections: Detection[]): Detection[] {
  const groups = new Map<string, Detection>();
  const counts = new Map<string, number>();

  for (const detection of detections) {
    const fallback = detection.ocr_normalized_text || detection.ocr_raw_text || detection.id;
    const key = detection.plate_annotation || fallback || detection.id;
    const current = groups.get(key);

    counts.set(key, (counts.get(key) ?? 0) + 1);
    if (!current || shouldPreferDetection(detection, current)) {
      groups.set(key, detection);
    }
  }

  const deduped = Array.from(groups.entries()).map(([key, detection]) => ({
    ...detection,
    plate_annotation: key,
    occurrences: counts.get(key) ?? 1,
  }));

  // Hide invalid OCR rows from the realtime feed.
  const validFirst = deduped.filter(
    (detection) =>
      detection.is_valid_ro_plate &&
      !isInvalidText(detection.ocr_normalized_text)
  );

  return validFirst.sort(
    (a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
  );
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

export async function createSession(
  file: File,
  fpsTarget: number,
  pose?: MobileLprPose | null,
  zoneId?: number | null,
): Promise<Session> {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("fps_target", String(fpsTarget));
  if (pose) {
    formData.append("gps_latitude", String(pose.latitude));
    formData.append("gps_longitude", String(pose.longitude));
    formData.append("gps_heading_deg", String(pose.headingDeg));
  }
  if (zoneId != null) {
    formData.append("zone_id", String(zoneId));
  }

  const session = await fetchApi<BackendSession>(`${API_BASE}/sessions`, {
    method: "POST",
    body: formData,
  });
  return mapSession(session);
}

export async function getParkingSpots(): Promise<ParkingSpot[]> {
  const spots = await fetchApi<BackendParkingSpot[]>(`${API_BASE}/parking/spots`);
  return spots.map(mapParkingSpot);
}

export async function getHardcodedTestFiles(): Promise<string[]> {
  return fetchApi<string[]>(`${API_BASE}/sessions/test-files`);
}

export async function createSessionFromHardcodedTest(
  filename: string,
  pose?: MobileLprPose | null,
  zoneId?: number | null,
): Promise<Session> {
  const formData = new FormData();
  if (pose) {
    formData.append("gps_latitude", String(pose.latitude));
    formData.append("gps_longitude", String(pose.longitude));
    formData.append("gps_heading_deg", String(pose.headingDeg));
  }
  if (zoneId != null) {
    formData.append("zone_id", String(zoneId));
  }

  const session = await fetchApi<BackendSession>(
    `${API_BASE}/sessions/test-files/${encodeURIComponent(filename)}`,
    { method: "POST", body: formData },
  );
  return mapSession(session);
}

export async function getParkingZones(): Promise<ParkingZone[]> {
  return fetchApi<ParkingZone[]>(`${API_BASE}/parking/zones`);
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

export async function getDetections(sessionId: string, since?: string): Promise<Detection[]> {
  const url = since
    ? `${API_BASE}/sessions/${sessionId}/detections?since=${encodeURIComponent(since)}`
    : `${API_BASE}/sessions/${sessionId}/detections`;
  const detections = await fetchApi<BackendDetection[]>(url);
  return dedupeDetectionsByAnnotation(detections.map(mapDetection));
}

export interface SessionEventHandlers {
  onFrameProcessed?: (data: { frames_processed: number; total_frames: number | null }) => void;
  onDetectionFinalized?: (data: { plate: string }) => void;
  onCompleted?: (data: { frames_processed: number; total_frames: number | null }) => void;
  onFailed?: (data: { error: string | null }) => void;
}

/** Open an SSE connection for live session events. Returns a close() function. */
export function subscribeToSessionEvents(
  sessionId: string,
  handlers: SessionEventHandlers,
): () => void {
  const es = new EventSource(`${API_BASE}/sessions/${sessionId}/events`);

  if (handlers.onFrameProcessed) {
    es.addEventListener("frame_processed", (e: MessageEvent) => {
      handlers.onFrameProcessed!(JSON.parse(e.data));
    });
  }
  if (handlers.onDetectionFinalized) {
    es.addEventListener("detection_finalized", (e: MessageEvent) => {
      handlers.onDetectionFinalized!(JSON.parse(e.data));
    });
  }
  if (handlers.onCompleted) {
    es.addEventListener("session_completed", (e: MessageEvent) => {
      handlers.onCompleted!(JSON.parse(e.data));
      es.close();
    });
  }
  if (handlers.onFailed) {
    es.addEventListener("session_failed", (e: MessageEvent) => {
      handlers.onFailed!(JSON.parse(e.data));
      es.close();
    });
  }

  return () => es.close();
}

export async function getSuspiciousOccupancies(
  flaggedOnly = true,
): Promise<SuspiciousOccupancy[]> {
  return fetchApi<SuspiciousOccupancy[]>(
    `${API_BASE}/parking/spots/suspicious?flagged_only=${flaggedOnly}`,
  );
}

export async function dismissSuspiciousOccupancy(recordId: number): Promise<void> {
  await fetchApi<void>(`${API_BASE}/parking/spots/suspicious/${recordId}`, {
    method: "DELETE",
  });
}

export async function searchPlates(
  query: string,
  county?: string,
  signal?: AbortSignal
): Promise<Plate[]> {
  const params = new URLSearchParams();
  if (query) params.append("q", query);
  if (county) params.append("county", county);

  const plates = await fetchApi<BackendPlate[]>(`${API_BASE}/plates?${params.toString()}`, {
    signal,
  });
  return plates.map(mapPlate);
}

export interface CameraIn {
  name: string;
  stream_url: string;
  parking_lot_id: number;
  is_active?: boolean;
}

export async function getCameras(): Promise<Camera[]> {
  return fetchApi<Camera[]>(`${API_BASE}/cameras`);
}

export async function getCamera(id: number): Promise<Camera> {
  return fetchApi<Camera>(`${API_BASE}/cameras/${id}`);
}

export async function createCamera(payload: CameraIn): Promise<Camera> {
  return fetchApi<Camera>(`${API_BASE}/cameras`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export async function updateCamera(id: number, payload: CameraIn): Promise<Camera> {
  return fetchApi<Camera>(`${API_BASE}/cameras/${id}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export async function deleteCamera(id: number): Promise<void> {
  await fetchApi<void>(`${API_BASE}/cameras/${id}`, { method: "DELETE" });
}
