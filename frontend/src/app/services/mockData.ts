import type { Session, Detection, Plate } from "../types";

export const MOCK_SESSIONS: Session[] = [
  {
    id: "1",
    source_filename: "parking_lot_video.mp4",
    status: "done",
    fps_target: 5,
    frames_processed: 450,
    frames_total: 450,
    error_message: null,
    created_at: new Date(Date.now() - 3600000).toISOString(),
    ended_at: new Date(Date.now() - 3000000).toISOString(),
  },
  {
    id: "2",
    source_filename: "street_monitoring.avi",
    status: "done",
    fps_target: 3,
    frames_processed: 280,
    frames_total: 280,
    error_message: null,
    created_at: new Date(Date.now() - 7200000).toISOString(),
    ended_at: new Date(Date.now() - 6800000).toISOString(),
  },
];

export const MOCK_DETECTIONS: Detection[] = [
  {
    id: "1",
    ocr_normalized_text: "B123ABC",
    ocr_raw_text: "B 123 ABC",
    is_valid_ro_plate: true,
    detection_confidence: 0.96,
    crop_image_url: "https://placehold.co/200x100/1e293b/94a3b8?text=B+123+ABC",
    ticket_status: "none",
    ticket_expires_at: null,
    created_at: new Date(Date.now() - 3500000).toISOString(),
  },
  {
    id: "2",
    ocr_normalized_text: "CJ456DEF",
    ocr_raw_text: "CJ 456 DEF",
    is_valid_ro_plate: true,
    detection_confidence: 0.94,
    crop_image_url: "https://placehold.co/200x100/1e293b/94a3b8?text=CJ+456+DEF",
    ticket_status: "active",
    ticket_expires_at: new Date(Date.now() + 3600000).toISOString(),
    created_at: new Date(Date.now() - 3400000).toISOString(),
  },
  {
    id: "3",
    ocr_normalized_text: "B789GHI",
    ocr_raw_text: "B789GHI",
    is_valid_ro_plate: true,
    detection_confidence: 0.89,
    crop_image_url: "https://placehold.co/200x100/1e293b/94a3b8?text=B+789+GHI",
    ticket_status: "none",
    ticket_expires_at: null,
    created_at: new Date(Date.now() - 3300000).toISOString(),
  },
  {
    id: "4",
    ocr_normalized_text: "B321XYZ",
    ocr_raw_text: "B 321 XYZ",
    is_valid_ro_plate: true,
    detection_confidence: 0.92,
    crop_image_url: "https://placehold.co/200x100/1e293b/94a3b8?text=B+321+XYZ",
    ticket_status: "subscription",
    ticket_expires_at: new Date(Date.now() + 2592000000).toISOString(),
    created_at: new Date(Date.now() - 3200000).toISOString(),
  },
  {
    id: "5",
    ocr_normalized_text: "IF654MNO",
    ocr_raw_text: "IF 654 MNO",
    is_valid_ro_plate: true,
    detection_confidence: 0.97,
    crop_image_url: "https://placehold.co/200x100/1e293b/94a3b8?text=IF+654+MNO",
    ticket_status: "grace",
    ticket_expires_at: new Date(Date.now() + 900000).toISOString(),
    created_at: new Date(Date.now() - 3100000).toISOString(),
  },
  {
    id: "6",
    ocr_normalized_text: "B555PQR",
    ocr_raw_text: "B555PQR",
    is_valid_ro_plate: true,
    detection_confidence: 0.88,
    crop_image_url: "https://placehold.co/200x100/1e293b/94a3b8?text=B+555+PQR",
    ticket_status: "none",
    ticket_expires_at: null,
    created_at: new Date(Date.now() - 3000000).toISOString(),
  },
];

export const MOCK_PLATES: Plate[] = [
  {
    id: "1",
    normalized_text: "B123ABC",
    county_code: "B",
    county_name: "București",
    plate_type: "standard",
    first_seen_at: new Date(Date.now() - 7200000).toISOString(),
    last_seen_at: new Date(Date.now() - 3500000).toISOString(),
    seen_count: 3,
    last_ticket_status: "none",
  },
  {
    id: "2",
    normalized_text: "CJ456DEF",
    county_code: "CJ",
    county_name: "Cluj",
    plate_type: "standard",
    first_seen_at: new Date(Date.now() - 10800000).toISOString(),
    last_seen_at: new Date(Date.now() - 3400000).toISOString(),
    seen_count: 2,
    last_ticket_status: "active",
  },
  {
    id: "3",
    normalized_text: "IF654MNO",
    county_code: "IF",
    county_name: "Ilfov",
    plate_type: "standard",
    first_seen_at: new Date(Date.now() - 5400000).toISOString(),
    last_seen_at: new Date(Date.now() - 3100000).toISOString(),
    seen_count: 1,
    last_ticket_status: "grace",
  },
];

export function generateMockSession(filename: string, fpsTarget: number): Session {
  return {
    id: String(Date.now()),
    source_filename: filename,
    status: "pending",
    fps_target: fpsTarget,
    frames_processed: 0,
    frames_total: 300,
    error_message: null,
    created_at: new Date().toISOString(),
    ended_at: null,
  };
}

export function simulateProcessing(session: Session): Session {
  const progress = session.frames_processed + Math.floor(Math.random() * 10) + 5;
  const isComplete = progress >= session.frames_total;

  return {
    ...session,
    status: isComplete ? "done" : "running",
    frames_processed: Math.min(progress, session.frames_total),
    ended_at: isComplete ? new Date().toISOString() : null,
  };
}

export function generateMockDetection(sessionId: string, index: number): Detection {
  const counties = ["B", "CJ", "IF", "TM", "CT"];
  const county = counties[Math.floor(Math.random() * counties.length)];
  const number = Math.floor(Math.random() * 900) + 100;
  const letters = String.fromCharCode(65 + Math.floor(Math.random() * 26)) +
    String.fromCharCode(65 + Math.floor(Math.random() * 26)) +
    String.fromCharCode(65 + Math.floor(Math.random() * 26));

  const plateText = `${county}${number}${letters}`;
  const statuses: Array<"active" | "subscription" | "grace" | "none" | "unknown"> =
    ["active", "subscription", "grace", "none", "none", "unknown"];
  const status = statuses[Math.floor(Math.random() * statuses.length)];

  return {
    id: String(Date.now() + index),
    ocr_normalized_text: plateText,
    ocr_raw_text: `${county} ${number} ${letters}`,
    is_valid_ro_plate: true,
    detection_confidence: 0.85 + Math.random() * 0.14,
    crop_image_url: `https://placehold.co/200x100/1e293b/94a3b8?text=${plateText.replace(/\s/g, '+')}`,
    ticket_status: status,
    ticket_expires_at: status !== "none" && status !== "unknown"
      ? new Date(Date.now() + Math.random() * 86400000).toISOString()
      : null,
    created_at: new Date().toISOString(),
  };
}
