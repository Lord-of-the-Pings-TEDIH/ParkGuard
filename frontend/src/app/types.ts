export type SessionStatus = "pending" | "running" | "done" | "failed";
export type TicketStatus = "active" | "subscription" | "grace" | "none" | "unknown";
export type PlateType = "standard" | "temporary" | "diplomatic" | "unknown";
export type VotingTag = "final" | "not_final";
export type SpotMatchStatus = "MATCH" | "WRONG_PLATE" | "NO_SPOT_FOUND";

export interface MobileLprPose {
  latitude: number;
  longitude: number;
  headingDeg: number;
}

export interface ParkingZone {
  id: number;
  name: string;
  code: string;
  grace_period_min: number;
  is_active: boolean;
  created_at: string;
}

export interface Detection {
  id: string;
  ocr_normalized_text: string;
  ocr_raw_text: string;
  is_valid_ro_plate: boolean;
  detection_confidence: number;
  crop_image_url: string;
  ticket_status: TicketStatus;
  ticket_expires_at: string | null;
  created_at: string;
  voting_tag: VotingTag;
  plate_annotation: string;
  occurrences: number;
  target_latitude: number | null;
  target_longitude: number | null;
  spot_match_status: SpotMatchStatus | null;
  target_distance_m: number | null;
  matched_spot_id: number | null;
}

export interface Session {
  id: string;
  source_filename: string;
  status: SessionStatus;
  fps_target: number;
  frames_processed: number;
  frames_total: number;
  error_message: string | null;
  created_at: string;
  ended_at: string | null;
  gps_latitude: number | null;
  gps_longitude: number | null;
  gps_heading_deg: number | null;
}

export interface ParkingSpot {
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

export interface Camera {
  id: number;
  name: string;
  stream_url: string;
  parking_lot_id: number;
  is_active: boolean;
  created_at: string;
}

export interface Plate {
  id: string;
  normalized_text: string;
  county_code: string;
  county_name: string;
  plate_type: PlateType;
  first_seen_at: string;
  last_seen_at: string;
  seen_count: number;
  last_ticket_status: TicketStatus;
}

export interface SessionStats {
  total_detections: number;
  unique_plates: number;
  unpaid_count: number;
  active_count: number;
}

export interface Alert {
  id: number;
  detection_id: string | null;
  alert_type: string;
  message: string | null;
  is_resolved: boolean;
  created_at: string;
}

export interface SuspiciousOccupancy {
  id: number;
  spot_id: number;
  spot_label: string | null;
  assigned_plate: string | null;
  plate_text: string;
  event_count: number;
  distinct_days: number;
  accumulated_score: number;
  hour_concentration: number | null;
  typical_hour: number | null;
  first_seen_at: string;
  last_seen_at: string;
  is_flagged: boolean;
}
