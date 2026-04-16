export type SessionStatus = "pending" | "running" | "done" | "failed";
export type TicketStatus = "active" | "subscription" | "grace" | "none" | "unknown";
export type PlateType = "standard" | "temporary" | "diplomatic" | "unknown";
export type VotingTag = "final" | "not_final";

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
