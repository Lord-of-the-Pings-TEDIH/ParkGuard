import { formatDistanceToNow, differenceInSeconds, format } from "date-fns";
import { ro } from "date-fns/locale";
import type { TicketStatus, SessionStats, Detection } from "../types";

export function formatTimeRemaining(expiresAt: string | null): string {
  if (!expiresAt) return "";

  const now = new Date();
  const expires = new Date(expiresAt);
  const seconds = differenceInSeconds(expires, now);

  if (seconds < 0) return "Expirat";
  if (seconds < 60) return `${seconds}s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h`;
  return `${Math.floor(seconds / 86400)}d`;
}

export function formatElapsedTime(startedAt: string): string {
  const now = new Date();
  const started = new Date(startedAt);
  const seconds = differenceInSeconds(now, started);

  if (seconds < 60) return `${seconds}s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
  return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`;
}

export function formatDuration(
  startedAt: string,
  endedAt: string | null,
): string | null {
  if (!endedAt) return null;
  const seconds = differenceInSeconds(new Date(endedAt), new Date(startedAt));
  if (!Number.isFinite(seconds) || seconds < 0) return null;
  if (seconds < 60) return `${seconds}s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
  return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`;
}

export function formatRelativeTime(dateString: string): string {
  return formatDistanceToNow(new Date(dateString), { addSuffix: true, locale: ro });
}

export function formatDateTime(dateString: string): string {
  return format(new Date(dateString), "MMM d, HH:mm", { locale: ro });
}

export function calculateStats(detections: Detection[]): SessionStats {
  const uniquePlates = new Set(detections.map(d => d.ocr_normalized_text));
  const unpaidCount = detections.filter(d => d.ticket_status === "none").length;
  const activeCount = detections.filter(d =>
    d.ticket_status === "active" ||
    d.ticket_status === "subscription"
  ).length;

  return {
    total_detections: detections.length,
    unique_plates: uniquePlates.size,
    unpaid_count: unpaidCount,
    active_count: activeCount,
  };
}

export function getTicketStatusColor(status: TicketStatus): string {
  switch (status) {
    case "active":
    case "subscription":
      return "green";
    case "grace":
      return "amber";
    case "none":
      return "red";
    case "unknown":
    default:
      return "gray";
  }
}

export function getTicketStatusLabel(status: TicketStatus): string {
  switch (status) {
    case "active":
      return "PLĂTIT";
    case "subscription":
      return "ABONAMENT";
    case "grace":
      return "GRAȚIE";
    case "none":
      return "⚠ NEPLĂTIT";
    case "unknown":
    default:
      return "?";
  }
}
