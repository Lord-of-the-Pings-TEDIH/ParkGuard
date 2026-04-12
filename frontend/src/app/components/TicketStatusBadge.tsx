import { getTicketStatusColor, getTicketStatusLabel, formatTimeRemaining } from "../utils/format";
import type { TicketStatus } from "../types";

interface TicketStatusBadgeProps {
  status: TicketStatus;
  expiresAt?: string | null;
  className?: string;
}

export function TicketStatusBadge({ status, expiresAt, className = "" }: TicketStatusBadgeProps) {
  const color = getTicketStatusColor(status);
  const label = getTicketStatusLabel(status);
  const timeRemaining = expiresAt ? formatTimeRemaining(expiresAt) : null;

  const colorClasses = {
    green: "bg-gradient-to-r from-green-100 to-emerald-100 text-green-700 border-green-400 dark:from-green-900 dark:to-emerald-900 dark:text-green-300 dark:border-green-600",
    amber: "bg-gradient-to-r from-amber-100 to-orange-100 text-amber-700 border-amber-400 dark:from-amber-900 dark:to-orange-900 dark:text-amber-300 dark:border-amber-600",
    red: "bg-gradient-to-r from-red-100 to-orange-100 text-red-700 border-red-400 dark:from-red-900 dark:to-orange-900 dark:text-red-300 dark:border-red-600",
    gray: "bg-gradient-to-r from-slate-100 to-gray-100 text-slate-700 border-slate-400 dark:from-slate-800 dark:to-gray-800 dark:text-slate-300 dark:border-slate-600",
  };

  return (
    <div className={`inline-flex items-center gap-1.5 rounded border px-2 py-1 text-xs font-bold ${colorClasses[color]} ${className}`}>
      <span>{label}</span>
      {timeRemaining && status !== "none" && status !== "unknown" && (
        <span className="opacity-75">({timeRemaining})</span>
      )}
    </div>
  );
}
