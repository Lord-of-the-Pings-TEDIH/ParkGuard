import { CheckCircle, AlertTriangle, MapPinOff } from "lucide-react";
import type { SpotMatchStatus } from "../types";

interface SpotMatchBadgeProps {
  status: SpotMatchStatus;
  distanceM?: number | null;
  className?: string;
}

const STYLES: Record<SpotMatchStatus, { label: string; classes: string; Icon: typeof CheckCircle }> = {
  MATCH: {
    label: "Loc corect",
    classes:
      "bg-gradient-to-r from-green-100 to-emerald-100 text-green-700 border-green-400 dark:from-green-900 dark:to-emerald-900 dark:text-green-300 dark:border-green-600",
    Icon: CheckCircle,
  },
  WRONG_PLATE: {
    label: "Loc greșit",
    classes:
      "bg-gradient-to-r from-red-100 to-orange-100 text-red-700 border-red-400 dark:from-red-900 dark:to-orange-900 dark:text-red-300 dark:border-red-600",
    Icon: AlertTriangle,
  },
  NO_SPOT_FOUND: {
    label: "Niciun loc înregistrat",
    classes:
      "bg-gradient-to-r from-slate-100 to-gray-100 text-slate-700 border-slate-400 dark:from-slate-800 dark:to-gray-800 dark:text-slate-300 dark:border-slate-600",
    Icon: MapPinOff,
  },
};

export function SpotMatchBadge({ status, distanceM, className = "" }: SpotMatchBadgeProps) {
  const { label, classes, Icon } = STYLES[status];
  return (
    <div
      className={`inline-flex items-center gap-1.5 rounded border px-2 py-1 text-xs font-bold ${classes} ${className}`}
      title={
        distanceM != null
          ? `Distanța până la cel mai apropiat loc: ${distanceM.toFixed(2)} m`
          : undefined
      }
    >
      <Icon className="h-3 w-3" />
      <span>{label}</span>
      {distanceM != null && status !== "NO_SPOT_FOUND" && (
        <span className="opacity-75">({distanceM.toFixed(1)} m)</span>
      )}
    </div>
  );
}
