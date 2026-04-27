import { useCallback, useEffect, useRef, useState } from "react";
import { AlertTriangle, RefreshCw, X } from "lucide-react";
import { Switch } from "./ui/switch";
import { Label } from "./ui/label";
import { getSuspiciousOccupancies, dismissSuspiciousOccupancy } from "../services/api";
import { formatRelativeTime } from "../utils/format";
import type { SuspiciousOccupancy } from "../types";

const POLL_INTERVAL_MS = 60_000;

function severity(score: number): { label: string; className: string } {
  if (score >= 12) return { label: "Critical", className: "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200" };
  if (score >= 8)  return { label: "High",     className: "bg-orange-100 text-orange-800 dark:bg-orange-900 dark:text-orange-200" };
  if (score >= 4)  return { label: "Medium",   className: "bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200" };
  return            { label: "Low",      className: "bg-muted text-muted-foreground" };
}

export function SuspiciousOccupancyPanel() {
  const [records, setRecords] = useState<SuspiciousOccupancy[]>([]);
  const [flaggedOnly, setFlaggedOnly] = useState(true);
  const [loading, setLoading] = useState(false);
  const [dismissing, setDismissing] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const fetchRecords = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await getSuspiciousOccupancies(flaggedOnly);
      setRecords(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load");
    } finally {
      setLoading(false);
    }
  }, [flaggedOnly]);

  useEffect(() => {
    void fetchRecords();
    timerRef.current = setInterval(() => void fetchRecords(), POLL_INTERVAL_MS);
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [fetchRecords]);

  const handleDismiss = async (rec: SuspiciousOccupancy) => {
    setDismissing(rec.id);
    try {
      await dismissSuspiciousOccupancy(rec.id);
      setRecords((prev) => prev.filter((r) => r.id !== rec.id));
    } catch {
      setError("Failed to dismiss record");
    } finally {
      setDismissing(null);
    }
  };

  return (
    <div className="flex h-full min-h-0 flex-col border-border bg-card lg:border-l">
      <div className="border-b border-border bg-gradient-to-r from-orange-50 to-red-50 p-3 dark:from-orange-950 dark:to-red-950 md:p-4">
        <div className="mb-3 flex items-center justify-between gap-2">
          <div className="flex items-center gap-1.5">
            <AlertTriangle className="h-4 w-4 text-orange-600 dark:text-orange-400" />
            <h3 className="font-medium text-foreground">Spot Misuse</h3>
          </div>
          <button
            type="button"
            onClick={() => void fetchRecords()}
            disabled={loading}
            className="rounded p-1 text-muted-foreground hover:text-foreground disabled:opacity-40"
            title="Refresh"
          >
            <RefreshCw className={`h-3.5 w-3.5 ${loading ? "animate-spin" : ""}`} />
          </button>
        </div>

        <div className="flex items-center gap-2">
          <Switch
            id="flagged-only"
            checked={flaggedOnly}
            onCheckedChange={setFlaggedOnly}
          />
          <Label htmlFor="flagged-only" className="text-xs text-muted-foreground">
            Flagged only
          </Label>
        </div>

        {records.length > 0 && (
          <p className="mt-2 text-xs text-muted-foreground">
            {records.length} record{records.length !== 1 ? "s" : ""}
          </p>
        )}
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto">
        <div className="space-y-2 p-3">
          {error && (
            <div className="rounded-lg border border-red-300 bg-red-50 p-3 text-xs text-red-700 dark:border-red-800 dark:bg-red-950/40 dark:text-red-300">
              {error}
            </div>
          )}

          {!loading && records.length === 0 && (
            <div className="py-12 text-center text-sm text-muted-foreground">
              {flaggedOnly ? "No flagged records" : "No records yet"}
            </div>
          )}

          {records.map((rec) => {
            const sev = severity(rec.accumulated_score);
            return (
              <div
                key={rec.id}
                className={`rounded-lg border p-3 shadow-sm ${
                  rec.is_flagged
                    ? "border-orange-300 bg-gradient-to-br from-orange-50 to-red-50 dark:border-orange-700 dark:from-orange-950 dark:to-red-950"
                    : "border-border bg-muted"
                }`}
              >
                {/* Header row: plate + severity + dismiss */}
                <div className="mb-2 flex items-center justify-between gap-2">
                  <span className="font-mono text-sm font-bold tracking-widest text-foreground">
                    {rec.plate_text}
                  </span>
                  <div className="flex items-center gap-1.5">
                    <span className={`rounded px-1.5 py-0.5 text-xs font-semibold ${sev.className}`}>
                      {sev.label} · {rec.accumulated_score.toFixed(1)}
                    </span>
                    <button
                      type="button"
                      title="Dismiss (false positive)"
                      disabled={dismissing === rec.id}
                      onClick={() => void handleDismiss(rec)}
                      className="rounded p-0.5 text-muted-foreground hover:text-foreground disabled:opacity-40"
                    >
                      <X className="h-3.5 w-3.5" />
                    </button>
                  </div>
                </div>

                <div className="space-y-1 text-xs text-muted-foreground">
                  {/* Spot label and rightful owner */}
                  <div className="flex justify-between">
                    <span>Spot</span>
                    <span className="font-medium text-foreground">
                      {rec.spot_label ?? `#${rec.spot_id}`}
                    </span>
                  </div>
                  {rec.assigned_plate && (
                    <div className="flex justify-between">
                      <span>Owner</span>
                      <span className="font-mono font-medium text-foreground">
                        {rec.assigned_plate}
                      </span>
                    </div>
                  )}
                  <div className="flex justify-between">
                    <span>Detections</span>
                    <span className="font-medium text-foreground">
                      {rec.event_count}× over {rec.distinct_days}d
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span>First seen</span>
                    <span>{formatRelativeTime(rec.first_seen_at)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Last seen</span>
                    <span>{formatRelativeTime(rec.last_seen_at)}</span>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
