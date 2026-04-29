import { useCallback, useEffect, useRef, useState } from "react";
import { AlertTriangle, Bell, RefreshCw, X } from "lucide-react";
import { Switch } from "./ui/switch";
import { Label } from "./ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./ui/tabs";
import {
  getSuspiciousOccupancies,
  dismissSuspiciousOccupancy,
  getAlerts,
  resolveAlert,
  deleteAlert,
} from "../services/api";
import { formatRelativeTime } from "../utils/format";
import type { Alert, SuspiciousOccupancy } from "../types";

const POLL_INTERVAL_MS = 60_000;

function severity(score: number): { label: string; className: string } {
  if (score >= 12) return { label: "Critical", className: "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200" };
  if (score >= 8)  return { label: "High",     className: "bg-orange-100 text-orange-800 dark:bg-orange-900 dark:text-orange-200" };
  if (score >= 4)  return { label: "Medium",   className: "bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200" };
  return            { label: "Low",      className: "bg-muted text-muted-foreground" };
}

function formatHour(h: number): string {
  const hh = Math.floor(h).toString().padStart(2, "0");
  const mm = Math.round((h % 1) * 60).toString().padStart(2, "0");
  return `${hh}:${mm}`;
}

export function SuspiciousOccupancyPanel() {
  const [records, setRecords] = useState<SuspiciousOccupancy[]>([]);
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [flaggedOnly, setFlaggedOnly] = useState(true);
  const [loading, setLoading] = useState(false);
  const [dismissing, setDismissing] = useState<number | null>(null);
  const [resolving, setResolving] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const fetchAll = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [occ, alr] = await Promise.all([
        getSuspiciousOccupancies(flaggedOnly),
        getAlerts(false),
      ]);
      setRecords(occ);
      setAlerts(alr);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load");
    } finally {
      setLoading(false);
    }
  }, [flaggedOnly]);

  useEffect(() => {
    void fetchAll();
    timerRef.current = setInterval(() => void fetchAll(), POLL_INTERVAL_MS);
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [fetchAll]);

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

  const handleResolveAlert = async (alert: Alert) => {
    setResolving(alert.id);
    try {
      await resolveAlert(alert.id);
      setAlerts((prev) => prev.filter((a) => a.id !== alert.id));
    } catch {
      setError("Failed to resolve alert");
    } finally {
      setResolving(null);
    }
  };

  const handleDeleteAlert = async (alert: Alert) => {
    setResolving(alert.id);
    try {
      await deleteAlert(alert.id);
      setAlerts((prev) => prev.filter((a) => a.id !== alert.id));
    } catch {
      setError("Failed to delete alert");
    } finally {
      setResolving(null);
    }
  };

  const unresolvedCount = alerts.length;

  return (
    <div className="flex h-full min-h-0 flex-col border-border bg-card lg:border-l">
      <div className="border-b border-border bg-gradient-to-r from-orange-50 to-red-50 px-3 py-2 dark:from-orange-950 dark:to-red-950 md:px-4 md:py-3">
        <div className="flex items-center justify-between gap-2">
          <div className="flex items-center gap-1.5">
            <AlertTriangle className="h-4 w-4 text-orange-600 dark:text-orange-400" />
            <h3 className="font-medium text-foreground">Spot Misuse</h3>
            {unresolvedCount > 0 && (
              <span className="rounded-full bg-red-500 px-1.5 py-0.5 text-xs font-bold text-white">
                {unresolvedCount}
              </span>
            )}
          </div>
          <button
            type="button"
            onClick={() => void fetchAll()}
            disabled={loading}
            className="rounded p-1 text-muted-foreground hover:text-foreground disabled:opacity-40"
            title="Refresh"
          >
            <RefreshCw className={`h-3.5 w-3.5 ${loading ? "animate-spin" : ""}`} />
          </button>
        </div>
      </div>

      <Tabs defaultValue="records" className="flex min-h-0 flex-1 flex-col">
        <TabsList className="mx-3 mt-2 grid w-auto grid-cols-2">
          <TabsTrigger value="records" className="text-xs">
            Records {records.length > 0 && `(${records.length})`}
          </TabsTrigger>
          <TabsTrigger value="alerts" className="text-xs">
            Alerts {unresolvedCount > 0 && `(${unresolvedCount})`}
          </TabsTrigger>
        </TabsList>

        {/* Occupancy Records tab */}
        <TabsContent value="records" className="min-h-0 flex-1 overflow-y-auto mt-0">
          <div className="px-3 pt-2 pb-1">
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
          </div>

          <div className="space-y-2 px-3 pb-3">
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
                    {rec.typical_hour !== null && rec.typical_hour !== undefined && (
                      <div className="flex justify-between">
                        <span>Typical time</span>
                        <span className="font-medium text-foreground">
                          {formatHour(rec.typical_hour)}
                          {rec.hour_concentration !== null && rec.hour_concentration !== undefined && (
                            <span className="ml-1 text-muted-foreground">
                              ({Math.round(rec.hour_concentration * 100)}% conc.)
                            </span>
                          )}
                        </span>
                      </div>
                    )}
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
        </TabsContent>

        {/* Alerts tab */}
        <TabsContent value="alerts" className="min-h-0 flex-1 overflow-y-auto mt-0">
          <div className="space-y-2 p-3">
            {error && (
              <div className="rounded-lg border border-red-300 bg-red-50 p-3 text-xs text-red-700 dark:border-red-800 dark:bg-red-950/40 dark:text-red-300">
                {error}
              </div>
            )}

            {!loading && alerts.length === 0 && (
              <div className="py-12 text-center text-sm text-muted-foreground">
                No open alerts
              </div>
            )}

            {alerts.map((alert) => (
              <div
                key={alert.id}
                className="rounded-lg border border-red-300 bg-gradient-to-br from-red-50 to-orange-50 p-3 shadow-sm dark:border-red-700 dark:from-red-950 dark:to-orange-950"
              >
                <div className="mb-1.5 flex items-start justify-between gap-2">
                  <div className="flex items-center gap-1.5">
                    <Bell className="mt-0.5 h-3.5 w-3.5 shrink-0 text-red-600 dark:text-red-400" />
                    <span className="text-xs font-semibold uppercase tracking-wide text-red-700 dark:text-red-300">
                      {alert.alert_type.replace(/_/g, " ")}
                    </span>
                  </div>
                  <div className="flex gap-1">
                    <button
                      type="button"
                      title="Mark resolved"
                      disabled={resolving === alert.id}
                      onClick={() => void handleResolveAlert(alert)}
                      className="rounded px-1.5 py-0.5 text-xs font-medium text-green-700 hover:bg-green-100 disabled:opacity-40 dark:text-green-400 dark:hover:bg-green-900"
                    >
                      Resolve
                    </button>
                    <button
                      type="button"
                      title="Delete alert"
                      disabled={resolving === alert.id}
                      onClick={() => void handleDeleteAlert(alert)}
                      className="rounded p-0.5 text-muted-foreground hover:text-foreground disabled:opacity-40"
                    >
                      <X className="h-3.5 w-3.5" />
                    </button>
                  </div>
                </div>

                {alert.message && (
                  <p className="text-xs text-foreground">{alert.message}</p>
                )}
                <p className="mt-1 text-xs text-muted-foreground">
                  {formatRelativeTime(alert.created_at)}
                </p>
              </div>
            ))}
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}
