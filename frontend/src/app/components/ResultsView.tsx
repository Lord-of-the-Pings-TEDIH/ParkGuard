import { useState } from "react";
import { motion } from "motion/react";
import { CheckCircle, Clock, Download } from "lucide-react";
import { DetectionCard } from "./DetectionCard";
import { calculateStats, formatDuration } from "../utils/format";
import type { Session, Detection } from "../types";

interface ResultsViewProps {
  session: Session;
  detections: Detection[];
  onReset: () => void;
}

type Filter = "all" | "unpaid" | "wrong";

const FILTERS: Array<{ id: Filter; label: string }> = [
  { id: "all",    label: "Toate"     },
  { id: "unpaid", label: "Neplătite" },
  { id: "wrong",  label: "Loc greșit" },
];

export function ResultsView({ session, detections, onReset }: ResultsViewProps) {
  const [filter, setFilter] = useState<Filter>("all");
  const stats = calculateStats(detections);
  const duration = formatDuration(session.created_at, session.ended_at);

  const filtered =
    filter === "all"    ? detections
    : filter === "unpaid" ? detections.filter((d) => d.ticket_status === "none")
    :                       detections.filter((d) => d.spot_match_status === "WRONG_PLATE");

  const handleExportCsv = () => {
    const header = "plate,ticket_status,spot_match_status,confidence,occurrences,voting_tag";
    const rows = detections.map((d) =>
      [
        d.ocr_normalized_text,
        d.ticket_status,
        d.spot_match_status ?? "",
        (d.detection_confidence * 100).toFixed(1),
        d.occurrences,
        d.voting_tag,
      ].join(","),
    );
    const blob = new Blob([[header, ...rows].join("\n")], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${session.source_filename.replace(/\.[^.]+$/, "")}_detections.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="flex h-full min-h-0 flex-col">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        className="border-b border-border bg-card flex-shrink-0"
        style={{ padding: "16px 24px" }}
      >
        {/* Title row */}
        <div className="flex items-start justify-between gap-3 mb-4">
          <div className="flex items-center gap-3 min-w-0">
            <div
              className="flex-shrink-0 flex items-center justify-center rounded-lg bg-green-500/10 border border-green-500/20"
              style={{ width: 28, height: 28 }}
            >
              <CheckCircle className="h-3.5 w-3.5 text-green-500" />
            </div>
            <div className="min-w-0">
              <div className="font-bold text-foreground truncate" style={{ fontSize: 16 }}>
                {session.source_filename}
              </div>
              <div className="flex items-center gap-2 text-xs text-muted-foreground mt-0.5">
                <span className="bg-green-500/10 text-green-600 dark:text-green-400 px-2 py-0.5 rounded-full text-[11px] font-medium">
                  Finalizat
                </span>
                {duration && (
                  <>
                    <span>·</span>
                    <span className="flex items-center gap-1">
                      <Clock className="h-3 w-3" />
                      {duration}
                    </span>
                  </>
                )}
              </div>
            </div>
          </div>
          <div className="flex gap-2 flex-shrink-0">
            <button
              type="button"
              onClick={handleExportCsv}
              className="border border-border bg-card text-muted-foreground hover:text-foreground rounded-lg px-3 py-1.5 text-xs cursor-pointer flex items-center gap-1.5 transition-colors"
            >
              <Download className="h-3.5 w-3.5" />
              Export CSV
            </button>
            <button
              type="button"
              onClick={onReset}
              className="rounded-lg bg-primary text-primary-foreground px-3 py-1.5 text-xs font-semibold cursor-pointer hover:opacity-90 transition-opacity"
            >
              ← Nouă sesiune
            </button>
          </div>
        </div>

        {/* Stat cards — flat with colored top border */}
        <div className="grid grid-cols-4 gap-3">
          <FlatStatCard label="Detectate"  value={stats.total_detections} accentColor="#2563eb" />
          <FlatStatCard label="Unice"      value={stats.unique_plates}    accentColor="#7c3aed" />
          <FlatStatCard label="Plătite"    value={stats.active_count}     accentColor="#16a34a" />
          <FlatStatCard label="Neplătite"  value={stats.unpaid_count}     accentColor="#dc2626" urgent={stats.unpaid_count > 0} />
        </div>
      </motion.div>

      {/* Filter + list */}
      <div className="min-h-0 flex-1 overflow-y-auto bg-background" style={{ padding: "14px 24px" }}>
        <div className="flex items-center justify-between mb-3">
          <span className="font-semibold text-foreground text-sm">
            Detectări ({filtered.length})
          </span>
          <div className="flex gap-1.5">
            {FILTERS.map((f) => (
              <button
                key={f.id}
                type="button"
                onClick={() => setFilter(f.id)}
                className={`rounded-lg border px-3 py-1 text-xs cursor-pointer transition-colors ${
                  filter === f.id
                    ? "border-primary bg-primary/10 text-primary font-semibold"
                    : "border-border bg-card text-muted-foreground hover:text-foreground"
                }`}
              >
                {f.label}
              </button>
            ))}
          </div>
        </div>

        {filtered.length === 0 ? (
          <div className="flex h-48 items-center justify-center text-sm text-muted-foreground">
            Nu s-au găsit detectări
          </div>
        ) : (
          <div className="space-y-2">
            {filtered.map((detection, index) => (
              <DetectionCard key={detection.id} detection={detection} index={index} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

interface FlatStatCardProps {
  label: string;
  value: number;
  accentColor: string;
  urgent?: boolean;
}

function FlatStatCard({ label, value, accentColor, urgent = false }: FlatStatCardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.97 }}
      animate={{ opacity: 1, scale: 1 }}
      className="relative rounded-lg border border-border bg-card shadow-sm"
      style={{ borderTop: `3px solid ${accentColor}`, padding: "14px 18px" }}
    >
      <div className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground mb-1.5">{label}</div>
      <div className="font-mono text-2xl font-bold text-foreground">{value}</div>
      {urgent && value > 0 && (
        <motion.div
          className="absolute right-2 top-2 h-2 w-2 rounded-full bg-destructive shadow"
          animate={{ scale: [1, 1.3, 1] }}
          transition={{ duration: 1.5, repeat: Infinity }}
        />
      )}
    </motion.div>
  );
}
