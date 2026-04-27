import { motion } from "motion/react";
import { RotateCcw, CheckCircle, Clock } from "lucide-react";
import { Button } from "./ui/button";
import { DetectionCard } from "./DetectionCard";
import { calculateStats, formatDuration } from "../utils/format";
import type { Session, Detection } from "../types";

interface ResultsViewProps {
  session: Session;
  detections: Detection[];
  onReset: () => void;
}

export function ResultsView({ session, detections, onReset }: ResultsViewProps) {
  const stats = calculateStats(detections);
  const duration = formatDuration(session.created_at, session.ended_at);

  return (
    <div className="flex h-full min-h-0 flex-col">
      {/* Header with stats */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="border-b border-border bg-card p-4 shadow-sm md:p-6"
      >
        <div className="mb-4 flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
          <div>
            <div className="flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-full bg-gradient-to-br from-green-500 to-emerald-600">
                <CheckCircle className="h-5 w-5 text-white" />
              </div>
              <div className="min-w-0 flex-1">
                <h2 className="truncate text-lg font-bold text-foreground md:text-xl">
                  {session.source_filename}
                </h2>
                <p className="flex items-center gap-1.5 text-xs text-muted-foreground md:text-sm">
                  <span>Complete</span>
                  {duration && (
                    <>
                      <span aria-hidden>·</span>
                      <span className="inline-flex items-center gap-1">
                        <Clock className="h-3 w-3" />
                        {duration}
                      </span>
                    </>
                  )}
                </p>
              </div>
            </div>
          </div>
          <Button
            variant="outline"
            onClick={onReset}
            className="border-blue-500 bg-gradient-to-r from-blue-50 to-indigo-50 text-blue-700 hover:from-blue-100 hover:to-indigo-100 dark:from-blue-950 dark:to-indigo-950 dark:text-blue-400"
          >
            <RotateCcw className="mr-2 h-4 w-4" />
            Nou
          </Button>
        </div>

        <div className="grid grid-cols-2 gap-2 md:grid-cols-4 md:gap-3">
          <StatCard
            label="Găsite"
            value={stats.total_detections}
            className="border-blue-200 bg-gradient-to-br from-blue-50 to-indigo-50 text-blue-700 dark:border-blue-800 dark:from-blue-950 dark:to-indigo-950 dark:text-blue-300"
          />
          <StatCard
            label="Unice"
            value={stats.unique_plates}
            className="border-purple-200 bg-gradient-to-br from-purple-50 to-pink-50 text-purple-700 dark:border-purple-800 dark:from-purple-950 dark:to-pink-950 dark:text-purple-300"
          />
          <StatCard
            label="Plătite"
            value={stats.active_count}
            className="border-green-200 bg-gradient-to-br from-green-50 to-emerald-50 text-green-700 dark:border-green-800 dark:from-green-950 dark:to-emerald-950 dark:text-green-300"
          />
          <StatCard
            label="Neplătite"
            value={stats.unpaid_count}
            className="border-red-200 bg-gradient-to-br from-red-50 to-orange-50 text-red-700 dark:border-red-800 dark:from-red-950 dark:to-orange-950 dark:text-red-300"
            urgent={stats.unpaid_count > 0}
          />
        </div>
      </motion.div>

      {/* Detection results */}
      <div className="min-h-0 flex-1 overflow-y-auto bg-background p-4 md:p-6">
        <div className="mb-4 flex flex-wrap items-center justify-between gap-2">
          <h3 className="font-medium text-foreground">Detectări</h3>
          <span className="rounded-full bg-gradient-to-r from-blue-100 to-indigo-100 px-3 py-1 text-sm font-medium text-blue-700 dark:from-blue-900 dark:to-indigo-900 dark:text-blue-300">
            {detections.length}
          </span>
        </div>

        {detections.length === 0 ? (
          <div className="flex h-64 items-center justify-center text-slate-500">
            Nu s-au găsit detectări
          </div>
        ) : (
          <div className="space-y-3">
            {detections.map((detection, index) => (
              <DetectionCard
                key={detection.id}
                detection={detection}
                index={index}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

interface StatCardProps {
  label: string;
  value: number;
  className?: string;
  urgent?: boolean;
}

function StatCard({ label, value, className = "", urgent = false }: StatCardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      className={`relative rounded-lg border p-3 shadow-lg md:p-4 ${className}`}
    >
      <div className="mb-1 text-xs opacity-75">{label}</div>
      <div className="font-mono text-xl font-bold md:text-2xl">{value}</div>
      {urgent && value > 0 && (
        <motion.div
          className="absolute right-2 top-2 h-2 w-2 rounded-full bg-gradient-to-r from-red-500 to-orange-500 shadow-lg"
          animate={{ scale: [1, 1.3, 1] }}
          transition={{ duration: 1.5, repeat: Infinity }}
        />
      )}
    </motion.div>
  );
}
