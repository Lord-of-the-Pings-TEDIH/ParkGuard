import { Trash2, PlayCircle, Loader2 } from "lucide-react";
import { motion } from "motion/react";
import { Button } from "./ui/button";
import { formatRelativeTime } from "../utils/format";
import type { Session } from "../types";

interface SessionHistoryProps {
  sessions: Session[];
  testFiles: string[];
  activeSessionId: string | null;
  pendingSessionId: string | null;
  pendingTestFile: string | null;
  onSelectSession: (id: string) => void;
  onDeleteSession: (id: string) => void;
  onRunHardcodedTest: (filename: string) => void;
}

export function SessionHistory({
  sessions,
  testFiles,
  activeSessionId,
  pendingSessionId,
  pendingTestFile,
  onSelectSession,
  onDeleteSession,
  onRunHardcodedTest,
}: SessionHistoryProps) {
  return (
    <div className="flex h-full min-h-0 flex-col border-border bg-card lg:border-l">
      <div className="border-b border-border bg-gradient-to-r from-purple-50 to-pink-50 p-3 dark:from-purple-950 dark:to-pink-950 md:p-4">
        <h3 className="font-medium text-foreground">Sessions</h3>
        <p className="text-xs text-muted-foreground">{sessions.length} procesări</p>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto">
        <div className="space-y-2 p-3">
          {testFiles.length > 0 && (
            <>
              <div className="mb-2 text-xs font-medium uppercase tracking-wide text-muted-foreground">
                Uploads hardcodate
              </div>
              {testFiles.map((filename) => (
                <button
                  key={filename}
                  type="button"
                  onClick={() => onRunHardcodedTest(filename)}
                  disabled={pendingTestFile !== null}
                  className={`w-full cursor-pointer rounded-lg border px-3 py-2 text-left text-sm font-medium transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-amber-500 active:scale-[0.99] ${
                    pendingTestFile === filename
                      ? "border-amber-500 bg-gradient-to-r from-amber-100 to-orange-100 text-amber-900 dark:border-amber-500 dark:from-amber-900 dark:to-orange-900 dark:text-amber-200"
                      : "border-amber-300 bg-gradient-to-r from-amber-50 to-orange-50 text-amber-800 hover:from-amber-100 hover:to-orange-100 dark:border-amber-700 dark:from-amber-950 dark:to-orange-950 dark:text-amber-300"
                  } disabled:cursor-not-allowed disabled:opacity-70`}
                  title={`Run ${filename}`}
                >
                  <div className="flex items-center gap-2">
                    {pendingTestFile === filename ? (
                      <Loader2 className="h-4 w-4 animate-spin" />
                    ) : (
                      <PlayCircle className="h-4 w-4" />
                    )}
                    <span className="truncate">{filename}</span>
                  </div>
                </button>
              ))}
              <div className="my-3 h-px bg-border" />
            </>
          )}

          {sessions.map((session, index) => {
            const isActive = session.id === activeSessionId;
            const isRunning = session.status === "running";
            const isPendingSelection = session.id === pendingSessionId;

            return (
              <motion.div
                key={session.id}
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: Math.min(index * 0.02, 0.12) }}
                className={`group relative cursor-pointer rounded-lg border p-3 shadow-sm transition-all hover:shadow-md ${
                  isPendingSelection
                    ? "border-blue-500 bg-gradient-to-br from-blue-100 to-indigo-100 dark:border-blue-500 dark:from-blue-900 dark:to-indigo-900"
                    : isActive
                    ? "border-blue-500 bg-gradient-to-br from-blue-50 to-indigo-50 dark:border-blue-600 dark:from-blue-950 dark:to-indigo-950"
                    : isRunning
                    ? "border-amber-500 bg-gradient-to-br from-amber-50 to-orange-50 dark:border-amber-600 dark:from-amber-950 dark:to-orange-950"
                    : "border-border bg-muted"
                }`}
                onClick={() => {
                  if (!isPendingSelection) {
                    onSelectSession(session.id);
                  }
                }}
              >
                {isRunning && (
                  <motion.div
                    className="absolute -left-1 top-1/2 h-2 w-2 -translate-y-1/2 rounded-full bg-gradient-to-r from-amber-500 to-orange-500"
                    animate={{ scale: [1, 1.3, 1] }}
                    transition={{ duration: 1.5, repeat: Infinity }}
                  />
                )}

                <div className="mb-2 flex items-start justify-between gap-2">
                  <div className="flex-1 truncate text-sm font-medium text-foreground">
                    {session.source_filename}
                  </div>
                  <Button
                    variant="ghost"
                    size="sm"
                    className="h-6 w-6 p-0 opacity-100 transition-opacity md:opacity-0 md:group-hover:opacity-100"
                    onClick={(e) => {
                      e.stopPropagation();
                      onDeleteSession(session.id);
                    }}
                    title="Delete session"
                  >
                    <Trash2 className="h-3 w-3" />
                  </Button>
                </div>

                <div className="mb-1 flex items-center gap-2">
                  <StatusBadge status={session.status} />
                  {session.status === "running" && (
                    <PlayCircle className="h-3 w-3 text-amber-600 dark:text-amber-400" />
                  )}
                  {isPendingSelection && (
                    <span className="inline-flex items-center gap-1 text-[10px] font-medium text-blue-700 dark:text-blue-300">
                      <Loader2 className="h-3 w-3 animate-spin" />
                      Loading
                    </span>
                  )}
                </div>

                <div className="grid grid-cols-2 gap-2 text-xs text-muted-foreground">
                  <div>Frames: {session.frames_processed}</div>
                  <div className="text-right">{formatRelativeTime(session.created_at)}</div>
                </div>
              </motion.div>
            );
          })}

          {sessions.length === 0 && (
            <div className="py-12 text-center text-sm text-slate-500">
              No sessions yet
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function StatusBadge({ status }: { status: Session["status"] }) {
  const config = {
    pending: { label: "Pending", className: "bg-gradient-to-r from-slate-100 to-gray-100 text-slate-700 dark:from-slate-800 dark:to-gray-800 dark:text-slate-300" },
    running: { label: "Running", className: "bg-gradient-to-r from-amber-100 to-orange-100 text-amber-700 dark:from-amber-900 dark:to-orange-900 dark:text-amber-300" },
    done: { label: "Done", className: "bg-gradient-to-r from-green-100 to-emerald-100 text-green-700 dark:from-green-900 dark:to-emerald-900 dark:text-green-300" },
    failed: { label: "Failed", className: "bg-gradient-to-r from-red-100 to-orange-100 text-red-700 dark:from-red-900 dark:to-orange-900 dark:text-red-300" },
  };

  const { label, className } = config[status];

  return (
    <span className={`rounded px-1.5 py-0.5 text-xs font-medium ${className}`}>
      {label}
    </span>
  );
}
