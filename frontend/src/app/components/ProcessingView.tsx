import { useEffect, useMemo, useState } from "react";
import { motion } from "motion/react";
import { XCircle, CheckCircle, AlertCircle } from "lucide-react";
import { Button } from "./ui/button";
import { Progress } from "./ui/progress";
import { DetectionCard } from "./DetectionCard";
import { formatElapsedTime, calculateStats } from "../utils/format";
import type { Session, Detection } from "../types";

interface ProcessingViewProps {
  session: Session;
  detections: Detection[];
  onCancel: () => void;
  onComplete: () => void;
}

export function ProcessingView({ session, detections, onCancel, onComplete }: ProcessingViewProps) {
  const [elapsed, setElapsed] = useState(0);
  const uniqueLiveDetections = useMemo(() => {
    const validOnly = detections.filter(
      (detection) =>
        detection.is_valid_ro_plate &&
        !detection.ocr_normalized_text.toUpperCase().startsWith("INVALID:")
    );
    return dedupeLiveDetections(validOnly);
  }, [detections]);

  useEffect(() => {
    const interval = setInterval(() => {
      setElapsed(prev => prev + 1);
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (session.status === "done") {
      setTimeout(onComplete, 1000);
    }
  }, [session.status, onComplete]);

  const progress = session.frames_total > 0
    ? (session.frames_processed / session.frames_total) * 100
    : 0;

  const stats = calculateStats(uniqueLiveDetections);

  const getStatusIcon = () => {
    switch (session.status) {
      case "done":
        return <CheckCircle className="h-5 w-5 text-green-500" />;
      case "failed":
        return <AlertCircle className="h-5 w-5 text-red-500" />;
      default:
        return null;
    }
  };

  return (
    <div className="flex h-full min-h-0 flex-col">
      {/* Header with progress */}
      <div className="border-b border-border bg-card p-4 shadow-sm md:p-6">
        <div className="mb-4 flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
          <div className="min-w-0 flex-1">
            <h2 className="truncate text-lg font-bold text-foreground md:text-xl">
              {session.source_filename}
            </h2>
          </div>
          <div className="flex items-center gap-3">
            {getStatusIcon()}
            {session.status === "running" && (
              <Button
                variant="outline"
                size="sm"
                onClick={onCancel}
                className="border-red-500 bg-gradient-to-r from-red-50 to-orange-50 text-red-700 hover:from-red-100 hover:to-orange-100 dark:from-red-950 dark:to-orange-950 dark:text-red-400"
              >
                <XCircle className="mr-2 h-4 w-4" />
                Anulează
              </Button>
            )}
          </div>
        </div>

        <Progress value={progress} className="mb-3 h-2" />

        <div className="grid grid-cols-2 gap-2 text-sm md:grid-cols-5 md:gap-3">
          <div className="rounded-lg border border-purple-200 bg-gradient-to-br from-purple-50 to-pink-50 p-2.5 dark:border-purple-800 dark:from-purple-950 dark:to-pink-950 md:p-3">
            <div className="mb-1 text-xs text-purple-700 dark:text-purple-300">Cadre</div>
            <div className="font-mono text-sm font-semibold text-purple-900 dark:text-purple-100 md:text-base">
              {session.frames_processed}/{session.frames_total}
            </div>
          </div>
          <div className="rounded-lg border border-cyan-200 bg-gradient-to-br from-cyan-50 to-blue-50 p-2.5 dark:border-cyan-800 dark:from-cyan-950 dark:to-blue-950 md:p-3">
            <div className="mb-1 text-xs text-cyan-700 dark:text-cyan-300">Timp</div>
            <div className="font-mono text-sm font-semibold text-cyan-900 dark:text-cyan-100 md:text-base">
              {formatElapsedTime(session.created_at)}
            </div>
          </div>
          <div className="rounded-lg border border-blue-200 bg-gradient-to-br from-blue-50 to-indigo-50 p-2.5 dark:border-blue-800 dark:from-blue-950 dark:to-indigo-950 md:p-3">
            <div className="mb-1 text-xs text-blue-700 dark:text-blue-300">Găsite</div>
            <div className="font-mono text-sm font-semibold text-blue-900 dark:text-blue-100 md:text-base">
              {stats.total_detections}
            </div>
          </div>
          <div className="rounded-lg border border-emerald-200 bg-gradient-to-br from-emerald-50 to-teal-50 p-2.5 dark:border-emerald-800 dark:from-emerald-950 dark:to-teal-950 md:p-3">
            <div className="mb-1 text-xs text-emerald-700 dark:text-emerald-300">Unice</div>
            <div className="font-mono text-sm font-semibold text-emerald-900 dark:text-emerald-100 md:text-base">
              {stats.unique_plates}
            </div>
          </div>
          <div className="col-span-2 rounded-lg border border-red-200 bg-gradient-to-br from-red-50 to-orange-50 p-2.5 dark:border-red-800 dark:from-red-950 dark:to-orange-950 md:col-span-1 md:p-3">
            <div className="mb-1 text-xs text-red-700 dark:text-red-300">Neplătite</div>
            <div className="font-mono text-sm font-semibold text-red-900 dark:text-red-100 md:text-base">
              {stats.unpaid_count}
            </div>
          </div>
        </div>

        {session.error_message && (
          <div className="mt-4 rounded-lg border border-red-400 bg-gradient-to-r from-red-50 to-orange-50 p-3 text-sm text-red-700 dark:border-red-600 dark:from-red-950 dark:to-orange-950 dark:text-red-300">
            <AlertCircle className="mr-2 inline h-4 w-4" />
            {session.error_message}
          </div>
        )}
      </div>

      {/* Live detection feed */}
      <div className="min-h-0 flex-1 overflow-y-auto bg-background p-4 md:p-6">
        <div className="mb-4 flex flex-wrap items-center justify-between gap-2">
          <h3 className="font-medium text-foreground">
            Detectări
          </h3>
          {session.status === "running" && (
            <motion.div
              className="flex items-center gap-2 rounded-full bg-gradient-to-r from-blue-500 to-indigo-600 px-3 py-1 text-sm text-white shadow-lg"
              animate={{ opacity: [1, 0.5, 1] }}
              transition={{ duration: 1.5, repeat: Infinity }}
            >
              <div className="h-2 w-2 rounded-full bg-white" />
              În direct
            </motion.div>
          )}
        </div>

        {uniqueLiveDetections.length === 0 ? (
          <div className="flex h-64 items-center justify-center text-slate-500">
            Se așteaptă detectări valide...
          </div>
        ) : (
          <div className="space-y-3">
            {uniqueLiveDetections.map((detection, index) => (
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

function dedupeLiveDetections(detections: Detection[]): Detection[] {
  const groups = new Map<string, GroupAggregate>();
  const sorted = [...detections].sort(
    (a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
  );

  for (const detection of sorted) {
    const parsed = parsePlateLike(detection.ocr_normalized_text) ?? parsePlateLike(detection.ocr_raw_text);
    const groupKey = parsed ? `plate:${parsed.anchor}` : `id:${detection.id}`;
    const variantKey = parsed ? parsed.compact : compactText(detection.ocr_normalized_text || detection.ocr_raw_text) || detection.id;

    const aggregate = groups.get(groupKey) ?? { variants: new Map() };
    const current = aggregate.variants.get(variantKey);
    const score = variantScore(detection);

    if (!current) {
      aggregate.variants.set(variantKey, {
        count: 1,
        bestDetection: detection,
        bestScore: score,
      });
      groups.set(groupKey, aggregate);
      continue;
    }

    current.count += 1;
    if (score > current.bestScore) {
      current.bestScore = score;
      current.bestDetection = detection;
    }
  }

  const deduped: Detection[] = [];
  for (const group of groups.values()) {
    const variants = Array.from(group.variants.values());
    variants.sort((a, b) => {
      if (b.count !== a.count) return b.count - a.count;
      if (b.bestDetection.is_valid_ro_plate !== a.bestDetection.is_valid_ro_plate) {
        return b.bestDetection.is_valid_ro_plate ? 1 : -1;
      }
      if (b.bestDetection.detection_confidence !== a.bestDetection.detection_confidence) {
        return b.bestDetection.detection_confidence - a.bestDetection.detection_confidence;
      }
      return new Date(b.bestDetection.created_at).getTime() - new Date(a.bestDetection.created_at).getTime();
    });
    deduped.push(variants[0].bestDetection);
  }

  return deduped.sort(
    (a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
  );
}

type ParsedPlate = {
  compact: string;
  anchor: string;
};

type GroupVariant = {
  count: number;
  bestDetection: Detection;
  bestScore: number;
};

type GroupAggregate = {
  variants: Map<string, GroupVariant>;
};

const DIGIT_TO_LETTER: Record<string, string> = {
  "0": "O",
  "1": "I",
  "2": "Z",
  "3": "B",
  "4": "A",
  "5": "S",
  "6": "G",
  "7": "T",
  "8": "B",
  "9": "P",
};

const LETTER_TO_DIGIT: Record<string, string> = {
  O: "0",
  Q: "0",
  I: "1",
  Z: "2",
  S: "5",
  G: "6",
  T: "7",
  B: "8",
  A: "4",
};

function compactText(text: string): string {
  return text.toUpperCase().replace(/[^A-Z0-9]/g, "");
}

function normalizeLetters(text: string): string {
  return text
    .split("")
    .map((char) => DIGIT_TO_LETTER[char] ?? char)
    .join("");
}

function normalizeDigits(text: string): string {
  return text
    .split("")
    .map((char) => LETTER_TO_DIGIT[char] ?? char)
    .join("");
}

function parsePlateLike(rawText: string): ParsedPlate | null {
  const clean = compactText(rawText || "");
  if (!clean) return null;

  if (clean.startsWith("B") || clean.startsWith("8")) {
    const body = clean.slice(1);
    const temporaryDigits = normalizeDigits(body);
    if (/^\d{3,}$/.test(temporaryDigits)) {
      return { compact: `B${temporaryDigits}`, anchor: `B${temporaryDigits}` };
    }
    if (body.length < 5) return null;

    const standardDigits = normalizeDigits(body.slice(0, -3));
    const standardLetters = normalizeLetters(body.slice(-3));
    if (!/^\d{2,3}$/.test(standardDigits) || !/^[A-Z]{3}$/.test(standardLetters)) {
      return null;
    }
    return {
      compact: `B${standardDigits}${standardLetters}`,
      anchor: `B${standardDigits}`,
    };
  }

  if (clean.length < 4) return null;
  const county = normalizeLetters(clean.slice(0, 2));
  if (!/^[A-Z]{2}$/.test(county)) {
    return null;
  }

  const body = clean.slice(2);
  const temporaryDigits = normalizeDigits(body);
  if (/^\d{3,}$/.test(temporaryDigits)) {
    return { compact: `${county}${temporaryDigits}`, anchor: `${county}${temporaryDigits}` };
  }

  if (body.length < 5) return null;
  const standardDigits = normalizeDigits(body.slice(0, 2));
  const standardLetters = normalizeLetters(body.slice(-3));
  if (!/^\d{2}$/.test(standardDigits) || !/^[A-Z]{3}$/.test(standardLetters)) {
    return null;
  }

  return {
    compact: `${county}${standardDigits}${standardLetters}`,
    anchor: `${county}${standardDigits}`,
  };
}

function variantScore(detection: Detection): number {
  return (
    detection.detection_confidence +
    (detection.is_valid_ro_plate ? 1 : 0) +
    (detection.ticket_status !== "unknown" ? 0.2 : 0)
  );
}
