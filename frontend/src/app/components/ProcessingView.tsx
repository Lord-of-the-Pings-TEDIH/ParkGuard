import { useEffect, useMemo, useState } from "react";
import { motion } from "motion/react";
import { XCircle, CheckCircle, AlertCircle } from "lucide-react";
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
      (d) => d.is_valid_ro_plate && !d.ocr_normalized_text.toUpperCase().startsWith("INVALID:"),
    );
    return dedupeLiveDetections(validOnly);
  }, [detections]);

  useEffect(() => {
    const interval = setInterval(() => setElapsed((p) => p + 1), 1000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (session.status === "done") setTimeout(onComplete, 1000);
  }, [session.status, onComplete]);

  const progress = session.frames_total > 0
    ? (session.frames_processed / session.frames_total) * 100
    : 0;

  const stats = calculateStats(uniqueLiveDetections);

  return (
    <div className="flex h-full min-h-0 flex-col">
      {/* Header */}
      <div className="border-b border-border bg-card flex-shrink-0" style={{ padding: "16px 24px" }}>
        <div className="flex items-start justify-between gap-3 mb-4">
          <div className="min-w-0">
            <div className="font-bold text-foreground truncate" style={{ fontSize: 16, marginBottom: 3 }}>
              {session.source_filename}
            </div>
            <div className="text-xs text-muted-foreground">
              {session.frames_processed}/{session.frames_total} cadre procesate · {formatElapsedTime(session.created_at)} elapsed
            </div>
          </div>
          <div className="flex items-center gap-2 flex-shrink-0">
            {session.status === "done" && <CheckCircle className="h-4 w-4 text-green-500" />}
            {session.status === "failed" && <AlertCircle className="h-4 w-4 text-destructive" />}
            {session.status === "running" && (
              <button
                type="button"
                onClick={onCancel}
                className="flex items-center gap-1.5 rounded-lg border border-destructive/40 bg-destructive/10 text-destructive px-3 py-1.5 text-xs font-semibold cursor-pointer hover:bg-destructive/20 transition-colors"
              >
                <XCircle className="h-3.5 w-3.5" />
                Anulează
              </button>
            )}
          </div>
        </div>

        {/* Progress bar */}
        <div className="rounded-full overflow-hidden bg-border mb-4" style={{ height: 5 }}>
          <div
            className="h-full rounded-full bg-primary transition-all duration-500"
            style={{ width: `${Math.min(100, progress)}%` }}
          />
        </div>

        {/* Stat cards */}
        <div className="grid gap-2.5" style={{ gridTemplateColumns: "repeat(5, 1fr)" }}>
          <MiniStat label="Cadre" value={`${session.frames_processed}/${session.frames_total}`} accent="text-primary" />
          <MiniStat label="Timp" value={`${elapsed}s`} accent="text-muted-foreground" />
          <MiniStat label="Găsite" value={String(stats.total_detections)} accent="text-primary" />
          <MiniStat label="Unice" value={String(stats.unique_plates)} accent="text-purple-600 dark:text-purple-400" />
          <MiniStat label="Neplătite" value={String(stats.unpaid_count)} accent="text-destructive" />
        </div>

        {session.error_message && (
          <div className="mt-3 rounded-lg border border-destructive/40 bg-destructive/10 p-3 text-xs text-destructive">
            <AlertCircle className="mr-1.5 inline h-3.5 w-3.5" />
            {session.error_message}
          </div>
        )}
      </div>

      {/* Live detection feed */}
      <div className="min-h-0 flex-1 overflow-y-auto bg-background" style={{ padding: "14px 24px" }}>
        <div className="flex items-center justify-between mb-3">
          <span className="font-semibold text-foreground text-sm">Detectări live</span>
          {session.status === "running" && (
            <motion.div
              className="flex items-center gap-1.5 rounded-full bg-primary px-2.5 py-1 text-xs font-semibold text-white"
              animate={{ opacity: [1, 0.5, 1] }}
              transition={{ duration: 1.5, repeat: Infinity }}
            >
              <div className="h-1.5 w-1.5 rounded-full bg-white/80" />
              În direct
            </motion.div>
          )}
        </div>

        {uniqueLiveDetections.length === 0 ? (
          <div className="flex h-48 items-center justify-center text-sm text-muted-foreground">
            Se așteaptă detectări valide...
          </div>
        ) : (
          <div className="space-y-2">
            {uniqueLiveDetections.map((detection, index) => (
              <DetectionCard key={detection.id} detection={detection} index={index} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function MiniStat({ label, value, accent }: { label: string; value: string; accent: string }) {
  return (
    <div className="rounded-lg border border-border bg-muted/50 px-3 py-2.5">
      <div className="text-[10px] uppercase tracking-widest text-muted-foreground font-medium mb-1">{label}</div>
      <div className={`font-mono font-bold text-sm ${accent}`}>{value}</div>
    </div>
  );
}

// ── dedup helpers (unchanged logic) ──────────────────────────────────────────

function dedupeLiveDetections(detections: Detection[]): Detection[] {
  const groups = new Map<string, GroupAggregate>();
  const sorted = [...detections].sort(
    (a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime(),
  );

  for (const detection of sorted) {
    const parsed = parsePlateLike(detection.ocr_normalized_text) ?? parsePlateLike(detection.ocr_raw_text);
    const groupKey = parsed ? `plate:${parsed.anchor}` : `id:${detection.id}`;
    const variantKey = parsed ? parsed.compact : compactText(detection.ocr_normalized_text || detection.ocr_raw_text) || detection.id;

    const aggregate = groups.get(groupKey) ?? { variants: new Map() };
    const current = aggregate.variants.get(variantKey);
    const score = variantScore(detection);

    if (!current) {
      aggregate.variants.set(variantKey, { count: 1, bestDetection: detection, bestScore: score });
      groups.set(groupKey, aggregate);
      continue;
    }
    current.count += 1;
    if (score > current.bestScore) { current.bestScore = score; current.bestDetection = detection; }
  }

  const deduped: Detection[] = [];
  for (const group of groups.values()) {
    const variants = Array.from(group.variants.values());
    variants.sort((a, b) => {
      if (b.count !== a.count) return b.count - a.count;
      if (b.bestDetection.is_valid_ro_plate !== a.bestDetection.is_valid_ro_plate)
        return b.bestDetection.is_valid_ro_plate ? 1 : -1;
      if (b.bestDetection.detection_confidence !== a.bestDetection.detection_confidence)
        return b.bestDetection.detection_confidence - a.bestDetection.detection_confidence;
      return new Date(b.bestDetection.created_at).getTime() - new Date(a.bestDetection.created_at).getTime();
    });
    deduped.push(variants[0].bestDetection);
  }
  return deduped.sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime());
}

type ParsedPlate = { compact: string; anchor: string };
type GroupVariant = { count: number; bestDetection: Detection; bestScore: number };
type GroupAggregate = { variants: Map<string, GroupVariant> };

const DIGIT_TO_LETTER: Record<string, string> = { "0":"O","1":"I","2":"Z","3":"B","4":"A","5":"S","6":"G","7":"T","8":"B","9":"P" };
const LETTER_TO_DIGIT: Record<string, string> = { O:"0",Q:"0",I:"1",Z:"2",S:"5",G:"6",T:"7",B:"8",A:"4" };

function compactText(text: string): string { return text.toUpperCase().replace(/[^A-Z0-9]/g, ""); }
function normalizeLetters(text: string): string { return text.split("").map((c) => DIGIT_TO_LETTER[c] ?? c).join(""); }
function normalizeDigits(text: string): string { return text.split("").map((c) => LETTER_TO_DIGIT[c] ?? c).join(""); }

function parsePlateLike(rawText: string): ParsedPlate | null {
  const clean = compactText(rawText || "");
  if (!clean) return null;
  if (clean.startsWith("B") || clean.startsWith("8")) {
    const body = clean.slice(1);
    const tmpDigits = normalizeDigits(body);
    if (/^\d{3,}$/.test(tmpDigits)) return { compact: `B${tmpDigits}`, anchor: `B${tmpDigits}` };
    if (body.length < 5) return null;
    const stdDigits = normalizeDigits(body.slice(0, -3));
    const stdLetters = normalizeLetters(body.slice(-3));
    if (!/^\d{2,3}$/.test(stdDigits) || !/^[A-Z]{3}$/.test(stdLetters)) return null;
    return { compact: `B${stdDigits}${stdLetters}`, anchor: `B${stdDigits}` };
  }
  if (clean.length < 4) return null;
  const county = normalizeLetters(clean.slice(0, 2));
  if (!/^[A-Z]{2}$/.test(county)) return null;
  const body = clean.slice(2);
  const tmpDigits = normalizeDigits(body);
  if (/^\d{3,}$/.test(tmpDigits)) return { compact: `${county}${tmpDigits}`, anchor: `${county}${tmpDigits}` };
  if (body.length < 5) return null;
  const stdDigits = normalizeDigits(body.slice(0, 2));
  const stdLetters = normalizeLetters(body.slice(-3));
  if (!/^\d{2}$/.test(stdDigits) || !/^[A-Z]{3}$/.test(stdLetters)) return null;
  return { compact: `${county}${stdDigits}${stdLetters}`, anchor: `${county}${stdDigits}` };
}

function variantScore(detection: Detection): number {
  return (
    detection.detection_confidence +
    (detection.is_valid_ro_plate ? 1 : 0) +
    (detection.ticket_status !== "unknown" ? 0.2 : 0)
  );
}
