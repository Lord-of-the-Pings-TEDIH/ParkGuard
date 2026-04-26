import { motion } from "motion/react";
import { ImageOff } from "lucide-react";
import { RomanianPlate } from "./RomanianPlate";
import { SpotMatchBadge } from "./SpotMatchBadge";
import { TicketStatusBadge } from "./TicketStatusBadge";
import type { Detection } from "../types";
import { useState } from "react";

interface DetectionCardProps {
  detection: Detection;
  index: number;
}

export function DetectionCard({ detection, index }: DetectionCardProps) {
  const [imageError, setImageError] = useState(false);
  const isUnpaid = detection.ticket_status === "none";
  const isWrongSpot = detection.spot_match_status === "WRONG_PLATE";
  const highlightAlert = isUnpaid || isWrongSpot;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, delay: index * 0.05 }}
      className={`relative rounded-lg border p-3 shadow-md transition-all md:p-4 ${
        highlightAlert
          ? "border-red-400 bg-gradient-to-br from-red-50 to-orange-50 dark:border-red-600 dark:from-red-950 dark:to-orange-950"
          : "border-border bg-card"
      }`}
    >
      <div className="flex flex-col gap-3 md:flex-row md:gap-4">
        <div className="flex-shrink-0">
          {imageError ? (
            <div className="flex h-20 w-full items-center justify-center rounded border border-border bg-muted md:h-20 md:w-32">
              <ImageOff className="h-6 w-6 text-muted-foreground" />
            </div>
          ) : (
            <img
              src={
                detection.crop_image_url.startsWith('http') || detection.crop_image_url.startsWith('/')
                  ? detection.crop_image_url
                  : `/api/crops/${detection.crop_image_url}`
              }
              alt={detection.ocr_normalized_text}
              className="h-20 w-full rounded border border-border object-cover shadow-sm md:w-32"
              onError={() => setImageError(true)}
            />
          )}
        </div>

        <div className="flex flex-1 flex-col justify-between gap-2">
          <div className="flex items-start justify-between gap-4">
            <RomanianPlate text={detection.ocr_normalized_text} />
            <div className="flex flex-col items-end gap-1.5">
              <TicketStatusBadge
                status={detection.ticket_status}
                expiresAt={detection.ticket_expires_at}
              />
              {detection.spot_match_status && (
                <SpotMatchBadge
                  status={detection.spot_match_status}
                  distanceM={detection.target_distance_m}
                />
              )}
            </div>
          </div>

          <div className="mt-2 flex flex-wrap items-center gap-3 text-xs text-muted-foreground">
            <span>Confidence: {(detection.detection_confidence * 100).toFixed(1)}%</span>
            <span className="uppercase tracking-wide">
              {detection.voting_tag === "final" ? "Final" : "Not final"}
            </span>
            <span>Seen: {detection.occurrences}x</span>
            {detection.ocr_raw_text !== detection.ocr_normalized_text && (
              <span className="italic">Raw: {detection.ocr_raw_text}</span>
            )}
          </div>
          {detection.target_latitude != null && detection.target_longitude != null && (
            <div className="mt-1 text-[10px] text-muted-foreground font-mono">
              GPS: {detection.target_latitude.toFixed(6)}, {detection.target_longitude.toFixed(6)}
            </div>
          )}
          <div className="mt-1 text-[10px] text-muted-foreground font-mono">
            Annotation: {detection.plate_annotation}
          </div>
        </div>
      </div>

      {highlightAlert && (
        <motion.div
          className="absolute -left-1 -top-1 h-2 w-2 rounded-full bg-gradient-to-r from-red-500 to-orange-500 shadow-lg"
          animate={{ scale: [1, 1.3, 1] }}
          transition={{ duration: 1.5, repeat: Infinity }}
        />
      )}
    </motion.div>
  );
}
