import { AnimatePresence, motion } from "motion/react";
import { ImageOff } from "lucide-react";
import { RomanianPlate } from "./RomanianPlate";
import { SpotMatchBadge } from "./SpotMatchBadge";
import { TicketStatusBadge } from "./TicketStatusBadge";
import type { Detection } from "../types";
import { useEffect, useState } from "react";

interface DetectionCardProps {
  detection: Detection;
  index: number;
}

export function DetectionCard({ detection, index }: DetectionCardProps) {
  const [imageError, setImageError] = useState(false);
  const [isExpanded, setIsExpanded] = useState(false);
  const isUnpaid = detection.ticket_status === "none";
  const isWrongSpot = detection.spot_match_status === "WRONG_PLATE";
  const highlightAlert = isUnpaid || isWrongSpot;

  const cropSrc =
    detection.crop_image_url.startsWith("http") ||
    detection.crop_image_url.startsWith("/")
      ? detection.crop_image_url
      : `/api/crops/${detection.crop_image_url}`;

  // Esc closes the lightbox.  motion's layoutId handles the bounce-back to the
  // card on close, so we only need to flip the open flag.
  useEffect(() => {
    if (!isExpanded) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setIsExpanded(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [isExpanded]);

  // Stable layoutId so motion animates the same crop between its card slot
  // and the fullscreen lightbox.  Falls back to id when annotation is missing.
  const layoutId = `crop-${detection.id}`;

  return (
    <>
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
            <motion.img
              layoutId={layoutId}
              src={cropSrc}
              alt={detection.ocr_normalized_text}
              className="h-20 w-full cursor-zoom-in rounded border border-border object-cover shadow-sm transition-shadow hover:shadow-lg md:w-32"
              onError={() => setImageError(true)}
              onClick={() => setIsExpanded(true)}
              role="button"
              tabIndex={0}
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") {
                  e.preventDefault();
                  setIsExpanded(true);
                }
              }}
              // Hide the in-card slot while the lightbox owns the shared
              // element — without this the small thumbnail would briefly
              // double up beside the expanded one mid-transition.
              style={isExpanded ? { visibility: "hidden" } : undefined}
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

    {/* Lightbox: shared layoutId on the <img> animates the crop from its
        slot in the card to a centred fullscreen view (and back when closed),
        so the user gets the "image flies in / flies back" effect requested. */}
    <AnimatePresence>
      {isExpanded && (
        <motion.div
          key="lightbox"
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-4 backdrop-blur-sm"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.2 }}
          onClick={() => setIsExpanded(false)}
          role="dialog"
          aria-modal="true"
          aria-label={`Plate ${detection.ocr_normalized_text} crop`}
        >
          <motion.img
            layoutId={layoutId}
            src={cropSrc}
            alt={detection.ocr_normalized_text}
            className="max-h-[90vh] max-w-[90vw] cursor-zoom-out rounded-lg border-2 border-white/40 object-contain shadow-2xl"
            onClick={(e) => {
              // Clicking the image itself shouldn't close — only the backdrop
              // should — so the user can read the plate without dismissing.
              e.stopPropagation();
            }}
          />
        </motion.div>
      )}
    </AnimatePresence>
    </>
  );
}
