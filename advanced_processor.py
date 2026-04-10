import argparse
import cv2
import logging
import time
from collections import defaultdict
from collections import Counter

import torch
from ultralytics import YOLO
from app.pipeline.ocr import PlateReader
from app.pipeline.plate_validator import normalize_plate
from app.pipeline.deskew import deskew_plate
from app.pipeline.lprnet import infer_lprnet_dummy

logging.basicConfig(level=logging.WARNING)

class AdvancedPlateProcessor:
    def __init__(
        self,
        yolo_model_path: str,
        sr_model_path: str,
        *,
        detector_conf: float = 0.20,
        yolo_device: str = "auto",
        blur_sharpness_threshold: float = 80.0,
        resolve_ambiguous_w: bool = True,
    ):
        print("Loading models...", flush=True)
        # 1. Load YOLOv8 for Plates
        self.detector = YOLO(yolo_model_path)
        self.detector_conf = float(detector_conf)
        self.yolo_device = self._resolve_device(yolo_device)
        self.blur_sharpness_threshold = float(blur_sharpness_threshold)
        self.resolve_ambiguous_w = bool(resolve_ambiguous_w)
        print(f"YOLO device: {self.yolo_device}", flush=True)
        
        # 2. Load OCR (EasyOCR tuned as our generic CRNN for now)
        self.reader = PlateReader(gpu=False)
        
        # 3. Load Super Resolution Model
        self.sr = cv2.dnn_superres.DnnSuperResImpl_create()
        self.sr.readModel(sr_model_path)
        self.sr.setModel("fsrcnn", 2) # FSRCNN scale 2x
        
        # Tracking history
        self.track_history = defaultdict(list)

    @staticmethod
    def _resolve_device(requested_device: str) -> str:
        value = (requested_device or "auto").strip().lower()
        if value != "auto":
            return value

        if torch.cuda.is_available():
            return "cuda:0"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @staticmethod
    def _sharpness_score(image) -> float:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    @staticmethod
    def _enhance_motion_blur(image):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        denoised = cv2.bilateralFilter(gray, 7, 50, 50)
        blurred = cv2.GaussianBlur(denoised, (0, 0), 1.0)
        sharpened = cv2.addWeighted(denoised, 1.6, blurred, -0.6, 0)
        return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def _resolve_last_char_w(counts: Counter[str]) -> str | None:
        """Resolve H/U/V/I ambiguity on the last char to W when pattern is strong."""
        prefix_votes: dict[str, Counter[str]] = defaultdict(Counter)
        for text, votes in counts.items():
            if len(text) != 7:
                continue
            prefix_votes[text[:6]][text[6]] += votes

        if not prefix_votes:
            return None

        prefix, suffix_counts = max(
            prefix_votes.items(), key=lambda item: sum(item[1].values())
        )
        total_votes = sum(suffix_counts.values())
        if total_votes < 8:
            return None

        ambiguous = {"W", "H", "U", "V", "I"}
        ambiguous_votes = sum(v for ch, v in suffix_counts.items() if ch in ambiguous)
        if ambiguous_votes / total_votes < 0.85:
            return None

        seen_ambiguous = {ch for ch, v in suffix_counts.items() if v > 0 and ch in ambiguous}
        if len(seen_ambiguous - {"W"}) < 2:
            return None
        if suffix_counts.get("U", 0) + suffix_counts.get("V", 0) == 0:
            return None

        top_char, _ = suffix_counts.most_common(1)[0]
        if top_char == "W":
            return f"{prefix}W"

        return f"{prefix}W"
        
    def process_video(
        self,
        video_path,
        frame_stride: int = 2,
        progress_every_frames: int = 240,
        ocr_every: int = 1,
        max_frames: int | None = None,
    ):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total_frames > 0:
            print(f"Total frames in video: {total_frames}", flush=True)
            
        print(
            "Processing frames with ByteTrack, Deskewing, SuperRes and Voting...",
            flush=True,
        )
        frame_idx = 0
        started_at = time.perf_counter()
        frame_stride = max(1, int(frame_stride))
        progress_every_frames = max(1, int(progress_every_frames))
        ocr_every = max(1, int(ocr_every))
        max_frames = None if not max_frames or max_frames <= 0 else int(max_frames)
        track_observations = defaultdict(int)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_idx += 1
            if max_frames is not None and frame_idx > max_frames:
                break
            if frame_idx % frame_stride != 0:
                continue  # Skip frames to process faster

            if frame_idx % progress_every_frames == 0:
                elapsed_s = max(1e-6, time.perf_counter() - started_at)
                processed_fps = frame_idx / elapsed_s
                if total_frames > 0:
                    progress = (frame_idx / total_frames) * 100
                    remaining_frames = max(0, total_frames - frame_idx)
                    eta_s = remaining_frames / max(1e-6, processed_fps)
                    print(
                        f"[{progress:5.1f}%] frame {frame_idx}/{total_frames} | "
                        f"tracks: {len(self.track_history)} | "
                        f"elapsed: {elapsed_s/60:.1f}m | ETA: {eta_s/60:.1f}m",
                        flush=True,
                    )
                else:
                    print(
                        f"Processed frame {frame_idx} | tracks: {len(self.track_history)}",
                        flush=True,
                    )
                
            # 1. Tracking with ByteTrack (Object Tracking)
            # This handles keeping the same ID for the same physical plate
            results = self.detector.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=self.detector_conf,
                verbose=False,
                device=self.yolo_device,
            )
            
            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.int().cpu().numpy()
                
                for box, track_id in zip(boxes, track_ids):
                    track_id = int(track_id)
                    track_observations[track_id] += 1
                    if track_observations[track_id] % ocr_every != 0:
                        continue

                    x1, y1, x2, y2 = map(int, box)
                    
                    # Prevent zero-size boxes
                    if y2 <= y1 or x2 <= x1:
                        continue
                        
                    crop = frame[y1:y2, x1:x2]
                    
                    # 2. Deskewing (Warp Perspective)
                    # Flattens the angled plate from the tow truck
                    deskewed = deskew_plate(crop)
                    
                    # 3. Super Resolution (AI Upscaling)
                    # Enhances the quality of small distant plates
                    try:
                        upscaled = self.sr.upsample(deskewed)
                    except Exception:
                        upscaled = deskewed # Fallback

                    # Motion blur mitigation: sharpen only on low-sharpness crops.
                    ocr_input = upscaled
                    sharpness = self._sharpness_score(upscaled)
                    if sharpness < self.blur_sharpness_threshold:
                        ocr_input = self._enhance_motion_blur(upscaled)

                    # 4. OCR / CRNN / LPRNet
                    # We first try our LPRNet interface, if weights missing, we use EasyOCR CRNN
                    lprnet_text, lpr_conf = infer_lprnet_dummy(ocr_input)
                    if lprnet_text:
                        raw_text = lprnet_text
                    else:
                        # Fallback to our tuned EasyOCR
                        raw_text, conf = self.reader.read_plate(ocr_input)
                        
                    if raw_text:
                        norm_text, is_valid = normalize_plate(raw_text)
                        if is_valid:
                            if not self.track_history[track_id]:
                                print(
                                    f"Track {track_id}: first valid read {norm_text} "
                                    f"(frame {frame_idx})",
                                    flush=True,
                                )
                            self.track_history[track_id].append(norm_text)
                            
        cap.release()
        
        # 5. Majority Voting
        print("\n=== FINAL RESULTS (AFTER MAJORITY VOTING) ===")
        unique_plates = set()
        
        for track_id, texts in self.track_history.items():
            if not texts:
                continue
                
            # Count the occurrences of each read for this specific tracked object
            counts = Counter(texts)
            # The most common read wins!
            best_text = counts.most_common(1)[0][0]
            corrected_text = None
            if self.resolve_ambiguous_w:
                corrected_text = self._resolve_last_char_w(counts)
                if corrected_text is not None:
                    best_text = corrected_text
            
            if best_text not in unique_plates:
                unique_plates.add(best_text)
                if corrected_text is not None:
                    print(
                        f"✅ Vehicle Track {track_id}: Plate {best_text} "
                        f"(W-disambiguated from votes: {dict(counts)})"
                    )
                else:
                    print(
                        f"✅ Vehicle Track {track_id}: Plate {best_text} "
                        f"(Voted from {len(texts)} reads: {dict(counts)})"
                    )
                
        if not unique_plates:
            print("No valid plates were tracked.")
        else:
            print(f"Total unique valid plates finalized: {len(unique_plates)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Advanced plate processor")
    parser.add_argument("--video", default="vid.mp4", help="Path to input video")
    parser.add_argument(
        "--model",
        default="models/yolo_plate.pt",
        help="Path to YOLO plate model",
    )
    parser.add_argument(
        "--sr-model",
        default="models/FSRCNN_x2.pb",
        help="Path to OpenCV super-resolution model",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=2,
        help="Process one frame every N frames (higher is faster, lower is more accurate)",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=240,
        help="Print progress every N source frames",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="YOLO device: auto, cpu, mps, cuda:0",
    )
    parser.add_argument(
        "--det-conf",
        type=float,
        default=0.20,
        help="YOLO confidence threshold for plate tracking",
    )
    parser.add_argument(
        "--blur-threshold",
        type=float,
        default=80.0,
        help="Laplacian sharpness threshold under which motion deblur sharpening is applied",
    )
    parser.add_argument(
        "--ocr-every",
        type=int,
        default=1,
        help="Run OCR every N observations per track (1 = full accuracy mode)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Optional source-frame limit for quick experiments (0 = full video)",
    )
    parser.add_argument(
        "--resolve-ambiguous-w",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resolve strong H/U/V/I last-char ambiguity to W at track voting stage",
    )
    args = parser.parse_args()

    processor = AdvancedPlateProcessor(
        args.model,
        args.sr_model,
        detector_conf=args.det_conf,
        yolo_device=args.device,
        blur_sharpness_threshold=args.blur_threshold,
        resolve_ambiguous_w=args.resolve_ambiguous_w,
    )
    processor.process_video(
        args.video,
        frame_stride=args.frame_stride,
        progress_every_frames=args.progress_every,
        ocr_every=args.ocr_every,
        max_frames=args.max_frames,
    )
