#!/usr/bin/env python3
"""Standalone visual debugger for a plate-recognition pipeline.

This script does NOT import from ParkGuard's `app/` package.
It runs end-to-end on a video and shows, step by step:
1) detection, 2) crop, 3) deskew, 4) OCR candidates, 5) voting per track.

Output:
- console logs for every processed detection
- a debug video with overlays and intermediate stages
- optional live window preview (`--show-window`)
"""

from __future__ import annotations

import argparse
import inspect
import math
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

DEBUG_WIDTH = 1280
FRAME_PANEL_H = 540
STRIP_PANEL_H = 180
TEXT_PANEL_H = 180
DEBUG_HEIGHT = FRAME_PANEL_H + STRIP_PANEL_H + TEXT_PANEL_H

NUM_TO_LET: dict[str, str] = {
    "0": "O",
    "1": "I",
    "2": "Z",
    "3": "E",
    "4": "A",
    "5": "S",
    "6": "G",
    "7": "T",
    "8": "B",
    "9": "G",
}

LET_TO_NUM: dict[str, str] = {
    "O": "0",
    "I": "1",
    "Z": "2",
    "E": "3",
    "B": "8",
    "S": "5",
    "G": "6",
    "T": "7",
    "A": "4",
    "Q": "0",
    "D": "0",
}

COUNTY_CODES: frozenset[str] = frozenset(
    {
        "AB",
        "AG",
        "AR",
        "B",
        "BC",
        "BH",
        "BN",
        "BR",
        "BT",
        "BV",
        "BZ",
        "CJ",
        "CL",
        "CS",
        "CT",
        "CV",
        "DB",
        "DJ",
        "GJ",
        "GL",
        "GR",
        "HD",
        "HR",
        "IF",
        "IL",
        "IS",
        "MH",
        "MM",
        "MS",
        "NT",
        "OT",
        "PH",
        "SB",
        "SJ",
        "SM",
        "SV",
        "TL",
        "TM",
        "TR",
        "VL",
        "VN",
        "VS",
    }
)

NON_ALNUM = re.compile(r"[^A-Z0-9]")


@dataclass
class OCRCandidate:
    raw_text: str = ""
    confidence: float = 0.0
    normalized: str = ""
    is_valid: bool = False
    angle: float = 0.0
    preprocess_name: str = ""


@dataclass
class PlateTrack:
    track_id: int
    bbox: tuple[int, int, int, int]
    last_frame: int
    observations: int = 0
    votes: Counter[str] = field(default_factory=Counter)
    conf_sums: dict[str, float] = field(default_factory=dict)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize plate pipeline stages on a video (standalone)."
    )
    parser.add_argument("--video", required=True, help="Path to input video.")
    parser.add_argument("--model", default="models/best.pt", help="YOLO model path.")
    parser.add_argument(
        "--out",
        default="pipeline_debug.mp4",
        help="Path to output debug video.",
    )
    parser.add_argument("--fps-target", type=int, default=5)
    parser.add_argument("--detection-conf", type=float, default=0.50)
    parser.add_argument("--ocr-min-conf", type=float, default=0.25)
    parser.add_argument("--ocr-angles", default="-15,-8,0,8,15")
    parser.add_argument(
        "--single-pass-ocr",
        action="store_true",
        help="Disable multi-pass preprocessing (CLAHE/adaptive/inverted).",
    )
    parser.add_argument("--min-sharpness", type=float, default=80.0)
    parser.add_argument("--track-max-age", type=int, default=18)
    parser.add_argument("--track-min-iou", type=float, default=0.20)
    parser.add_argument("--min-track-votes", type=int, default=3)
    parser.add_argument(
        "--skip-ocr-when-stable",
        action="store_true",
        help="Mimic production optimization: stop OCR after track reaches min votes.",
    )
    parser.add_argument(
        "--max-sampled-frames",
        type=int,
        default=0,
        help="0 = process all sampled frames.",
    )
    parser.add_argument(
        "--no-ocr",
        action="store_true",
        help="Disable OCR (keeps detection/crop/tracking visualization).",
    )
    parser.add_argument(
        "--show-window",
        action="store_true",
        help="Show live window (requires non-headless OpenCV).",
    )
    return parser.parse_args()


def _apply_map(text: str, mapping: dict[str, str]) -> str:
    return "".join(mapping.get(ch, ch) for ch in text)


def compact_plate(text: str) -> str:
    return NON_ALNUM.sub("", text.upper())


def normalize_plate(raw_text: str) -> str:
    """Standalone normalizer for common Romanian standard plates."""
    cleaned = compact_plate(raw_text)
    if not cleaned:
        return "INVALID: empty"
    if len(cleaned) < 6:
        return "INVALID: too short"

    tail3 = _apply_map(cleaned[-3:], NUM_TO_LET)
    if not tail3.isalpha():
        return "INVALID: tail must be letters"

    prefix = cleaned[:-3]
    if not prefix:
        return "INVALID: missing prefix"

    if prefix[0] in {"B", "8"}:
        county = "B"
        digits = _apply_map(prefix[1:], LET_TO_NUM)
        if not digits.isdigit() or not (2 <= len(digits) <= 3):
            return "INVALID: Bucharest needs 2-3 digits"
        return f"{county} {digits} {tail3}"

    if len(prefix) < 4:
        return "INVALID: county prefix too short"

    county = _apply_map(prefix[:2], NUM_TO_LET)
    digits = _apply_map(prefix[2:], LET_TO_NUM)
    if county not in COUNTY_CODES:
        return "INVALID: invalid county"
    if not digits.isdigit() or len(digits) != 2:
        return "INVALID: county plates need 2 digits"
    return f"{county} {digits} {tail3}"


def is_invalid_plate(normalized: str) -> bool:
    return normalized.startswith("INVALID:")


def parse_ocr_angles(value: str) -> tuple[float, ...]:
    angles: list[float] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        angle = float(token)
        if angle < -45.0 or angle > 45.0:
            raise ValueError("OCR angles must be in [-45, 45].")
        if all(abs(existing - angle) > 1e-9 for existing in angles):
            angles.append(angle)
    if not angles:
        angles = [0.0]
    if all(abs(angle) > 1e-9 for angle in angles):
        angles.append(0.0)
    return tuple(angles)


def iter_sampled_frames(
    video_path: str, fps_target: int
) -> tuple[dict[str, float], list[tuple[int, int, np.ndarray]]]:
    if fps_target <= 0:
        raise ValueError("fps_target must be > 0")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path!r}")
    try:
        src_fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(src_fps) or src_fps <= 0:
            src_fps = 25.0
        total_frames = int(max(0, cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        frame_interval = max(1, round(src_fps / fps_target))

        frames: list[tuple[int, int, np.ndarray]] = []
        frame_index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index % frame_interval == 0:
                pts_ms = round(frame_index / src_fps * 1000)
                frames.append((frame_index, pts_ms, frame))
            frame_index += 1
    finally:
        cap.release()

    info = {
        "src_fps": src_fps,
        "total_frames": float(total_frames),
        "sampled_frames": float(len(frames)),
    }
    return info, frames


def to_scalar(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        return float(value.reshape(-1)[0])
    if isinstance(value, (list, tuple)) and value:
        return to_scalar(value[0])
    if hasattr(value, "item"):
        try:
            return float(value.item())
        except Exception:
            return None
    return None


def detect_plates(
    model: YOLO, frame: np.ndarray, conf_threshold: float
) -> list[tuple[tuple[int, int, int, int], float]]:
    frame_h, frame_w = frame.shape[:2]
    results = model(frame, conf=conf_threshold, verbose=False)
    if not results:
        return []
    boxes = getattr(results[0], "boxes", None)
    if boxes is None:
        return []

    detections: list[tuple[tuple[int, int, int, int], float]] = []
    for box in boxes:
        cls_id = to_scalar(getattr(box, "cls", None))
        conf = to_scalar(getattr(box, "conf", None))
        xyxy = getattr(box, "xyxy", None)
        if cls_id is None or conf is None or xyxy is None:
            continue
        if conf < conf_threshold:
            continue
        if int(cls_id) != 0:
            continue
        coords = np.asarray(xyxy).reshape(-1)
        if coords.size < 4:
            continue
        x1, y1, x2, y2 = map(float, coords[:4])
        left = max(0, int(np.floor(min(x1, x2))))
        top = max(0, int(np.floor(min(y1, y2))))
        right = min(frame_w, int(np.ceil(max(x1, x2))))
        bottom = min(frame_h, int(np.ceil(max(y1, y2))))
        if right <= left or bottom <= top:
            continue
        detections.append(((left, top, right - left, bottom - top), float(conf)))

    detections.sort(key=lambda item: item[1], reverse=True)
    return detections


def crop_plate(frame: np.ndarray, bbox: tuple[int, int, int, int], padding: int) -> np.ndarray:
    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        raise ValueError("Invalid bbox for crop")
    img_h, img_w = frame.shape[:2]
    p = max(0, int(padding))
    x1 = max(0, x - p)
    y1 = max(0, y - p)
    x2 = min(img_w, x + w + p)
    y2 = min(img_h, y + h + p)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        raise ValueError("Empty crop")
    return crop


def rotate_bound(image: np.ndarray, angle: float) -> np.ndarray:
    if abs(angle) < 1e-9:
        return image
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    m = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(m[0, 0])
    sin = abs(m[0, 1])
    bound_w = int((h * sin) + (w * cos))
    bound_h = int((h * cos) + (w * sin))
    m[0, 2] += bound_w / 2.0 - center[0]
    m[1, 2] += bound_h / 2.0 - center[1]
    return cv2.warpAffine(
        image,
        m,
        (bound_w, bound_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def sharpness_score(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def enhance_motion_blur(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    denoised = cv2.bilateralFilter(gray, 7, 50, 50)
    blurred = cv2.GaussianBlur(denoised, (0, 0), 1.0)
    sharpened = cv2.addWeighted(denoised, 1.6, blurred, -0.6, 0)
    return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)


def _resize_to_target(gray: np.ndarray, target_height: int = 64) -> np.ndarray:
    h, w = gray.shape[:2]
    scale = target_height / max(1, h)
    target_width = max(1, int(round(w * scale)))
    interpolation = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
    return cv2.resize(gray, (target_width, target_height), interpolation=interpolation)


def preprocess_clahe(crop: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    gray = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    return _resize_to_target(gray)


def preprocess_adaptive(crop: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    gray = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=15,
        C=5,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    return _resize_to_target(binary)


def preprocess_inverted(crop: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    gray = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=15,
        C=5,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    return _resize_to_target(binary)


def preprocess_passes(use_multi_pass: bool) -> list[tuple[str, Any]]:
    if use_multi_pass:
        return [
            ("clahe", preprocess_clahe),
            ("adaptive", preprocess_adaptive),
            ("inverted", preprocess_inverted),
        ]
    return [("clahe", preprocess_clahe)]


def _perspective_warp(image: np.ndarray, quad: np.ndarray) -> np.ndarray | None:
    pts = quad.reshape(4, 2).astype("float32")
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]
    tl, tr, br, bl = rect
    width_a = np.hypot(br[0] - bl[0], br[1] - bl[1])
    width_b = np.hypot(tr[0] - tl[0], tr[1] - tl[1])
    max_w = max(int(width_a), int(width_b))
    height_a = np.hypot(tr[0] - br[0], tr[1] - br[1])
    height_b = np.hypot(tl[0] - bl[0], tl[1] - bl[1])
    max_h = max(int(height_a), int(height_b))
    if max_w < 4 or max_h < 4:
        return None
    dst = np.array(
        [[0, 0], [max_w - 1, 0], [max_w - 1, max_h - 1], [0, max_h - 1]],
        dtype="float32",
    )
    m = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, m, (max_w, max_h))


def _estimate_skew_angle(edges: np.ndarray) -> float | None:
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=30,
        minLineLength=max(10, edges.shape[1] // 4),
        maxLineGap=10,
    )
    if lines is None or len(lines) == 0:
        return None
    angles: list[float] = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = x2 - x1
        dy = y2 - y1
        if abs(dx) < 2:
            continue
        angle = math.degrees(math.atan2(dy, dx))
        if abs(angle) < 30:
            angles.append(angle)
    if not angles:
        return None
    angles.sort()
    return angles[len(angles) // 2]


def deskew_plate(crop: np.ndarray) -> np.ndarray:
    if crop.size == 0:
        return crop
    h, w = crop.shape[:2]
    if h < 10 or w < 10:
        return crop

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blur, 50, 200)
    contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    crop_area = h * w
    best_quad: np.ndarray | None = None
    best_score = 0.0

    for eps_frac in (0.02, 0.03, 0.05):
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, eps_frac * peri, True)
            if len(approx) != 4:
                continue
            area = cv2.contourArea(approx)
            area_ratio = area / max(1, crop_area)
            if area_ratio < 0.15:
                continue
            rect = cv2.minAreaRect(approx)
            rect_w, rect_h = rect[1]
            if rect_w == 0 or rect_h == 0:
                continue
            ar = max(rect_w, rect_h) / min(rect_w, rect_h)
            if ar < 2.0 or ar > 7.0:
                continue
            score = area_ratio * ar
            if score > best_score:
                best_score = score
                best_quad = approx
        if best_quad is not None:
            break

    if best_quad is not None:
        warped = _perspective_warp(crop, best_quad)
        if warped is not None and warped.size > 0:
            return warped

    angle = _estimate_skew_angle(edged)
    if angle is not None and abs(angle) > 0.5:
        h2, w2 = crop.shape[:2]
        center = (w2 / 2.0, h2 / 2.0)
        m = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(
            crop,
            m,
            (w2, h2),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
    return crop


def _bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    union = max(1, aw * ah + bw * bh - inter)
    return float(inter / union)


def _prune_stale_tracks(active_tracks: dict[int, PlateTrack], frame_index: int, max_age: int) -> None:
    stale = [
        track_id
        for track_id, track in active_tracks.items()
        if frame_index - track.last_frame > max_age
    ]
    for track_id in stale:
        del active_tracks[track_id]


def assign_track(
    bbox: tuple[int, int, int, int],
    frame_index: int,
    active_tracks: dict[int, PlateTrack],
    all_tracks: dict[int, PlateTrack],
    next_track_id: int,
    max_age: int,
    min_iou: float,
) -> tuple[PlateTrack, int]:
    _prune_stale_tracks(active_tracks, frame_index, max_age)
    best_track: PlateTrack | None = None
    best_iou = 0.0
    for track in active_tracks.values():
        iou = _bbox_iou(bbox, track.bbox)
        if iou > best_iou:
            best_iou = iou
            best_track = track
    if best_track is None or best_iou < min_iou:
        track = PlateTrack(
            track_id=next_track_id,
            bbox=bbox,
            last_frame=frame_index,
            observations=1,
        )
        active_tracks[next_track_id] = track
        all_tracks[next_track_id] = track
        return track, next_track_id + 1

    best_track.bbox = bbox
    best_track.last_frame = frame_index
    best_track.observations += 1
    return best_track, next_track_id


def _levenshtein_distance(a: str, b: str) -> int:
    if len(a) < len(b):
        return _levenshtein_distance(b, a)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            cost = 0 if ca == cb else 1
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + cost))
        prev = curr
    return prev[-1]


def _find_near_match(track: PlateTrack, plate: str) -> str | None:
    if plate in track.votes:
        return plate
    for existing in track.votes:
        if len(existing) == len(plate) and _levenshtein_distance(existing, plate) <= 1:
            return existing
    return None


def register_vote(track: PlateTrack, plate: str, confidence: float, weight: float = 1.0) -> None:
    near = _find_near_match(track, plate)
    if near is not None and near != plate:
        existing_votes = track.votes[near]
        if weight > existing_votes:
            track.votes[plate] = track.votes.pop(near) + 1
            track.conf_sums[plate] = track.conf_sums.pop(near, 0.0) + confidence
        else:
            track.votes[near] += 1
            track.conf_sums[near] = track.conf_sums.get(near, 0.0) + confidence
    else:
        track.votes[plate] += 1
        track.conf_sums[plate] = track.conf_sums.get(plate, 0.0) + confidence


def track_best_plate(track: PlateTrack) -> tuple[str | None, int, float, float]:
    if not track.votes:
        return None, 0, 0.0, 0.0

    def _score(item: tuple[str, int]) -> tuple[int, float, str]:
        plate, votes = item
        avg_conf = track.conf_sums.get(plate, 0.0) / max(1, votes)
        return votes, avg_conf, plate

    best_plate, best_votes = max(track.votes.items(), key=_score)
    avg_conf = track.conf_sums.get(best_plate, 0.0) / max(1, best_votes)
    stability = best_votes / max(1, track.observations)
    return best_plate, best_votes, avg_conf, stability


def collect_ocr_tokens(node: Any, out: list[tuple[str, float]]) -> None:
    if node is None:
        return
    if isinstance(node, dict):
        rec_texts = node.get("rec_texts")
        rec_scores = node.get("rec_scores")
        if isinstance(rec_texts, (list, tuple)):
            for idx, text in enumerate(rec_texts):
                if not isinstance(text, str):
                    continue
                score = 0.0
                if (
                    isinstance(rec_scores, (list, tuple))
                    and idx < len(rec_scores)
                    and isinstance(rec_scores[idx], (int, float))
                ):
                    score = float(rec_scores[idx])
                out.append((text, score))
            return
        rec_text = node.get("rec_text")
        rec_score = node.get("rec_score")
        if isinstance(rec_text, str) and isinstance(rec_score, (int, float)):
            out.append((rec_text, float(rec_score)))
            return
        if isinstance(rec_text, (list, tuple)) and rec_text and isinstance(rec_text[0], str):
            score = float(rec_score) if isinstance(rec_score, (int, float)) else 0.0
            out.append((rec_text[0], score))
            return
        for value in node.values():
            collect_ocr_tokens(value, out)
        return
    if not isinstance(node, (list, tuple)):
        if hasattr(node, "json"):
            try:
                collect_ocr_tokens(getattr(node, "json"), out)
            except Exception:
                return
        return
    if (
        len(node) >= 2
        and isinstance(node[1], (list, tuple))
        and len(node[1]) >= 2
        and isinstance(node[1][0], str)
        and isinstance(node[1][1], (int, float))
    ):
        out.append((node[1][0], float(node[1][1])))
        return
    if len(node) >= 2 and isinstance(node[0], str) and isinstance(node[1], (int, float)):
        out.append((node[0], float(node[1])))
        return
    for item in node:
        collect_ocr_tokens(item, out)


def _build_paddle_ocr() -> Any:
    os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

    from paddleocr import PaddleOCR

    init_signature = inspect.signature(PaddleOCR.__init__)
    init_kwargs: dict[str, Any] = {"lang": "en"}

    if "use_textline_orientation" in init_signature.parameters:
        init_kwargs["use_textline_orientation"] = True
    elif "use_angle_cls" in init_signature.parameters:
        init_kwargs["use_angle_cls"] = True

    if "show_log" in init_signature.parameters:
        init_kwargs["show_log"] = False

    return PaddleOCR(**init_kwargs)


class OCRReader:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.reader: Any | None = None
        self.error: str | None = None
        if not enabled:
            return
        try:
            self.reader = _build_paddle_ocr()
        except Exception as exc:
            self.error = str(exc)
            self.enabled = False

    def read(self, image: np.ndarray) -> tuple[str, float]:
        if not self.enabled or self.reader is None:
            return "", 0.0
        result: Any | None = None
        if hasattr(self.reader, "predict"):
            result = self.reader.predict(image)
        elif hasattr(self.reader, "ocr"):
            try:
                result = self.reader.ocr(image, det=False, rec=True, cls=False)
            except TypeError:
                result = self.reader.ocr(image)
        if result is None:
            return "", 0.0
        tokens: list[tuple[str, float]] = []
        collect_ocr_tokens(result, tokens)
        if not tokens:
            return "", 0.0
        text = NON_ALNUM.sub("", "".join(token for token, _ in tokens).upper())
        confidence = float(sum(conf for _, conf in tokens) / len(tokens))
        return text, confidence


def best_ocr_candidate(
    reader: OCRReader,
    crop: np.ndarray,
    ocr_angles: tuple[float, ...],
    use_multi_pass: bool,
) -> tuple[OCRCandidate, list[OCRCandidate]]:
    best_valid = OCRCandidate()
    best_any = OCRCandidate()
    all_candidates: list[OCRCandidate] = []
    for pass_name, preprocess_fn in preprocess_passes(use_multi_pass):
        try:
            processed = preprocess_fn(crop)
        except Exception:
            continue

        for angle in ocr_angles:
            candidate_image = rotate_bound(processed, angle)
            ocr_input = (
                cv2.cvtColor(candidate_image, cv2.COLOR_GRAY2BGR)
                if candidate_image.ndim == 2
                else candidate_image
            )
            raw_text, confidence = reader.read(ocr_input)
            if raw_text:
                normalized = normalize_plate(raw_text)
                valid = not is_invalid_plate(normalized)
                candidate = OCRCandidate(
                    raw_text=raw_text,
                    confidence=confidence,
                    normalized=normalized,
                    is_valid=valid,
                    angle=angle,
                    preprocess_name=pass_name,
                )
            else:
                candidate = OCRCandidate(
                    angle=angle,
                    preprocess_name=pass_name,
                )
            all_candidates.append(candidate)

            if candidate.raw_text and candidate.confidence > best_any.confidence:
                best_any = candidate
            if candidate.is_valid and candidate.confidence > best_valid.confidence:
                best_valid = candidate
    chosen = best_valid if best_valid.raw_text else best_any
    return chosen, all_candidates


def detect_character_boxes(crop: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    if crop.size == 0:
        return crop, []
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop.copy()
    denoised = cv2.bilateralFilter(gray, 7, 50, 50)
    _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = binary.shape[:2]
    boxes: list[tuple[int, int, int, int]] = []
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        area = cw * ch
        if area < 20:
            continue
        if ch < int(h * 0.35):
            continue
        if cw < 2 or cw > int(w * 0.35):
            continue
        boxes.append((x, y, cw, ch))
    boxes.sort(key=lambda b: b[0])

    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for idx, (x, y, cw, ch) in enumerate(boxes):
        cv2.rectangle(vis, (x, y), (x + cw, y + ch), (0, 255, 0), 1)
        cv2.putText(
            vis,
            str(idx + 1),
            (x, max(0, y - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return vis, boxes


def resize_with_padding(image: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    if image.size == 0:
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    h, w = image.shape[:2]
    scale = min(target_w / max(1, w), target_h / max(1, h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x = (target_w - new_w) // 2
    y = (target_h - new_h) // 2
    canvas[y : y + new_h, x : x + new_w] = resized
    return canvas


def add_panel_label(image: np.ndarray, label: str) -> np.ndarray:
    out = image.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(
        out,
        label,
        (10, 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return out


def render_text_panel(lines: list[str], width: int, height: int) -> np.ndarray:
    panel = np.full((height, width, 3), 20, dtype=np.uint8)
    y = 26
    for line in lines:
        if y > height - 10:
            break
        cv2.putText(
            panel,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        y += 24
    return panel


def make_debug_canvas(
    frame_view: np.ndarray,
    crop: np.ndarray,
    rectified: np.ndarray,
    char_vis: np.ndarray,
    text_lines: list[str],
) -> np.ndarray:
    top = resize_with_padding(frame_view, DEBUG_WIDTH, FRAME_PANEL_H)
    third = DEBUG_WIDTH // 3
    c1 = add_panel_label(resize_with_padding(crop, third, STRIP_PANEL_H), "CROP")
    c2 = add_panel_label(resize_with_padding(rectified, third, STRIP_PANEL_H), "DESKEW")
    c3 = add_panel_label(
        resize_with_padding(char_vis, DEBUG_WIDTH - 2 * third, STRIP_PANEL_H),
        "CHAR PROPOSALS",
    )
    strip = np.concatenate([c1, c2, c3], axis=1)
    text = render_text_panel(text_lines, DEBUG_WIDTH, TEXT_PANEL_H)
    return np.concatenate([top, strip, text], axis=0)


def main() -> None:
    args = parse_args()
    video_path = Path(args.video)
    model_path = Path(args.model)
    out_path = Path(args.out)

    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}")
    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    info, sampled_frames = iter_sampled_frames(str(video_path), args.fps_target)
    if args.max_sampled_frames > 0:
        sampled_frames = sampled_frames[: args.max_sampled_frames]
    if not sampled_frames:
        raise SystemExit("No sampled frames available.")

    print(
        f"[INFO] video={video_path} src_fps={info['src_fps']:.2f} "
        f"total_frames={int(info['total_frames'])} sampled={len(sampled_frames)}"
    )

    model = YOLO(str(model_path))
    reader = OCRReader(enabled=not args.no_ocr)
    if not reader.enabled and not args.no_ocr:
        print(f"[WARN] OCR disabled automatically: {reader.error}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(out_path),
        fourcc,
        float(max(1, args.fps_target)),
        (DEBUG_WIDTH, DEBUG_HEIGHT),
    )
    if not writer.isOpened():
        raise SystemExit(f"Cannot open output video for writing: {out_path}")

    ocr_angles = parse_ocr_angles(args.ocr_angles)
    min_ocr_conf = max(0.0, min(1.0, float(args.ocr_min_conf)))
    use_multi_pass = not args.single_pass_ocr
    track_max_age = max(1, int(args.track_max_age))
    track_min_iou = max(0.01, min(0.99, float(args.track_min_iou)))
    min_track_votes = max(1, int(args.min_track_votes))

    active_tracks: dict[int, PlateTrack] = {}
    all_tracks: dict[int, PlateTrack] = {}
    next_track_id = 1
    total_detections = 0

    for sampled_idx, (frame_index, pts_ms, frame) in enumerate(sampled_frames, start=1):
        detections = detect_plates(model, frame, conf_threshold=float(args.detection_conf))

        if not detections:
            no_det_view = frame.copy()
            cv2.putText(
                no_det_view,
                f"Frame {frame_index} ({pts_ms} ms) - no plate detections",
                (20, 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            canvas = make_debug_canvas(
                frame_view=no_det_view,
                crop=np.zeros((60, 200, 3), dtype=np.uint8),
                rectified=np.zeros((60, 200, 3), dtype=np.uint8),
                char_vis=np.zeros((60, 200, 3), dtype=np.uint8),
                text_lines=[
                    f"sample {sampled_idx}/{len(sampled_frames)} frame={frame_index} pts={pts_ms}ms",
                    "detections: 0",
                    "status: no plate in this sampled frame",
                ],
            )
            writer.write(canvas)
            if args.show_window:
                cv2.imshow("Pipeline Flow Debugger", canvas)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
            continue

        for det_idx, (bbox, det_conf) in enumerate(detections, start=1):
            total_detections += 1
            x, y, w, h = bbox

            track, next_track_id = assign_track(
                bbox=bbox,
                frame_index=frame_index,
                active_tracks=active_tracks,
                all_tracks=all_tracks,
                next_track_id=next_track_id,
                max_age=track_max_age,
                min_iou=track_min_iou,
            )
            pre_plate, pre_votes, _, _ = track_best_plate(track)
            track_already_stable = bool(pre_plate and pre_votes >= min_track_votes)

            pad = max(2, min(8, int(min(w, h) * 0.08)))
            try:
                crop = crop_plate(frame, bbox, padding=pad)
            except ValueError:
                continue

            rectified = deskew_plate(crop)
            if rectified.size == 0:
                rectified = crop

            sharp = sharpness_score(rectified)
            ocr_input = rectified
            if sharp >= args.min_sharpness and sharp < args.min_sharpness * 1.35:
                ocr_input = enhance_motion_blur(rectified)

            chosen = OCRCandidate()
            angle_candidates: list[OCRCandidate] = []
            should_run_ocr = reader.enabled and (
                (not args.skip_ocr_when_stable) or (not track_already_stable)
            )
            if should_run_ocr:
                chosen, angle_candidates = best_ocr_candidate(
                    reader,
                    ocr_input,
                    ocr_angles=ocr_angles,
                    use_multi_pass=use_multi_pass,
                )
                if chosen.raw_text and chosen.confidence >= min_ocr_conf:
                    if chosen.is_valid:
                        register_vote(track, compact_plate(chosen.normalized), chosen.confidence)
                    elif chosen.confidence >= 0.70 and not chosen.normalized.startswith("INVALID:"):
                        compact_raw = compact_plate(chosen.raw_text)
                        if 4 <= len(compact_raw) <= 10:
                            register_vote(track, compact_raw, chosen.confidence, weight=0.3)

            best_plate, best_votes, best_avg_conf, stability = track_best_plate(track)
            final_plate = best_plate if best_plate and best_votes >= min_track_votes else None

            char_vis, char_boxes = detect_character_boxes(ocr_input)

            frame_view = frame.copy()
            cv2.rectangle(frame_view, (x, y), (x + w, y + h), (0, 255, 0), 2)
            title = (
                f"T{track.track_id} det={det_conf:.2f} "
                f"votes={best_votes} final={final_plate or '-'}"
            )
            cv2.putText(
                frame_view,
                title,
                (x, max(20, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            votes_sorted = sorted(
                track.votes.items(),
                key=lambda it: (it[1], track.conf_sums.get(it[0], 0.0)),
                reverse=True,
            )
            vote_line = " | ".join(f"{plate}:{votes}" for plate, votes in votes_sorted[:4]) or "-"

            ocr_lines = []
            shown_candidates = sorted(
                angle_candidates,
                key=lambda c: (c.is_valid, c.confidence),
                reverse=True,
            )
            for cand in shown_candidates[:6]:
                status = "OK" if cand.is_valid else "NO"
                ocr_lines.append(
                    f"pass={cand.preprocess_name or '-'} a={cand.angle:+.0f} "
                    f"txt={cand.raw_text or '-'} conf={cand.confidence:.2f} {status}"
                )
            if len(angle_candidates) > 6:
                ocr_lines.append(f"... +{len(angle_candidates) - 6} candidate(s)")
            if not ocr_lines:
                if not reader.enabled:
                    ocr_lines.append("ocr: disabled")
                elif track_already_stable and args.skip_ocr_when_stable:
                    ocr_lines.append("ocr: skipped (track already stable)")
                else:
                    ocr_lines.append("ocr: no candidates")

            text_lines = [
                f"sample {sampled_idx}/{len(sampled_frames)} frame={frame_index} pts={pts_ms}ms det={det_idx}/{len(detections)}",
                f"track=T{track.track_id} stable_before={track_already_stable} det_conf={det_conf:.3f} sharpness={sharp:.1f}",
                f"ocr_selected raw={chosen.raw_text or '-'} conf={chosen.confidence:.2f} norm={chosen.normalized or '-'}",
                f"selected_pass={chosen.preprocess_name or '-'} ocr_mode={'multi-pass' if use_multi_pass else 'single-pass'}",
                f"run_ocr_this_sample={should_run_ocr} skip_when_stable={args.skip_ocr_when_stable}",
                f"votes: {vote_line}",
                f"best_plate={best_plate or '-'} best_votes={best_votes} avg_conf={best_avg_conf:.2f} stability={stability:.2f}",
                f"char_proposals={len(char_boxes)} min_track_votes={min_track_votes} final_plate={final_plate or '-'}",
            ] + ocr_lines

            canvas = make_debug_canvas(
                frame_view=frame_view,
                crop=crop,
                rectified=ocr_input,
                char_vis=char_vis,
                text_lines=text_lines,
            )
            writer.write(canvas)

            print(
                f"[STEP] f={frame_index} det={det_idx}/{len(detections)} "
                f"track=T{track.track_id} conf={det_conf:.2f} sharp={sharp:.1f} "
                f"ocr='{chosen.raw_text or '-'}' norm='{chosen.normalized or '-'}' "
                f"pass='{chosen.preprocess_name or '-'}' "
                f"votes={best_votes} final='{final_plate or '-'}'"
            )

            if args.show_window:
                cv2.imshow("Pipeline Flow Debugger", canvas)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    writer.release()
                    cv2.destroyAllWindows()
                    print("[INFO] interrupted by user")
                    return

    writer.release()
    if args.show_window:
        cv2.destroyAllWindows()

    stable_tracks = sum(
        1 for track in all_tracks.values() if track_best_plate(track)[1] >= min_track_votes
    )
    print(
        f"[DONE] detections={total_detections} tracks={len(all_tracks)} "
        f"stable_tracks={stable_tracks} debug_video={out_path}"
    )


if __name__ == "__main__":
    main()
