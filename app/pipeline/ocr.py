"""OCR image preprocessing and PaddleOCR wrapper utilities.

Multi-pass preprocessing strategy
----------------------------------
    Pass 1 — CLAHE (standard, good for most lighting conditions)
    Pass 2 — Adaptive-threshold binarisation (handles uneven illumination)
    Pass 3 — Inverted binarisation (handles dark-background plates)

Each pass is fed to PaddleOCR independently.  The result with the highest
confidence that produces a valid Romanian plate is selected.  If no valid
plate is found across all passes, the highest-confidence raw text from any
pass is returned (to let the downstream normaliser attempt correction).
"""

from __future__ import annotations

import inspect
import logging
import os
import re
import sys
import time
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)
_NON_ALNUM_RE = re.compile(r"[^A-Z0-9]")


# ---------------------------------------------------------------------------
# Multi-pass preprocessing pipelines
# ---------------------------------------------------------------------------


def preprocess_clahe(crop: np.ndarray) -> np.ndarray:
    """Standard CLAHE preprocessing: grayscale → bilateral denoise → CLAHE → resize."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop

    # Bilateral filter — reduces noise while preserving edges (character strokes).
    gray = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    return _resize_to_target(gray)


def preprocess_adaptive(crop: np.ndarray) -> np.ndarray:
    """Adaptive-threshold binarisation: handles uneven illumination."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop

    # Bilateral filter first.
    gray = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)

    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=15,
        C=5,
    )

    # Morphological opening — remove small noise dots.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    return _resize_to_target(binary)


def preprocess_inverted(crop: np.ndarray) -> np.ndarray:
    """Inverted binarisation: for dark-background or inverted plates."""
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


def _resize_to_target(gray: np.ndarray) -> np.ndarray:
    """Resize a grayscale image to a fixed 64px height."""
    h, w = gray.shape[:2]
    target_height = 64
    scale = target_height / max(1, h)
    target_width = max(1, int(round(w * scale)))
    interpolation = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
    return cv2.resize(gray, (target_width, target_height), interpolation=interpolation)


# Keep the old name around for backward compatibility (tests, external callers).
def preprocess_crop(crop: np.ndarray) -> np.ndarray:
    """Legacy alias — delegates to :func:`preprocess_clahe`."""
    return preprocess_clahe(crop)


# Ordered list of preprocessing functions for multi-pass OCR.
_PREPROCESS_PASSES: list[tuple[str, Any]] = [
    ("clahe", preprocess_clahe),
    ("adaptive", preprocess_adaptive),
    ("inverted", preprocess_inverted),
]


# ---------------------------------------------------------------------------
# PaddleOCR builder
# ---------------------------------------------------------------------------


def _build_paddle_ocr(*, use_angle_cls: bool, lang: str) -> Any:
    from paddleocr import PaddleOCR

    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

    init_signature = inspect.signature(PaddleOCR.__init__)
    init_kwargs: dict[str, Any] = {"lang": lang}

    if "use_textline_orientation" in init_signature.parameters:
        init_kwargs["use_textline_orientation"] = use_angle_cls
    elif "use_angle_cls" in init_signature.parameters:
        init_kwargs["use_angle_cls"] = use_angle_cls

    if "show_log" in init_signature.parameters:
        init_kwargs["show_log"] = False

    return PaddleOCR(**init_kwargs)


# ---------------------------------------------------------------------------
# Token collector (handles all PaddleOCR output formats)
# ---------------------------------------------------------------------------


def _collect_ocr_tokens(node: Any, out: list[tuple[str, float]]) -> None:
    if node is None:
        return

    if isinstance(node, dict):
        rec_texts = node.get("rec_texts")
        rec_scores = node.get("rec_scores")
        if isinstance(rec_texts, (list, tuple)):
            for idx, text in enumerate(rec_texts):
                if not isinstance(text, str):
                    continue
                confidence = 0.0
                if (
                    isinstance(rec_scores, (list, tuple))
                    and idx < len(rec_scores)
                    and isinstance(rec_scores[idx], (int, float))
                ):
                    confidence = float(rec_scores[idx])
                out.append((text, confidence))
            return

        rec_text = node.get("rec_text")
        rec_score = node.get("rec_score")
        if isinstance(rec_text, str) and isinstance(rec_score, (int, float)):
            out.append((rec_text, float(rec_score)))
            return
        if (
            isinstance(rec_text, (list, tuple))
            and rec_text
            and isinstance(rec_text[0], str)
        ):
            confidence = float(rec_score) if isinstance(rec_score, (int, float)) else 0.0
            out.append((rec_text[0], confidence))
            return

        for value in node.values():
            _collect_ocr_tokens(value, out)
        return

    if not isinstance(node, (list, tuple)):
        if hasattr(node, "json"):
            try:
                _collect_ocr_tokens(getattr(node, "json"), out)
            except Exception:
                return
        return

    if (
        len(node) >= 2
        and isinstance(node[0], str)
        and isinstance(node[1], (int, float))
    ):
        out.append((node[0], float(node[1])))
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

    for item in node:
        _collect_ocr_tokens(item, out)


# ---------------------------------------------------------------------------
# PlateReader
# ---------------------------------------------------------------------------


class PlateReader:
    """Read plate text from image crops with PaddleOCR.

    Parameters
    ----------
    gpu:
        Kept for backward-compatibility; currently ignored.
    use_angle_cls:
        Whether to enable textline orientation detection.
    lang:
        Language model passed to PaddleOCR.
    use_fsm_decode:
        When ``True``, the raw OCR output is decoded through the FSM-guided
        Prefix Beam Search (:mod:`app.pipeline.ctc_decoder`).
    fsm_beam_width:
        Beam width for the FSM decoder.
    use_multi_pass:
        When ``True`` (default), runs OCR with 3 preprocessing variants
        (CLAHE, adaptive threshold, inverted) and picks the best result.
        Set to ``False`` for single-pass (faster, for tests).
    use_super_res:
        When ``True``, small crops are upscaled via AI super-resolution.
    super_res_min_height:
        Pixel threshold below which SR is applied.
    super_res_model:
        SR model name (e.g. "espcn_x2").
    super_res_allow_download:
        Allow automatic model download.
    """

    def __init__(
        self,
        gpu: bool = False,
        *,
        use_angle_cls: bool = True,
        lang: str = "en",
        # --- FSM Beam Search ---
        use_fsm_decode: bool = False,
        fsm_beam_width: int = 5,
        # --- Multi-pass ---
        use_multi_pass: bool = False,
        # --- AI Super-Resolution ---
        use_super_res: bool = False,
        super_res_min_height: int = 32,
        super_res_model: str = "espcn_x2",
        super_res_allow_download: bool = True,
    ) -> None:
        _ = gpu  # kept for backward-compatibility in call sites/tests
        try:
            self.reader = _build_paddle_ocr(use_angle_cls=use_angle_cls, lang=lang)
        except ModuleNotFoundError as exc:
            if exc.name == "paddle":
                pyver = f"{sys.version_info.major}.{sys.version_info.minor}"
                version_hint = ""
                if sys.version_info >= (3, 13):
                    version_hint = (
                        f" Current Python is {pyver}; PaddlePaddle wheels are often "
                        "unavailable for 3.13+."
                    )
                raise RuntimeError(
                    "paddlepaddle is required by PaddleOCR. Install it with: "
                    f"venv/bin/pip install paddlepaddle.{version_hint}"
                ) from exc
            raise

        # --- FSM Beam Search decoder ---
        self._use_fsm = use_fsm_decode
        self._fsm_beam_width = fsm_beam_width

        # --- Multi-pass ---
        self._use_multi_pass = use_multi_pass

        # --- AI Super-Resolution ---
        self._super_resolver = None
        self._sr_min_height = super_res_min_height
        if use_super_res:
            try:
                from app.pipeline.super_resolution import SuperResolver

                self._super_resolver = SuperResolver(
                    model_name=super_res_model,
                    enabled=True,
                    allow_download=super_res_allow_download,
                )
            except Exception as exc:
                logger.warning(
                    "PlateReader: SuperResolver initialisation failed (%s); "
                    "super-resolution disabled",
                    exc,
                )

    @classmethod
    def from_settings(cls) -> "PlateReader":
        """Create a :class:`PlateReader` configured from application settings.

        This factory reads ``app.core.config.settings`` so the import is
        deferred and tests that only import :mod:`app.pipeline.ocr` directly
        are not required to provide a ``DATABASE_URL`` environment variable.
        """
        from app.core.config import settings

        return cls(
            gpu=False,
            use_angle_cls=settings.OCR_USE_ANGLE_CLS,
            lang="en",
            use_fsm_decode=settings.OCR_FSM_DECODE,
            fsm_beam_width=settings.OCR_FSM_BEAM_WIDTH,
            use_multi_pass=settings.OCR_MULTI_PASS,
            use_super_res=settings.OCR_SUPER_RES,
            super_res_min_height=settings.OCR_SUPER_RES_MIN_HEIGHT,
            super_res_model=settings.OCR_SUPER_RES_MODEL,
            super_res_allow_download=settings.OCR_SUPER_RES_ALLOW_DOWNLOAD,
        )

    # ------------------------------------------------------------------
    # Single-pass OCR (one preprocessed image → text + confidence)
    # ------------------------------------------------------------------

    def _ocr_single(self, ocr_input: np.ndarray) -> tuple[str, float]:
        """Run PaddleOCR on a single preprocessed image."""
        result: Any | None = None
        if hasattr(self.reader, "ocr"):
            try:
                result = self.reader.ocr(ocr_input, det=False, rec=True, cls=False)
            except TypeError as exc:
                message = str(exc)
                unsupported_kw = "unexpected keyword argument" in message and (
                    "'det'" in message or "'rec'" in message or "'cls'" in message
                )
                if not unsupported_kw:
                    raise
                result = self.reader.ocr(ocr_input)

        if result is None and hasattr(self.reader, "predict"):
            result = self.reader.predict(ocr_input)

        if result is None:
            raise RuntimeError("PaddleOCR reader has no supported OCR method")

        tokens: list[tuple[str, float]] = []
        _collect_ocr_tokens(result, tokens)

        if not tokens:
            return "", 0.0

        raw_text = _NON_ALNUM_RE.sub(
            "", "".join(token for token, _ in tokens).upper()
        )
        confidence = float(sum(conf for _, conf in tokens) / len(tokens))
        return raw_text, confidence

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def read_plate(
        self, crop: np.ndarray, *, plate_type: str | None = None
    ) -> tuple[str, float]:
        """Read plate text from a BGR crop image.

        Parameters
        ----------
        crop:
            BGR image array of the plate crop.
        plate_type:
            Optional color-based plate type hint (e.g. ``"standard"``,
            ``"temporary"``).  Passed to the FSM beam decoder and
            normaliser to resolve ambiguity.

        The pipeline:
          1. (Optional) AI Super-Resolution on small crops
          2. Single-pass or multi-pass preprocessing
          3. PaddleOCR inference per pass
          4. (Optional) FSM Prefix Beam Search correction
          5. Select best valid result across passes
        """
        started_at = time.perf_counter()

        # Step 1 — AI Super-Resolution on small crops (optional).
        if self._super_resolver is not None:
            crop = self._super_resolver.upscale_if_small(
                crop, min_height=self._sr_min_height
            )

        # Step 2+3 — Preprocessing + OCR (multi-pass or single-pass).
        if self._use_multi_pass:
            raw_text, confidence = self._multi_pass_ocr(crop, plate_type=plate_type)
        else:
            processed = preprocess_clahe(crop)
            ocr_input = (
                cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
                if processed.ndim == 2
                else processed
            )
            raw_text, confidence = self._ocr_single(ocr_input)

        # Step 4 — FSM Prefix Beam Search decoder (optional).
        if self._use_fsm and raw_text:
            from app.pipeline.ctc_decoder import fsm_beam_decode

            raw_text, confidence = fsm_beam_decode(
                raw_text,
                confidence,
                beam_width=self._fsm_beam_width,
                plate_type=plate_type,
            )

        elapsed_ms = (time.perf_counter() - started_at) * 1000
        logger.debug(
            'OCR: raw="%s" conf=%.2f plate_type=%s time=%.1fms',
            raw_text, confidence, plate_type, elapsed_ms,
        )
        return raw_text, confidence

    def _multi_pass_ocr(
        self, crop: np.ndarray, *, plate_type: str | None = None
    ) -> tuple[str, float]:
        """Run OCR with multiple preprocessing variants and pick the best.

        Selection priority:
          1. Valid Romanian plate with highest confidence
          2. Any non-empty text with highest confidence
        """
        from app.pipeline.plate_validator import is_invalid_plate, normalize_plate

        best_valid_text = ""
        best_valid_conf = 0.0
        best_any_text = ""
        best_any_conf = 0.0

        for pass_name, preprocess_fn in _PREPROCESS_PASSES:
            try:
                processed = preprocess_fn(crop)
                ocr_input = (
                    cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
                    if processed.ndim == 2
                    else processed
                )
                text, conf = self._ocr_single(ocr_input)
            except Exception as exc:
                logger.debug(
                    "OCR multi-pass: %s failed: %s", pass_name, exc
                )
                continue

            if not text:
                continue

            # Track best-any (fallback).
            if conf > best_any_conf:
                best_any_text = text
                best_any_conf = conf

            # Check if it normalises to a valid plate.
            normalized = normalize_plate(text, plate_type=plate_type)
            if not is_invalid_plate(normalized) and conf > best_valid_conf:
                best_valid_text = text
                best_valid_conf = conf
                logger.debug(
                    "OCR multi-pass: %s produced valid plate %r (conf=%.3f)",
                    pass_name,
                    text,
                    conf,
                )

        # Prefer valid over any.
        if best_valid_text:
            return best_valid_text, best_valid_conf
        return best_any_text, best_any_conf
