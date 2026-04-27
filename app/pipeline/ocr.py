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
import threading
import time
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)
_NON_ALNUM_RE = re.compile(r"[^A-Z0-9]")


# ---------------------------------------------------------------------------
# Multi-pass preprocessing pipelines
# ---------------------------------------------------------------------------


def _to_gray_denoised(crop: np.ndarray) -> np.ndarray:
    """Convert to grayscale and bilateral-filter once — result is shared across passes.

    Bilateral filter is expensive; computing it here and passing the result as
    ``gray_denoised`` to each preprocessing function avoids running it 3× per
    detection during multi-pass OCR.
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    return cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)


def preprocess_clahe(
    crop: np.ndarray, *, gray_denoised: np.ndarray | None = None
) -> np.ndarray:
    """Standard CLAHE preprocessing: grayscale → bilateral denoise → CLAHE → resize.

    Parameters
    ----------
    gray_denoised:
        Pre-computed grayscale + bilateral-filtered image.  Pass this when
        calling multiple preprocessing variants on the same crop to avoid
        repeating the expensive bilateral filter.
    """
    denoised = gray_denoised if gray_denoised is not None else _to_gray_denoised(crop)
    # Smaller crops need a tighter tile grid and higher clip limit for contrast.
    h = denoised.shape[0]
    clip_limit = 3.0 if h < 40 else 2.0
    tile_grid = (4, 4) if h < 40 else (8, 8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    return _resize_to_target(clahe.apply(denoised))


def preprocess_adaptive(
    crop: np.ndarray, *, gray_denoised: np.ndarray | None = None
) -> np.ndarray:
    """Adaptive-threshold binarisation: handles uneven illumination."""
    denoised = gray_denoised if gray_denoised is not None else _to_gray_denoised(crop)
    # Smaller blockSize for low-height crops; must be odd and >= 3.
    h = denoised.shape[0]
    block_size = 11 if h < 40 else 15

    binary = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=block_size,
        C=5,
    )

    # Morphological opening — remove small noise dots.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    return _resize_to_target(binary)


def preprocess_inverted(
    crop: np.ndarray, *, gray_denoised: np.ndarray | None = None
) -> np.ndarray:
    """Inverted binarisation: for dark-background or inverted plates."""
    denoised = gray_denoised if gray_denoised is not None else _to_gray_denoised(crop)
    h = denoised.shape[0]
    block_size = 11 if h < 40 else 15

    binary = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=block_size,
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


def _build_paddle_ocr(*, use_angle_cls: bool, lang: str, use_gpu: bool = False) -> Any:
    os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
    os.environ["FLAGS_use_mkldnn"] = "0"
    os.environ["FLAGS_use_pir_api"] = "0"

    # We only feed PaddleOCR pre-cropped plate images, so the full OCR
    # pipeline's text-detection stage is wasted work — and on some
    # paddlepaddle wheels it crashes the PIR runtime
    # (NotImplementedError: ConvertPirAttribute2RuntimeAttribute).
    # Use the TextRecognition predictor directly when it is available.
    try:
        from paddleocr import TextRecognition  # type: ignore[attr-defined]
    except Exception:
        TextRecognition = None  # type: ignore[assignment]

    if TextRecognition is not None:
        try:
            tr_kwargs: dict[str, Any] = {}
            tr_sig = inspect.signature(TextRecognition.__init__)
            if "device" in tr_sig.parameters and use_gpu:
                tr_kwargs["device"] = "gpu"
            if "enable_mkldnn" in tr_sig.parameters:
                tr_kwargs["enable_mkldnn"] = False
            reader = TextRecognition(**tr_kwargs)
            logger.info("OCR backend: paddleocr.TextRecognition (det stage skipped)")
            return reader
        except Exception as exc:
            logger.warning(
                "TextRecognition init failed (%s); falling back to PaddleOCR pipeline",
                exc,
            )

    from paddleocr import PaddleOCR

    init_signature = inspect.signature(PaddleOCR.__init__)
    init_kwargs: dict[str, Any] = {"lang": lang}

    if "use_textline_orientation" in init_signature.parameters:
        init_kwargs["use_textline_orientation"] = use_angle_cls
    elif "use_angle_cls" in init_signature.parameters:
        init_kwargs["use_angle_cls"] = use_angle_cls

    if "show_log" in init_signature.parameters:
        init_kwargs["show_log"] = False

    # Belt-and-braces: also pass enable_mkldnn=False if the constructor
    # accepts it (newer PaddleOCR builds expose it directly).
    if "enable_mkldnn" in init_signature.parameters:
        init_kwargs["enable_mkldnn"] = False
    if "text_detection_enable_mkldnn" in init_signature.parameters:
        init_kwargs["text_detection_enable_mkldnn"] = False
    if "text_recognition_enable_mkldnn" in init_signature.parameters:
        init_kwargs["text_recognition_enable_mkldnn"] = False

    # GPU device selection — newer API uses "device", older uses "use_gpu".
    if use_gpu:
        if "device" in init_signature.parameters:
            init_kwargs["device"] = "gpu"
        elif "use_gpu" in init_signature.parameters:
            init_kwargs["use_gpu"] = True

    try:
        return PaddleOCR(**init_kwargs)
    except Exception as exc:
        if use_gpu:
            import logging as _log
            _log.getLogger(__name__).warning(
                "PaddleOCR GPU init failed (%s); retrying on CPU", exc
            )
            init_kwargs.pop("device", None)
            init_kwargs.pop("use_gpu", None)
            return PaddleOCR(**init_kwargs)
        raise


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
        # --- Early-exit on first valid + high-conf pass ---
        early_exit_conf: float = 0.85,
        # --- AI Super-Resolution ---
        use_super_res: bool = False,
        super_res_min_height: int = 32,
        super_res_model: str = "espcn_x2",
        super_res_allow_download: bool = True,
    ) -> None:
        try:
            self.reader = _build_paddle_ocr(use_angle_cls=use_angle_cls, lang=lang, use_gpu=gpu)
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

        # Resolved on first OCR call: "legacy" (det/rec/cls kwargs),
        # "ocr" (positional only), or "predict".
        self._ocr_call_mode: str | None = None

        # PaddleOCR's TextRecognition predictor holds mutable internal buffers
        # and is not safe under concurrent access.  When multiple processing
        # sessions dispatch OCR through asyncio.to_thread they all hit the
        # singleton PlateReader from different threads — without this lock the
        # second caller can corrupt the first one's intermediate tensors and
        # produce empty / wrong text or crash the runtime.
        self._infer_lock = threading.Lock()

        # --- FSM Beam Search decoder ---
        self._use_fsm = use_fsm_decode
        self._fsm_beam_width = fsm_beam_width

        # --- Multi-pass ---
        self._use_multi_pass = use_multi_pass
        self._early_exit_conf = max(0.0, min(1.0, float(early_exit_conf)))

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
            gpu=settings.OCR_USE_GPU,
            use_angle_cls=settings.OCR_USE_ANGLE_CLS,
            lang="en",
            use_fsm_decode=settings.OCR_FSM_DECODE,
            fsm_beam_width=settings.OCR_FSM_BEAM_WIDTH,
            use_multi_pass=settings.OCR_MULTI_PASS,
            early_exit_conf=settings.OCR_EARLY_EXIT_CONF,
            use_super_res=settings.OCR_SUPER_RES,
            super_res_min_height=settings.OCR_SUPER_RES_MIN_HEIGHT,
            super_res_model=settings.OCR_SUPER_RES_MODEL,
            super_res_allow_download=settings.OCR_SUPER_RES_ALLOW_DOWNLOAD,
        )

    # ------------------------------------------------------------------
    # Single-pass OCR (one preprocessed image → text + confidence)
    # ------------------------------------------------------------------

    def _invoke_ocr(self, ocr_input: np.ndarray) -> Any:
        """Call the underlying PaddleOCR reader, caching which API works."""
        with self._infer_lock:
            mode = self._ocr_call_mode
            if mode == "legacy":
                return self.reader.ocr(ocr_input, det=False, rec=True, cls=False)
            if mode == "ocr":
                return self.reader.ocr(ocr_input)
            if mode == "predict":
                return self.reader.predict(ocr_input)

            # First call: probe once and remember.
            if hasattr(self.reader, "ocr"):
                try:
                    result = self.reader.ocr(ocr_input, det=False, rec=True, cls=False)
                    self._ocr_call_mode = "legacy"
                    return result
                except TypeError as exc:
                    msg = str(exc)
                    if "unexpected keyword argument" not in msg:
                        raise
                result = self.reader.ocr(ocr_input)
                self._ocr_call_mode = "ocr"
                return result
            if hasattr(self.reader, "predict"):
                result = self.reader.predict(ocr_input)
                self._ocr_call_mode = "predict"
                return result
            raise RuntimeError("PaddleOCR reader has no supported OCR method")

    def _invoke_ocr_batch(self, inputs: list[np.ndarray]) -> list[Any]:
        """Run a single batched OCR call covering ``inputs`` in order.

        Returns a list of per-image raw outputs aligned with ``inputs``.  When
        the underlying reader does not accept a list (older PaddleOCR builds),
        we transparently fall back to N single-image calls so the caller does
        not need to special-case the API surface.
        """
        if not inputs:
            return []

        # Probe the call mode against the first crop so subsequent batch calls
        # take the cached fast path.
        if self._ocr_call_mode is None:
            self._invoke_ocr(inputs[0])

        mode = self._ocr_call_mode
        try:
            with self._infer_lock:
                if mode == "legacy":
                    result = self.reader.ocr(inputs, det=False, rec=True, cls=False)
                elif mode == "ocr":
                    result = self.reader.ocr(inputs)
                elif mode == "predict":
                    result = self.reader.predict(inputs)
                else:
                    raise RuntimeError("PaddleOCR reader has no supported OCR method")
        except Exception as exc:
            logger.debug(
                "Batched OCR failed (mode=%s, n=%d): %s — falling back to per-image",
                mode, len(inputs), exc,
            )
            return [self._invoke_ocr(img) for img in inputs]

        # PaddleOCR usually returns a list with one entry per input image.
        # If the shapes don't line up we fall back to per-image so we never
        # silently mis-pair OCR text to bboxes.
        if isinstance(result, list) and len(result) == len(inputs):
            return result
        logger.debug(
            "Batched OCR returned %d entries for %d inputs — falling back to per-image",
            len(result) if isinstance(result, list) else -1, len(inputs),
        )
        return [self._invoke_ocr(img) for img in inputs]

    def _ocr_single(self, ocr_input: np.ndarray) -> tuple[str, float]:
        """Run PaddleOCR on a single preprocessed image."""
        try:
            result = self._invoke_ocr(ocr_input)
        except Exception as exc:
            logger.warning("PaddleOCR call failed: %s", exc, exc_info=True)
            return "", 0.0

        tokens: list[tuple[str, float]] = []
        _collect_ocr_tokens(result, tokens)

        if not tokens:
            logger.debug(
                "OCR returned no tokens (mode=%s, raw=%r)",
                self._ocr_call_mode, result,
            )
            return "", 0.0

        raw_text = _NON_ALNUM_RE.sub(
            "", "".join(token for token, _ in tokens).upper()
        )
        # Aggregate token confidences with the *minimum* — for a license plate
        # one mis-read character invalidates the whole string, so the weakest
        # token is the honest signal. Mean used to mask single-character
        # failures (e.g. mean(0.99, 0.99, 0.30) = 0.76 looked confident).
        confidence = float(min(conf for _, conf in tokens))
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

    def read_plates_batch(
        self,
        crops: list[np.ndarray],
        *,
        plate_types: list[str | None] | None = None,
    ) -> list[tuple[str, float]]:
        """Single-pass batched OCR over multiple crops.

        Runs CLAHE preprocessing on every crop, fires *one* PaddleOCR call for
        the whole batch, then returns ``(text, confidence)`` per crop in the
        same order.  This amortises the Python/Paddle per-call overhead — on
        frames with 2+ plates the speed-up is 20–40 % over per-crop calls.

        Multi-pass (adaptive/inverted) and multi-angle sweeps are *not* applied
        here — the caller is expected to fall back to :meth:`read_plate` for
        any crop where the batched single-pass result is empty or low-confidence.
        """
        if not crops:
            return []
        if plate_types is None:
            plate_types = [None] * len(crops)
        elif len(plate_types) != len(crops):
            raise ValueError("plate_types length must match crops")

        inputs: list[np.ndarray] = []
        for crop in crops:
            if self._super_resolver is not None:
                crop = self._super_resolver.upscale_if_small(
                    crop, min_height=self._sr_min_height
                )
            processed = preprocess_clahe(crop)
            ocr_input = (
                cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
                if processed.ndim == 2
                else processed
            )
            inputs.append(ocr_input)

        try:
            raw_results = self._invoke_ocr_batch(inputs)
        except Exception as exc:
            logger.warning("Batched PaddleOCR call failed: %s", exc, exc_info=True)
            return [("", 0.0)] * len(crops)

        results: list[tuple[str, float]] = []
        for idx, raw in enumerate(raw_results):
            tokens: list[tuple[str, float]] = []
            _collect_ocr_tokens(raw, tokens)
            if not tokens:
                results.append(("", 0.0))
                continue
            text = _NON_ALNUM_RE.sub(
                "", "".join(token for token, _ in tokens).upper()
            )
            confidence = float(min(conf for _, conf in tokens))

            if self._use_fsm and text:
                from app.pipeline.ctc_decoder import fsm_beam_decode

                text, confidence = fsm_beam_decode(
                    text,
                    confidence,
                    beam_width=self._fsm_beam_width,
                    plate_type=plate_types[idx],
                )
            results.append((text, confidence))

        return results

    def _multi_pass_ocr(
        self, crop: np.ndarray, *, plate_type: str | None = None
    ) -> tuple[str, float]:
        """Run OCR with multiple preprocessing variants and pick the best.

        Selection priority:
          1. Valid Romanian plate with highest confidence
          2. Any non-empty text with highest confidence

        When the first pass already returns a *valid* plate at confidence
        >= ``early_exit_conf``, we short-circuit instead of running the
        remaining preprocessing variants.  CLAHE (the first pass) handles
        ~70% of typical lighting conditions on its own; the adaptive and
        inverted passes are there as fallbacks for the harder cases.
        """
        from app.pipeline.plate_validator import is_invalid_plate, normalize_plate

        best_valid_text = ""
        best_valid_conf = 0.0
        best_any_text = ""
        best_any_conf = 0.0

        # Compute grayscale + bilateral denoise once — reused by all three passes
        # instead of running the expensive bilateral filter 3× per detection.
        gray_denoised = _to_gray_denoised(crop)

        for pass_name, preprocess_fn in _PREPROCESS_PASSES:
            try:
                processed = preprocess_fn(crop, gray_denoised=gray_denoised)
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
                if best_valid_conf >= self._early_exit_conf:
                    return best_valid_text, best_valid_conf

        # Prefer valid over any.
        if best_valid_text:
            return best_valid_text, best_valid_conf
        return best_any_text, best_any_conf
