"""OCR image preprocessing and PaddleOCR wrapper utilities."""

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


def preprocess_crop(crop: np.ndarray) -> np.ndarray:
    """Apply OCR preprocessing to a plate crop."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    h, w = gray.shape[:2]
    target_height = 64
    scale = target_height / max(1, h)
    target_width = max(1, int(round(w * scale)))
    interpolation = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
    return cv2.resize(gray, (target_width, target_height), interpolation=interpolation)


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


class PlateReader:
    """Read plate text from image crops with PaddleOCR."""

    def __init__(
        self,
        gpu: bool = False,
        *,
        use_angle_cls: bool = True,
        lang: str = "en",
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

    def read_plate(self, crop: np.ndarray) -> tuple[str, float]:
        started_at = time.perf_counter()
        processed = preprocess_crop(crop)
        ocr_input = (
            cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
            if processed.ndim == 2
            else processed
        )

        if hasattr(self.reader, "predict"):
            result = self.reader.predict(ocr_input)
        else:
            result = self.reader.ocr(ocr_input, det=False, rec=True, cls=False)

        tokens: list[tuple[str, float]] = []
        _collect_ocr_tokens(result, tokens)

        if not tokens:
            raw_text = ""
            confidence = 0.0
        else:
            raw_text = _NON_ALNUM_RE.sub(
                "", "".join(token for token, _ in tokens).upper()
            )
            confidence = float(sum(conf for _, conf in tokens) / len(tokens))

        elapsed_ms = (time.perf_counter() - started_at) * 1000
        logger.debug('OCR: raw="%s" conf=%.2f time=%.1fms', raw_text, confidence, elapsed_ms)
        return raw_text, confidence
