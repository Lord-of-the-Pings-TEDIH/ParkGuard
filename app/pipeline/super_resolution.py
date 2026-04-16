"""app/pipeline/super_resolution.py

AI-based Super-Resolution upscaling for small license-plate crops.

Uses OpenCV's ``cv2.dnn_superres`` module (available in
``opencv-contrib-python-headless``) with an ESPCN model for fast, on-device
inference.  On first use the model weights are downloaded automatically to
``~/.parkguard/sr_models/`` and cached for all subsequent runs.

When the DNN backend is unavailable (model not downloaded, or the installed
OpenCV flavour lacks ``dnn_superres``), the module falls back transparently
to Lanczos4 upscaling followed by a mild unsharp-mask sharpening pass —
which is already a tangible improvement over plain bicubic interpolation.

Typical integration
-------------------
    resolver = SuperResolver(model_name="espcn_x2", enabled=True)
    # Apply SR only if the crop is small (avoids overhead on clear footage):
    upscaled = resolver.upscale_if_small(crop, min_height=32)
"""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path
from typing import ClassVar

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model registry
# Each entry: algorithm_name, scale_factor, download_url, local_filename
# ---------------------------------------------------------------------------
_MODELS: dict[str, tuple[str, int, str, str]] = {
    "espcn_x2": (
        "espcn",
        2,
        "https://raw.githubusercontent.com/fannymonori/TF-ESPCN/master/export/ESPCN_x2.pb",
        "ESPCN_x2.pb",
    ),
    "espcn_x4": (
        "espcn",
        4,
        "https://raw.githubusercontent.com/fannymonori/TF-ESPCN/master/export/ESPCN_x4.pb",
        "ESPCN_x4.pb",
    ),
    "edsr_x2": (
        "edsr",
        2,
        "https://github.com/Saafke/EDSR_Tensorflow/raw/master/models/EDSR_x2.pb",
        "EDSR_x2.pb",
    ),
    "edsr_x4": (
        "edsr",
        4,
        "https://github.com/Saafke/EDSR_Tensorflow/raw/master/models/EDSR_x4.pb",
        "EDSR_x4.pb",
    ),
}

_DEFAULT_MODELS_DIR: Path = Path.home() / ".parkguard" / "sr_models"


class SuperResolver:
    """AI super-resolution wrapper with automatic model management.

    Parameters
    ----------
    model_name:
        One of ``"espcn_x2"``, ``"espcn_x4"``, ``"edsr_x2"``, ``"edsr_x4"``.
        ``"espcn_x2"`` is the recommended default — it is small (~4 MB),
        fast on CPU, and delivers good sharpness for plate text at 2× scale.
    models_dir:
        Directory where downloaded model weights are cached.
        Defaults to ``~/.parkguard/sr_models/``.
    enabled:
        When ``False`` the resolver acts as a no-op (returns input unchanged).
        Useful for disabling SR from configuration without code changes.
    allow_download:
        When ``True`` (default) the model weights are downloaded automatically
        if not already cached.  Set to ``False`` in offline / test environments.
    """

    # Class-level cache: one instance per unique model key.
    _instance_cache: ClassVar[dict[str, SuperResolver]] = {}

    def __init__(
        self,
        model_name: str = "espcn_x2",
        *,
        models_dir: Path | str | None = None,
        enabled: bool = True,
        allow_download: bool = True,
    ) -> None:
        self._enabled = enabled
        self._scale: int = 2
        self._sr: object | None = None  # cv2.dnn_superres.DnnSuperResImpl

        if not enabled:
            logger.debug("SuperResolver: disabled by configuration")
            return

        model_key = model_name.lower().replace("-", "_")
        if model_key not in _MODELS:
            logger.warning(
                "SuperResolver: unknown model %r — SR disabled. "
                "Valid choices: %s",
                model_name,
                ", ".join(_MODELS),
            )
            return

        algorithm, scale, url, filename = _MODELS[model_key]
        self._scale = scale

        models_dir_path = Path(models_dir) if models_dir else _DEFAULT_MODELS_DIR
        model_path = models_dir_path / filename

        if not model_path.exists():
            if allow_download:
                model_path = self._try_download(url, model_path) or model_path
            else:
                logger.info(
                    "SuperResolver: model %s not found and download is disabled; "
                    "using enhanced Lanczos fallback",
                    filename,
                )
                return

        if not model_path.exists():
            logger.warning(
                "SuperResolver: model %s unavailable; using enhanced Lanczos fallback",
                filename,
            )
            return

        self._sr = self._load_sr_model(algorithm, scale, model_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _try_download(url: str, dest: Path) -> Path | None:
        """Download the model file to *dest*; returns the path on success."""
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            logger.info("SuperResolver: downloading model from %s …", url)
            urllib.request.urlretrieve(url, dest)
            logger.info("SuperResolver: model cached at %s", dest)
            return dest
        except Exception as exc:
            logger.warning("SuperResolver: download failed — %s", exc)
            return None

    @staticmethod
    def _load_sr_model(algorithm: str, scale: int, path: Path) -> object | None:
        """Load and return a ``DnnSuperResImpl`` instance, or ``None``."""
        try:
            # cv2.dnn_superres is only present in opencv-contrib builds.
            sr = cv2.dnn_superres.DnnSuperResImpl_create()  # type: ignore[attr-defined]
            sr.readModel(str(path))
            sr.setModel(algorithm, scale)
            logger.info(
                "SuperResolver: loaded %s x%d from %s",
                algorithm.upper(),
                scale,
                path.name,
            )
            return sr
        except AttributeError:
            logger.warning(
                "cv2.dnn_superres is not available in the installed OpenCV build. "
                "Switch to opencv-contrib-python-headless for DNN super-resolution. "
                "Using enhanced Lanczos fallback instead."
            )
            return None
        except Exception as exc:
            logger.warning(
                "SuperResolver: could not load model %s — %s; "
                "using enhanced Lanczos fallback",
                path.name,
                exc,
            )
            return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_available(self) -> bool:
        """``True`` when the DNN model is loaded and ready."""
        return self._sr is not None

    def upscale(self, img: np.ndarray) -> np.ndarray:
        """Upscale *img* by the configured scale factor.

        Uses the DNN super-resolution model when available, otherwise falls
        back to Lanczos4 interpolation + unsharp-mask sharpening.

        Parameters
        ----------
        img:
            BGR or grayscale ``numpy`` array.

        Returns
        -------
        np.ndarray
            Upscaled image in the same colour space as the input.
        """
        if self._sr is not None:
            try:
                if img.ndim == 2:  # Grayscale — DNN SR expects BGR.
                    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    upscaled = self._sr.upsample(bgr)  # type: ignore[union-attr]
                    return cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
                return self._sr.upsample(img)  # type: ignore[union-attr]
            except Exception as exc:
                logger.warning(
                    "SuperResolver: DNN upsample failed (%s); using enhanced Lanczos",
                    exc,
                )

        return _enhanced_lanczos(img, self._scale)

    def upscale_if_small(
        self, img: np.ndarray, *, min_height: int = 32
    ) -> np.ndarray:
        """Upscale *img* only when its height is below *min_height*.

        This guard prevents unnecessary overhead on large, clear crops where
        bicubic resizing inside ``preprocess_crop`` is already adequate.

        Parameters
        ----------
        img:
            Input crop (BGR or grayscale).
        min_height:
            Height threshold in pixels.  Crops with ``h >= min_height`` are
            returned unchanged.

        Returns
        -------
        np.ndarray
            Upscaled image if ``h < min_height``, otherwise the original.
        """
        if not self._enabled:
            return img
        h = img.shape[0]
        if h < min_height:
            logger.debug(
                "SuperResolver: crop h=%d < min_height=%d — applying SR x%d",
                h,
                min_height,
                self._scale,
            )
            return self.upscale(img)
        return img


# ---------------------------------------------------------------------------
# Enhanced Lanczos fallback (no model required)
# ---------------------------------------------------------------------------


def _enhanced_lanczos(img: np.ndarray, scale: int) -> np.ndarray:
    """Lanczos4 upscale + unsharp mask — better than plain bicubic on text.

    Not true AI super-resolution, but a meaningful improvement over the
    raw bicubic / area interpolation used before plate crops reach the OCR.
    """
    h, w = img.shape[:2]
    up = cv2.resize(
        img,
        (w * scale, h * scale),
        interpolation=cv2.INTER_LANCZOS4,
    )
    # Mild unsharp mask to enhance character edges.
    blurred = cv2.GaussianBlur(up, (0, 0), sigmaX=1.0)
    sharpened = cv2.addWeighted(up, 1.5, blurred, -0.5, 0)
    return sharpened
