from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    DATABASE_URL: str
    MODEL_PATH: str = "models/best.pt"
    CROPS_DIR: str = "./crops"
    UPLOAD_DIR: str = "./uploads"
    FPS_TARGET: int = 5
    DETECTION_CONF: float = 0.50
    OCR_USE_ANGLE_CLS: bool = True
    OCR_MIN_CONF: float = 0.25
    OCR_ANGLES: str = "-15,-8,0,8,15"
    OCR_MIN_SHARPNESS: float = 80.0
    TRACK_MAX_AGE: int = 18
    TRACK_MIN_IOU: float = 0.20
    MIN_TRACK_VOTES: int = 3

    # --- FSM Prefix Beam Search decoder ---
    # Replaces greedy OCR + post-hoc regex repair with a mathematically
    # constrained beam search over the Romanian plate grammar.
    OCR_FSM_DECODE: bool = True
    OCR_FSM_BEAM_WIDTH: int = 5

    # --- Multi-pass OCR preprocessing ---
    # Runs OCR with 3 preprocessing variants (CLAHE, adaptive threshold,
    # inverted) and selects the best valid result.  Improves accuracy
    # in challenging lighting conditions at the cost of ~3× OCR time.
    OCR_MULTI_PASS: bool = True

    # --- Mobile-LPR (Mobile License Plate Recognition) ---
    # Defaults for the police-car camera intrinsics used by
    # PlateGeolocationCalculator.  Override per-deployment in .env.  The
    # geolocation step is only enabled when a Session is created with
    # GPS pose; otherwise these values are unused.
    MOBILE_LPR_CAMERA_HEIGHT_M: float = 1.5
    MOBILE_LPR_CAMERA_FOCAL_PX: float = 1000.0
    MOBILE_LPR_CAMERA_PITCH_DEG: float = 5.0
    # Plate centre is assumed at this height above the ground.
    MOBILE_LPR_PLATE_HEIGHT_M: float = 0.6
    # Search radius (metres) used by validate_parking_at_location to bind
    # a projected GPS point to a registered ParkingSpot.
    MOBILE_LPR_SEARCH_RADIUS_M: float = 3.0

    # --- AI Super-Resolution ---
    # When enabled, crops smaller than OCR_SUPER_RES_MIN_HEIGHT are upscaled
    # by an AI model (ESPCN by default) before OCR, improving accuracy on
    # distant vehicles.  The model weights are downloaded automatically on
    # first use to ~/.parkguard/sr_models/.
    # Requires opencv-contrib-python-headless (see requirements.txt).
    # Set to False to use the enhanced Lanczos fallback instead.
    OCR_SUPER_RES: bool = False
    OCR_SUPER_RES_MIN_HEIGHT: int = 32
    # Available: "espcn_x2", "espcn_x4", "edsr_x2", "edsr_x4"
    # espcn_x2 is recommended: ~4 MB, fast on CPU.
    OCR_SUPER_RES_MODEL: str = "espcn_x2"
    OCR_SUPER_RES_ALLOW_DOWNLOAD: bool = True

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
