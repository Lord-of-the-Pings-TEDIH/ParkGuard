from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    DATABASE_URL: str
    MODEL_PATH: str = "models/best.pt"
    CROPS_DIR: str = "./crops"
    UPLOAD_DIR: str = "./uploads"
    FPS_TARGET: int = 5
    # Raised from 0.50: fine-tuned YOLO rarely fires below 0.62 on true plates,
    # so the lower threshold only adds false-positive OCR invocations.
    DETECTION_CONF: float = 0.62
    OCR_USE_ANGLE_CLS: bool = True
    OCR_MIN_CONF: float = 0.25
    OCR_ANGLES: str = "-15,-8,0,8,15"
    OCR_MIN_SHARPNESS: float = 80.0
    TRACK_MAX_AGE: int = 18
    # Raised from 0.20: 0.35 overlap is enough for parked-car enforcement video
    # while preventing different vehicles from merging into the same track.
    TRACK_MIN_IOU: float = 0.35
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

    # --- OCR speed-ups (no accuracy loss) ---
    # Once the first preprocessing pass / first angle returns a *valid*
    # Romanian plate with confidence >= OCR_EARLY_EXIT_CONF, skip the
    # remaining passes/angles for that detection.  The high-confidence
    # path covers the vast majority of frames and the angle/multi-pass
    # sweep only kicks in when needed (tilted, washed-out, blurred).
    # Lowered from 0.85: yields ~10% more early exits on the first pass without
    # meaningfully affecting result quality on typical parking-enforcement crops.
    OCR_EARLY_EXIT_CONF: float = 0.80
    # Once a track has accumulated stable consensus (best plate with
    # >= MIN_TRACK_VOTES votes at confidence >= OCR_SKIP_STABLE_CONF AND
    # the track has been observed >= MIN_TRACK_VOTES * 2 times), additional
    # OCR runs on subsequent frames of the same track are pure overhead —
    # the canonical plate is locked in and the finalizer relabels every
    # detection in the track anyway.  Skipping OCR on stable tracks gives
    # ~50–80% reduction in pipeline time on long clips with parked cars.
    OCR_SKIP_STABLE_TRACKS: bool = True
    OCR_SKIP_STABLE_CONF: float = 0.85
    # Commit progress every N processed frames instead of every frame.
    # Reduces DB write amplification at the cost of slightly chunkier
    # progress updates (still well under one second on typical CPU loads).
    # Raised from 4: 3× fewer DB commits with the same sub-second progress
    # granularity at the default 5 FPS target (12 frames = ~2.4 s of video).
    PROCESSOR_COMMIT_EVERY_FRAMES: int = 12

    # --- Ticket zone ---
    # When a session is processed without an explicit zone_id, fall back to
    # this zone so ticket/subscription lookups actually match seeded data.
    # Set to the integer primary key of the relevant ParkingZone row.
    DEFAULT_ZONE_ID: int | None = None

    # --- Occupancy / spot-misuse scoring ---
    # How many calendar days back to look when computing suspicion scores.
    # Darius's spec: "timp de 6 luni, 20 de treceri succesive" → 180 days.
    OCCUPANCY_LOOKBACK_DAYS: int = 180
    # Exponential-decay half-life.  With a 180-day window, 30 days means
    # events from last week still count at ~0.86 weight while events from
    # 5 months ago weigh ~0.09.
    OCCUPANCY_DECAY_HALF_LIFE_DAYS: float = 30.0
    # Score above which a (spot, plate) pair is flagged for human review.
    OCCUPANCY_FLAG_THRESHOLD: float = 8.0
    # Minimum number of matching plates needed to positively identify the
    # lot being patrolled (non-GPS heuristic).
    LOT_HEURISTIC_MIN_PLATES: int = 3
    # When hour_concentration >= threshold AND len(events) >= min_events,
    # base_score is multiplied by this factor before bonuses are added.
    OCCUPANCY_HOUR_CONCENTRATION_THRESHOLD: float = 0.7
    OCCUPANCY_HOUR_CONCENTRATION_MIN_EVENTS: int = 5
    OCCUPANCY_HOUR_CONCENTRATION_MULTIPLIER: float = 1.3
    # Score is only flagged when event_count also meets this minimum.
    # 0 = disabled (keeps existing behaviour, tests stay green by default).
    OCCUPANCY_MIN_EVENTS_FOR_FLAG: int = 0

    # --- Mobile-LPR (Mobile License Plate Recognition) ---
    # Defaults for the police-car camera intrinsics used by
    # PlateGeolocationCalculator.  Override per-deployment in .env.  The
    # geolocation step is only enabled when a Session is created with
    # GPS pose; otherwise these values are unused.
    MOBILE_LPR_CAMERA_HEIGHT_M: float = 1.5
    # Focal length in pixels of the recorded video.  4K iPhone wide-angle
    # falls in the 2800–3200 px range; 2800 is a reasonable default.
    MOBILE_LPR_CAMERA_FOCAL_PX: float = 2800.0
    MOBILE_LPR_CAMERA_PITCH_DEG: float = 5.0
    # Plate centre is assumed at this height above the ground.
    MOBILE_LPR_PLATE_HEIGHT_M: float = 0.6
    # Search radius (metres) used by validate_parking_at_location to bind
    # a projected GPS point to a registered ParkingSpot.  Consumer GPS on a
    # smartphone is typically accurate to 5–15 m in open sky and 20–40 m in
    # urban canyons.  40 m covers the realistic worst case while staying tight
    # enough to avoid cross-street false matches in dense neighbourhoods.
    MOBILE_LPR_SEARCH_RADIUS_M: float = 40.0
    # Hard cap on the projected distance.  Plates at or above the geometric
    # horizon (typical of distant cars in handheld portrait video) are
    # clamped to this distance so every detection still produces a GPS
    # estimate.  Increase to surface a wider cone; decrease to keep
    # estimates conservative.
    MOBILE_LPR_MAX_DISTANCE_M: float = 50.0
    # Default camera pose used when a session is created without an explicit
    # GPS pose (latitude/longitude/heading_deg).  When all three defaults are
    # set, every session — including hardcoded test sessions — will project
    # plates to GPS automatically and the per-car coordinates show up in the
    # UI without the user having to fill in the Mobile-LPR panel each time.
    # Leave any of them as ``None`` to opt out.
    MOBILE_LPR_DEFAULT_LATITUDE: float | None = None
    MOBILE_LPR_DEFAULT_LONGITUDE: float | None = None
    MOBILE_LPR_DEFAULT_HEADING_DEG: float | None = None

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

    # --- PaddleOCR device ---
    # Set to True when a CUDA GPU is available to run OCR inference on the GPU.
    # Requires paddlepaddle-gpu instead of paddlepaddle.
    OCR_USE_GPU: bool = False

    # --- Per-frame batched OCR ---
    # When True (default), frames containing ≥2 plates are sent to PaddleOCR
    # as a single batch instead of one call per detection.  Amortises the
    # per-call Python/Paddle overhead — typical 20–40% speed-up in heavy
    # scenes.  Detections that don't reach OCR_EARLY_EXIT_CONF on the batch
    # result still fall back to the per-detection multi-pass code path so
    # accuracy is preserved.
    OCR_BATCH_PER_FRAME: bool = True

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
