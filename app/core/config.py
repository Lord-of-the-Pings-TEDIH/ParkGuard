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

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
