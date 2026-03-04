from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    DATABASE_URL: str
    MODEL_PATH: str
    CROPS_DIR: str = "./crops"
    UPLOAD_DIR: str = "./uploads"
    FPS_TARGET: int = 5
    DETECTION_CONF: float = 0.50

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
