from pydantic_settings import BaseSettings
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    app_name: str = "AREP Student Dropout Analytics"
    app_version: str = "1.0.0"
    debug: bool = False
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:3000"]
    dataset_path: str = str(BASE_DIR / "data" / "dataset.csv")
    models_dir: str = str(BASE_DIR / "data" / "trained_models")

    class Config:
        env_file = ".env"


settings = Settings()
