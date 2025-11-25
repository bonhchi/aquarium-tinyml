# ai_gateway/config.py
from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DATA_ROOT: Path = Path("./dataset")  # nơi sẽ lưu raw để train
    MODEL_ROOT: Path = Path("./ml/artifacts")  # nơi train.py export .tflite
    META_DIR: Path = Path("./ai_gateway/meta")  # nơi lưu metadata model
    TRAINING_DATA_FILE: Path = Path(
        "./dataset/interim/aquarium_tinyml_features.csv"
    )  # default file làm đầu vào train

    class Config:
        env_file = ".env"


def _resolve_training_file(cfg: Settings) -> Path:
    """
    Nếu file default chưa tồn tại, thử fallback sang sample.
    Giữ reference để API dataset có thể thông báo đường dẫn đang dùng.
    """
    if cfg.TRAINING_DATA_FILE.exists():
        return cfg.TRAINING_DATA_FILE

    fallback = cfg.DATA_ROOT / "interim" / "aquarium_tinyml_interim_sample.csv"
    if fallback.exists():
        return fallback
    return cfg.TRAINING_DATA_FILE


settings = Settings()
settings.DATA_ROOT.mkdir(parents=True, exist_ok=True)
(settings.DATA_ROOT / "raw").mkdir(parents=True, exist_ok=True)
settings.META_DIR.mkdir(parents=True, exist_ok=True)
settings.TRAINING_DATA_FILE = _resolve_training_file(settings)
