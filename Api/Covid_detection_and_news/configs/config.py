from pathlib import Path


class Config:
    WEIGHT_PATH = Path("./weights")
    WEIGHT_PATH.mkdir(parents=True, exist_ok=True)
    SCALER_PATH = WEIGHT_PATH / "scalers"
    MODEL_PATH = WEIGHT_PATH / "models"

class AssetsConfig:
    
    ASSETS_PATH = Path("./assets")
    ASSETS_PATH.mkdir(parents=True, exist_ok=True)
    AUDIO_PATH = ASSETS_PATH / "audio"
    AUDIO_PATH.mkdir(parents=True, exist_ok=True)
    META_PATH = ASSETS_PATH / "metadata"
    META_PATH.mkdir(parents=True, exist_ok=True)
