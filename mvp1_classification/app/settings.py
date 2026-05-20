from pathlib import Path
from typing import Union
import os


APP_DIR = Path(__file__).resolve().parent
SERVICE_ROOT = APP_DIR.parent
PROJECT_ROOT = SERVICE_ROOT.parent


def _csv(name: str, default: str) -> list[str]:
    return [item.strip() for item in os.getenv(name, default).split(",") if item.strip()]


def _path(name: str, default: Union[Path, str] = "") -> str:
    value = os.getenv(name)
    if value:
        return value
    return str(default) if default else ""


# Shared parameter folders. Keep pretrained/base weights separate from trained app heads.
PARAMETERS_DIR = Path(_path("DFU_PARAMETERS_DIR", PROJECT_ROOT / "parameters"))
APP_MODELS_DIR = Path(_path("DFU_APP_MODELS_DIR", PARAMETERS_DIR / "app_models"))

# Legacy single-classification endpoint settings.
MODEL_PATH = _path("MODEL_PATH", APP_MODELS_DIR / "legacy_classifier.pt")
MODEL_LABELS = _csv("MODEL_LABELS", "W0,W1,W2,W3,W4")
DEFAULT_CLASS = os.getenv("DEFAULT_CLASS", "W1")

# Task-specific classification settings used by /api/analyze.
CLASSIFIER_BACKEND = os.getenv("CLASSIFIER_BACKEND", "dummy").strip().lower()
FOOT_MODEL_PATH = _path("FOOT_MODEL_PATH", APP_MODELS_DIR / "foot_classifier.pt")
DFU_MODEL_PATH = _path("DFU_MODEL_PATH", APP_MODELS_DIR / "dfu_classifier.pt")
WAGNER_MODEL_PATH = _path("WAGNER_MODEL_PATH", APP_MODELS_DIR / "wagner_classifier.pt")
SINBAD_MODEL_PATH = _path("SINBAD_MODEL_PATH", APP_MODELS_DIR / "sinbad_classifier.pt")

FOOT_LABELS = _csv("FOOT_LABELS", "not_foot,foot")
DFU_LABELS = _csv("DFU_LABELS", "other_injury,dfu")
WAGNER_LABELS = _csv("WAGNER_LABELS", "W0,W1,W2,W3,W4,W5")
SINBAD_LABELS = _csv("SINBAD_LABELS", "low,moderate,high")

FOOT_DEFAULT_CLASS = os.getenv("FOOT_DEFAULT_CLASS", "foot")
DFU_DEFAULT_CLASS = os.getenv("DFU_DEFAULT_CLASS", "dfu")
WAGNER_DEFAULT_CLASS = os.getenv("WAGNER_DEFAULT_CLASS", "W1")
SINBAD_DEFAULT_CLASS = os.getenv("SINBAD_DEFAULT_CLASS", "moderate")

# Optional custom classifier hook: module:Class. The class should implement BaseClassifier.
CUSTOM_CLASSIFIER = os.getenv("CUSTOM_CLASSIFIER", "")

# Segmentation settings. The default demo backend keeps the web/API runnable before
# detectron2 and trained model weights are installed.
SEG_DEFAULT_BACKEND = os.getenv("SEG_DEFAULT_BACKEND", "demo").strip().lower()
SEG_CONFIG_PATH = _path(
    "SEG_CONFIG_PATH",
    PROJECT_ROOT / "Model_training" / "configs" / "custom" / "wound_instance_swinb.yaml",
)
SEG_WEIGHTS_PATH = _path(
    "SEG_WEIGHTS_PATH",
    PARAMETERS_DIR / "Fine-tuned_pth" / "wound_segmenter" / "model_final.pth",
)
SEG_DEVICE = os.getenv("SEG_DEVICE", "cpu")
SEG_SCORE_THRESH = float(os.getenv("SEG_SCORE_THRESH", "0.50"))
SEG_AREA_THRESHOLD = float(os.getenv("SEG_AREA_THRESHOLD", "0.002"))
SEG_DINO_IMAGE_SIZE = int(os.getenv("SEG_DINO_IMAGE_SIZE", "512"))
CUSTOM_SEGMENTER = os.getenv("CUSTOM_SEGMENTER", "")

# DINOv3 backbone and visualization settings.
DINO_MODEL_NAME = os.getenv("DINO_MODEL_NAME", "vit_base_patch16_dinov3.lvd1689m")
DINO_WEIGHTS_PATH = _path(
    "DINO_WEIGHTS_PATH",
    PARAMETERS_DIR / "DINOv3_pth" / "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
)
DINO_DEVICE = os.getenv("DINO_DEVICE", "cpu")
DINO_IMAGE_SIZE = int(os.getenv("DINO_IMAGE_SIZE", "512"))
DINO_SEG_THRESHOLD = float(os.getenv("DINO_SEG_THRESHOLD", "0.50"))
DINO_M2F_CONFIG_PATH = _path(
    "DINO_M2F_CONFIG_PATH",
    PROJECT_ROOT / "Model_training" / "configs" / "custom" / "dino_v3_mask2former_wound_instance.yaml",
)
DINO_M2F_WEIGHTS_PATH = _path(
    "DINO_M2F_WEIGHTS_PATH",
    PARAMETERS_DIR / "Fine-tuned_pth" / "wound_dino_m2f" / "model_final.pth",
)
PCA_BLEND = float(os.getenv("PCA_BLEND", "0.42"))
PCA_BRIGHTNESS = float(os.getenv("PCA_BRIGHTNESS", "1.0"))

# Hugging Face access token. Do not hardcode; set env var.
HUGGINGFACE_HUB_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN", "")

CORS_ORIGINS = _csv("CORS_ORIGINS", "")
