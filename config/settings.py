"""
config/settings.py
==================
Central configuration for Road Damage Detection.
All constants and defaults live here — never hardcoded in UI or inference code.
"""

from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Project root is two levels up from this file (config/settings.py → root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR = PROJECT_ROOT / "data"

# ---------------------------------------------------------------------------
# Available models
# ---------------------------------------------------------------------------

AVAILABLE_MODELS: dict[str, Path] = {
    "YOLO26 — yolo26.pt (new)": PROJECT_ROOT / "yolo26.pt",
    "YOLOv8n — pothole_model.pt": MODELS_DIR / "pothole_model.pt",
    "YOLO26 — yolo26_model.pt": MODELS_DIR / "yolo26_model.pt",
}

# ---------------------------------------------------------------------------
# Damage class names (must match training label order)
# ---------------------------------------------------------------------------

DAMAGE_CLASSES: list[str] = [
    "Longitudinal Crack",
    "Transverse Crack",
    "Alligator Crack",
    "Pothole",
]

# ---------------------------------------------------------------------------
# Default inference hyperparameters
# ---------------------------------------------------------------------------

@dataclass
class InferenceConfig:
    """Inference settings with sensible defaults."""

    confidence_threshold: float = 0.12
    iou_threshold: float = 0.45
    image_size: int = 640
    enhance_contrast: bool = False
    tracker: str = "bytetrack.yaml"

    # Sliders / selectbox boundaries exposed to the UI
    CONF_MIN: float = field(default=0.1, init=False, repr=False)
    CONF_MAX: float = field(default=1.0, init=False, repr=False)
    CONF_STEP: float = field(default=0.05, init=False, repr=False)

    IOU_MIN: float = field(default=0.1, init=False, repr=False)
    IOU_MAX: float = field(default=1.0, init=False, repr=False)
    IOU_STEP: float = field(default=0.05, init=False, repr=False)

    IMAGE_SIZE_OPTIONS: tuple = field(
        default=(640, 1280, 1920), init=False, repr=False
    )


# Singleton default — import this everywhere
DEFAULT_SETTINGS = InferenceConfig()

# Backward-compatible alias used by older imports.
Settings = InferenceConfig

# ---------------------------------------------------------------------------
# Contrast enhancement parameters
# ---------------------------------------------------------------------------

CONTRAST_FACTOR: float = 2.5
SHARPNESS_FACTOR: float = 2.0

# ---------------------------------------------------------------------------
# Supported file extensions
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS: list[str] = ["png", "jpg", "jpeg"]
VIDEO_EXTENSIONS: list[str] = ["mp4", "mov", "avi"]
ALL_EXTENSIONS: list[str] = IMAGE_EXTENSIONS + VIDEO_EXTENSIONS
