# Road Damage Detection — Trained Model Weights

This directory contains the trained `.pt` model weight files used by the application.

## Models

| File | Architecture | Type | Optimizer | Epochs | Notes |
|------|-------------|------|-----------|--------|-------|
| `yolo26_model.pt` | YOLO26 (Nano) | Anchor-based | SGD | 30 | **Primary model** — full fine-tune, all layers unfrozen |
| `pothole_model.pt` | YOLOv8n (Nano) | Anchor-free | SGD | 50 | Baseline comparison model |

## Dataset

Both models were trained on the **Roboflow Universe Road Damage Detection** dataset:

- **Classes (4)**: Longitudinal Crack, Transverse Crack, Alligator Crack, Pothole
- **Split**: 70% train / 20% val / 10% test
- **Input size**: 640 × 640
- **Training environment**: Kaggle (Dual T4 GPU)

## Augmentations Applied (YOLO26 fine-tune)

| Augmentation | Value | Purpose |
|---|---|---|
| Mosaic | 1.0 (100%) | 4-image stitching for complex context |
| Mixup | 0.1 (10%) | Prevents asphalt texture overfitting |
| HSV Hue | 0.015 | Sensor variance across regions |
| HSV Saturation | 0.7 | Color variance |
| HSV Value | 0.4 | Wet asphalt, shadows, and glare |
| Horizontal Flip | 0.5 (50%) | Road direction invariance |
| Scale | 0.5 | Varied camera distances |
| Vertical Flip | 0.0 | Disabled — roads stay grounded |

## Adding New Models

1. Place your new `.pt` file in this directory.
2. Register it in [`config/settings.py`](../config/settings.py) under `AVAILABLE_MODELS`.
3. The model will automatically appear in the app's sidebar dropdown.
