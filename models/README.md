# Road Damage Detection — Trained Model Weights

This directory contains the trained `.pt` model weight files used by the application.

## Models

| File | Architecture | Optimizer | Epochs | mAP50-95 | Notes |
|------|-------------|-----------|--------|----------|-------|
| `pothole_model.pt` | YOLOv8n (Nano) | SGD | 50 | 0.62 | Anchor-free, primary model |
| `yolo26_model.pt` | YOLO26 / YOLOv5nu | Adam | 50 | 0.58 | Anchor-based, comparison model |

## Dataset

Both models were trained on the **Roboflow Universe Road Damage Detection** dataset:
- **Classes**: Longitudinal Crack, Transverse Crack, Alligator Crack, Pothole
- **Split**: 70% train / 20% val / 10% test
- **Input size**: 640 × 640

## Adding New Models

1. Place your new `.pt` file in this directory.
2. Register it in [`config/settings.py`](../config/settings.py) under `AVAILABLE_MODELS`.
3. The model will automatically appear in the app's sidebar dropdown.
