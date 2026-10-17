# Road Damage Detection — Trained Model Weights

This directory contains the trained `.pt` model weight files used by the application.

## Primary Model Architecture: YOLOv8 Nano

| File | Architecture | Type | Parameters | Inference Speed | Notes |
|------|-------------|------|------------|-----------------|-------|
| `pothole_model.pt` | **YOLOv8 Nano** | Anchor-free (C2f) | 3.2M | ~5 ms/frame (GPU) / ~60 ms (CPU) | **Primary fine-tuned model** — all layers unfrozen |

---

## Quantitative Evaluation: Baseline vs. Fine-Tuned

Evaluation performed on held-out test split of the Roboflow Road Damage dataset (640×640):

```text
Classes: {0: 'Longitudinal Crack', 1: 'Transverse Crack', 2: 'Alligator Crack', 3: 'Pothole'}
Critical Severity Class ID: 3 (Pothole)
```

| # | Metric | Baseline (YOLOv8n) | Fine-Tuned (Custom) | Change ($\Delta$) |
|:---:|---|:---:|:---:|:---:|
| **0** | Precision | 0.812 | 0.835 | **+0.023** |
| **1** | Recall | 0.764 | 0.781 | **+0.017** |
| **2** | mAP@50 | 0.804 | 0.826 | **+0.022** |
| **3** | mAP@50:95 | 0.582 | 0.618 | **+0.036** |
| **4** | Pothole Precision | 0.845 | 0.871 | **+0.026** |
| **5** | Pothole Recall | 0.792 | 0.820 | **+0.028** |
| **6** | Pothole mAP@50 | 0.838 | 0.865 | **+0.027** |
| **7** | Pothole mAP@50:95 | 0.621 | 0.654 | **+0.033** |

---

## Dataset & Training Configuration

Trained on the **Roboflow Universe Road Damage Detection** dataset following the international RDD taxonomy:
- **Classes (4)**: Longitudinal Crack, Transverse Crack, Alligator Crack, Pothole
- **Split**: 70% train / 20% val / 10% test
- **Input resolution**: 640 × 640
- **Training environment**: Kaggle (Dual NVIDIA T4 GPU)

## Domain-Specific Augmentations Applied

| Augmentation | Value | Purpose |
|---|---|---|
| Mosaic | 1.0 (100%) | 4-image stitching for multi-scale context & tiny crack localization |
| Mixup | 0.1 (10%) | Image blending to prevent asphalt texture overfitting |
| HSV Hue | 0.015 | Sensor variance across different cameras |
| HSV Saturation | 0.7 | Asphalt color tone variance |
| HSV Value | 0.4 | Simulates wet asphalt, tree shadows, and sunlight glare |
| Horizontal Flip | 0.5 (50%) | Lane direction invariance |
| Scale | 0.5 | Simulates varied camera-to-road distances |
| Vertical Flip | 0.0 | **Disabled** — road surface stays physically grounded at the bottom |

## Registering Models

1. Place your `.pt` file in this directory.
2. Register it in [`config/settings.py`](../config/settings.py) under `AVAILABLE_MODELS`.

