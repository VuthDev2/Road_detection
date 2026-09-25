# Data Directory

This directory holds all dataset files for the Road Damage Detection project.

> **Note**: Raw datasets are excluded from git (see `.gitignore`). Follow the steps below to download them.

## Directory Structure

```
data/
├── raw/         # Original downloaded dataset (git-ignored)
├── processed/   # YOLO-format labels and images (git-ignored)
└── samples/     # Demo footage for quick testing
    └── real_road_test.mp4
```

## Downloading the Dataset

The dataset is hosted on **Roboflow Universe**. Use the Roboflow CLI or the provided notebook:

```bash
# Option 1: via Roboflow CLI
pip install roboflow
python -c "
from roboflow import Roboflow
rf = Roboflow(api_key='YOUR_API_KEY')
project = rf.workspace().project('road-damage-detection')
dataset = project.version(1).download('yolov8', location='data/raw/')
"

# Option 2: Run notebooks/01_data_preparation.ipynb
```

## YOLO Format (processed/)

After downloading, the `processed/` folder should follow this layout:

```
data/processed/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```
