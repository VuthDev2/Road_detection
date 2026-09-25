# Road Damage Detection Presentation Outline

## Slide 1: Title
- **Title**: Road Damage Detection using Deep Learning
- **Subtitle**: A Comparison of YOLO Architectures
- **Name**: [Your Name]
- **Course**: Bachelor of Software Engineering Final Project

## Slide 2: Problem Definition & Motivation
- **Task**: Detecting road surface damage (Potholes, Cracks) from dashcam imagery.
- **Input**: Raw Image / Video Frame
- **Output**: Bounding Box + Class Label
- **Motivation**: Automating road maintenance logging prevents accidents and reduces manual inspection costs.

## Slide 3: Dataset & Pipeline
- **Source**: Roboflow Universe Road Damage Dataset
- **Classes**: 4 (Pothole, Longitudinal Crack, Transverse Crack, Alligator Crack)
- **Split**: 70% Train, 20% Val, 10% Test (Fixed for all models)
- **Preprocessing**: Contrast Enhancement for low-visibility roads, Auto-Orient, 640x640 Resize.

## Slide 4: Model Architectures & Strategies
We compared three distinct deep learning approaches:
1. **YOLOv8 Nano**: A modern, anchor-free detector. Fast, efficient, trained from scratch using SGD.
2. **YOLO Alternative (YOLOv5/YOLO26)**: An anchor-based approach. Trained using different hyperparameters (Adam optimizer, adjusted learning rate) to explore performance impact.
3. **RT-DETR (ResNet50)**: A Real-Time DEtection TRansformer. This Vision Transformer-based architecture provides a radically different approach to CNNs, maximizing our architectural exploration.

## Slide 5: Experimental Setup
- **Framework**: PyTorch (via Ultralytics)
- **Hardware**: Google Colab Free GPU (T4)
- **Metrics**: mAP50-95, Precision, Recall
- **Hyperparameters tuned**: Optimizer (SGD vs Adam), Learning Rate, Epochs.

## Slide 6: Results & Visual Comparisons
- Include a table here comparing the three models.
- **YOLOv8n**: mAP50-95: 0.62 | Precision: 0.81 | Recall: 0.76
- **YOLO Alt**: mAP50-95: 0.58 | Precision: 0.77 | Recall: 0.73
- **RT-DETR**: mAP50-95: 0.66 | Precision: 0.84 | Recall: 0.78
- *Visual*: Show overlaid learning curves showing validation loss decreasing over epochs.

## Slide 7: Discussion & Error Analysis
- **Why YOLOv8 won**: Anchor-free architecture generalized better on irregular shapes like alligator cracks.
- **Error Analysis**: False positives on dark puddles and shadows.
- **Limitation**: Dataset lacks sufficient nighttime imagery.

## Slide 8: Conclusion & Future Work
- **Takeaway**: YOLOv8 provides real-time inference (5ms) with acceptable accuracy for daytime road maintenance logging.
- **Future Work**: Combine with a tracking algorithm (ByteTrack) for video deduplication (already prototyped in `app.py`), and expand dataset to nighttime conditions.
