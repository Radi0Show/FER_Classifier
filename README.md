# FER_Classifier



This repository contains the implementation of a face emotion recognition system designed to identify emotions from video inputs using deep learning models. The system focuses on analyzing central regions of video frames where faces are most likely to appear.

## Emotion Classes

The model classifies emotions into eight categories:

0. Neutral
1. Happy
2. Angry
3. Sad
4. Fear
5. Surprise
6. Disgust
7. Contempt

## Best CNN Model Details

The system utilizes various models pre-trained on different datasets with the following performance metrics:

| Model                  | AffectNet (8 classes) | AffectNet (7 classes) | AFEW  | VGAF  | LSD   | MTL   | Inference Time (ms) | Model Size (MB) |
|------------------------|-----------------------|-----------------------|-------|-------|-------|-------|---------------------|-----------------|
| `mobilenet_7.h5`       | -                     | 64.71%                | 55.35%| 68.92%| -     | 1.099 | 16 ± 5              | 14              |
| `enet_b0_8_best_afew.pt`| 60.95%               | 64.63%                | 59.89%| 66.80%| 59.32%| 1.110 | 59 ± 26             | 16              |
| `enet_b0_8_best_vgaf.pt`| 61.32%               | 64.57%                | 55.14%| 68.29%| 59.72%| 1.123 | 59 ± 26             | 16              |
| `enet_b0_8_va_mtl.pt`  | 61.93%                | 64.94%                | 56.73%| 66.58%| 60.94%| 1.276 | 60 ± 32             | 16              |
| `enet_b0_7.pt`         | -                     | 65.74%                | 56.99%| 65.18%| -     | 1.111 | 59 ± 26             | 16              |
| `enet_b2_7.pt`         | -                     | 66.34%                | 59.63%| 69.84%| -     | 1.134 | 191 ± 18            | 30              |
| `enet_b2_8.pt`         | 63.03%                | 66.29%                | 57.78%| 70.23%| 52.06%| 1.147 | 191 ± 18            | 30              |
| `enet_b2_8_best.pt`    | 63.125%               | 66.51%                | 56.73%| 71.12%| -     | -     | 191 ± 18            | 30              |

## Implementation

The implementation processes videos by cropping the center of each frame (224x224 pixels) to focus on the main area where faces are likely to be present. Predictions are updated every 15 frames, and results are displayed continuously until the next update. This ensures the system can adapt to real-time changes in expressions while maintaining low computational overhead.

### Requirements

To run the model, ensure you have the following setup:

- Python 3.8 or newer
- PyTorch 1.7 or newer
- OpenCV 4.5 or newer

### Running the Model

To run the emotion recognition, use the following command:

```bash
python emotion_recognition.ipynb --video_path 'path_to_your_video.mp4'
