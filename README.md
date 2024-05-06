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

| Methods            | UAR DFEW  | WAR DFEW  |
|--------------------|-----------|-----------|
| C3D                | 42.74     | 53.54     |
| P3D                | 43.97     | 54.47     |
| I3D-RGB            | 43.40     | 54.27     |
| 3D ResNet18        | 46.52     | 58.27     |
| R(2+1)D18          | 42.79     | 53.22     |
| ResNet18-LSTM      | 51.32     | 63.85     |
| ResNet18-ViT       | 55.36     | 67.56     |
|                    |           |           |
| FER-CLIP           | 59.61     | 71.25     |
| enet_b2_8_best.pt  | 63.125%   | 66.51%    |
| ViT-B/16/SAM       | 0.5310    | 0.6220    |


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
