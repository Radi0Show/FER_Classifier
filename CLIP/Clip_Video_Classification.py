import torch
import cv2

# Load the model
model_path = '"D:/aidan/Downloads/DFEW-set5-model.pth"'
model = torch.load(model_path, map_location='cpu')  # Load to CPU
model.eval()

# Setup video capture
video_path = 'SmileVideo.mp4'
cap = cv2.VideoCapture(video_path)

# Get video properties for output
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Setup video writer
output_video_path = 'output.mp4'
out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

frames_to_skip = 15  # Update predictions every 60 frames
frame_counter = 0

# Adjust crop size for 360p
crop_size = 224  # Keep crop size if model needs this size
x_start = (frame_width - crop_size) // 2
y_start = (frame_height - crop_size) // 2

last_prediction = None  # Store the last prediction

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Crop to the center of the frame
    center_crop = frame[y_start:y_start+crop_size, x_start:x_start+crop_size]

    if frame_counter % frames_to_skip == 0:
        # Preprocess the cropped frame
        processed_frame = center_crop.transpose((2, 0, 1))
        processed_frame = torch.tensor(processed_frame).unsqueeze(0).float()

        # Model inference
        with torch.no_grad():
            output = model(processed_frame)

        # Extract prediction data from model output
        # Assuming model output is a classification index or similar
        last_prediction = output.argmax(1).item()

    # Display last known prediction on the frame
    if last_prediction is not None:
        prediction_text = f"Prediction: {last_prediction}"
        cv2.putText(frame, prediction_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Optionally, draw a rectangle on the original frame to show the crop area
    cv2.rectangle(frame, (x_start, y_start), (x_start+crop_size, y_start+crop_size), (255, 0, 0), 2)

    out.write(frame)
    frame_counter += 1

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Processed video saved to {output_video_path}")
