import torch
import cv2

# Load the model
model_path = 'enet_b2_8_best.pt'
model = torch.load(model_path, map_location='cpu')  # Load to CPU
model.eval()

# Setup webcam capture
cap = cv2.VideoCapture(0)  # Use 0 for the default webcam

# Get default webcam properties (might need adjustment depending on the webcam)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define crop size and starting points based on the model's requirements
crop_size = 224  # Adjust if the model needs a different size
x_start = (frame_width - crop_size) // 2
y_start = (frame_height - crop_size) // 2

frames_to_skip = 15  # Update predictions at a defined interval
frame_counter = 0

last_prediction = None  # Store the last prediction

while True:
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
        last_prediction = output.argmax(1).item()

    # Display last known prediction on the frame
    if last_prediction is not None:
        prediction_text = f"Prediction: {last_prediction}"
        cv2.putText(frame, prediction_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Optionally, draw a rectangle on the original frame to show the crop area
    cv2.rectangle(frame, (x_start, y_start), (x_start+crop_size, y_start+crop_size), (255, 0, 0), 2)

    # Show the frame
    cv2.imshow('Webcam Feed', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit on 'q' key
        break

    frame_counter += 1

cap.release()
cv2.destroyAllWindows()
