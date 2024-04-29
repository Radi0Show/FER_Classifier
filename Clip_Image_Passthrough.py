import torch
from PIL import Image
import clip
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Define transformations
transform = Compose([
    Resize((224, 224)),  # Resize the image to 224x224 pixels
    ToTensor(),          # Convert the image to a torch.Tensor
    Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
])

# Load pre-saved text embeddings
def load_text_embeddings():
    save_dir = "embeddings"
    text_embeddings_path = os.path.join(save_dir, "text.pt")
    if torch.cuda.is_available():
        text_features = torch.load(text_embeddings_path).to(device)
    else:
        text_features = torch.load(text_embeddings_path, map_location=torch.device('cpu'))
    return text_features

text_features = load_text_embeddings()

# Function to predict emotion from an image file
def predict_emotion_from_image(image_path):
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    image = preprocess(image).unsqueeze(0).to(device)

    # Calculate image features
    with torch.no_grad():
        image_features = model.encode_image(image)

    # Compute similarities and select the highest one
    similarities = (image_features @ text_features.T).softmax(dim=-1)
    predicted_class_idx = similarities.argmax().item()

    # Map index to human-readable form
    emotions = [
        "angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"
    ]
    predicted_emotion = emotions[predicted_class_idx]
    return predicted_emotion

# Example usage: Predict the emotion of a specific image
image_path = "path_to_your_image.jpg"
emotion = predict_emotion_from_image(image_path)
print(f"The predicted emotion is: {emotion}")
