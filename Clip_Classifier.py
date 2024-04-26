import torch
import clip
from PIL import Image
from datasets import load_dataset
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


dataset = load_dataset("FER-Universe/DiffusionFER", 'default')
print(dataset)  

# Define transformations 
transform = Compose([
    Resize((224, 224)),  # Resize the image to 224x224 pixels
    ToTensor(),          # Convert the image to a torch.Tensor
    Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
])

# Inference function to predict emotion
def predict_emotion(image_path):
    image = Image.open(image_path).convert("RGB")
    image = preprocess(image).unsqueeze(0).to(device)

    # Text descriptions for each emotion class
    text_descriptions = [
        "A face with a neutral expression.",
        "A face showing happiness, smiling.",
        "A sad face with possible tears.",
        "A surprised face with wide eyes.",
        "A face expressing fear.",
        "A disgusted face with a frown.",
        "An angry face with a frown and intense eyes."
    ]
    text = clip.tokenize(text_descriptions).to(device)

    # Calculate image and text features
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

    # Compute similarities and select the highest one
    similarities = (image_features @ text_features.T).softmax(dim=-1)
    predicted_class_idx = similarities.argmax().item()

    return text_descriptions[predicted_class_idx]

# Example usage
# Adjust the image path to point to a valid image in your dataset directory
image_path = r'C:\Users\aidan\Documents\GitHub\FER_Classifier\happy-smile.webp'
emotion = predict_emotion(image_path)
print(f"Predicted Emotion: {emotion}")
