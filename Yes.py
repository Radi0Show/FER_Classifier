import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

def load_image(image_path):
    return Image.open(image_path)

def predict_emotion(image_path, model_path, text_labels):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the fine-tuned model and processor
    model = CLIPModel.from_pretrained(model_path)
    processor = CLIPProcessor.from_pretrained(model_path)
    model.to(device)
    model.eval()

    image = load_image(image_path)
    inputs = processor(text=text_labels, images=image, return_tensors="pt", padding=True, truncation=True).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
        predicted_label_idx = probs.argmax().item()

    return text_labels[predicted_label_idx]

model_path = r'C:\Users\aidan\Documents\GitHub\FER_Classifier\model'
image_path = r'C:\Users\aidan\Documents\GitHub\FER_Classifier\l.jpg'
text_labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

predicted_emotion = predict_emotion(image_path, model_path, text_labels)
print(f"The predicted emotion is: {predicted_emotion}")
