import torch
from PIL import Image
import clip
from datasets import load_dataset
from transformers import CLIPProcessor, CLIPModel
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn.functional import cosine_similarity

def cosine_similarity_contrastive_loss(image_features, text_features, margin=0.3):
    # Cosine similarity ranges from -1 to 1; higher means more similar
    cosine_sim = cosine_similarity(image_features, text_features)

    # Labels for similarity; assuming the simplest case where each image matches its corresponding text
    labels = torch.eye(cosine_sim.size(0)).to(cosine_sim.device)

    # Contrastive loss: if labels=1 (positive pairs), we want cosine_sim to be high (close to 1)
    positive_loss = (1 - cosine_sim) * labels

    # If labels=0 (negative pairs), we want cosine_sim to be low (close to -1 or below margin)
    negative_loss = torch.clamp(cosine_sim - margin, min=0) * (1 - labels)

    # Combine losses
    loss = torch.mean(positive_loss + negative_loss)
    return loss


# Define label to text mapping based on your dataset specifics
label_to_text = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "neutral",
    5: "sad",
    6: "surprise"
}

# Load dataset
dataset = load_dataset("FER-Universe/DiffusionFER", 'default')
train_dataset = dataset['train']

# DataLoader and collation
def collate_fn(batch):
    images = [item['image'] for item in batch]
    texts = [label_to_text[item['label']] for item in batch]  # Convert labels to text
    return {"images": images, "labels": texts}

train_loader = DataLoader(train_dataset, batch_size=4, collate_fn=collate_fn)

# Load CLIP model and processor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.to(device)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-5)

# Number of epochs
num_epochs = 3

# Debugging model outputs
def debug_model_outputs(image_features, text_features):
    print("Image Features Norm:", image_features.norm(dim=1))
    print("Text Features Norm:", text_features.norm(dim=1))
    print("Sample Image Features:", image_features[0][:10])
    print("Sample Text Features:", text_features[0][:10])

# Update training loop with debugging
model.train()
for epoch in range(num_epochs):
    for batch in train_loader:
        images, texts = batch['images'], batch['labels']

        inputs = processor(text=texts, images=images, return_tensors="pt", padding=True, truncation=True).to(device)
        outputs = model(**inputs)

        # Calculate loss using the revised contrastive loss
        loss = cosine_similarity_contrastive_loss(outputs.image_embeds, outputs.text_embeds)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}, Batch Loss: {loss.item()}")

# Save the fine-tuned model

model.save_pretrained(r'C:\Users\aidan\Documents\GitHub\FER_Classifier\model')
processor.save_pretrained(r'C:\Users\aidan\Documents\GitHub\FER_Classifier\model')
