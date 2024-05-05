from transformers import ViTModel, ViTConfig, ViTFeatureExtractor
import torch
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torch.utils.data import DataLoader
from transformers import ViTModel, ViTConfig, ViTFeatureExtractor

# Assuming you've saved your model in the following directory
model_directory = 'C:/Users/rusha/models/vit-base-patch16-224'

# Load the model and configuration from local files
config = ViTConfig.from_pretrained(model_directory)
model = ViTModel.from_pretrained(model_directory, config=config)

# If you also downloaded a feature extractor
feature_extractor = ViTFeatureExtractor.from_pretrained(model_directory)


transform = Compose([
    Resize((224, 224)),  # Resize images to the size expected by ViT
    ToTensor(),  # Convert images to PyTorch tensors
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize using ImageNet stats
])

# Load dataset
dataset = ImageFolder(root='C:/Users/rusha/CSC 561/Final Project/archive (1)/train', transform=transform)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

def extract_features(data_loader, model):
    model.eval()
    features = []
    labels = []

    with torch.no_grad():
        for imgs, lbls in data_loader:
            # Assume the model outputs features from the last layer
            outputs = model(imgs).last_hidden_state[:, 0, :]
            features.append(outputs)
            labels.append(lbls)

    features = torch.cat(features)
    labels = torch.cat(labels)
    return features, labels

features, labels = extract_features(data_loader, model)

class MLPClassifier(torch.nn.Module):
    def __init__(self, input_size, num_classes):
        super(MLPClassifier, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, 512)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(512, num_classes)
        self.output = torch.nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.output(x)
        return x

# Initialize MLP
mlp = MLPClassifier(input_size=features.shape[1], num_classes=len(dataset.classes))
mlp.train()

# Training code here
optimizer = torch.optim.Adam(mlp.parameters(), lr=0.001)
criterion = torch.nn.NLLLoss()

for epoch in range(10):  # number of epochs
    for batch_features, batch_labels in zip(features, labels):
        optimizer.zero_grad()
        outputs = mlp(batch_features)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")
