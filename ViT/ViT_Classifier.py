import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from transformers import ViTModel

class MLPClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 1024)  # Correct hidden units
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)  # Updated dropout rate
        self.fc2 = nn.Linear(1024, num_classes)
        self.output = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return self.output(x)

def extract_features(data_loader, model, device):
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for imgs, lbls in data_loader:
            imgs = imgs.to(device)
            lbls = lbls.to(device)
            outputs = model(imgs).last_hidden_state[:, 0, :]  # Extracting the [CLS] token representation
            features.append(outputs)
            labels.append(lbls)
    return torch.cat(features), torch.cat(labels)

model_name = "google/vit-base-patch16-224"
vit_model = ViTModel.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vit_model = vit_model.to(device)

transform = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = ImageFolder(root='images/train', transform=transform)
test_dataset = ImageFolder(root='images/validation', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

mlp = MLPClassifier(input_size=768, num_classes=len(train_dataset.classes))
mlp = mlp.to(device)

weights_path = 'mlp_classifier_best.pth'
if os.path.exists(weights_path):
    mlp.load_state_dict(torch.load(weights_path, map_location=device))
    print("Loaded weights from file:", weights_path)

optimizer = optim.Adam(mlp.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
criterion = nn.NLLLoss()

best_accuracy = 0
for epoch in range(15):
    mlp.train()
    total_loss = 0
    for imgs, labels in train_loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        # Directly using extracted features for forward pass
        with torch.no_grad():
            features = vit_model(imgs).last_hidden_state[:, 0, :]
        outputs = mlp(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()

    scheduler.step()
    print(f"Epoch {epoch+1}, Average Loss: {total_loss / len(train_loader)}")

    mlp.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            features = vit_model(imgs).last_hidden_state[:, 0, :]
            outputs = mlp(features)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy}%")
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(mlp.state_dict(), weights_path)
        print("Saved best model weights with accuracy: {:.2f}%".format(best_accuracy))

print("Training complete. Best model saved.")
