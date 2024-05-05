import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from transformers import ViTModel, ViTFeatureExtractor

class MLPClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 1024)  # Increased hidden units
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
            outputs = model(imgs).last_hidden_state[:, 0, :]
            features.append(outputs)
            labels.append(lbls)
    return torch.cat(features), torch.cat(labels)

model_name = "google/vit-base-patch16-224"
model = ViTModel.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

transform = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = ImageFolder(root='images/train', transform=transform)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

train_features, train_labels = extract_features(train_loader, model, device)
test_features, test_labels = extract_features(test_loader, model, device)

mlp = MLPClassifier(input_size=train_features.shape[1], num_classes=len(dataset.classes))
mlp = mlp.to(device)
optimizer = optim.Adam(mlp.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
criterion = nn.NLLLoss()

best_accuracy = 0
for epoch in range(15):
    mlp.train()
    total_loss = 0
    for i in range(0, train_features.size(0), 32):
        batch_features = train_features[i:i+32].to(device)
        batch_labels = train_labels[i:i+32].to(device)

        optimizer.zero_grad()
        outputs = mlp(batch_features)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    scheduler.step()
    print(f"Epoch {epoch+1}, Average Loss: {total_loss / (i // 32 + 1)}")

    mlp.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(0, test_features.size(0), 32):
            batch_features = test_features[i:i+32].to(device)
            batch_labels = test_labels[i:i+32].to(device)
            outputs = mlp(batch_features)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy}%")
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(mlp.state_dict(), 'mlp_classifier_best.pth')
        print("Saved best model weights with accuracy: {:.2f}%".format(best_accuracy))

print("Training complete. Best model saved.")
