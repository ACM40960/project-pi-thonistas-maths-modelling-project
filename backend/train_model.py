import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
from torchvision.models import resnet18, ResNet18_Weights

class SketchResNet(nn.Module):
    def __init__(self):
        super(SketchResNet, self).__init__()
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, 10)  # 10 classes

    def forward(self, x):
        return self.model(x)


def train_model(data_dir, epochs=30, batch_size=64):
    transform = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder(data_dir, transform=transform)
    print("Class order:", dataset.classes)

    # Train/Val/Test Split (70/15/15)
    total_len = len(dataset)
    train_len = int(0.7 * total_len)
    val_len = int(0.15 * total_len)
    test_len = total_len - train_len - val_len
    train_ds, val_ds, test_ds = torch.utils.data.random_split(dataset, [train_len, val_len, test_len])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = SketchResNet()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    train_losses, val_losses = [] , []
    train_accuracies, val_accuracies = [], []

    for epoch in range(epochs):
        # Training
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # Validation
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = outputs.max(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_loss /= len(val_loader)
        val_acc = correct / total
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}")

    # Save model
    torch.save(model.state_dict(), "model.pth")
    print("Model saved to backend/model.pth")

    # Plot curves
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(train_accuracies, label="Train Acc")
    plt.plot(val_accuracies, label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig("loss_accuracy_plot.png")
    plt.show()

    # Final test accuracy
    print("\nEvaluating on test data...")
    model.eval()
    correct, total = 0, 0
    all_preds, all_true = [], []

    for idx, (imgs, labels) in enumerate(test_loader):
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_true.extend(labels.cpu().numpy())
        
        if idx % 10 == 0 or idx == len(test_loader) - 1:
            print(f"Processed batch {idx+1}/{len(test_loader)}")

    test_acc = correct / total
    print(f"\n✅ Final Test Accuracy: {test_acc:.2f}")


if __name__ == "__main__":
    train_model("image_data", epochs=30)
