import os
import csv
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
# model
class SketchCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x32

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16x16

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 8x8

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)   # 4x4
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),             # 256 * 4 * 4 = 4096
            nn.Linear(4096, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)  # 20 classes
        )

    def forward(self, x):
        x = self.cnn_layers(x)
        x = self.fc_layers(x)
        return x

#plotting

def plot_loss_accuracy(train_losses, val_losses, train_accs, val_accs, outdir):
    os.makedirs(outdir, exist_ok=True)

    #loss
    plt.figure(figsize=(6,4))
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "loss_curve.png"))
    plt.close()

    #accuracy
    plt.figure(figsize=(6,4))
    plt.plot(train_accs, label="Train")
    plt.plot(val_accs, label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "accuracy_curve.png"))
    plt.close()

    #combined
    fig = plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss")

    plt.subplot(1,2,2)
    plt.plot(train_accs, label="Train")
    plt.plot(val_accs, label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy")

    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "loss_accuracy_plot.png"))
    plt.close(fig)


def plot_confusion_matrix(cm, class_names, outpath, normalize=True):
    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-12)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           ylabel='True label',
           xlabel='Predicted label',
           title='Confusion Matrix' + (' (Normalized)' if normalize else ''))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i == j or cm[i, j] > 0.1:  
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black", fontsize=7)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def plot_f1_bar(f1_per_class, class_names, outpath):
    idx = np.arange(len(class_names))
    fig, ax = plt.subplots(figsize=(12,5))
    ax.bar(idx, f1_per_class)
    ax.set_xticks(idx)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("F1-score")
    ax.set_title("Per-class F1")
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def plot_pr_curve_micro(y_true, y_score, outpath):
    """
    y_true: (N,) int labels
    y_score: (N, C) probabilities
    """
    classes = y_score.shape[1]
    y_true_bin = label_binarize(y_true, classes=list(range(classes)))

    precision, recall, _ = precision_recall_curve(y_true_bin.ravel(), y_score.ravel())
    ap_micro = average_precision_score(y_true_bin, y_score, average="micro")

    fig, ax = plt.subplots(figsize=(6,5))
    ax.plot(recall, precision, lw=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision–Recall (micro-average), AP={ap_micro:.3f}")
    ax.grid(True, ls="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)

#evaluation
@torch.no_grad()
def evaluate_and_report(model, loader, device, class_names, outdir="reports"):
    os.makedirs(outdir, exist_ok=True)

    y_true = []
    y_pred = []
    y_prob = []

    model.eval()
    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        logits = model(imgs)
        probs = F.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        y_true.append(labels.cpu().numpy())
        y_pred.append(preds.cpu().numpy())
        y_prob.append(probs.cpu().numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    y_prob = np.concatenate(y_prob)

    # raw predictions for reproducibility
    with open(os.path.join(outdir, "test_predictions.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true_label", "pred_label"])
        for t, p in zip(y_true, y_pred):
            writer.writerow([class_names[t], class_names[p]])

    #confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, class_names, os.path.join(outdir, "confusion_matrix.png"), normalize=True)

    #classification report (precision/recall/F1/support)
    report_txt = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    with open(os.path.join(outdir, "classification_report.txt"), "w") as f:
        f.write(report_txt)

    #F1 per class bar
    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, digits=4)
    f1_per_class = [report_dict[name]["f1-score"] for name in class_names]
    plot_f1_bar(f1_per_class, class_names, os.path.join(outdir, "f1_per_class_bar.png"))

    #precision recall 
    plot_pr_curve_micro(y_true, y_prob, os.path.join(outdir, "pr_curve_micro.png"))

    # Return overall metrics if you want to print
    micro_f1 = report_dict["accuracy"]  # accuracy equals micro-F1 in multi-class single-label
    macro_f1 = report_dict["macro avg"]["f1-score"]
    weighted_f1 = report_dict["weighted avg"]["f1-score"]
    return {
        "accuracy": float(report_dict["accuracy"]),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1)
    }
# Training

def train_model(data_dir, epochs=30, batch_size=64, outdir="reports"):
    os.makedirs(outdir, exist_ok=True)

    transform = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder(data_dir, transform=transform)
    class_names = dataset.classes
    print("Class order:", class_names)
    num_classes = len(class_names)

    total_len = len(dataset)
    train_len = int(0.7 * total_len)
    val_len = int(0.15 * total_len)
    test_len = total_len - train_len - val_len
    train_ds, val_ds, test_ds = torch.utils.data.random_split(dataset, [train_len, val_len, test_len])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = SketchCNN(num_classes=num_classes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-4)

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    #csv log
    log_path = os.path.join(outdir, "training_log.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "train_acc", "val_acc"])

    for epoch in range(epochs):
        #Train 
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
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / max(1, len(train_loader))
        train_acc = correct / max(1, total)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        #validate
        model.eval()
        val_running, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_running += loss.item()
                preds = outputs.argmax(dim=1)
                v_correct += (preds == labels).sum().item()
                v_total += labels.size(0)
        val_loss = val_running / max(1, len(val_loader))
        val_acc = v_correct / max(1, v_total)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch+1}/{epochs}: "
              f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.3f}, "
              f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.3f}")

        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, f"{train_loss:.6f}", f"{val_loss:.6f}", f"{train_acc:.6f}", f"{val_acc:.6f}"])

    #save model
    torch.save(model.state_dict(), "model.pth")
    print("Model saved to model.pth")

    plot_loss_accuracy(train_losses, val_losses, train_accuracies, val_accuracies, outdir)

    #test evaluation 
    print("\nEvaluating on test data...")
    metrics_dict = evaluate_and_report(model, test_loader, device, class_names, outdir=outdir)
    print(f"Final Test Accuracy: {metrics_dict['accuracy']:.3f} | "
          f"Macro-F1: {metrics_dict['macro_f1']:.3f} | Weighted-F1: {metrics_dict['weighted_f1']:.3f}")


if __name__ == "__main__":
    train_model("image_data", epochs=30, batch_size=64, outdir="reports")