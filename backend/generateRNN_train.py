import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from generateRNN_model import GenerateRNN
from generateRNN_preprocess import StrokeDataset
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Device setup
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")


def mdn_loss(y_pred, target, num_mixtures=20):
    # Unpack MDN parameters
    pi_hat, mu_x, mu_y, log_sigma_x, log_sigma_y, rho_hat, q_logits = torch.split(
        y_pred, [num_mixtures]*6 + [3], dim=-1
    )

    pi = F.softmax(pi_hat, dim=-1)
    sigma_x = torch.clamp(torch.exp(log_sigma_x), min=1e-3)
    sigma_y = torch.clamp(torch.exp(log_sigma_y), min=1e-3)
    rho = torch.tanh(rho_hat)

    dx = target[:, :, 0].unsqueeze(-1)
    dy = target[:, :, 1].unsqueeze(-1)
    pen = target[:, :, 2:]

    norm_x = (dx - mu_x) / sigma_x
    norm_y = (dy - mu_y) / sigma_y

    z_xy = norm_x**2 + norm_y**2 - 2 * rho * norm_x * norm_y
    denom = 2 * (1 - rho**2)
    exp_term = torch.exp(-z_xy / denom)
    coeff = 1 / (2 * torch.pi * sigma_x * sigma_y * torch.sqrt(1 - rho**2 + 1e-6))

    gmm = coeff * exp_term
    gmm = torch.sum(pi * gmm, dim=-1)
    gmm = torch.clamp(gmm, min=1e-6)  # Avoid log(0)

    loss_stroke = -torch.log(gmm)
    loss_pen = F.cross_entropy(
        q_logits.view(-1, 3),
        torch.argmax(pen, dim=-1).view(-1),
        reduction='none'
    ).view_as(loss_stroke)

    return torch.mean(loss_stroke + loss_pen)


def kl_loss(mu, logvar):
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def train_conditional_model(tensor_data_path="tensor_data/all_classes.pt",
                            save_path="model_store/conditional_generateRNN.pth",
                            epochs=100, batch_size=64, lr=0.0001):  # ⬅ Lowered LR for stability

    # Load dataset
    data = torch.load(tensor_data_path)
    strokes, labels = data["strokes"], data["labels"]
    dataset = StrokeDataset(strokes, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model + Optimizer
    model = GenerateRNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    total_losses = []
    recon_losses = []
    kl_losses = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        loss_r_total = 0
        loss_k_total = 0

        for batch_strokes, batch_class_onehot in loader:
            batch_strokes = batch_strokes.to(device)
            batch_class_onehot = batch_class_onehot.to(device)

            # Expand class info to match stroke length
            class_info = batch_class_onehot.unsqueeze(1).repeat(1, batch_strokes.size(1), 1)
            input_with_class = torch.cat([batch_strokes, class_info], dim=2)

            optimizer.zero_grad()
            output, mu, logvar = model(input_with_class)

            loss_r = mdn_loss(output, batch_strokes)
            loss_k = kl_loss(mu, logvar)
            loss = loss_r + loss_k
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            loss_r_total += loss_r.item()
            loss_k_total += loss_k.item()

        avg_loss = total_loss / len(loader)
        avg_loss_r = loss_r_total / len(loader)
        avg_loss_k = loss_k_total / len(loader)

        total_losses.append(avg_loss)
        recon_losses.append(avg_loss_r)
        kl_losses.append(avg_loss_k)

        print(f"Epoch {epoch+1}/{epochs}, Total: {avg_loss:.4f}, Recon: {avg_loss_r:.4f}, KL: {avg_loss_k:.4f}")

    # Save model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    # Save loss plot
    os.makedirs("loss_plots", exist_ok=True)
    plt.plot(total_losses, label="Total Loss")
    plt.plot(recon_losses, label="Reconstruction Loss (MDN)")
    plt.plot(kl_losses, label="KL Divergence")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Combined Loss Curve (All Classes)")
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_plots/combined_loss_curve.png")
    plt.close()


if __name__ == "__main__":
    train_conditional_model()
