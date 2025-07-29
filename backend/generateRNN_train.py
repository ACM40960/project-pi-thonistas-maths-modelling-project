import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from generateRNN_model import GenerateRNN
from generateRNN_preprocess import StrokeDataset
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt


def mdn_loss(y_pred, target, num_mixtures=20):
    z = y_pred
    pi_hat, mu_x, mu_y, log_sigma_x, log_sigma_y, rho_hat, q_logits = torch.split(
        z, [num_mixtures, num_mixtures, num_mixtures, num_mixtures, num_mixtures, num_mixtures, 3], dim=-1)

    pi = F.softmax(pi_hat, dim=-1)
    sigma_x = torch.exp(log_sigma_x)
    sigma_y = torch.exp(log_sigma_y)
    rho = torch.tanh(rho_hat)

    dx = target[:, :, 0].unsqueeze(-1)
    dy = target[:, :, 1].unsqueeze(-1)
    pen = target[:, :, 2:]

    norm_x = (dx - mu_x) / sigma_x
    norm_y = (dy - mu_y) / sigma_y

    z_xy = norm_x**2 + norm_y**2 - 2*rho*norm_x*norm_y
    denom = 2 * (1 - rho**2)
    exp_term = torch.exp(-z_xy / denom)
    coeff = 1 / (2 * math.pi * sigma_x * sigma_y * torch.sqrt(1 - rho**2))

    gmm = coeff * exp_term
    gmm = torch.sum(pi * gmm, dim=-1) + 1e-6

    loss_stroke = -torch.log(gmm)
    loss_pen = F.cross_entropy(q_logits.view(-1, 3), torch.argmax(pen, dim=-1).view(-1), reduction='none')
    loss_pen = loss_pen.view_as(loss_stroke)

    return torch.mean(loss_stroke + loss_pen)


def kl_loss(mu, logvar):
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def train_one_class(class_name, data_path="tensor_data", save_path="model_store",
                    epochs=20, batch_size=64, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on class: {class_name}")

    strokes = torch.load(os.path.join(data_path, f"{class_name}.pt"))
    dataset = StrokeDataset(strokes)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

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

        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            output, mu, logvar = model(batch)
            loss_r = mdn_loss(output, batch)
            loss_k = kl_loss(mu, logvar)
            loss = loss_r + loss_k
            loss.backward()
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

    os.makedirs(save_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_path, f"{class_name}_generateRNN.pth"))
    print(f"Model saved to {save_path}/{class_name}_generateRNN.pth")

    # Plot losses
    os.makedirs("loss_plots", exist_ok=True)
    plt.plot(total_losses, label="Total Loss")
    plt.plot(recon_losses, label="Reconstruction Loss (MDN)")
    plt.plot(kl_losses, label="KL Divergence")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss Curve for {class_name}")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"loss_plots/{class_name}_loss_curve.png")
    plt.close()


if __name__ == "__main__":
    classes = ["clock", "door", "bat", "bicycle", "paintbrush", "cactus", "lightbulb", "smileyface", "bus", "guitar"]
    for cls in classes:
        train_one_class(cls)