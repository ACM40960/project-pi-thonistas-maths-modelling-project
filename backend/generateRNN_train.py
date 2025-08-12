import os
import csv
import math
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from generateRNN_model import GenerateRNN
from generateRNN_preprocess import StrokeDataset

#losses & metrics
def mdn_loss(y_pred, target, num_mixtures=20):
    """
    Reconstruction = MDN NLL for (Δx, Δy) + CE loss for pen state.
    Returns:
      recon_loss (scalar tensor),
      q_logits (B,T,3) pen logits,
      pen_onehot (B,T,3) true pen one-hot
    """
    z = y_pred
    pi_hat, mu_x, mu_y, log_sigma_x, log_sigma_y, rho_hat, q_logits = torch.split(
        z, [num_mixtures, num_mixtures, num_mixtures, num_mixtures, num_mixtures, num_mixtures, 3], dim=-1
    )

    # mixture params
    pi = F.softmax(pi_hat, dim=-1)
    sigma_x = torch.exp(log_sigma_x).clamp_min(1e-3)
    sigma_y = torch.exp(log_sigma_y).clamp_min(1e-3)
    rho = torch.tanh(rho_hat)

    dx = target[:, :, 0].unsqueeze(-1)  # (B,T,1)
    dy = target[:, :, 1].unsqueeze(-1)  # (B,T,1)
    pen = target[:, :, 2:]              # (B,T,3) one-hot

    # bivariate normal NLL
    norm_x = (dx - mu_x) / sigma_x
    norm_y = (dy - mu_y) / sigma_y
    z_xy = norm_x**2 + norm_y**2 - 2 * rho * norm_x * norm_y
    denom = 2 * (1 - rho**2 + 1e-6)
    exp_term = torch.exp(-z_xy / denom)
    coeff = 1.0 / (2 * math.pi * sigma_x * sigma_y * torch.sqrt(1 - rho**2 + 1e-6))
    gmm = torch.sum(pi * (coeff * exp_term), dim=-1) + 1e-6  # (B,T)
    loss_stroke = -torch.log(gmm)

    # pen CE
    pen_ce = F.cross_entropy(
        q_logits.reshape(-1, 3),
        torch.argmax(pen, dim=-1).reshape(-1),
        reduction='none'
    ).reshape_as(loss_stroke)

    recon = torch.mean(loss_stroke + pen_ce)
    return recon, q_logits, pen


def kl_loss(mu, logvar):
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def pen_accuracy_from_logits(q_logits, pen_onehot):
    """Simple 'accuracy' for the generator: pen-state prediction accuracy."""
    pred = q_logits.argmax(dim=-1)          # (B,T)
    true = pen_onehot.argmax(dim=-1)        # (B,T)
    # include all steps; you could mask end-token if you want
    correct = (pred == true).sum().item()
    total = np.prod(true.shape)
    return correct / max(1, total)


#plotting
def plot_losses(out_png, totals, recons, kls, title="Loss (Total / Recon / KL)"):
    plt.figure(figsize=(6,4))
    plt.plot(totals, label="Total", linewidth=2)
    plt.plot(recons, label="Recon (MDN + pen CE)")
    plt.plot(kls, label="KL")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


def plot_pen_acc(out_png, accs):
    plt.figure(figsize=(6,4))
    plt.plot(accs, label="Pen-state accuracy", linewidth=2)
    plt.xlabel("Epoch"); plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.title("Pen-state accuracy per epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


#train one class
def train_one_class(class_name,
                    data_path="tensor_data",
                    save_path="model_store",
                    report_root="Generate_report",
                    epochs=20,
                    batch_size=64,
                    lr=1e-3,
                    beta_kl_warmup=5):
    """
    Trains one per-class generator and writes:
      Generate_report/<class>/train_log.csv
      Generate_report/<class>/loss_curves.png
      Generate_report/<class>/pen_accuracy.png
      Generate_report/<class>/model_path.txt
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(save_path, exist_ok=True)
    class_dir = os.path.join(report_root, class_name)
    os.makedirs(class_dir, exist_ok=True)

    # data
    strokes = torch.load(os.path.join(data_path, f"{class_name}.pt"))
    dataset = StrokeDataset(strokes)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # model/opt
    model = GenerateRNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # logs
    log_csv = os.path.join(class_dir, "train_log.csv")
    with open(log_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "total_loss", "recon_loss", "kl_loss", "pen_acc", "beta"])

    totals, recons, kls, pen_accs = [], [], [], []

    for epoch in range(epochs):
        model.train()
        total_loss_sum = recon_sum = kl_sum = 0.0
        pen_acc_sum = 0.0
        steps = 0

        # KL warm-up: beta from 0→1 over first beta_kl_warmup epochs
        beta = 1.0
        if beta_kl_warmup and epoch < beta_kl_warmup:
            beta = (epoch + 1) / beta_kl_warmup

        for batch in loader:
            steps += 1
            batch = batch.to(device)
            optimizer.zero_grad()

            y_pred, mu, logvar = model(batch)
            recon, q_logits, pen = mdn_loss(y_pred, batch)
            kl = kl_loss(mu, logvar)
            loss = recon + beta * kl

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss_sum += loss.item()
            recon_sum += recon.item()
            kl_sum += kl.item()
            pen_acc_sum += pen_accuracy_from_logits(q_logits, pen)

        avg_total = total_loss_sum / steps
        avg_recon = recon_sum / steps
        avg_kl = kl_sum / steps
        avg_pen = pen_acc_sum / steps

        totals.append(avg_total); recons.append(avg_recon); kls.append(avg_kl); pen_accs.append(avg_pen)

        with open(log_csv, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([epoch+1, f"{avg_total:.6f}", f"{avg_recon:.6f}", f"{avg_kl:.6f}", f"{avg_pen:.6f}", f"{beta:.3f}"])

        print(f"[{class_name}] Epoch {epoch+1}/{epochs} | Total {avg_total:.4f} | Recon {avg_recon:.4f} | KL {avg_kl:.4f} | PenAcc {avg_pen:.3f} | beta {beta:.2f}")

    #savemodel
    model_path = os.path.join(save_path, f"{class_name}_generateRNN.pth")
    torch.save(model.state_dict(), model_path)
    with open(os.path.join(class_dir, "model_path.txt"), "w") as f:
        f.write(model_path)

    plot_losses(os.path.join(class_dir, "loss_curves.png"), totals, recons, kls,
                title=f"{class_name} — Training Loss")
    plot_pen_acc(os.path.join(class_dir, "pen_accuracy.png"), pen_accs)

#latent t-SNE (all classes)

def compute_encoder_mus_for_class(model, strokes, device, max_items=300):
    """Return a (N, latent_dim) matrix of μ for up to max_items samples."""
    model.eval()
    if len(strokes) > max_items:
        idxs = np.random.choice(len(strokes), max_items, replace=False)
        strokes = [strokes[i] for i in idxs]
    ds = StrokeDataset(strokes)
    dl = DataLoader(ds, batch_size=64, shuffle=False)

    mus = []
    with torch.no_grad():
        for batch in dl:
            batch = batch.to(device)
            # used encoder LSTM directly to get μ (same as forward does internally)
            _, (h_n, _) = model.encoder.lstm(batch)  # h_n: [2, B, H]
            h = torch.cat([h_n[0], h_n[1]], dim=1)   # [B, 2H]
            mu = model.encoder.fc_mu(h)              # [B, latent_dim]
            mus.append(mu.cpu().numpy())
    if mus:
        return np.concatenate(mus, axis=0)
    return np.zeros((0, 128), dtype=np.float32)


def plot_latent_tsne(class_names, report_root="Generate_report", data_path="tensor_data", model_store="model_store"):
    """Creates Generate_report/latent_tsne.png colored by class."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_mu = []
    labels = []
    for i, cls in enumerate(class_names):
        # load trained model if available; else untrained (not ideal but works)
        model = GenerateRNN().to(device)
        pth = os.path.join(model_store, f"{cls}_generateRNN.pth")
        if os.path.exists(pth):
            model.load_state_dict(torch.load(pth, map_location=device))
        strokes = torch.load(os.path.join(data_path, f"{cls}.pt"))
        mus = compute_encoder_mus_for_class(model, strokes, device, max_items=300)
        if mus.shape[0] > 0:
            all_mu.append(mus)
            labels += [i] * mus.shape[0]

    if not all_mu:
        print("No μ collected for t-SNE.")
        return

    X = np.vstack(all_mu)
    y = np.array(labels)

    tsne = TSNE(n_components=2, perplexity=30, learning_rate='auto', init='pca', random_state=42)
    Z = tsne.fit_transform(X)

    plt.figure(figsize=(8,6))
    for i, cls in enumerate(class_names):
        mask = (y == i)
        plt.scatter(Z[mask, 0], Z[mask, 1], s=8, alpha=0.7, label=cls)
    plt.title("Latent Space (t-SNE of encoder μ)")
    plt.legend(markerscale=2, fontsize=8, frameon=False, ncol=2)
    plt.tight_layout()
    os.makedirs(report_root, exist_ok=True)
    plt.savefig(os.path.join(report_root, "latent_tsne.png"), dpi=240)
    plt.close()

#summary loss plots
def summary_loss_plots(class_names, report_root="Generate_report"):
    """Reads each class's train_log.csv and plots one summary per loss type."""
    series = {}
    for cls in class_names:
        log_csv = os.path.join(report_root, cls, "train_log.csv")
        if not os.path.exists(log_csv):
            continue
        with open(log_csv, "r") as f:
            r = csv.DictReader(f)
            epochs, total, recon, kl = [], [], [], []
            for row in r:
                epochs.append(int(row["epoch"]))
                total.append(float(row["total_loss"]))
                recon.append(float(row["recon_loss"]))
                kl.append(float(row["kl_loss"]))
        series[cls] = {"epoch": epochs, "total": total, "recon": recon, "kl": kl}

    if not series:
        print("No per-class logs found for summary plots.")
        return

    # Total
    plt.figure(figsize=(10,6))
    for cls, s in series.items():
        plt.plot(s["epoch"], s["total"], label=f"{cls}", linewidth=1.2)
    plt.title("Total Loss by Class")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.legend(ncol=2, fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(report_root, "summary_total_loss.png"), dpi=240)
    plt.close()

    # Recon
    plt.figure(figsize=(10,6))
    for cls, s in series.items():
        plt.plot(s["epoch"], s["recon"], label=f"{cls}", linewidth=1.2)
    plt.title("Reconstruction (MDN + pen CE) by Class")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.legend(ncol=2, fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(report_root, "summary_recon_loss.png"), dpi=240)
    plt.close()

    # KL
    plt.figure(figsize=(10,6))
    for cls, s in series.items():
        plt.plot(s["epoch"], s["kl"], label=f"{cls}", linewidth=1.2)
    plt.title("KL Divergence by Class")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.legend(ncol=2, fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(report_root, "summary_kl_loss.png"), dpi=240)
    plt.close()

if __name__ == "__main__":
    # your 10 classes
    classes = ["clock", "door", "bat", "bicycle", "paintbrush",
               "cactus", "lightbulb", "smileyface", "bus", "guitar"]

    # train each class & write per-class reports
    for cls in classes:
        train_one_class(cls,
                        data_path="tensor_data",
                        save_path="model_store",
                        report_root="Generate_report",
                        epochs=20,
                        batch_size=64,
                        lr=1e-3,
                        beta_kl_warmup=5)

    # after all trainings, make global visuals
    plot_latent_tsne(classes, report_root="Generate_report",
                     data_path="tensor_data", model_store="model_store")
    summary_loss_plots(classes, report_root="Generate_report")
