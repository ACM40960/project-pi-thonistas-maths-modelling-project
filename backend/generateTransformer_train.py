import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt

from generateTransformer_model import TransformerModel
from generateRNN_preprocess import StrokeDataset
from generateRNN_classmap import CLASS_TO_INDEX

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# === Loss Functions ===
def mdn_loss(pi, normal, pred_pen, target):
    dx = target[:, :, 0].unsqueeze(-1)
    dy = target[:, :, 1].unsqueeze(-1)
    pen = target[:, :, 2:].contiguous()

    norm_x = (dx - normal.loc[:, :, :, 0]) / normal.scale_tril[:, :, :, 0, 0]
    norm_y = (dy - normal.loc[:, :, :, 1]) / normal.scale_tril[:, :, :, 1, 1]

    z_xy = norm_x ** 2 + norm_y ** 2 - 2 * normal.scale_tril[:, :, :, 0, 1] * norm_x * norm_y
    denom = 2 * (1 - normal.scale_tril[:, :, :, 0, 1] ** 2 + 1e-6)
    exp_term = torch.exp(-z_xy / denom)
    coeff = 1 / (2 * torch.pi * normal.scale_tril[:, :, :, 0, 0] * normal.scale_tril[:, :, :, 1, 1] * torch.sqrt(1 - normal.scale_tril[:, :, :, 0, 1] ** 2 + 1e-6))

    gmm = coeff * exp_term
    gmm = torch.sum(pi.probs * gmm, dim=-1)
    gmm = torch.clamp(gmm, min=1e-6)

    loss_stroke = -torch.log(gmm)

    # If pred_pen is a Categorical distribution, use logits
    if hasattr(pred_pen, "logits"):
        logits = pred_pen.logits
    else:
        logits = pred_pen  # fallback

    loss_pen = F.cross_entropy(
        logits.view(-1, 3),
        torch.argmax(pen, dim=-1).view(-1),
        reduction='none'
    ).view_as(loss_stroke)


    return torch.mean(loss_stroke + loss_pen)

# === Training Function ===
def train_model(tensor_data_path="tensor_data/all_classes.pt",
                save_path="model_store/transformer_generate.pth",
                resume_from=None,
                epochs=50, batch_size=64, lr=1e-4,
                checkpoint_interval=5):

    # Load data
    data = torch.load(tensor_data_path)
    strokes, labels = data["strokes"], data["labels"]
    dataset = StrokeDataset(strokes, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = TransformerModel().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    losses = []
    start_epoch = 0

    # Resume logic
    if resume_from:
        print(f"Resuming from {resume_from}")
        model.load_state_dict(torch.load(resume_from, map_location=device))
        # try to extract last epoch number
        try:
            start_epoch = int(resume_from.split("_")[-1].replace(".pth", ""))
            print(f"Resuming from epoch {start_epoch}")
        except:
            print("Could not infer epoch number from filename.")

    try:
        for epoch in range(start_epoch, epochs):
            model.train()
            epoch_loss = 0

            for batch_strokes, batch_classes in loader:
                batch_strokes = batch_strokes.to(device)
                input_seq = batch_strokes[:, :-1, :]
                target_seq = batch_strokes[:, 1:, :]

                optimizer.zero_grad()
                pi, normal, pen_logits = model(input_seq, batch_classes)
                loss = mdn_loss(pi, normal, pen_logits, target_seq)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()
                # print(f"  Batch Loss: {loss.item():.4f}", flush=True)

            avg = epoch_loss / len(loader)
            print(f"Epoch {epoch + 1}/{epochs} — Loss: {avg:.4f}")
            losses.append(avg)

            # Save checkpoint
            if (epoch + 1) % checkpoint_interval == 0:
                checkpoint_file = f"model_store/transformer_epoch_{epoch+1}.pth"
                torch.save(model.state_dict(), checkpoint_file)
                print(f"[Checkpoint] Saved model at {checkpoint_file}")

    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving current model state...")
        torch.save(model.state_dict(), "model_store/transformer_interrupted.pth")
        print("Saved as transformer_interrupted.pth")

    # Final save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    # Plot loss curve
    os.makedirs("loss_plots", exist_ok=True)
    plt.plot(losses, label="Total Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Transformer Sketch Training Loss")
    plt.grid(True)
    plt.legend()
    plt.savefig("loss_plots/transformer_loss_curve.png")
    plt.close()

if __name__ == "__main__":
    train_model(resume_from=None, epochs=50)


