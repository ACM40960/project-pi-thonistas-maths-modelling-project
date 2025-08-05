import torch
from torch.utils.data import DataLoader, TensorDataset
import os
from transformer_model import TransformerSketch
from transformer_utils import *

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def train_for_class(class_name, epochs=10):
    print(f"\n▶ Training for class: {class_name}")
    drawings = load_ndjson_class_data(class_name, limit=10000)
    strokes, lengths = create_dataset(drawings)
    mask = np.arange(strokes.shape[1])[None, :] < lengths[:, None]
    
    X = torch.tensor(strokes[:-1], device=device)
    Y = torch.tensor(strokes[1:], device=device)
    M = torch.tensor(mask[:-1], device=device)

    ds = TensorDataset(X, Y, M)
    dl = DataLoader(ds, batch_size=64, shuffle=True)

    model = TransformerSketch().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)

    losses = []
    for epoch in range(epochs):
        total = 0
        for x, y, m in dl:
            loss = model.compute_loss(x, y, m)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total += loss.item()
        print(f"Epoch {epoch+1} - Loss: {total/len(dl):.4f}")
        losses.append(total/len(dl))

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), f"models/transformer_{class_name}.pth")

    os.makedirs("loss_plots", exist_ok=True)
    plot_loss_curve(losses, f"loss_plots/transformer_{class_name}_loss.png")

if __name__ == "__main__":
    classes = ['bat', 'bicycle', 'bus', 'cactus', 'clock', 'door', 'guitar', 'lightbulb', 'paintbrush', 'smileyface']
    for c in classes:
        train_for_class(c, epochs=20)
