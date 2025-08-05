import torch
from transformer_model import TransformerSketch
import numpy as np

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def sample_sketch_autoload(class_name, max_len=128):
    model = TransformerSketch().to(device)
    model.load_state_dict(torch.load(f"models/transformer_{class_name}.pth", map_location=device))
    model.eval()

    strokes = torch.zeros((1, 1, 3), device=device)
    for _ in range(max_len - 1):
        with torch.no_grad():
            pi, mu, sigma, pen_logits = model(strokes)
            i = torch.distributions.Categorical(logits=pi[0, -1]).sample()
            stroke = torch.normal(mu[0, -1, i], sigma[0, -1, i])
            pen = torch.distributions.Categorical(logits=pen_logits[0, -1]).sample()
            next = torch.cat([stroke, pen.unsqueeze(0).float()])
            strokes = torch.cat([strokes, next.view(1, 1, 3)], dim=1)
            if pen.item() == 2:
                break
    return strokes.squeeze().cpu().numpy()
