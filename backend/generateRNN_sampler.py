import torch
import numpy as np
from generateRNN_model import GenerateRNN
from generateRNN_utils import sample_next_point, render_strokes_to_image, strokes_to_absolute
import io
from PIL import Image
import base64
import os
from datetime import datetime

def load_model(class_name, model_path="model_store", device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GenerateRNN().to(device)

    model.load_state_dict(torch.load(f"{model_path}/{class_name}_generateRNN.pth", map_location=device))
    model.eval()
    return model


def sample_sketch(model, max_len=200, temperature=0.65, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Start token: [0, 0, 1, 0, 0] = pen-down
    prev = torch.tensor([[0, 0, 1, 0, 0]], dtype=torch.float).unsqueeze(0).to(device)  # [1, 1, 5]
    z = torch.randn((1, 128)).to(device)
    strokes = []

    hidden = None
    for _ in range(max_len):
        out = model.decoder(prev, z)  # [1, 1, M*6+3]
        params = out.squeeze(0).squeeze(0)  # [M*6+3]
        next_point = sample_next_point(params, temperature)
        strokes.append(next_point)

        prev = torch.tensor(next_point, dtype=torch.float).unsqueeze(0).unsqueeze(0).to(device)

        if next_point[-1] == 1:  # p3 = end
            break

    return np.array(strokes)  # shape: [T, 5]


def generate_sketch_image_base64(class_name, temperature=0.6):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(class_name, device=device)
    strokes = sample_sketch(model, temperature=temperature, device=device)

    # Save strokes to debug if empty
    if len(strokes) == 0:
        print(f"[ERROR] Empty stroke for class '{class_name}' — saving raw strokes.")
        with open(f"user_drawings/{class_name}_empty.txt", "w") as f:
            f.write("[]")
        raise ValueError("Sketch contains no strokes")

    abs_strokes = strokes_to_absolute(strokes)
    img = render_strokes_to_image(abs_strokes)

    # 🔽 Save PNG to disk
    os.makedirs("generated_debug", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    img.save(f"generated_debug/{class_name}_{timestamp}.png")

    # Return base64
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    base64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return "data:image/png;base64," + base64_str

