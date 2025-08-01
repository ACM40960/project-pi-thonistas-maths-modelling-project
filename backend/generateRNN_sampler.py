import torch
import numpy as np
from generateRNN_model import GenerateRNN
from generateRNN_utils import sample_next_point, render_strokes_to_image, strokes_to_absolute
import io
from PIL import Image
import base64
import os
from datetime import datetime
from generateRNN_preprocess import CLASS_TO_INDEX

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

def load_model(model_path="model_store/conditional_generateRNN.pth"):
    model = GenerateRNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def sample_sketch_autoload(class_label, max_len=200, temperature=0.65):
    model = load_model()
    return sample_sketch(model, class_label, max_len=max_len, temperature=temperature)

def sample_sketch(model, class_label, max_len=200, temperature=0.65):
    # Convert label to one-hot vector
    class_idx = CLASS_TO_INDEX[class_label]
    class_onehot = torch.zeros(1, 10, dtype=torch.float, device=device)
    class_onehot[0, class_idx] = 1.0

    # Start token: [0, 0, 1, 0, 0]
    prev = torch.tensor([[0, 0, 1, 0, 0]], dtype=torch.float, device=device).unsqueeze(0)  # [1, 1, 5]
    z = torch.randn((1, 128), device=device)
    strokes = []

    hidden = None
    for _ in range(max_len):
        class_info = class_onehot.unsqueeze(1)  # [1, 1, 10]
        decoder_input = torch.cat([prev, class_info], dim=2)  # [1, 1, 15]

        out = model.decoder(decoder_input, z)  # [1, 1, M*6+3]
        params = out.squeeze(0).squeeze(0)  # [M*6+3]
        next_point = sample_next_point(params, temperature)
        strokes.append(next_point)

        prev = torch.tensor(next_point, dtype=torch.float, device=device).unsqueeze(0).unsqueeze(0)

        if next_point[-1] == 1:  # p3 = end
            break

    return np.array(strokes)  # [T, 5]

def generate_sketch_image_base64(class_name, temperature=0.3):
    model = load_model()
    strokes = sample_sketch(model, class_label=class_name, temperature=temperature)

    if len(strokes) == 0:
        print(f"[ERROR] Empty stroke for class '{class_name}' — saving raw strokes.")
        with open(f"user_drawings/{class_name}_empty.txt", "w") as f:
            f.write("[]")
        raise ValueError("Sketch contains no strokes")

    abs_strokes = strokes_to_absolute(strokes)
    img = render_strokes_to_image(abs_strokes)

    os.makedirs("generated_debug", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    img.save(f"generated_debug/{class_name}_{timestamp}.png")

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    base64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return "data:image/png;base64," + base64_str
