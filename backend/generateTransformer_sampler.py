import torch
import numpy as np
from generateTransformer_model import TransformerModel
from generateRNN_utils import render_strokes_to_image, strokes_to_absolute
from generateRNN_classmap import CLASS_TO_INDEX
import io
from PIL import Image
import base64
import os
from datetime import datetime


# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load model from file
# def load_model(model_path="model_store/transformer_generate.pth"):
def load_model(model_path="model_store/transformer_epoch_35.pth"):
    model = TransformerModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def sample_sketch(model, class_label, max_len=200, temperature=1.2):
    class_idx = CLASS_TO_INDEX[class_label]
    class_tensor = torch.tensor([class_idx], device=device)

    prev = torch.tensor([[[0.0, 0.0, 1.0]]], dtype=torch.float32, device=device)
    strokes = []

    for _ in range(max_len):
        with torch.no_grad():
            pi_dist, normal_dist, pen_dist = model(prev, class_tensor, tau=temperature)
            print(type(pi_dist), type(normal_dist), type(pen_dist))

            # Sample from pi distribution (OneHotCategorical)
            mixture_idx = pi_dist.sample()[:, -1].nonzero(as_tuple=True)[1].item()

            # Sample from selected Gaussian
            stroke_sample = normal_dist.sample()[:, -1, mixture_idx, :]
            dx, dy = stroke_sample[0].tolist()

            # Sample pen state
            pen_state = pen_dist.sample()[:, -1].item()

        # Convert scalar pen_state to one-hot format
        p1, p2, p3 = 0, 0, 0
        if pen_state == 0:
            p1 = 1  # pen down
        elif pen_state == 1:
            p2 = 1  # pen up
        elif pen_state == 2:
            p3 = 1  # end

        strokes.append([dx, dy, p1, p2, p3])

        next_input = torch.tensor([[[dx, dy, pen_state]]], dtype=torch.float32, device=device)
        prev = torch.cat([prev, next_input], dim=1)

        if len(strokes) >= 15 and pen_state == 2:
            break

    return np.array(strokes)


# Helper for autoload
def sample_sketch_autoload(class_label, temperature=1.2):
    model = load_model()
    return sample_sketch(model, class_label, temperature=temperature)

# Convert strokes to PNG and base64-encoded image
def generate_sketch_image_base64(class_name, temperature=1.2):
    model = load_model()
    strokes = sample_sketch(model, class_name, temperature=temperature)

    if len(strokes) == 0:
        raise ValueError(f"Sketch for class '{class_name}' is empty")

    abs_strokes = strokes_to_absolute(strokes)
    img = render_strokes_to_image(abs_strokes)

    os.makedirs("generated_debug", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    img.save(f"generated_debug/{class_name}_{timestamp}.png")

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    base64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return "data:image/png;base64," + base64_str

# Run directly
if __name__ == "__main__":
    label = "guitar"
    try:
        strokes = sample_sketch_autoload(label)
        print(f"Sampled {len(strokes)} strokes for '{label}'")
        print(strokes)

        # Generate preview
        img_data = generate_sketch_image_base64(label)
        with open(f"generated_debug/preview_{label}.html", "w") as f:
            f.write(f"<img src='{img_data}'/>")
        print(f"Preview saved to generated_debug/preview_{label}.html")

    except Exception as e:
        print("Error during generation:", e)
