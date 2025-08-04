from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import base64
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import time
import os
from generateRNN_sampler import generate_sketch_image_base64
from generateRNN_sampler import sample_sketch 
from generateTransformer_sampler import sample_sketch_autoload

import json
import random

# Import your custom CNN
from train_model import SketchCNN

app = Flask(__name__)
CORS(app)

# Load the SketchCNN model instead of SketchResNet
model = SketchCNN()
model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))
model.eval()

def preprocess_user_image(pil_img):
    # Convert to grayscale
    img = np.array(pil_img.convert("L"))

    # Threshold to binarize (remove gray/anti-aliasing)
    _, img = cv2.threshold(img, 220, 255, cv2.THRESH_BINARY)

    # Dilation to thicken lines
    kernel = np.ones((2, 2), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)

    # Find bounding box and crop
    coords = cv2.findNonZero(255 - img)  # Because black stroke on white bg
    if coords is None:
        return None

    x, y, w, h = cv2.boundingRect(coords)
    cropped = img[y:y+h, x:x+w]

    # Pad to square
    size = max(w, h)
    padded = 255 * np.ones((size, size), dtype=np.uint8)
    x_offset = (size - w) // 2
    y_offset = (size - h) // 2
    padded[y_offset:y_offset+h, x_offset:x_offset+w] = cropped

    # Resize to 64x64
    resized = cv2.resize(padded, (64, 64), interpolation=cv2.INTER_AREA)

    # Normalize and convert to tensor
    final = resized.astype(np.float32) / 255.0
    tensor = torch.from_numpy(final).unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, 64, 64]
    return tensor


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)

    if "image" not in data:
        return jsonify({"error": "Missing 'image' key in request"}), 400

    try:
        image_data = base64.b64decode(data["image"].split(",")[1])
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        os.makedirs("user_drawings", exist_ok=True)
        filename = f"user_drawings/drawing_{int(time.time())}.png"
        image.save(filename)

        image_tensor = preprocess_user_image(image)
        if image_tensor is None:
            return jsonify({"error": "Blank image received"}), 400
        
        SAVE_DEBUG_IMAGES = True
        if SAVE_DEBUG_IMAGES:
            degraded_np = image_tensor.squeeze().numpy() * 255
            degraded_np = degraded_np.astype(np.uint8)
            cv2.imwrite(f"user_drawings/degraded_{int(time.time())}.png", degraded_np)

        with torch.no_grad():
            outputs = model(image_tensor)
            probs = F.softmax(outputs, dim=1)
            confidence, prediction = torch.max(probs, dim=1)

        class_names = ['bat', 'bicycle', 'bus', 'cactus', 'clock', 'door', 'guitar', 'lightbulb', 'paintbrush', 'smileyface']
        predicted_label = class_names[prediction.item()]
        conf_value = confidence.item()

        print(f"Predicted: {predicted_label} (Confidence: {conf_value:.2f})")

        if conf_value < 0.6:
            return jsonify({"label": "Not recognized"})
        else:
            return jsonify({"label": predicted_label})

    except Exception as e:
        print("ERROR during prediction:", e)
        return jsonify({"error": str(e)}), 500


@app.route("/category", methods=["POST"])
def receive_category():
    data = request.get_json()
    selected_class = data.get("label")

    valid_classes = [
        'bat', 'bicycle', 'bus', 'cactus', 'clock', 'door',
        'guitar', 'lightbulb', 'paintbrush', 'smileyface'
    ]

    if selected_class not in valid_classes:
        return jsonify({"error": "Invalid category"}), 400

    print(f"User selected category: {selected_class}")
    try:
        image_data = generate_sketch_image_base64(selected_class)
        return jsonify({"image": image_data})
    except Exception as e:
        print("Generation error:", e)
        return jsonify({"error": str(e)}), 500

@app.route("/ndjson-strokes", methods=["POST"])
def return_example_stroke():
    data = request.get_json()
    class_name = data.get("label")

    file_path = f"data/{class_name}.ndjson"
    if not os.path.exists(file_path):
        return jsonify({ "error": "Class not found" }), 404

    with open(file_path, "r") as f:
        lines = f.readlines()
        random.shuffle(lines)  # Shuffle for variety

        for line in lines:
            sketch = json.loads(line)
            if sketch.get("recognized", False):
                return jsonify({ "drawing": sketch["drawing"] })

    return jsonify({ "error": "No valid sketch found" }), 500

@app.route("/generate-strokes", methods=["POST"])
def generate_strokes():
    data = request.json
    label = data.get("label", "")
    print("[DEBUG] Label received:", label)

    try:
        strokes = sample_sketch_autoload(label)
        print("[DEBUG] Sampled strokes shape:", np.array(strokes).shape)
        return jsonify({ "drawing": strokes.tolist() })
    except Exception as e:
        print("[ERROR]", e)
        return jsonify({ "error": str(e) }), 500



if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5050)

