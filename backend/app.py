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

from train_model import SketchResNet

app = Flask(__name__)
CORS(app)

# Load model
model = SketchResNet()
model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))
model.eval()

# Degradation-based preprocessing
def degrade_image(pil_img):
    # Convert to grayscale numpy array
    img = np.array(pil_img.convert("L"))

    # Simulate pixelation
    img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_LINEAR)
    img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)

    # Gaussian blur
    img = cv2.GaussianBlur(img, (3, 3), sigmaX=0.3)

    # Erode to reduce stroke smoothness
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.erode(img, kernel, iterations=1)

    # Normalize and convert to tensor
    img = img.astype(np.float32) / 255.0
    tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, 64, 64]
    return tensor

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)

    if "image" not in data:
        return jsonify({"error": "Missing 'image' key in request"}), 400

    try:
        # Decode base64 to PIL image
        image_data = base64.b64decode(data["image"].split(",")[1])
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # Save the raw user image
        os.makedirs("user_drawings", exist_ok=True)
        filename = f"user_drawings/drawing_{int(time.time())}.png"
        image.save(filename)

        # Degrade and preprocess the image
        image_tensor = degrade_image(image)
        # Save degraded image for inspection (optional)
        degraded_np = image_tensor.squeeze().numpy() * 255  # [64, 64], rescale back to 0–255
        degraded_np = degraded_np.astype(np.uint8)
        cv2.imwrite(f"user_drawings/degraded_{int(time.time())}.png", degraded_np)


        with torch.no_grad():
            outputs = model(image_tensor)
            probs = F.softmax(outputs, dim=1)
            confidence, prediction = torch.max(probs, dim=1)

        class_names = ["clock", "door", "bat", "bicycle", "paintbrush", "cactus", "lightbulb", "smileyface", "bus", "guitar"]
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

    if selected_class not in ["clock", "door", "bat", "bicycle", "paintbrush",
                              "cactus", "lightbulb", "smileyface", "bus", "guitar"]:
        return jsonify({"error": "Invalid category"}), 400
    
    print(f"User selected category: {selected_class}")

    return jsonify({"strokes": 0})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5050)
