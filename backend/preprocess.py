import numpy as np
import cv2
import json
import os

def vector_to_image(drawing, image_size=64, padding=20):
    canvas = np.ones((256, 256), dtype=np.uint8) * 255  # base resolution
    min_x = min(pt for stroke in drawing for pt in stroke[0])
    max_x = max(pt for stroke in drawing for pt in stroke[0])
    min_y = min(pt for stroke in drawing for pt in stroke[1])
    max_y = max(pt for stroke in drawing for pt in stroke[1])

    # Normalize coordinates into a square box with padding
    scale = (256 - 2 * padding) / max(max_x - min_x + 1, max_y - min_y + 1)
    
    for stroke in drawing:
        for i in range(len(stroke[0]) - 1):
            x1 = int((stroke[0][i] - min_x) * scale + padding)
            y1 = int((stroke[1][i] - min_y) * scale + padding)
            x2 = int((stroke[0][i + 1] - min_x) * scale + padding)
            y2 = int((stroke[1][i + 1] - min_y) * scale + padding)
            cv2.line(canvas, (x1, y1), (x2, y2), 0, 2)

    # Resize to desired model input size
    canvas = cv2.resize(canvas, (image_size, image_size), interpolation=cv2.INTER_AREA)
    return canvas



def save_images_from_ndjson(ndjson_path, out_dir, label, max_images=200):
    os.makedirs(out_dir, exist_ok=True)
    count = 0
    with open(ndjson_path, 'r') as f:
        for i, line in enumerate(f):
            if count >= max_images:
                break
            data = json.loads(line)
            if data['recognized']:
                img = vector_to_image(data['drawing'])
                path = os.path.join(out_dir, f"{label}_{count}.png")
                cv2.imwrite(path, img)
                count += 1

if __name__ == "__main__":
    classes = ["clock", "door", "bat", "bicycle", "paintbrush", "cactus", "lightbulb", "smileyface", "bus", "guitar"]
    for cls in classes:
        ndjson_path = f"data/{cls}.ndjson"
        out_dir = f"image_data/{cls}"
        save_images_from_ndjson(ndjson_path, out_dir, cls, max_images=1000)
