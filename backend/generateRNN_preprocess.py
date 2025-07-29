import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset


class StrokeDataset(Dataset):
    def __init__(self, stroke_tensors):
        self.data = stroke_tensors

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def drawing_to_stroke_sequence(drawing, max_len=200):
    sequence = []
    prev_x, prev_y = 0, 0

    for stroke in drawing:
        for i in range(len(stroke[0])):
            x = stroke[0][i]
            y = stroke[1][i]
            dx = x - prev_x
            dy = y - prev_y
            prev_x, prev_y = x, y

            if i == len(stroke[0]) - 1:
                p = [0, 1, 0]  # pen-up
            else:
                p = [1, 0, 0]  # pen-down
            sequence.append([dx, dy] + p)

    sequence.append([0, 0, 0, 0, 1])  # End of drawing

    # Pad or trim
    if len(sequence) < max_len:
        pad = [[0, 0, 0, 0, 1]] * (max_len - len(sequence))
        sequence += pad
    else:
        sequence = sequence[:max_len]

    return torch.tensor(sequence, dtype=torch.float)


def parse_ndjson_to_tensor(ndjson_path, max_drawings=1000, max_len=200):
    stroke_tensors = []
    with open(ndjson_path, 'r') as f:
        for line in f:
            if len(stroke_tensors) >= max_drawings:
                break
            data = json.loads(line)
            if data.get("recognized", False):
                drawing = data['drawing']
                stroke_seq = drawing_to_stroke_sequence(drawing, max_len)
                stroke_tensors.append(stroke_seq)
    return stroke_tensors


def save_stroke_tensors_for_classes(class_list, input_dir="data", output_dir="tensor_data"):
    os.makedirs(output_dir, exist_ok=True)
    for cls in class_list:
        ndjson_path = os.path.join(input_dir, f"{cls}.ndjson")
        output_path = os.path.join(output_dir, f"{cls}.pt")
        stroke_tensors = parse_ndjson_to_tensor(ndjson_path)
        torch.save(stroke_tensors, output_path)
        print(f"Saved {len(stroke_tensors)} sequences for '{cls}' to {output_path}")


if __name__ == "__main__":
    class_names = ['bat', 'bicycle', 'bus', 'cactus', 'clock', 'door','guitar', 'lightbulb', 'paintbrush', 'smileyface']
    save_stroke_tensors_for_classes(class_names)
