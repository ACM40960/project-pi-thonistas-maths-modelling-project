import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F

from generateRNN_classmap import CLASS_TO_INDEX

class StrokeDataset(Dataset):
    def __init__(self, stroke_tensors, class_labels, num_classes=10):
        self.data = stroke_tensors
        self.labels = class_labels
        self.num_classes = num_classes

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        stroke = self.data[idx]  # [T, 5]
        class_idx = self.labels[idx]
        class_onehot = F.one_hot(torch.tensor(class_idx), num_classes=self.num_classes).float()  # [10]
        return stroke, class_onehot


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

    return torch.tensor(sequence, dtype=torch.float)  # [max_len, 5]


def parse_all_classes(input_dir="data", class_list=None, max_drawings=3000, max_len=200):
    stroke_tensors = []
    class_labels = []

    for cls in class_list:
        path = os.path.join(input_dir, f"{cls}.ndjson")
        count = 0
        with open(path, 'r') as f:
            for line in f:
                if count >= max_drawings:
                    break
                data = json.loads(line)
                if data.get("recognized", False):
                    drawing = data['drawing']
                    seq = drawing_to_stroke_sequence(drawing, max_len)
                    stroke_tensors.append(seq)
                    class_labels.append(CLASS_TO_INDEX[cls])
                    count += 1

    return stroke_tensors, class_labels


def save_combined_tensor_data(class_list, input_dir="data", output_path="tensor_data/all_classes.pt"):
    strokes, labels = parse_all_classes(input_dir, class_list)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save({"strokes": strokes, "labels": labels}, output_path)
    print(f"Saved {len(strokes)} sequences across {len(class_list)} classes to {output_path}")


if __name__ == "__main__":
    class_names = list(CLASS_TO_INDEX.keys())
    save_combined_tensor_data(class_names)
