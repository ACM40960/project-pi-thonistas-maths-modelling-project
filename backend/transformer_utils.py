import os
import json
import numpy as np
import matplotlib.pyplot as plt
import torch

def load_ndjson_class_data(class_name, limit=None):
    path = f"data/{class_name}.ndjson"
    with open(path, "r") as f:
        drawings = [json.loads(line)["drawing"] for line in f if json.loads(line)["recognized"]]
    if limit:
        drawings = drawings[:limit]
    return drawings

def convert_to_stroke3(drawing):
    strokes = []
    for stroke in drawing:
        if not isinstance(stroke, list) or len(stroke) != 2:
            continue  # Skip malformed stroke
        x_list, y_list = stroke
        if len(x_list) != len(y_list):
            continue  # Skip inconsistent stroke
        for j in range(len(x_list)):
            dx = x_list[j] - x_list[j - 1] if j > 0 else 0
            dy = y_list[j] - y_list[j - 1] if j > 0 else 0
            pen = 1 if j == len(x_list) - 1 else 0
            strokes.append([dx, dy, pen])
    if strokes:
        strokes[-1][2] = 2  # End of sketch
    else:
        strokes.append([0, 0, 2])
    return np.array(strokes, dtype=np.float32)


def pad_stroke(stroke, max_len=128):
    padded = np.zeros((max_len, 3), dtype=np.float32)
    length = min(len(stroke), max_len)
    padded[:length] = stroke[:length]
    return padded, length

def create_dataset(drawings, max_len=128):
    strokes, lengths = [], []
    for d in drawings:
        s3 = convert_to_stroke3(d)
        padded, l = pad_stroke(s3, max_len)
        strokes.append(padded)
        lengths.append(l)
    return np.stack(strokes), np.array(lengths)

def plot_loss_curve(losses, save_path):
    plt.figure()
    plt.plot(losses, label="Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Transformer Sketch Training Loss")
    plt.legend()
    plt.savefig(save_path)
    plt.close()
