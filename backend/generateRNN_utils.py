import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
import base64
import torch
import torch.nn.functional as F


def sample_next_point(params, temperature=0.65, num_mixtures=20):
    # Split mixture params and pen logits
    split = torch.split(params, [num_mixtures]*6 + [3], dim=0)
    pi_hat, mu_x, mu_y, log_sigma_x, log_sigma_y, rho_hat, pen_logits = split

    # Apply temperature
    pi = F.softmax(pi_hat / temperature, dim=0)
    rho = torch.tanh(rho_hat)
    sigma_x = torch.exp(log_sigma_x) * np.sqrt(temperature)
    sigma_y = torch.exp(log_sigma_y) * np.sqrt(temperature)

    # Sample a mixture component
    m = torch.distributions.Categorical(pi)
    idx = m.sample().item()

    # Sample (Δx, Δy) from bivariate Gaussian
    mean = [mu_x[idx].item(), mu_y[idx].item()]
    std_x = sigma_x[idx].item()
    std_y = sigma_y[idx].item()
    rho_val = rho[idx].item()

    cov = [[std_x**2, rho_val * std_x * std_y],
           [rho_val * std_x * std_y, std_y**2]]

    dx, dy = np.random.multivariate_normal(mean, cov)

    # Pen state
    pen_probs = F.softmax(pen_logits / temperature, dim=0).detach().cpu().numpy()
    pen_state = np.random.choice(3, p=pen_probs)
    p = [0, 0, 0]
    p[pen_state] = 1

    return [dx, dy] + p


def strokes_to_absolute(stroke_data):
    abs_data = []
    x, y = 0, 0
    for i in range(len(stroke_data)):
        dx, dy, p1, p2, p3 = stroke_data[i]
        x += dx
        y += dy
        abs_data.append([x, y, p1, p2, p3])
    return np.array(abs_data)


def render_strokes_to_image(stroke_data, canvas_size=256):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from io import BytesIO
    from PIL import Image

    segments = []
    x, y = 0, 0
    segment = []

    for dx, dy, p1, p2, p3 in stroke_data:
        prev_x, prev_y = x, y
        x += dx
        y += dy
        if p1 == 1:
            segment.append(((prev_x, prev_y), (x, y)))
        if p2 == 1 or p3 == 1:
            if segment:
                segments.append(segment)
            segment = []

    all_points = [pt for seg in segments for line in seg for pt in line]
    xs, ys = zip(*all_points)

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    def scale(v, min_val, max_val):
        return (v - min_val) / (max_val - min_val + 1e-8) * canvas_size

    fig, ax = plt.subplots()
    ax.set_xlim(0, canvas_size)
    ax.set_ylim(canvas_size, 0)  # Flip Y for visual match
    ax.axis("off")

    for segment in segments:
        for (x0, y0), (x1, y1) in segment:
            x0 = scale(x0, min_x, max_x)
            y0 = scale(y0, min_y, max_y)
            x1 = scale(x1, min_x, max_x)
            y1 = scale(y1, min_y, max_y)
            ax.plot([x0, x1], [y0, y1], color='black', linewidth=2)

    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)

    return Image.open(buf).convert("RGB").resize((canvas_size, canvas_size), Image.Resampling.LANCZOS)
