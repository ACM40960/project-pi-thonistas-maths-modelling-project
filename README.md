# SketchCoder

A **real-time web application** for interactive sketch **classification** and **generation**. Built with a React frontend and a Flask backend, the system demonstrates end-to-end pipelines for recognizing hand-drawn sketches and generating class-conditioned doodles using deep learning models.

---

## 📖 Project Description

SketchCoder combines two core tasks:

* **Classification**: Users can draw on a canvas, and the trained CNN model predicts the object class among 10 categories.
* **Generation**: Users can select a category (e.g., *bus*, *guitar*) and the generative RNN will produce a novel sketch in that class.

The project demonstrates a **full ML lifecycle**: dataset preparation, preprocessing, model training , backend APIs, and an interactive frontend.
![home page](backend/screenshots/Screenshot%202025-08-18%20at%2020.18.45.png)
![sketch](backend/screenshots/Screenshot%202025-08-18%20at%2020.20.23.png)
![sketch output](backend/screenshots/Screenshot%202025-08-18%20at%2020.20.32.png)
![generate select category](backend/screenshots/Screenshot%202025-08-18%20at%2020.20.42.png)
![generate output](backend/screenshots/Untitled%20design.png)

---

## 📂 Dataset

We use the Google Quick, Draw! dataset, a large-scale crowdsourced collection of millions of doodles across hundreds of categories. Each doodle is stored in NDJSON format (newline-delimited JSON), where each line corresponds to one drawing.
* **Structure**: Each record contains metadata (`key_id`, `word`/category label, `countrycode`, `timestamp`, recognition flag) and the actual drawing strokes.
* **Drawing representation**: The `drawing` field is an array of stroke sequences, where each stroke is a list of x-coordinates, y-coordinates, and pen states. This makes it suitable for both raster classification and sequence-based generative modeling.
* **Classes used (10)**: `bat`, `bicycle`, `bus`, `cactus`, `clock`, `door`, `guitar`, `lightbulb`, `paintbrush`, `smileyface`.
* **Train/Test split**: \~80/20 per class.

---

## 🏗️ Architectures & Data Pipelines

### Classification: SketchCNN

[SketchCNN Architecture](backend/diagrams/SketchCNN-1.png)

**Pipeline:**  
[Classification Pipeline](backend/diagrams/Screenshot%202025-08-18%20at%2019.38.15.png)

---

### Generation: GenerateRNN (Sketch-RNN style)

[GenerateRNN Architecture](backend/diagrams/GenerateRNN-1.png)

**Pipeline:**  
[Generation Pipeline](backend/diagrams/Screenshot%202025-08-18%20at%2019.38.24.png)

---

## 📊 Results

### Classification Performance

* **Accuracy**: 96.0%
* **Macro F1**: 0.959
* **Average Precision (micro)**: 0.991

**Per-class metrics:**

| Class      | Precision | Recall | F1-score |
| ---------- | --------- | ------ | -------- |
| bat        | 0.939     | 0.939  | 0.939    |
| bicycle    | 0.994     | 0.963  | 0.978    |
| bus        | 0.978     | 1.000  | 0.989    |
| cactus     | 0.984     | 0.892  | 0.936    |
| clock      | 0.987     | 0.936  | 0.961    |
| door       | 0.982     | 0.994  | 0.988    |
| guitar     | 0.935     | 0.988  | 0.961    |
| lightbulb  | 0.944     | 0.978  | 0.961    |
| paintbrush | 0.882     | 0.940  | 0.910    |
| smileyface | 0.981     | 0.963  | 0.972    |

## 📊 Results  

### Classification Plots  

- **Confusion Matrix** –  
  <img src="backend/reports/confusion_matrix.png" width="450">  
  The model shows strong class separation, with most predictions lying on the diagonal.  
  Misclassifications are minimal, and the lowest recall is for *cactus* (0.892), which overlaps slightly with other categories.  

- **Precision–Recall Curve (AP = 0.991)** –  
  <img src="backend/reports/pr_curve_micro.png" width="450">  
  The PR curve stays near the top-right corner, confirming excellent performance across thresholds.  
  The micro-average precision of **0.991** indicates strong robustness against class imbalance.  

- **Per-class F1 Scores** –  
  <img src="backend/reports/f1_per_class_bar.png" width="450">  
  Most classes achieve F1-scores above **0.95**, with *bus* and *door* nearly perfect.  
  The lowest-performing class is *paintbrush* (0.910), likely due to its visual similarity with *guitar* and *lightbulb*.  

- **Accuracy Curve** –  
  <img src="backend/reports/accuracy_curve.png" width="450">  
  Training accuracy converges close to **100%**, while validation stabilizes around **96%**,  
  suggesting good generalization with minimal overfitting.  

- **Loss Curve** –  
  <img src="backend/reports/loss_curve.png" width="450">  
  Training loss decreases smoothly and approaches near-zero, while validation loss stabilizes at a low level  
  with minor fluctuations, indicating consistent learning and no major divergence.  

---

### Generation Plots  

- **Latent Space (t-SNE of encoder μ)** –  
  <img src="backend/Generate_report/latent_tsne.png" width="450">  
  The embeddings form well-separated clusters for each class, showing that the encoder learns a structured latent space.  
  Visually similar objects (*paintbrush* and *guitar*) are closer but still separable, indicating good class-specific representation.  

- **KL Divergence by Class** –  
  <img src="backend/Generate_report/summary_kl_loss.png" width="450">  
  The KL term drops sharply in the first few epochs and stabilizes near zero, confirming that the variational latent space converges smoothly.  
  Some classes (*door*, *clock*) stabilize faster than others.  

- **Reconstruction Loss (MDN + Pen CE)** –  
  <img src="backend/Generate_report/summary_recon_loss.png" width="450">  
  The reconstruction loss decreases consistently across classes, with minor oscillations.  
  This shows that the decoder progressively learns to reproduce stroke patterns with higher fidelity.  

- **Total Loss by Class** –  
  <img src="backend/Generate_report/summary_total_loss.png" width="450">  
  Overall loss follows a downward trend, reflecting combined improvements in both latent regularization (KL) and reconstruction.  
  The small per-class variations highlight that some sketches (*bus*, *paintbrush*) are harder to generate than others.  

---

## ⚙️ Setup & Installation

### Using Conda (recommended)

```bash
# Recreate environment
conda env create -f environment.yml

# Activate environment
conda activate sketchbot-env

# Run backend
cd ~/Desktop/SktechBot/backend
python app.py

# Deactivate when done
conda deactivate
```

### Using requirements.txt

```bash
cd backend
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Access at: [http://localhost:3000](http://localhost:3000)

---

## Future Work

* Extend to **Transformer/attention-based generators** for richer sketches.
* Add **stroke-by-stroke playback animation** in frontend.
* Confidence calibration for **reliable rejection of uncertain predictions**.
* Expand dataset to more categories.
* Deploy with **Docker + cloud hosting** for broader access.

---

## Acknowledgments

* **Professor Sarp Akcay**, University College Dublin — for invaluable guidance and supervision throughout the course project.
* **Google Quick, Draw! dataset** — for providing large-scale sketch data.
* Research inspiration from **SketchRNN** (Ha & Eck, 2017).

