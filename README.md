# Explainability-Aware Model Selection for MCI Subtype Classification in Brain MRI
### Balancing Diagnostic Accuracy and Clinical Interpretability in Deep Learning

**Authors:** Ahana Priyanka N, Anushya V, Faatina S  
**Published at:** ICMLAI-2026 — International Conference on Machine Learning & Artificial Intelligence (Pangea Global Events × SSN)

---

## Overview

This repository contains the implementation for a systematic study on optimizing deep learning components for Alzheimer's Disease stage classification from MRI scans, with explainability as a first-class evaluation criterion.

Rather than proposing a new architecture, this work addresses a gap in existing literature: **the lack of systematic optimization of key model components** (activation functions, optimizers, loss functions) combined with quantitative XAI evaluation. A pretrained ResNet-18 is used as the backbone, and 45 model configurations are benchmarked across both predictive and explainability metrics to guide context-aware clinical deployment.

**Target Classes (ADNI Dataset):**
- `EMCI` — Early Mild Cognitive Impairment (500 images)
- `MCI` — Mild Cognitive Impairment (500 images)
- `LMCI` — Late Mild Cognitive Impairment (500 images)

---

## Research Gap Addressed

Prior work (El-Assy et al. 2024; Ali et al. 2024; Zarei et al. 2024; Altwijri et al. 2023; Zolfaghari et al. 2025) focused on CNN architecture design or feature extraction improvements, but:
- Lacked **systematic component-level optimization** (activations, optimizers, loss functions)
- Offered **limited explainability** beyond ad-hoc visualizations
- Had **limited multi-class coverage** of EMCI/LMCI/MCI specifically

---

## Methodology

### Pipeline Overview

```
ADNI MRI Dataset
      │
      ▼
Dataset Preparation ──► 70% Train / 15% Val / 15% Test (balanced per class)
      │
      ▼
Preprocessing ──► Resize (224×224) · ImageNet Normalization · Augmentation (Flip, Rotation)
      │
      ▼
Model Configuration ──► 45 combinations: 5 activations × 3 optimizers × 3 loss functions
      │
      ▼
ResNet-18 (Pretrained) ──► Modified FC → 3-class output
      │
      ▼
Training ──► 50 epochs · Early Stopping (patience=15) · Grad Clipping · ReduceLROnPlateau
      │
      ▼
Evaluation ──► Accuracy · Precision · Recall · F1-score
      │
      ▼
Explainability Module ──► Integrated Gradients · Saliency · Occlusion · Input×Gradient
      │
      ▼
Explainability Metrics ──► Faithfulness · Stability · Gini · Sparsity · Entropy
      │
      ▼
Model Comparison & Selection ──► Clinical Score · Research Score · Screening Score
```

### Dataset

| Property | Value |
|---|---|
| Source | Alzheimer's Disease Neuroimaging Initiative (ADNI) |
| Modality | T1-weighted MRI scans |
| Total Images | 1,500 (500 per class, balanced) |
| Train Split | 70% — 1,050 images |
| Validation Split | 15% — 225 images |
| Test Split | 15% — 225 images |

### Model Configurations (45 Total)

| Dimension | Options |
|---|---|
| Activation Functions (5) | ReLU, GELU, Swish, Mish, Leaky ReLU |
| Optimizers (3) | Adam, AdamW, SGD |
| Loss Functions (3) | Cross-Entropy, Focal Loss, Weighted Cross-Entropy |

### Hyperparameters

| Parameter | Value |
|---|---|
| Batch Size | 32 |
| Initial Learning Rate | 0.001 |
| Maximum Epochs | 50 |
| Early Stopping Patience | 15 |
| Gradient Clipping | max_norm = 1.0 |
| LR Scheduler | ReduceLROnPlateau (factor=0.5, patience=5) |
| Cross-Validation | 10-fold |

### Explainability Methods (Captum)

Four XAI attribution methods are applied per model to generate brain region heatmaps:
- **Integrated Gradients (IG)** — primary attribution method
- **Saliency Maps**
- **Occlusion**
- **Input × Gradient**

### Explainability Metrics

| Metric | What It Measures | Clinical Target |
|---|---|---|
| **Faithfulness** | Prediction drop when important regions removed (Insertion/Deletion AUC) | 0.20 – 0.30 |
| **Stability** | Pearson correlation of attributions under input noise | 0.15 – 0.25 |
| **Gini (Focus)** | Concentration of attributions on specific regions | 0.65 – 0.75 |
| **Sparsity** | Fraction of near-zero attribution pixels | 0.90 – 0.95 |
| **Entropy** | Uncertainty/spread in the attribution map | 0.88 – 0.93 |
| **Method Agreement** | Correlation across IG, Saliency, and Occlusion | ≥ 0.70 |

### Clinical Decision Scoring

Models are ranked with composite scores for three deployment contexts:

| Use Case | Formula |
|---|---|
| **Clinical Diagnosis** | 50% Faithfulness + 30% Stability + 20% Accuracy |
| **Research** | 40% Stability + 30% Accuracy + 30% Faithfulness |
| **Population Screening** | 70% Accuracy + 30% Faithfulness |

---

## Results

### Model Comparison

| Model | Accuracy | F1 | Faithfulness | Stability | Focus (Gini) |
|---|---|---|---|---|---|
| **Adam_Relu** | **92.9%** | **0.929** | 0.205 (GOOD) | 0.189 (GOOD) | 0.621 (GOOD) |
| AdamW_Relu | 92.0% | 0.920 | 0.123 (FAIR) | 0.220 (GOOD) | 0.672 (GOOD) |
| Adam_Relu_Focal | 92.0% | 0.920 | 0.080 (POOR) | 0.069 (FAIR) | 0.668 (GOOD) |
| Adam_Mish | 91.6% | 0.916 | 0.309 (EXCELLENT) | 0.258 (HIGH) | 0.635 (GOOD) |
| Adam_GELU | 91.6% | 0.916 | 0.161 (FAIR) | 0.331 (HIGH) | 0.648 (GOOD) |
| Adam_Relu_Weighted | 91.1% | 0.911 | 0.260 (GOOD) | 0.206 (GOOD) | 0.730 (GOOD) |
| Adam_Swish | 90.7% | 0.906 | 0.331 (EXCELLENT) | 0.347 (HIGH) | 0.606 (GOOD) |
| Adam_LeakyRelu | 88.9% | 0.889 | 0.155 (FAIR) | 0.204 (GOOD) | 0.692 (GOOD) |
| SGD_Relu | 81.3% | 0.813 | 0.080 (POOR) | 0.076 (FAIR) | 0.431 (FAIR) |

### Composite Scores

| Model | Clinical Score | Research Score | Screening Score |
|---|---|---|---|
| **Adam_Relu** | **0.579** | 0.623 | **0.850** |
| Adam_GELU | 0.839 | **0.813** | 0.894 |
| Adam_LeakyRelu | 0.962 | 0.942 | 0.865 |
| Adam_Mish | 0.677 | 0.667 | 0.808 |
| Adam_Swish | 0.622 | 0.740 | 0.716 |
| AdamW_Relu | 0.433 | 0.546 | 0.697 |
| SGD_Relu | 0.185 | 0.277 | 0.646 |

### Best Model: Adam + ReLU

Adam_ReLU achieved the highest overall test accuracy (92.9%, F1=0.929) and converges fastest to peak validation accuracy (~95%), making it the best balanced choice. Training convergence analysis confirms it reaches high accuracy earlier than AdamW_Relu, Adam_GELU, and SGD_Relu.

### Key Findings by Activation Function

| Activation | Avg Accuracy | Faithfulness | Best Use Case |
|---|---|---|---|
| ReLU | 92.9% | 0.205 | Screening & general deployment |
| Mish | 91.6% | 0.309 | Clinical diagnosis |
| GELU | 91.6% | 0.161 | Research (highest stability: 0.331) |
| Swish | 90.7% | 0.331 | Clinical diagnosis (most faithful) |
| Leaky ReLU | 88.9% | 0.155 | Balanced use |

- **Swish / Mish** — best faithfulness; recommended for clinical diagnosis where explanation trust matters
- **ReLU** — best raw accuracy + convergence speed; recommended for screening
- **GELU** — highest stability; best suited for reproducible research
- **SGD** — underperforms across all axes; not recommended for medical imaging tasks

---

## Requirements

### Environment
- Google Colab (GPU runtime, CUDA 11.8 — tested on T4/A100)
- Python 3.10+

### Installation

```bash
pip install numpy==2.0.2
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install captum scikit-learn matplotlib pandas seaborn tqdm opencv-python-headless
```

| Package | Purpose |
|---|---|
| `torch` / `torchvision` | Model training, transforms, pretrained weights |
| `captum` | XAI attribution methods (IG, Saliency, GradSHAP, Occlusion) |
| `scikit-learn` | Evaluation metrics, normalization |
| `matplotlib` / `seaborn` | Visualizations and clinical plots |
| `scipy` | Pearson/Spearman correlation for stability |

---

## Dataset Setup

Request access to the ADNI dataset at [adni.loni.usc.edu](https://adni.loni.usc.edu), then organize T1-weighted MRI scans in Google Drive:

```
MyDrive/Wholebrain/
├── MCI/       (500 images)
├── LMCI/      (500 images)
└── EMCI/      (500 images)
```

The notebook auto-splits data and copies it to `MyDrive/Alzheimer3Class/`. Accepted formats: `.jpg`, `.jpeg`, `.png`.

---

## Usage

1. Open the notebook in Google Colab with GPU runtime enabled
2. Mount Google Drive and verify the dataset path (`/content/drive/MyDrive/Wholebrain/`)
3. Run all cells in order — the pipeline handles splitting, training (9 core experiments), evaluation, XAI analysis, and clinical scoring sequentially
4. All models, training histories, and reports save automatically to Google Drive

---

## Outputs

```
Alzheimer_Optimized_Results/
├── {ModelName}.pth                  # Best model weights per experiment
├── {ModelName}_history.csv          # Per-epoch: loss, val_accuracy, grad_norm, lr
└── experiment_summary.csv           # Aggregated metrics across all experiments

Alzheimer_Explainability_Results/
├── clinical_explainability_results.csv   # Raw explainability scores per model
└── clinical_interpretations.csv         # Annotated clinical interpretations
```

---

## Citation

If you use this code or build on this work, please cite:

```
Ahana Priyanka N, Anushya V, Faatina S. "Explainability-Aware Model Selection for MCI 
Subtype Classification in Brain MRI: Balancing Diagnostic Accuracy and Clinical 
Interpretability in Deep Learning." ICMLAI-2026, Pangea Global Events × SSN, 2026.
```

---

## License

This project is intended for academic research. ADNI data access requires registration and compliance with ADNI's data use agreement. Please ensure compliance before use.
