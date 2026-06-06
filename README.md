# Explainability-Aware Model Selection for MCI Subtype Classification in Brain MRI

A research framework for classifying Alzheimer's disease subtypes from brain MRI images using ResNet-18, with a systematic evaluation of model configurations guided by both predictive performance and explainability metrics.

---

## Overview

This project addresses a critical gap in clinical AI: selecting deep learning models not just by accuracy, but by how faithfully and transparently they explain their predictions. The pipeline trains and evaluates multiple ResNet-18 variants across different optimizers, activation functions, and loss functions on a 3-class MCI dataset, then ranks them using explainability-aware scoring for different clinical use cases.

**Target Classes:**
- `EMCI` — Early Mild Cognitive Impairment
- `LMCI` — Late Mild Cognitive Impairment
- `MCI` — Mild Cognitive Impairment

---

## Repository Structure

```
notebook.ipynb          # Full pipeline: training → evaluation → explainability analysis
README.md               # This file
```

**Google Drive outputs (generated at runtime):**
```
Alzheimer3Class/               # Prepared dataset (train/val/test splits)
Alzheimer_Optimized_Results/   # Saved model weights (.pth) + training histories (.csv)
Alzheimer_Explainability_Results/  # Clinical explainability report CSVs
```

---

## Requirements

### Environment
- Google Colab (GPU runtime recommended — tested on T4/A100 with CUDA 11.8)
- Python 3.10+

### Key Dependencies

```bash
pip install numpy==2.0.2
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install captum scikit-learn matplotlib pandas seaborn tqdm opencv-python-headless
```

| Package | Purpose |
|---|---|
| `torch` / `torchvision` | Model training and transforms |
| `captum` | Explainability methods (IG, Saliency, GradSHAP, Occlusion) |
| `scikit-learn` | Metrics and normalization |
| `matplotlib` / `seaborn` | Visualizations |
| `scipy` | Statistical correlation (Pearson, Spearman) |

---

## Dataset Setup

The dataset should be organized in Google Drive at `/content/drive/MyDrive/Wholebrain/` with one subfolder per class:

```
Wholebrain/
├── MCI/
├── LMCI/
└── EMCI/
```

The notebook automatically splits images into `train/val/test` (70/15/15) and copies them to `Alzheimer3Class/`.

- Accepted formats: `.jpg`, `.jpeg`, `.png`
- Input resolution: images are resized to `224×224`
- Normalization: ImageNet mean/std (`[0.485, 0.456, 0.406]`, `[0.229, 0.224, 0.225]`)

---

## Pipeline

### 1. Data Preparation
Stratified random split with `random.seed(42)` for reproducibility.

### 2. Model Configuration
Base architecture: **ResNet-18** (ImageNet pretrained), final FC replaced with a 3-class head.

Experiments systematically vary:

| Dimension | Options |
|---|---|
| **Optimizer** | Adam, AdamW, SGD |
| **Activation** | ReLU, GELU, Swish, Mish, Leaky ReLU |
| **Loss Function** | Cross-Entropy, Focal Loss, Weighted Cross-Entropy |

Custom activations (`Swish`, `Mish`) and `FocalLoss` are implemented from scratch.

### 3. Training
- 50 epochs max, early stopping (patience = 15)
- `ReduceLROnPlateau` scheduler (factor 0.5, patience 5)
- Gradient clipping (`max_norm=1.0`)
- Batch size: 32

### 4. Performance Evaluation
Metrics computed on the held-out test set:
- Accuracy, Precision, Recall, F1-score (macro)
- Convergence speed (epochs to 95% of peak validation accuracy)

### 5. Explainability Analysis
Per-model explainability metrics computed using **Captum**:

| Metric | Description | Clinical Target |
|---|---|---|
| **Faithfulness** | Insertion/Deletion AUC — does the model use what it highlights? | 0.20 – 0.30 |
| **Stability** | Pearson correlation of attributions under input noise | 0.15 – 0.25 |
| **Gini Coefficient** | Concentration of attributions on specific regions | 0.65 – 0.75 |
| **Sparsity** | Fraction of near-zero attribution pixels | 0.90 – 0.95 |
| **Entropy** | Uncertainty/spread of the attribution map | 0.88 – 0.93 |
| **Method Agreement** | Correlation across IG, Saliency, and Occlusion | ≥ 0.70 |

### 6. Clinical Decision Scoring
Models are ranked with composite scores for three deployment contexts:

| Use Case | Weighting |
|---|---|
| **Clinical Diagnosis** | 50% Faithfulness · 30% Stability · 20% Accuracy |
| **Research** | 40% Stability · 30% Accuracy · 30% Faithfulness |
| **Population Screening** | 70% Accuracy · 30% Faithfulness |

---

## Key Results

| Model | Test Acc | Faithfulness | Stability |
|---|---|---|---|
| Adam_Relu | 92.9% | 0.205 | — |
| Adam_Swish | 90.7% | 0.331 | — |
| Adam_Mish | 91.6% | 0.309 | — |
| AdamW_Relu | 92.0% | 0.123 | — |
| SGD_Relu | 81.3% | 0.080 | — |

**Activation findings:**
- **Swish / Mish** — best faithfulness; recommended for clinical diagnosis
- **ReLU / Leaky ReLU** — best raw accuracy; recommended for screening
- **GELU** — balanced; suitable for research contexts
- **SGD** — underperforms on both axes; not recommended for medical imaging

---

## Outputs

After running the notebook:

```
Alzheimer_Optimized_Results/
├── Adam_Relu.pth                  # Best model weights per experiment
├── Adam_Relu_history.csv          # Epoch-level training metrics
├── experiment_summary.csv         # Aggregated results across all experiments

Alzheimer_Explainability_Results/
├── clinical_explainability_results.csv   # Raw explainability metrics
├── clinical_interpretations.csv         # Annotated clinical interpretations
```

Figures saved locally:
- `clinical_decision_support.png` — Accuracy vs Faithfulness scatter with clinical zones
- `activation_impact.png` — Activation function comparison bar chart
- `validation_accuracy_actual.png` — Training curves from saved histories

---

## Usage

1. Open the notebook in Google Colab
2. Mount your Google Drive and verify the dataset path (`/content/drive/MyDrive/Wholebrain/`)
3. Run all cells sequentially — the pipeline handles splitting, training, evaluation, and explainability in order
4. Trained models and CSVs are saved automatically to Google Drive

---

## Citation

If you use this codebase in your research, please cite appropriately and acknowledge the use of [Captum](https://captum.ai/) for attribution methods.

---

## License

This project is for academic research purposes. Please ensure compliance with dataset licensing terms before use.
