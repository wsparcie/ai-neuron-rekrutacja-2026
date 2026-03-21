# ADHD detection from EEG

Binary classification of ADHD vs control using resting-state EEG.

Dataset: https://www.kaggle.com/datasets/danizo/eeg-dataset-for-adhd/data

---

## Setup

```bash
python -m venv venv
venv\Scripts\activate      # windows
# source venv/bin/activate  # linux/mac

pip install -r requirements.txt
```

Download `adhdata.csv` from Kaggle and put it at `adhd_classifier/data/adhdata.csv`.

## Notebooks

- `notebooks/eda.ipynb` – signal quality checks, band powers, distributions, topomaps
- `notebooks/experiments.ipynb` – feature extraction, model training, evaluation

## Results

5-fold GroupKFold CV (subject-level, so same person can't leak into val):

| model             | accuracy      | f1            | roc auc       |
| ----------------- | ------------- | ------------- | ------------- |
| SVM (RBF)         | 0.800 ± 0.056 | 0.800 ± 0.067 | 0.871 ± 0.054 |
| Random Forest     | 0.825 ± 0.062 | 0.822 ± 0.072 | 0.893 ± 0.041 |
| Gradient Boosting | 0.811 ± 0.066 | 0.808 ± 0.073 | 0.879 ± 0.047 |
| LDA               | 0.759 ± 0.047 | 0.753 ± 0.060 | 0.830 ± 0.053 |

Hold-out test set (25 subjects, subject-level aggregate, seed=42):

| model                                      | accuracy | f1 (ADHD) | roc auc   |
| ------------------------------------------ | -------- | --------- | --------- |
| SVM (RBF)                                  | 0.76     | 0.786     | 0.776     |
| Random Forest                              | 0.76     | 0.786     | 0.776     |
| Gradient Boosting                          | 0.76     | 0.786     | 0.776     |
| LDA                                        | 0.72     | 0.741     | 0.769     |
| MLP small (128→64, d=0.3, early stop)      | 0.72     | 0.741     | 0.769     |
| MLP medium (256→128→64, d=0.3, early stop) | 0.72     | 0.741     | 0.731     |
| MLP medium (256→128→64, d=0.5, early stop) | 0.72     | 0.720     | 0.750     |
| **CNN (dropout=0.3, seed=42)**             | **0.80** | **0.783** | **0.885** |
| CNN (dropout=0.5, seed=42)                 | 0.52     | 0.143     | 0.878     |

## Features

447 features per 2-second window extracted across 19 EEG channels:

- absolute band power δ θ α β γ (95 = 19 ch × 5 bands)
- relative band power (95)
- per-channel: mean, std, skew, kurtosis (76)
- per-channel: spectral entropy, SEF95 (38)
- per-channel: Hjorth activity, mobility, complexity (57)
- per-channel: zero-crossing rate, peak-to-peak amplitude (38)
- global ratios: theta/alpha, theta/beta, slow-wave dominance (3)
- frontal alpha asymmetry: F4–F3, F8–F7 (2)
- frontal-posterior theta coherence: Fz/Pz, F3/P3, F4/P4 (3)
- interhemispheric coherence (8 pairs × 5 bands = 40)

## Structure

```
adhd_classifier/
├── data_loader.py        # csv loading, subject-level split, windowing
├── preprocessing.py      # artefact clipping (5σ MAD), RobustScaler
├── feature_extraction.py # Welch PSD → band powers + stats + TAR
├── baseline.py           # SVM / RF / GBM / LDA + GroupKFold CV
├── neural_net.py         # MLP (ADHDNet) + 1D temporal CNN (EEGConvNet)
├── evaluation.py         # metrics, confusion matrix, ROC
├── tests/
│   └── test_features.py
└── notebooks/
    ├── eda.ipynb          # signal quality, PSD, TAR statistical test
    └── experiments.ipynb  # feature extraction, model training, evaluation
```
