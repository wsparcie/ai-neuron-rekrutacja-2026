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
| SVM (RBF)         | 0.655 ± 0.095 | 0.678 ± 0.103 | 0.747 ± 0.115 |
| Random Forest     | 0.696 ± 0.110 | 0.695 ± 0.144 | 0.738 ± 0.061 |
| Gradient Boosting | 0.696 ± 0.114 | 0.691 ± 0.142 | 0.723 ± 0.066 |
| LDA               | 0.573 ± 0.089 | 0.564 ± 0.130 | 0.534 ± 0.143 |

Hold-out test (feature matrix scaled with RobustScaler after extraction):

| model                                    | accuracy | f1 (ADHD) | roc auc   |
| ---------------------------------------- | -------- | --------- | --------- |
| SVM (RBF)                                | 0.68     | 0.667     | 0.596     |
| Random Forest                            | 0.68     | 0.667     | 0.644     |
| Gradient Boosting                        | 0.60     | 0.615     | 0.660     |
| LDA                                      | 0.52     | 0.538     | 0.436     |
| MLP small (128→64, dropout=0.3)          | 0.68     | 0.636     | 0.737     |
| **MLP medium (256→128→64, dropout=0.3)** | **0.72** | **0.741** | **0.744** |
| MLP medium (256→128→64, dropout=0.5)     | 0.68     | 0.667     | 0.744     |
| CNN (dropout=0.3)                        | \*       | \*        | \*        |
| CNN (dropout=0.5)                        | \*       | \*        | \*        |

\* CNN results are run-dependent (random weight init + no seed in `train_cnn`). Run `notebooks/experiments.ipynb` to reproduce — the final cell prints subject-level metrics for all models.

## Features

229 features per subject extracted with Welch PSD across 19 channels:

- absolute band power δ θ α β (76)
- relative band power (76)
- per-channel mean, std, skew, kurtosis (76)
- theta/alpha ratio (1)

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
