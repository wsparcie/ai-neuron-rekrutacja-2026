<p align="center">
  <h1 align="center">ADHD Detection from EEG</h1>
</p>

<p align="center">
    <strong>Binary classification of ADHD vs Control using 19-channel EEG spectral features</strong>
</p>

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python)](https://www.python.org/) [![NumPy](https://img.shields.io/badge/NumPy-2.4+-013243?logo=numpy)](https://numpy.org/) [![Pandas](https://img.shields.io/badge/Pandas-3.0+-150458?logo=pandas)](https://pandas.pydata.org/)

[![SciPy](https://img.shields.io/badge/SciPy-1.17+-8CAAE6?logo=scipy)](https://scipy.org/) [![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8+-F7931E?logo=scikitlearn)](https://scikit-learn.org/) [![Matplotlib](https://img.shields.io/badge/Matplotlib-3.10+-11557c)](https://matplotlib.org/) [![Seaborn](https://img.shields.io/badge/Seaborn-0.13+-9cf)](https://seaborn.pydata.org/)

[![PyTorch](https://img.shields.io/badge/PyTorch-2.10+-EE4C2C?logo=pytorch)](https://pytorch.org/) [![Jupyter](https://img.shields.io/badge/Jupyter-Lab-F37626?logo=jupyter)](https://jupyter.org/)

[![Status](https://img.shields.io/badge/Status-Alpha-yellow)]() [![License](https://img.shields.io/badge/License-MIT-blue)](LICENSE)

</div>

---

#### Dataset: [EEG Dataset for ADHD](https://www.kaggle.com/datasets/danizo/eeg-dataset-for-adhd/data)

121 subjects (61 ADHD, 60 Control)
128 Hz
19 channels

## Setup

```bash
python -m venv venv

# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

Download `adhdata.csv` from Kaggle and place it at `data/adhdata.csv`.

## Notebooks

| notebook                      | description                                      |
| ----------------------------- | ------------------------------------------------ |
| `notebooks/eda.ipynb`         | signal quality, band powers, class distributions |
| `notebooks/experiments.ipynb` | preprocessing → features → models → comparison   |

## Results

5-fold GroupKFold CV:

| model             | accuracy      | f1            | roc auc       |
| ----------------- | ------------- | ------------- | ------------- |
| SVM (RBF)         | 0.655 ± 0.095 | 0.678 ± 0.103 | 0.747 ± 0.115 |
| Random Forest     | 0.696 ± 0.110 | 0.695 ± 0.144 | 0.738 ± 0.061 |
| Gradient Boosting | 0.696 ± 0.114 | 0.691 ± 0.142 | 0.723 ± 0.066 |
| LDA               | 0.573 ± 0.089 | 0.564 ± 0.130 | 0.534 ± 0.143 |

Hold-out test set:

| model                              | accuracy | f1 (ADHD) | roc auc   |
| ---------------------------------- | -------- | --------- | --------- |
| SVM (RBF)                          | 0.68     | 0.667     | 0.596     |
| Random Forest                      | 0.68     | 0.667     | 0.644     |
| Gradient Boosting                  | 0.60     | 0.615     | 0.660     |
| LDA                                | 0.52     | 0.538     | 0.436     |
| MLP-small (128→64, d=0.3)          | 0.68     | 0.636     | 0.737     |
| **MLP-medium (256→128→64, d=0.3)** | **0.72** | **0.741** | **0.744** |
| MLP-medium (256→128→64, d=0.5)     | 0.68     | 0.667     | 0.744     |

## Features

229 features per subject, extracted via Welch PSD over 19 channels:

- absolute band power (δ, θ, α, β): 76 features
- relative band power: 76 features
- per-channel stats (mean, std, skew, kurtosis): 76 features
- theta/alpha ratio (TAR): 1 feature

## Structure

```
adhd_classifier/
├── data_loader.py          # loading, subject-level split, windowing
├── preprocessing.py        # artefact clipping (5σ), RobustScaler
├── feature_extraction.py   # Welch PSD band powers + stats + TAR (229 features)
├── baseline.py             # SVM, RF, GBM, LDA + GroupKFold CV
├── neural_net.py           # PyTorch MLP (BatchNorm + Dropout)
├── evaluation.py           # metrics, confusion matrix, ROC, model comparison
├── tests/
│   └── test_features.py
└── notebooks/
    ├── eda.ipynb
    └── experiments.ipynb
```
