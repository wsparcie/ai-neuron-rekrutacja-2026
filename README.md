# Wykrywanie ADHD z sygnału EEG

Binarna klasyfikacja ADHD/kontrola na podstawie spoczynkowego EEG. 121 podmiotów, 19 kanałów, 128 Hz.

Dataset: https://www.kaggle.com/datasets/danizo/eeg-dataset-for-adhd/data

## Uruchomienie

```bash
python -m venv venv
venv\Scripts\activate        # windows
# source venv/bin/activate   # linux/mac
pip install -r requirements.txt
```

Pobierz `adhdata.csv` z Kaggle i wrzuć do `data/`.

## Notebooki

- `notebooks/eda.ipynb` — PSD, topomaps, TAR, koherencja
- `notebooks/experiments.ipynb` — ekstrakcja cech, trening modeli, ewaluacja

## Wyniki

5-fold GroupKFold CV (podział per pacjent):

| model         | accuracy      | f1            | roc auc       |
| ------------- | ------------- | ------------- | ------------- |
| Random Forest | 0.825 ± 0.062 | 0.830 ± 0.076 | 0.910 ± 0.055 |
| GBM           | 0.811 ± 0.066 | 0.822 ± 0.077 | 0.890 ± 0.066 |
| SVM (RBF)     | 0.800 ± 0.056 | 0.811 ± 0.075 | 0.884 ± 0.070 |
| LDA           | 0.759 ± 0.047 | 0.782 ± 0.068 | 0.802 ± 0.048 |

Hold-out testowy (25 podmiotów, głosowanie większościowe per pacjent):

| model           | accuracy | f1    | roc auc |
| --------------- | -------- | ----- | ------- |
| **CNN (d=0.5)** | **0.80** | 0.783 | 0.872   |
| CNN (d=0.3)     | 0.76     | 0.727 | 0.885   |
| SVM / RF / GBM  | 0.76     | 0.786 | 0.776   |
| MLP             | 0.72     | ~0.73 | ~0.75   |
| LDA             | 0.72     | 0.741 | 0.769   |
