# Wykrywanie ADHD z sygnału EEG

Klasyfikacja binarna ADHD/kontrola na EEG z zadania uwagi wzrokowej.  
121 podmiotów · 19 kanałów · 128 Hz  
Dane: Nasrabadi et al., Shahed University — [IEEE DataPort DOI: 10.21227/rzfh-zn36](https://dx.doi.org/10.21227/rzfh-zn36) / [Kaggle mirror](https://www.kaggle.com/datasets/danizo/eeg-dataset-for-adhd/data)

---

## Tech Stack

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python)](https://www.python.org/) [![NumPy](https://img.shields.io/badge/NumPy-2.4+-013243?logo=numpy)](https://numpy.org/) [![Pandas](https://img.shields.io/badge/Pandas-3.0+-150458?logo=pandas)](https://pandas.pydata.org/) [![SciPy](https://img.shields.io/badge/SciPy-1.17+-8CAAE6?logo=scipy)](https://scipy.org/)

[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8+-F7931E?logo=scikitlearn)](https://scikit-learn.org/) [![PyTorch](https://img.shields.io/badge/PyTorch-2.10+-EE4C2C?logo=pytorch)](https://pytorch.org/) [![Matplotlib](https://img.shields.io/badge/Matplotlib-3.10+-11557c)](https://matplotlib.org/) [![Seaborn](https://img.shields.io/badge/Seaborn-0.13+-9cf)](https://seaborn.pydata.org/)

[![Jupyter](https://img.shields.io/badge/Jupyter-Lab-F37626?logo=jupyter)](https://jupyter.org/) [![pytest](https://img.shields.io/badge/pytest-8.0+-0A9EDC?logo=pytest)](https://pytest.org/) [![Status](https://img.shields.io/badge/Status-Research-yellow)]() [![Dataset%20License](https://img.shields.io/badge/Dataset%20License-CC--BY-blue)](https://creativecommons.org/licenses/by/4.0/)

</div>

---

## O projekcie

W pełni zaimplementowany pipeline, czyli od surowego CSV przez preprocessing i ekstrakcję cech, aż po porównanie klasyfikatorów (sklearn + PyTorch).

Podział następuje na poziomie podmiotów (subject split), nie okien. Wszystkie okna tego samego pacjenta lądują w jednym zbiorze; bez tego AUC byłoby zawyżone o kilkanaście punktów procentowych.

Ekstrakcja cech liczy ~350 cech na okno (moce pasm, parametry Hjortha, TAR/TBR, asymetria alfa, koherencja frontoparietalna i interhemisferyczna). Obok tego pipeline CNN operuje bezpośrednio na surowych oknach bez ręcznych cech.

Wynik końcowy jest agregowany per pacjent (głosowanie większościowe / uśrednianie prawdopodobieństwa), co jest bardziej sensowną metryką niż dokładność per-okno.

---

## Start

### 1. Klonowanie i środowisko

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

Pobierz `adhdata.csv` z Kaggle i wrzuć do `data/`.

## Metodologia

### Dane i preprocessing

| Parametr                  | Wartość                                                                |
| ------------------------- | ---------------------------------------------------------------------- |
| Liczba podmiotów          | 121 (61 ADHD, 60 Control; chłopcy i dziewczynki, wiek 7–12 lat)        |
| Diagnoza ADHD             | Psychiatra wg kryteriów DSM-IV; dzieci leczone Ritalinem do 6 miesięcy |
| Protokół zapisu           | Zadanie uwagi wzrokowej (liczenie postaci z bajek)                     |
| Kanały EEG                | 19; referencja A1/A2                                                   |
| Częstotliwość próbkowania | 128 Hz                                                                 |

> Dzieci z ADHD były leczone Ritalinem do 6 miesięcy przed nagraniem. Ritalin normalizuje moc theta, stąd klasyczny wzorzec "więcej theta u ADHD" nie jest widoczny w bezwzględnych mocach pasm, więc sygnał diagnostyczny przenosi się do cech względnych (TAR, TBR) i topograficznych (koherencja frontoparietalna).

**Preprocessing:**

1. `load_data()`: wczytanie CSV, walidacja kolumn, parsowanie etykiet.
2. `subject_split()`: podział 80/20 na poziomie podmiotów (stratyfikowany po klasie).
3. `drop_missing()`: usunięcie wierszy z NaN.
4. `clip_artefacts()`: obcięcie amplitud > mediana ± 5×MAD (liczone tylko na train). Artefakty ruchowe zostają przycięte bez jakiegokolwiek filtrowania sygnału.

### Ekstrakcja cech

Sygnał dzielony na okna 256 próbek (2 s), krok 128 (50% overlap).

| Kategoria               | Opis                                                     | Liczba cech |
| ----------------------- | -------------------------------------------------------- | ----------- |
| Moc pasm (bezwzględna)  | δ, θ, α, β, γ dla każdego kanału                         | 19 × 5 = 95 |
| Moc pasm (względna)     | j.w. znormalizowana przez moc całkowitą                  | 19 × 5 = 95 |
| Statystyki czasowe      | Mean, Std, Skew, Kurtosis                                | 19 × 4 = 76 |
| Parametry Hjortha       | Activity, Mobility, Complexity                           | 19 × 3 = 57 |
| Entropia spektralna     | Shannon entropy widma mocy                               | 19          |
| SEF95                   | Częstotliwość graniczna 95% mocy skumulowanej            | 19          |
| Zero-crossing rate      | Prędkość zmiany znaku sygnału                            | 19          |
| Peak-to-peak            | Zakres amplitud w oknie                                  | 19          |
| TAR / TBR / SWDR        | Theta/Alpha Ratio, Theta/Beta Ratio, Slow-Wave Dominance | 3           |
| Asymetria alfa          | log(alfa_prawy) − log(alfa_lewy) dla par F4/F3, F8/F7    | 2           |
| Koherencja fp           | Koherencja theta Fz–Pz, F3–P3, F4–P4                     | 3           |
| Koherencja interhemisf. | Wszystkie pasma dla 8 par kanałów                        | 40          |
| **Łącznie**             |                                                          | **~350+**   |

Cechy oparte na PSD (band power, TAR, TBR, SEF95) liczone metodą Welcha (`scipy.signal.welch`, nperseg=256).

### Modele

| Model                   | Wejście                     | Biblioteka | Szczegóły                                                                               |
| ----------------------- | --------------------------- | ---------- | --------------------------------------------------------------------------------------- |
| **SVM (RBF)**           | cechy                       | sklearn    | C=1.0, StandardScaler w Pipeline                                                        |
| **Random Forest**       | cechy                       | sklearn    | 200 drzew, brak ograniczenia głębokości                                                 |
| **Gradient Boosting**   | cechy                       | sklearn    | 100 drzew, lr=0.1, max_depth=3                                                          |
| **LDA**                 | cechy                       | sklearn    | baseline liniowy                                                                        |
| **MLP (ADHDNet)**       | cechy                       | PyTorch    | warstwy FC + BatchNorm + ReLU + Dropout, BCEWithLogitsLoss z pos_weight, early stopping |
| **1D-CNN (EEGConvNet)** | surowe okna (kanały × czas) | PyTorch    | 3× Conv1d (32→64→128), BN, ELU, AdaptiveAvgPool, head FC                                |

Wszystkie modele PyTorch trenowane są z:

- ważoną funkcją straty `BCEWithLogitsLoss(pos_weight=n_neg/n_pos)`,
- optymalizatorem Adam z `weight_decay=1e-4`,
- early stopping na zbiorze walidacyjnym (wewnętrzny subject split 80/20 ze zbioru treningowego).

### Ewaluacja

- CV: 5-fold `GroupKFold` (grupa = ID podmiotu) na zbiorze treningowym
- Hold-out: 25 podmiotów (20%), wynik agregowany per pacjent
- Metryki: Accuracy, F1 (klasa ADHD), ROC-AUC
- Confusion Matrix na poziomie podmiotów (nie okien)

---

## Wyniki

### 5-fold GroupKFold CV (zbiór treningowy)

| Model             | Accuracy      | F1            | ROC-AUC           |
| ----------------- | ------------- | ------------- | ----------------- |
| **Random Forest** | 0.825 ± 0.062 | 0.830 ± 0.076 | **0.910 ± 0.055** |
| Gradient Boosting | 0.811 ± 0.066 | 0.822 ± 0.077 | 0.890 ± 0.066     |
| SVM (RBF)         | 0.800 ± 0.056 | 0.811 ± 0.075 | 0.884 ± 0.070     |
| LDA               | 0.759 ± 0.047 | 0.782 ± 0.068 | 0.802 ± 0.048     |

### Hold-out (25 podmiotów, agregacja per pacjent)

| Model                 | Accuracy | F1        | ROC-AUC   |
| --------------------- | -------- | --------- | --------- |
| **CNN (dropout=0.5)** | **0.80** | **0.783** | 0.872     |
| CNN (dropout=0.3)     | 0.76     | 0.727     | **0.885** |
| SVM / RF / GBM        | 0.76     | 0.786     | 0.776     |
| MLP (256-128-64)      | 0.72     | ~0.73     | ~0.75     |
| LDA                   | 0.72     | 0.741     | 0.769     |

RF wygrywa w CV (AUC = 0,91); przy ~100 podmiotach w treningu drzewa dobrze radzą sobie z szumem cechowym. CNN dorównuje mu na hold-oucie (accuracy 0,80) i robi to bez ręcznej ekstrakcji cech, co sugeruje że surowy sygnał zawiera informację, której estymowane cechy nie wychwytują w pełni. LDA wyraźnie odstaje, gdyż przestrzeń cech jest nieliniowa.

---

## Testy

```bash
python -m pytest tests/ -v
```

---

## Struktura projektu

```
adhd-detector/
├── data/
│   └── adhdata.csv           # dane wejściowe
├── models/                   # zapisane wytrenowane modele
│   ├── best_cnn.pt
│   ├── best_sklearn.joblib
│   └── feat_scaler.joblib
├── notebooks/
│   ├── eda.ipynb             # analiza eksploracyjna
│   └── experiments.ipynb     # eksperymenty modelowania
├── tests/
│   └── test_features.py      # testy jednostkowe
├── __init__.py
├── baseline.py               # modele sklearn + GroupKFold CV
├── data_loader.py            # ładowanie danych, Subject Split, okienkowanie
├── evaluation.py             # metryki, CM, ROC, porównanie modeli
├── feature_extraction.py     # ekstrakcja 350+ cech z okien EEG
├── neural_net.py             # MLP (ADHDNet) + 1D-CNN (EEGConvNet) w PyTorch
├── preprocessing.py          # usuwanie NaN, clipping artefaktów (MAD)
├── requirements.txt
└── README.md
run.py                        # skrypt odtwarzający pełne eksperymenty
```

---

## Źródło danych

Nasrabadi, A. M., Allahverdy, A., Samavati, M., & Mohammadi, M. R. (2020).  
_EEG data for ADHD / Control children._ IEEE DataPort.  
https://dx.doi.org/10.21227/rzfh-zn36  
Licencja: Creative Commons Attribution
