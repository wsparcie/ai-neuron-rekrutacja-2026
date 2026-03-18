import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import skew, kurtosis

from data_loader import EEG_CHANNELS, SAMPLING_FREQ


FREQ_BANDS = {
    'delta': (0.5, 4.0),
    'theta': (4.0, 8.0),
    'alpha': (8.0, 13.0),
    'beta':  (13.0, 30.0),
}


def band_power(psd: np.ndarray, freqs: np.ndarray, low: float, high: float) -> float:
    mask = (freqs >= low) & (freqs < high)
    return float(np.trapezoid(psd[mask], freqs[mask]))


def extract_features_from_window(window: np.ndarray) -> np.ndarray:
    features = []
    n_samples = window.shape[0]

    for ch_idx in range(window.shape[1]):
        signal = window[:, ch_idx]

        freqs, psd = welch(signal, fs=SAMPLING_FREQ, nperseg=min(n_samples, SAMPLING_FREQ * 2))

        for band, (low, high) in FREQ_BANDS.items():
            bp = band_power(psd, freqs, low, high)
            features.append(bp)

        total = band_power(psd, freqs, 0.5, 30.0) + 1e-12
        for band, (low, high) in FREQ_BANDS.items():
            bp = band_power(psd, freqs, low, high)
            features.append(bp / total)

        features.extend([
            float(np.mean(signal)),
            float(np.std(signal)),
            float(skew(signal)),
            float(kurtosis(signal)),
        ])

    theta_sum, alpha_sum = 0.0, 0.0
    for ch_idx in range(window.shape[1]):
        signal = window[:, ch_idx]
        freqs, psd = welch(signal, fs=SAMPLING_FREQ, nperseg=min(n_samples, SAMPLING_FREQ * 2))
        theta_sum += band_power(psd, freqs, *FREQ_BANDS['theta'])
        alpha_sum += band_power(psd, freqs, *FREQ_BANDS['alpha'])
    features.append(theta_sum / (alpha_sum + 1e-12))

    return np.array(features, dtype=np.float32)


def build_feature_names() -> list[str]:
    names = []
    for ch in EEG_CHANNELS:
        for band in FREQ_BANDS:
            names.append(f"{ch}_{band}_power")
        for band in FREQ_BANDS:
            names.append(f"{ch}_{band}_rel")
        names.extend([f"{ch}_mean", f"{ch}_std", f"{ch}_skew", f"{ch}_kurtosis"])
    names.append('theta_alpha_ratio')
    return names


def extract_subject_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    X_rows, y_rows = [], []

    for subject_id, group in df.groupby('ID'):
        window = group[EEG_CHANNELS].values
        features = extract_features_from_window(window)
        X_rows.append(features)
        y_rows.append(group['label'].iloc[0])

    return np.stack(X_rows), np.array(y_rows)


def extract_window_features(windows: np.ndarray) -> np.ndarray:
    return np.stack([extract_features_from_window(w) for w in windows])
