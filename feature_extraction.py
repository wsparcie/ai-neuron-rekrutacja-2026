import numpy as np
import pandas as pd
from scipy.signal import welch, coherence
from scipy.stats import skew, kurtosis

from data_loader import EEG_CHANNELS, SAMPLING_FREQ

try:
    _trapz = np.trapezoid
except AttributeError:
    _trapz = np.trapz


FREQ_BANDS = {
    'delta': (0.5, 4.0),
    'theta': (4.0, 8.0),
    'alpha': (8.0, 13.0),
    'beta':  (13.0, 30.0),
    'gamma': (30.0, 45.0),
}

_CH = {ch: i for i, ch in enumerate(EEG_CHANNELS)}

_ASYMMETRY_PAIRS = [('F4', 'F3'), ('F8', 'F7')]

_FP_THETA_PAIRS = [('Fz', 'Pz'), ('F3', 'P3'), ('F4', 'P4')]

_INTER_PAIRS = [
    ('Fp1', 'Fp2'), ('F3', 'F4'), ('F7', 'F8'),
    ('C3', 'C4'), ('T7', 'T8'), ('P3', 'P4'),
    ('P7', 'P8'), ('O1', 'O2'),
]


def band_power(psd: np.ndarray, freqs: np.ndarray, low: float, high: float) -> float:
    mask = (freqs >= low) & (freqs < high)
    return float(_trapz(psd[mask], freqs[mask]))


def extract_features_from_window(window: np.ndarray) -> np.ndarray:
    features: list[float] = []
    n_samples = window.shape[0]
    nperseg = min(n_samples, SAMPLING_FREQ * 2)

    theta_sum = alpha_sum = beta_sum = delta_sum = 0.0
    alpha_power: dict[str, float] = {}

    for ch_idx, ch_name in enumerate(EEG_CHANNELS):
        signal = window[:, ch_idx]
        freqs, psd = welch(signal, fs=SAMPLING_FREQ, nperseg=nperseg)

        bp = {band: band_power(psd, freqs, low, high)
              for band, (low, high) in FREQ_BANDS.items()}
        for band in FREQ_BANDS:
            features.append(bp[band])

        total = sum(bp.values()) + 1e-12
        for band in FREQ_BANDS:
            features.append(bp[band] / total)

        features.extend([
            float(np.mean(signal)),
            float(np.std(signal)),
            float(skew(signal)),
            float(kurtosis(signal)),
        ])

        psd_norm = psd / (psd.sum() + 1e-12)
        features.append(float(-np.sum(psd_norm * np.log(psd_norm + 1e-12))))

        cumulative = np.cumsum(psd)
        sef_idx = min(int(np.searchsorted(cumulative, 0.95 * cumulative[-1])), len(freqs) - 1)
        features.append(float(freqs[sef_idx]))

        dx = np.diff(signal)
        ddx = np.diff(dx)
        activity = float(np.var(signal))
        var_dx = float(np.var(dx))
        var_ddx = float(np.var(ddx))
        mobility = float(np.sqrt(var_dx / (activity + 1e-12)))
        complexity = float(np.sqrt(var_ddx / (var_dx + 1e-12)) / (mobility + 1e-12))
        features.extend([activity, mobility, complexity])

        features.append(float(np.sum(np.diff(np.sign(signal)) != 0) / len(signal)))

        features.append(float(np.max(signal) - np.min(signal)))

        theta_sum += bp['theta']
        alpha_sum += bp['alpha']
        beta_sum  += bp['beta']
        delta_sum += bp['delta']
        alpha_power[ch_name] = bp['alpha']

    features.append(theta_sum / (alpha_sum + 1e-12))
    features.append(theta_sum / (beta_sum + 1e-12))
    features.append((delta_sum + theta_sum) / (alpha_sum + beta_sum + 1e-12))

    for r_ch, l_ch in _ASYMMETRY_PAIRS:
        features.append(
            float(np.log(alpha_power[r_ch] + 1e-12) - np.log(alpha_power[l_ch] + 1e-12))
        )

    for f_ch, p_ch in _FP_THETA_PAIRS:
        coh_freqs, coh = coherence(
            window[:, _CH[f_ch]], window[:, _CH[p_ch]],
            fs=SAMPLING_FREQ, nperseg=nperseg,
        )
        theta_mask = (coh_freqs >= 4.0) & (coh_freqs < 8.0)
        features.append(float(np.mean(coh[theta_mask])) if theta_mask.any() else 0.0)

    for l_ch, r_ch in _INTER_PAIRS:
        coh_freqs, coh = coherence(
            window[:, _CH[l_ch]], window[:, _CH[r_ch]],
            fs=SAMPLING_FREQ, nperseg=nperseg,
        )
        for low, high in FREQ_BANDS.values():
            mask = (coh_freqs >= low) & (coh_freqs < high)
            features.append(float(np.mean(coh[mask])) if mask.any() else 0.0)

    return np.array(features, dtype=np.float32)


def build_feature_names() -> list[str]:
    names = []
    for ch in EEG_CHANNELS:
        for band in FREQ_BANDS:
            names.append(f"{ch}_{band}_power")
        for band in FREQ_BANDS:
            names.append(f"{ch}_{band}_rel")
        names.extend([
            f"{ch}_mean", f"{ch}_std", f"{ch}_skew", f"{ch}_kurtosis",
            f"{ch}_spec_entropy", f"{ch}_sef95",
            f"{ch}_hjorth_activity", f"{ch}_hjorth_mobility", f"{ch}_hjorth_complexity",
            f"{ch}_zcr", f"{ch}_ptp",
        ])
    names.extend(['theta_alpha_ratio', 'theta_beta_ratio', 'slow_wave_dominance'])
    for r_ch, l_ch in _ASYMMETRY_PAIRS:
        names.append(f"alpha_asymmetry_{r_ch}_{l_ch}")
    for f_ch, p_ch in _FP_THETA_PAIRS:
        names.append(f"theta_coh_{f_ch}_{p_ch}")
    for l_ch, r_ch in _INTER_PAIRS:
        for band in FREQ_BANDS:
            names.append(f"inter_coh_{l_ch}_{r_ch}_{band}")
    return names


def extract_subject_features(
    df: pd.DataFrame,
    window_size: int = 256,
    step: int = 128,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from data_loader import get_subject_windows

    X_windows, y, groups = get_subject_windows(df, window_size=window_size, step=step)
    X = extract_window_features(X_windows)
    return X, y, groups


def extract_window_features(windows: np.ndarray) -> np.ndarray:
    return np.stack([extract_features_from_window(w) for w in windows])
