import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from data_loader import EEG_CHANNELS
from feature_extraction import (
    band_power,
    extract_features_from_window,
    extract_window_features,
    build_feature_names,
)


N_CHANNELS = len(EEG_CHANNELS)
WINDOW_SIZE = 256


def test_band_power_nonnegative():
    rng = np.random.default_rng(0)
    freqs = np.linspace(0, 64, 256)
    psd = rng.random(256)
    result = band_power(psd, freqs, 4.0, 8.0)
    assert result >= 0.0, f"band_power should be non-negative, got {result}"
    print("band_power returns non-negative value")


def test_feature_shape():
    rng = np.random.default_rng(0)
    window = rng.standard_normal((WINDOW_SIZE, N_CHANNELS)).astype(np.float32)

    features = extract_features_from_window(window)
    expected = len(build_feature_names())

    assert features.shape == (expected,), (
        f"expected ({expected},), got {features.shape}"
    )
    print(f"feature shape correct: {features.shape}  ({expected} features per window)")


def test_no_nans_in_features():
    rng = np.random.default_rng(1)
    window = rng.standard_normal((WINDOW_SIZE, N_CHANNELS)).astype(np.float32)
    features = extract_features_from_window(window)

    nan_count = int(np.isnan(features).sum())
    assert nan_count == 0, f"found {nan_count} NaN values in extracted features"
    print("no NaN values in extracted features")


def test_feature_names_count():
    names = build_feature_names()
    rng = np.random.default_rng(2)
    window = rng.standard_normal((WINDOW_SIZE, N_CHANNELS)).astype(np.float32)
    features = extract_features_from_window(window)

    assert len(names) == len(features), (
        f"name count ({len(names)}) != feature count ({len(features)})"
    )
    print(f"feature name count matches: {len(names)}")


def test_theta_alpha_ratio_in_names():
    names = build_feature_names()
    assert "theta_alpha_ratio" in names, "theta_alpha_ratio missing from feature names"
    assert names[-1] == "theta_alpha_ratio", "theta_alpha_ratio should be the last feature"
    print("theta_alpha_ratio present as last feature")


def test_batch_extraction_shape():
    rng = np.random.default_rng(3)
    n_windows = 10
    windows = rng.standard_normal((n_windows, WINDOW_SIZE, N_CHANNELS)).astype(np.float32)

    X = extract_window_features(windows)
    expected_features = len(build_feature_names())

    assert X.shape == (n_windows, expected_features), (
        f"expected ({n_windows}, {expected_features}), got {X.shape}"
    )
    assert not np.isnan(X).any(), "NaN values found in batch-extracted features"
    print(f"batch extraction shape correct: {X.shape}")


def test_features_vary_across_windows():
    rng = np.random.default_rng(4)
    w1 = rng.standard_normal((WINDOW_SIZE, N_CHANNELS)).astype(np.float32)
    w2 = rng.standard_normal((WINDOW_SIZE, N_CHANNELS)).astype(np.float32)

    f1 = extract_features_from_window(w1)
    f2 = extract_features_from_window(w2)

    assert not np.allclose(f1, f2), "different windows produced identical features"
    print("features vary across different windows")


if __name__ == "__main__":
    print(f"running feature extraction tests  ({N_CHANNELS} channels, window={WINDOW_SIZE})\n")

    tests = [
        test_band_power_nonnegative,
        test_feature_shape,
        test_no_nans_in_features,
        test_feature_names_count,
        test_theta_alpha_ratio_in_names,
        test_batch_extraction_shape,
        test_features_vary_across_windows,
    ]

    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except AssertionError as e:
            print(f"✗ {t.__name__}: {e}")

    print(f"\n{passed}/{len(tests)} tests passed")
    if passed != len(tests):
        sys.exit(1)
