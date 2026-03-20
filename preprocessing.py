import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from .data_loader import EEG_CHANNELS


def drop_missing(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.dropna(subset=EEG_CHANNELS)
    dropped = before - len(df)
    if dropped:
        print(f"dropped {dropped} rows with NaN values")
    return df.reset_index(drop=True)


def scale_channels(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scaler = RobustScaler()
    train_df = train_df.copy()
    test_df = test_df.copy()

    train_df[EEG_CHANNELS] = scaler.fit_transform(train_df[EEG_CHANNELS])
    test_df[EEG_CHANNELS] = scaler.transform(test_df[EEG_CHANNELS])

    return train_df, test_df


def clip_artefacts(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    threshold: float = 5.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = train_df.copy()
    test_df = test_df.copy()
    for ch in EEG_CHANNELS:
        median = train_df[ch].median()
        mad = (train_df[ch] - median).abs().median()
        lo = median - threshold * mad
        hi = median + threshold * mad
        train_df[ch] = train_df[ch].clip(lower=lo, upper=hi)
        test_df[ch] = test_df[ch].clip(lower=lo, upper=hi)
    return train_df, test_df
