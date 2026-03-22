import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split


EEG_CHANNELS = [
    'Fp1', 'Fp2',
    'F3', 'F4', 'F7', 'F8',
    'C3', 'C4', 'Cz',
    'T7', 'T8',
    'P3', 'P4', 'P7', 'P8', 'Pz',
    'O1', 'O2',
    'Fz',
]

SAMPLING_FREQ = 128


def load_data(csv_path: str | Path) -> pd.DataFrame:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"dataset not found at: {csv_path}")

    df = pd.read_csv(csv_path)

    missing = [c for c in EEG_CHANNELS + ['Class', 'ID'] if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing expected columns: {missing}")

    df['label'] = (df['Class'].str.strip() == 'ADHD').astype(int)
    df['ID'] = df['ID'].astype(str).str.strip()

    return df


def subject_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    subject_labels = df.groupby('ID')['label'].first().reset_index()

    train_subjects, test_subjects = train_test_split(
        subject_labels,
        test_size=test_size,
        stratify=subject_labels['label'],
        random_state=random_state,
    )

    train_ids = set(train_subjects['ID'])
    test_ids = set(test_subjects['ID'])

    train_df = df[df['ID'].isin(train_ids)].reset_index(drop=True)
    test_df = df[df['ID'].isin(test_ids)].reset_index(drop=True)

    return train_df, test_df


def get_subject_windows(
    df: pd.DataFrame,
    window_size: int = 256,
    step: int = 128,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_list, y_list, groups_list = [], [], []

    for subject_id, group in df.groupby('ID'):
        signal = group[EEG_CHANNELS].values
        label = group['label'].iloc[0]

        n_samples = len(signal)
        starts = range(0, n_samples - window_size + 1, step)

        for start in starts:
            window = signal[start: start + window_size]
            X_list.append(window)
            y_list.append(label)
            groups_list.append(subject_id)

    X = np.stack(X_list)
    y = np.array(y_list)
    groups = np.array(groups_list)

    return X, y, groups
