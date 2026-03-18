import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit


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
        raise FileNotFoundError(f"Dataset not found at: {csv_path}")

    df = pd.read_csv(csv_path)

    missing = [c for c in EEG_CHANNELS + ['Class', 'ID'] if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing expected columns: {missing}")

    df['label'] = (df['Class'].str.strip() == 'ADHD').astype(int)
    df['ID'] = df['ID'].astype(str).str.strip()

    n_subjects = df['ID'].nunique()
    n_adhd = df.groupby('ID')['label'].first().sum()
    print(f"Loaded {len(df):,} samples | {n_subjects} subjects ({n_adhd} ADHD, {n_subjects - n_adhd} Control)")

    return df


def subject_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    subject_labels = df.groupby('ID')['label'].first().reset_index()

    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(sss.split(subject_labels['ID'], subject_labels['label']))

    train_ids = set(subject_labels.iloc[train_idx]['ID'])
    test_ids = set(subject_labels.iloc[test_idx]['ID'])

    train_df = df[df['ID'].isin(train_ids)].reset_index(drop=True)
    test_df = df[df['ID'].isin(test_ids)].reset_index(drop=True)

    _print_split_info(subject_labels, train_ids, test_ids)

    return train_df, test_df


def _print_split_info(
    subject_labels: pd.DataFrame,
    train_ids: set,
    test_ids: set,
) -> None:
    train_labels = subject_labels[subject_labels['ID'].isin(train_ids)]['label']
    test_labels = subject_labels[subject_labels['ID'].isin(test_ids)]['label']

    print(
        f"Train: {len(train_ids)} subjects "
        f"({train_labels.sum()} ADHD, {(~train_labels.astype(bool)).sum()} Control)"
    )
    print(
        f"Test:  {len(test_ids)} subjects "
        f"({test_labels.sum()} ADHD, {(~test_labels.astype(bool)).sum()} Control)"
    )


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

    print(f"Created {len(X)} windows of size {window_size} (step={step})")
    return X, y, groups
