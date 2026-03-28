import argparse
import pathlib

import joblib
import numpy as np
import pandas as pd
import torch

from data_loader import (
    load_data,
    get_subject_windows,
    EEG_CHANNELS,
)
from preprocessing import drop_missing
from feature_extraction import extract_subject_features, extract_window_features
from neural_net import EEGConvNet, prepare_cnn_input
from evaluation import evaluate, aggregate_by_subject

MODELS_DIR = pathlib.Path(__file__).parent / "models"


def load_sklearn(models_dir: pathlib.Path):
    model = joblib.load(models_dir / "best_sklearn.joblib")
    scaler = joblib.load(models_dir / "feat_scaler.joblib")
    return model, scaler


def load_cnn(models_dir: pathlib.Path, n_channels: int = 19):
    model = EEGConvNet(n_channels=n_channels)
    state = torch.load(models_dir / "best_cnn.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def predict_sklearn(
    df: pd.DataFrame,
    model,
    scaler,
    window_size: int = 256,
    step: int = 128,
) -> dict[str, dict]:
    X, y, groups = extract_subject_features(df, window_size=window_size, step=step)
    X = scaler.transform(X)

    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None

    return _aggregate_per_subject(y, y_pred, y_proba, groups)


def predict_cnn(
    df: pd.DataFrame,
    model: EEGConvNet,
    window_size: int = 256,
    step: int = 128,
) -> dict[str, dict]:
    X_raw, y, groups = get_subject_windows(df, window_size=window_size, step=step)
    X_cnn = prepare_cnn_input(X_raw)

    device = next(model.parameters()).device
    with torch.no_grad():
        logits = model(torch.tensor(X_cnn, dtype=torch.float32).to(device))
        y_proba = torch.sigmoid(logits).cpu().numpy()
        y_pred = (y_proba > 0.5).astype(int)

    return _aggregate_per_subject(y, y_pred, y_proba, groups)


def _aggregate_per_subject(
    y: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None,
    groups: np.ndarray,
) -> dict[str, dict]:
    results: dict[str, dict] = {}
    for subj in np.unique(groups):
        mask = groups == subj
        pred_mean = y_pred[mask].mean()
        results[str(subj)] = {
            "label_true": int(round(float(y[mask].mean()))),
            "label_pred": int(round(pred_mean)),
            "proba": float(y_proba[mask].mean()) if y_proba is not None else float(pred_mean),
            "n_windows": int(mask.sum()),
        }
    return results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ADHD EEG inference")
    p.add_argument("--data", required=True, help="path to CSV with EEG data")
    p.add_argument(
        "--model",
        choices=["sklearn", "cnn"],
        default="sklearn",
        help="which saved model to use (default: sklearn)",
    )
    p.add_argument("--models-dir", default=str(MODELS_DIR), help="directory with saved models")
    p.add_argument("--verbose", action="store_true", help="print per-subject detail")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    models_dir = pathlib.Path(args.models_dir)

    df = load_data(args.data)
    df = drop_missing(df)
    has_labels = "Class" in df.columns

    if args.model == "sklearn":
        model, scaler = load_sklearn(models_dir)
        results = predict_sklearn(df, model, scaler)
    else:
        model = load_cnn(models_dir, n_channels=len(EEG_CHANNELS))
        results = predict_cnn(df, model)

    out_rows = []
    for subj, info in results.items():
        row = {
            "subject_id": subj,
            "prediction": "ADHD" if info["label_pred"] == 1 else "Control",
            "probability": f"{info['proba']:.3f}",
            "n_windows": info["n_windows"],
        }
        if has_labels:
            row["ground_truth"] = "ADHD" if info["label_true"] == 1 else "Control"
            row["correct"] = info["label_pred"] == info["label_true"]
        out_rows.append(row)

    out_df = pd.DataFrame(out_rows)
    if args.verbose:
        print(out_df.to_string(index=False))
    else:
        print(out_df[["subject_id", "prediction", "probability"]].to_string(index=False))

    if has_labels:
        y_true = np.array([r["label_true"] for r in results.values()])
        y_pred = np.array([r["label_pred"] for r in results.values()])
        y_proba = np.array([r["proba"] for r in results.values()])
        print("\n--- evaluation ---")
        evaluate(y_true, y_pred, y_proba)


if __name__ == "__main__":
    main()
