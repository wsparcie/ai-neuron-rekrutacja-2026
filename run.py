import argparse
import pathlib

import joblib
import numpy as np
import pandas as pd
import torch

from data_loader import (
    load_data,
    subject_split,
    get_subject_windows,
    EEG_CHANNELS,
)
from preprocessing import drop_missing, clip_artefacts
from feature_extraction import extract_subject_features, build_feature_names
from baseline import train_baselines, cross_validate_classifiers
from neural_net import (
    ADHDNet,
    EEGConvNet,
    train_nn,
    train_cnn,
    prepare_cnn_input,
)
from evaluation import evaluate, aggregate_by_subject, compare_models

from sklearn.preprocessing import RobustScaler


DEFAULT_DATA = pathlib.Path(__file__).parent / "data" / "adhdata.csv"
MODELS_DIR = pathlib.Path(__file__).parent / "models"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ADHD EEG classification pipeline")
    p.add_argument("--data", type=str, default=str(DEFAULT_DATA), help="path to adhdata.csv")
    p.add_argument("--no-nn", action="store_true", help="skip neural network training")
    p.add_argument("--seed", type=int, default=42, help="global random seed")
    return p.parse_args()


def _section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    _section("1 / 6  Loading & preprocessing")
    df = load_data(args.data)
    df = drop_missing(df)

    train_df, test_df = subject_split(df, test_size=0.2, random_state=args.seed)
    del df
    train_df, test_df = clip_artefacts(train_df, test_df, threshold=5.0)
    print(f"train subjects: {train_df['ID'].nunique()}"
          f"  |  test subjects: {test_df['ID'].nunique()}")

    _section("2 / 6  Feature extraction (256-sample windows, 50% overlap)")
    X_train, y_train, groups_train = extract_subject_features(train_df)
    X_test,  y_test,  groups_test  = extract_subject_features(test_df)

    feat_scaler = RobustScaler()
    X_train = feat_scaler.fit_transform(X_train)
    X_test  = feat_scaler.transform(X_test)
    print(f"features: {X_train.shape[1]}  |  "
          f"train windows: {len(X_train)}  |  test windows: {len(X_test)}")

    X_train_cnn = X_train_cnn_inner = X_val_cnn = X_test_cnn = None
    y_train_inner = y_val = None

    if not args.no_nn:
        X_train_raw, _, _ = get_subject_windows(train_df, window_size=256, step=128)
        X_test_raw,  _, _ = get_subject_windows(test_df,  window_size=256, step=128)
        X_train_cnn = prepare_cnn_input(X_train_raw)
        X_test_cnn  = prepare_cnn_input(X_test_raw)

        _train_inner_df, _val_df = subject_split(train_df, test_size=0.2, random_state=0)
        X_train_inner, y_train_inner, _ = extract_subject_features(_train_inner_df)
        X_val,         y_val,         _ = extract_subject_features(_val_df)
        X_train_inner = feat_scaler.transform(X_train_inner)
        X_val         = feat_scaler.transform(X_val)
        _inner_raw, _, _ = get_subject_windows(_train_inner_df, window_size=256, step=128)
        _val_raw,   _, _ = get_subject_windows(_val_df,         window_size=256, step=128)
        X_train_cnn_inner = prepare_cnn_input(_inner_raw)
        X_val_cnn         = prepare_cnn_input(_val_raw)

    all_metrics: dict[str, dict] = {}

    _section("3 / 6  5-fold GroupKFold cross-validation (sklearn)")
    cv_summary = cross_validate_classifiers(
        X_train, y_train, groups=groups_train, n_splits=5)

    rows = []
    for name, metrics in cv_summary.items():
        row = {"Model": name}
        for metric, (mean, std) in metrics.items():
            row[metric] = f"{mean:.3f} ± {std:.3f}"
        rows.append(row)
    cv_df = pd.DataFrame(rows).set_index("Model")
    print(cv_df.to_string())

    _section("4 / 6  Sklearn baselines on hold-out test set (subject-level)")
    baseline_results = train_baselines(X_train, y_train, X_test, y_test)
    for name, res in baseline_results.items():
        print(f"\n{name}")
        agg_true, agg_pred, agg_proba = aggregate_by_subject(
            y_test, res["y_pred"], groups_test, res["y_proba"])
        metrics = evaluate(agg_true, agg_pred, agg_proba)
        all_metrics[name] = metrics

    if not args.no_nn:
        _section("5 / 6  MLP training")
        nn_configs = [
            {"hidden_dims": [128, 64],      "dropout": 0.3, "label": "MLP-small (128-64, d=0.3)"},
            {"hidden_dims": [256, 128, 64], "dropout": 0.3, "label": "MLP-medium (256-128-64, d=0.3)"},
            {"hidden_dims": [256, 128, 64], "dropout": 0.5, "label": "MLP-medium (256-128-64, d=0.5)"},
        ]
        for cfg in nn_configs:
            print(f"\ntraining {cfg['label']}")
            model, _ = train_nn(
                X_train_inner, y_train_inner, X_val, y_val,
                hidden_dims=cfg["hidden_dims"], dropout=cfg["dropout"],
                epochs=150, lr=1e-3, batch_size=32, patience=15, seed=args.seed,
            )
            device = next(model.parameters()).device
            model.eval()
            with torch.no_grad():
                logits = model(torch.tensor(X_test, dtype=torch.float32).to(device))
                proba  = torch.sigmoid(logits).cpu().numpy()
                preds  = (proba > 0.5).astype(int)
            agg_true, agg_pred, agg_proba = aggregate_by_subject(
                y_test, preds, groups_test, proba)
            print(f"\n{cfg['label']} (subject-level)")
            metrics = evaluate(agg_true, agg_pred, agg_proba)
            all_metrics[cfg["label"]] = metrics

        _section("5b / 6  1D-CNN training")
        cnn_configs = [
            {"dropout": 0.3, "label": "CNN (dropout=0.3)"},
            {"dropout": 0.5, "label": "CNN (dropout=0.5)"},
        ]
        cnn_results: dict[str, dict] = {}
        for cfg in cnn_configs:
            print(f"\ntraining {cfg['label']}")
            model, _ = train_cnn(
                X_train_cnn_inner, y_train_inner,
                X_val_cnn,         y_val,
                n_channels=len(EEG_CHANNELS),
                dropout=cfg["dropout"],
                epochs=50, lr=1e-3, batch_size=64, patience=10, seed=args.seed,
            )
            device = next(model.parameters()).device
            model.eval()
            with torch.no_grad():
                logits = model(torch.tensor(X_test_cnn, dtype=torch.float32).to(device))
                proba  = torch.sigmoid(logits).cpu().numpy()
                preds  = (proba > 0.5).astype(int)
            agg_true, agg_pred, agg_proba = aggregate_by_subject(
                y_test, preds, groups_test, proba)
            print(f"\n{cfg['label']} (subject-level)")
            metrics = evaluate(agg_true, agg_pred, agg_proba)
            all_metrics[cfg["label"]] = metrics
            cnn_results[cfg["label"]] = {"model": model, "metrics": metrics}

    _section("6 / 6  Model comparison & saving")
    compare_models(all_metrics)

    MODELS_DIR.mkdir(exist_ok=True)
    from sklearn.base import clone
    from baseline import CLASSIFIERS

    sklearn_names = list(baseline_results.keys())
    best_sk_name = max(sklearn_names, key=lambda n: all_metrics[n].get("roc_auc", 0))
    joblib.dump(baseline_results[best_sk_name]["model"], MODELS_DIR / "best_sklearn.joblib")
    joblib.dump(feat_scaler, MODELS_DIR / "feat_scaler.joblib")
    print(f"saved: best sklearn = {best_sk_name}")

    if not args.no_nn and cnn_results:
        best_cnn_name = max(cnn_results.keys(),
                            key=lambda n: all_metrics[n].get("roc_auc", 0))
        torch.save(cnn_results[best_cnn_name]["model"].state_dict(),
                   MODELS_DIR / "best_cnn.pt")
        print(f"saved: best CNN = {best_cnn_name}")

    print("\nDone.")


if __name__ == "__main__":
    main()
