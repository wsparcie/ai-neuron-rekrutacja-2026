import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay,
)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None = None) -> dict:
    report = classification_report(y_true, y_pred, target_names=['Control', 'ADHD'], output_dict=True)
    print(classification_report(y_true, y_pred, target_names=['Control', 'ADHD']))

    metrics = {
        'accuracy': report['accuracy'],
        'f1_adhd': report['ADHD']['f1-score'],
        'precision_adhd': report['ADHD']['precision'],
        'recall_adhd': report['ADHD']['recall'],
    }

    if y_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
        print(f"ROC AUC: {metrics['roc_auc']:.4f}")

    return metrics


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, title: str = '') -> None:
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=['Control', 'ADHD'])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    if title:
        ax.set_title(title)
    plt.tight_layout()
    plt.show()


def plot_roc_curve(y_true: np.ndarray, y_proba: np.ndarray, label: str = '') -> None:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"{label} (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.tight_layout()
    plt.show()


def aggregate_by_subject(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    unique_subjects = np.unique(groups)
    agg_true, agg_pred = [], []
    agg_proba: list[float] | None = [] if y_proba is not None else None

    for subj in unique_subjects:
        mask = groups == subj
        agg_true.append(int(round(float(y_true[mask].mean()))))
        agg_pred.append(int(round(float(y_pred[mask].mean()))))
        if agg_proba is not None and y_proba is not None:
            agg_proba.append(float(y_proba[mask].mean()))

    return (
        np.array(agg_true),
        np.array(agg_pred),
        np.array(agg_proba) if agg_proba is not None else None,
    )


def compare_models(results: dict[str, dict]) -> None:
    df_results = {name: metrics for name, metrics in results.items()}
    df = pd.DataFrame(df_results).T
    print("\nmodel comparison:")
    print(df.round(4).to_string())

    metric_cols = [c for c in df.columns if c in ['accuracy', 'f1_adhd', 'roc_auc']]
    df[metric_cols].plot(kind='bar', figsize=(8, 5), ylim=(0, 1))
    plt.title('Model Comparison')
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.show()
