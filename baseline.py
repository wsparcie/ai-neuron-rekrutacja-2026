import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.base import clone
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score


CLASSIFIERS = {
    'SVM (RBF)': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVC(kernel='rbf', C=1.0, probability=True, random_state=42)),
    ]),
    'Random Forest': RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42,
    ),
    'LDA': LinearDiscriminantAnalysis(),
}


def train_baselines(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    results = {}

    for name, clf in CLASSIFIERS.items():
        model = clone(clf)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

        results[name] = {'model': model, 'y_pred': y_pred, 'y_proba': y_proba}
        print(f"trained: {name}")

    return results


def cross_validate_classifiers(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
) -> dict:
    gkf = GroupKFold(n_splits=n_splits)
    cv_results = {name: {'accuracy': [], 'f1': [], 'roc_auc': []} for name in CLASSIFIERS}

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups), 1):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        for name, clf in CLASSIFIERS.items():
            model = clone(clf)
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_val)
            y_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else None

            cv_results[name]['accuracy'].append(accuracy_score(y_val, y_pred))
            cv_results[name]['f1'].append(f1_score(y_val, y_pred, zero_division=0))
            if y_proba is not None:
                cv_results[name]['roc_auc'].append(roc_auc_score(y_val, y_proba))

        print(f"folded: {fold}/{n_splits}")

    summary = {}
    for name, scores in cv_results.items():
        summary[name] = {
            metric: (np.mean(vals), np.std(vals))
            for metric, vals in scores.items() if vals
        }
    return summary
