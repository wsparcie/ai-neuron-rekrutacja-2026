import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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

    from sklearn.base import clone

    results = {}

    for name, clf in CLASSIFIERS.items():
        model = clone(clf)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_proba = None

        results[name] = {'model': model, 'y_pred': y_pred, 'y_proba': y_proba}
        print(f"Trained: {name}")

    return results
