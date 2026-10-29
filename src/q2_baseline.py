"""Leakage-safe classical baseline helpers for PS-Q2."""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score, f1_score

SEED = 42


def make_search(cv_splits: int = 5) -> GridSearchCV:
    """Create a training-only CV search; scaling is fitted inside each fold."""
    pipeline = Pipeline([
        ("scale", StandardScaler()),
        ("svc", SVC(kernel="rbf", class_weight="balanced")),
    ])
    return GridSearchCV(
        pipeline,
        {"svc__C": [0.1, 1.0, 10.0], "svc__gamma": ["scale", 0.01, 0.1]},
        scoring="balanced_accuracy",
        cv=StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=SEED),
        n_jobs=-1,
        refit=True,
    )


def fit_predict(X_train, y_train, X_test):
    """Tune only on training CV, then return test labels and audit metadata."""
    search = make_search()
    search.fit(np.asarray(X_train), np.asarray(y_train))
    predictions = search.predict(np.asarray(X_test))
    return predictions, {
        "best_params": search.best_params_,
        "cv_balanced_accuracy": float(search.best_score_),
        "seed": SEED,
        "model": "RBF SVC with fold-local StandardScaler",
    }


def summarize(y_true, y_pred):
    """Compute the official metric and reference F1 on held-out fold predictions."""
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, pos_label="B")),
    }
