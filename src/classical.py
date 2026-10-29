"""Nested-CV classical baselines for QFF2026 PS-Q2."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

from src.data import FEATURE_COLUMNS, SIGNAL_FEATURES, SEED, load_q2_data, pca_spectrum

C_VALUES = [1.0, 10.0]
GAMMA_VALUES = ["scale", 0.1]
K_VALUES = [4, 5, 6, 8]


def _make_estimator(model_name: str, k: int | None = None) -> tuple[Pipeline, dict]:
    steps = []
    if model_name.startswith("PCA-"):
        steps.append(("pca", PCA(n_components=int(k), random_state=SEED)))
        # This transformation is fitted separately within each CV training fold.
        steps.append(("scale", MinMaxScaler(feature_range=(0.0, 1.0))))
        if model_name.startswith("PCA-RBF"):
            steps.append(("svc", SVC(kernel="rbf", class_weight="balanced")))
            grid = {"svc__C": C_VALUES, "svc__gamma": GAMMA_VALUES}
        else:
            steps.append(("svc", SVC(kernel="poly", class_weight="balanced")))
            grid = {"svc__C": C_VALUES, "svc__gamma": GAMMA_VALUES, "svc__degree": [2]}
    elif model_name == "RBF-SVM all 50 features":
        steps = [("scale", StandardScaler()), ("svc", SVC(kernel="rbf", class_weight="balanced"))]
        grid = {"svc__C": C_VALUES, "svc__gamma": GAMMA_VALUES}
    elif model_name == "RBF-SVM x01-x06 ablation":
        steps = [("scale", StandardScaler()), ("svc", SVC(kernel="rbf", class_weight="balanced"))]
        grid = {"svc__C": C_VALUES, "svc__gamma": GAMMA_VALUES}
    elif model_name == "Logistic regression":
        steps = [("scale", StandardScaler()), ("lr", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=SEED))]
        grid = {"lr__C": C_VALUES}
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return Pipeline(steps), grid


def run_classical_baselines(root: str | Path | None = None) -> pd.DataFrame:
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    X, y, _, _, _, _ = load_q2_data(root)
    X = X.loc[:, FEATURE_COLUMNS]
    y = np.asarray(y, dtype=int)
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows = []

    configs = [("RBF-SVM all 50 features", None), ("Logistic regression", None), ("RBF-SVM x01-x06 ablation", None)]
    for k in K_VALUES:
        configs.extend([(f"PCA-RBF-SVM k={k}", k), (f"PCA-polynomial-SVM k={k}", k)])

    for model_name, k in configs:
        if model_name == "RBF-SVM x01-x06 ablation":
            X_model = X.loc[:, SIGNAL_FEATURES]
        else:
            X_model = X
        estimator, param_grid = _make_estimator(model_name, k)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        search = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            scoring="balanced_accuracy",
            cv=cv,
            n_jobs=1,
            refit=True,
            error_score="raise",
            return_train_score=False,
        )
        search.fit(X_model, y)
        # Out-of-fold predictions with the CV-selected hyperparameters, using the
        # same fixed stratified folds for every baseline.
        oof_pred = cross_val_predict(search.best_estimator_, X_model, y, cv=cv, n_jobs=1, method="predict")
        fold_scores = []
        for fold, (_, valid_idx) in enumerate(cv.split(X_model, y), start=1):
            fold_scores.append({
                "fold": fold,
                "balanced_accuracy": balanced_accuracy_score(y[valid_idx], oof_pred[valid_idx]),
                "f1": f1_score(y[valid_idx], oof_pred[valid_idx], pos_label=1, zero_division=0),
            })
        ba = np.array([s["balanced_accuracy"] for s in fold_scores], dtype=float)
        f1 = np.array([s["f1"] for s in fold_scores], dtype=float)
        rows.append({
            "model": model_name,
            "pca_components": k if k is not None else "",
            "balanced_accuracy_mean": ba.mean(),
            "balanced_accuracy_std": ba.std(ddof=1),
            "f1_mean": f1.mean(),
            "f1_std": f1.std(ddof=1),
            "fold_balanced_accuracy": json.dumps([round(float(v), 8) for v in ba]),
            "fold_f1": json.dumps([round(float(v), 8) for v in f1]),
            "best_params_by_fold": json.dumps(search.best_params_, sort_keys=True, default=str),
            "grid_search_best_cv_balanced_accuracy": float(search.best_score_),
            "seed": SEED,
            "cv": "StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",
        })
        print(f"Completed: {model_name}", flush=True)

    result = pd.DataFrame(rows)
    out = root / "results" / "tables" / "classical_baselines.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)
    spectrum = pca_spectrum(X, root / "results" / "figures" / "pca_spectrum.png")
    spectrum.to_csv(root / "results" / "tables" / "pca_spectrum.csv", index=False)
    return result


if __name__ == "__main__":
    df = run_classical_baselines()
    print("\nClassical baselines (5-fold CV; mean ± sample SD):")
    display = df[["model", "balanced_accuracy_mean", "balanced_accuracy_std", "f1_mean", "f1_std"]].copy()
    for metric in ["balanced_accuracy_mean", "balanced_accuracy_std", "f1_mean", "f1_std"]:
        display[metric] = display[metric].map(lambda x: f"{x:.4f}")
    print(display.to_string(index=False))
