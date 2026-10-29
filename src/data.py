"""Data loading and leakage-safe preprocessing for QFF2026 PS-Q2."""
from __future__ import annotations

from pathlib import Path
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

SEED = 42
LABEL_TO_INT = {"A": 0, "B": 1}
INT_TO_LABEL = {value: key for key, value in LABEL_TO_INT.items()}
FEATURE_COLUMNS = [f"x{i:02d}" for i in range(1, 51)]
SIGNAL_FEATURES = [f"x{i:02d}" for i in range(1, 7)]


def load_q2_data(root: str | Path | None = None):
    """Return X_train, y_train (A->0/B->1), X_test, sample_ids, and mappings."""
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    train_path = root / "data" / "q2" / "hd_data_train.csv"
    test_path = root / "data" / "q2" / "hd_data_test_features.csv"
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    if "label" not in train.columns:
        raise ValueError("Training data must contain a 'label' column.")
    missing = sorted(set(FEATURE_COLUMNS) - set(train.columns))
    if missing:
        raise ValueError(f"Training data is missing features: {missing}")
    missing_test = sorted(set(FEATURE_COLUMNS) - set(test.columns))
    if missing_test:
        raise ValueError(f"Test data is missing features: {missing_test}")
    labels = set(train["label"].dropna().unique())
    if not labels <= set(LABEL_TO_INT):
        raise ValueError(f"Unexpected labels: {sorted(labels)}")
    X_train = train[FEATURE_COLUMNS].copy()
    y_train = train["label"].map(LABEL_TO_INT).astype(int)
    # sample_id is intentionally excluded from the feature matrix.
    X_test = test[FEATURE_COLUMNS].copy()
    sample_ids = test["sample_id"].copy() if "sample_id" in test.columns else pd.Series(range(len(test)), name="sample_id")
    return X_train, y_train, X_test, sample_ids, LABEL_TO_INT.copy(), INT_TO_LABEL.copy()


def make_pca_minmax_pipeline(k: int, s: float = 1.0) -> Pipeline:
    """PCA followed by fold-local MinMax scaling to [0, s]."""
    if k not in (4, 5, 6, 8):
        raise ValueError("k must be one of 4, 5, 6, or 8")
    if s <= 0:
        raise ValueError("s must be positive")
    return Pipeline([
        ("pca", PCA(n_components=k, random_state=SEED)),
        ("minmax", MinMaxScaler(feature_range=(0.0, float(s)))),
    ])


def pca_spectrum(X_train, output_path: str | Path, max_components: int = 50) -> pd.DataFrame:
    """Fit PCA on training data only, save the explained-variance spectrum figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model = PCA(n_components=min(max_components, X_train.shape[1]), random_state=SEED)
    model.fit(X_train)
    ratios = model.explained_variance_ratio_
    table = pd.DataFrame({
        "component": range(1, len(ratios) + 1),
        "explained_variance_ratio": ratios,
        "cumulative_explained_variance": ratios.cumsum(),
    })
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(table["component"], table["explained_variance_ratio"], marker="o", markersize=3, label="Individual")
    ax.plot(table["component"], table["cumulative_explained_variance"], marker=".", markersize=3, label="Cumulative")
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance ratio")
    ax.set_title("PCA explained-variance spectrum (training data only)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return table
