"""Leakage-aware final Q2 model selection and prediction generation.

The existing ablation CSV is used only to shortlist configurations. Each shortlisted
configuration is re-scored with fold-local PCA and MinMax scaling before selection.
Test features are transformed only after the configuration is selected; test labels
are never loaded or used.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score, f1_score
from src.data import load_q2_data, INT_TO_LABEL, SEED
from src.quantum_kernel import ideal_kernel

C_VALUES = (0.1, 1.0, 10.0)
SHORTLIST_SIZE = 12


def _fold_local_score(X, y, config, cv):
    """Evaluate one kernel config with PCA/scaling fitted inside each CV training fold."""
    n = int(config.n_qubits)
    reps = int(config.reps)
    ent = str(config.entanglement)
    bandwidth = float(config.bandwidth_s)
    fold_data = []
    for tr, va in cv.split(X, y):
        pca = PCA(n_components=n, random_state=SEED).fit(X[tr])
        ztr0 = pca.transform(X[tr])
        zva0 = pca.transform(X[va])
        scaler = MinMaxScaler(feature_range=(0.0, bandwidth)).fit(ztr0)
        ztr = scaler.transform(ztr0)
        zva = scaler.transform(zva0)
        ktr = ideal_kernel(ztr, reps=reps, entanglement=ent)
        kva = ideal_kernel(zva, ztr, reps=reps, entanglement=ent)
        fold_data.append((ktr, kva, y[tr], y[va]))

    candidates = []
    for c in C_VALUES:
        bas, f1s = [], []
        for ktr, kva, ytr, yva in fold_data:
            clf = SVC(kernel="precomputed", C=c, class_weight="balanced")
            clf.fit(ktr, ytr)
            pred = clf.predict(kva)
            bas.append(balanced_accuracy_score(yva, pred))
            f1s.append(f1_score(yva, pred, pos_label=1, zero_division=0))
        candidates.append({"C": c, "balanced_accuracy_mean": float(np.mean(bas)),
                           "balanced_accuracy_std": float(np.std(bas, ddof=1)),
                           "f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s, ddof=1))})
    return sorted(candidates, key=lambda r: (r["balanced_accuracy_mean"], -r["balanced_accuracy_std"]), reverse=True)[0]


def run(root=None):
    root = Path(root) if root else Path(__file__).resolve().parents[1]
    tabs = root / "results" / "tables"
    tabs.mkdir(parents=True, exist_ok=True)
    X, y, Xtest, ids, _, _ = load_q2_data(root)
    X = np.asarray(X, dtype=float)
    Xtest = np.asarray(Xtest, dtype=float)
    y = np.asarray(y, dtype=int)

    ablation_path = tabs / "ideal_kernel_ablation.csv"
    if not ablation_path.exists():
        raise FileNotFoundError("Missing results/tables/ideal_kernel_ablation.csv; run `python -m src.quantum_kernel` first.")
    ab = pd.read_csv(ablation_path)
    # Shortlist candidates using training-only ablation results, then re-score each with
    # fold-local preprocessing. This limits expensive fold-specific kernel calculations.
    shortlist = ab.sort_values(["cv_balanced_accuracy", "cv_balanced_accuracy_std"],
                               ascending=[False, True]).head(SHORTLIST_SIZE)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    evaluated = []
    for _, cfg in shortlist.iterrows():
        score = _fold_local_score(X, y, cfg, cv)
        evaluated.append({"entanglement": str(cfg.entanglement), "reps": int(cfg.reps),
                          "n_qubits": int(cfg.n_qubits), "bandwidth_s": float(cfg.bandwidth_s),
                          **score, "shortlist_cv_balanced_accuracy": float(cfg.cv_balanced_accuracy)})
        print(f"Leakage-safe CV: ent={cfg.entanglement}, reps={int(cfg.reps)}, k={int(cfg.n_qubits)}, "
              f"s={float(cfg.bandwidth_s):g}, C={score['C']:g}, BA={score['balanced_accuracy_mean']:.4f}", flush=True)
    evaluated_df = pd.DataFrame(evaluated).sort_values(
        ["balanced_accuracy_mean", "balanced_accuracy_std"], ascending=[False, True])
    evaluated_df.to_csv(tabs / "leakage_safe_kernel_shortlist.csv", index=False)
    best = evaluated_df.iloc[0]
    n, reps, ent = int(best.n_qubits), int(best.reps), str(best.entanglement)
    bandwidth, C = float(best.bandwidth_s), float(best.C)

    # Final refit uses all labelled training rows; test data are used only for predictions.
    pca = PCA(n_components=n, random_state=SEED).fit(X)
    ztr0 = pca.transform(X)
    zte0 = pca.transform(Xtest)
    scaler = MinMaxScaler(feature_range=(0.0, bandwidth)).fit(ztr0)
    Z = scaler.transform(ztr0)
    Ztest = scaler.transform(zte0)
    K = ideal_kernel(Z, reps=reps, entanglement=ent)
    Ktest = ideal_kernel(Ztest, Z, reps=reps, entanglement=ent)
    model = SVC(kernel="precomputed", C=C, class_weight="balanced").fit(K, y)
    pred = model.predict(Ktest)
    pd.DataFrame({"sample_id": ids, "label": [INT_TO_LABEL[int(v)] for v in pred]}).to_csv(
        root / "predictions_q2.csv", index=False)

    rows = []
    classical_path = tabs / "classical_baselines.csv"
    if classical_path.exists():
        classical = pd.read_csv(classical_path)
        for _, r in classical.iterrows():
            rows.append({"model": r["model"], "balanced_accuracy_cv": r["balanced_accuracy_mean"],
                         "balanced_accuracy_std": r["balanced_accuracy_std"], "f1_cv": r["f1_mean"],
                         "f1_std": r["f1_std"], "status": "measured; see baseline validation protocol"})
    rows.append({"model": f"Ideal quantum kernel ({ent}, reps={reps}, k={n}, s={bandwidth:g})",
                 "balanced_accuracy_cv": float(best.balanced_accuracy_mean),
                 "balanced_accuracy_std": float(best.balanced_accuracy_std),
                 "f1_cv": float(best.f1_mean), "f1_std": float(best.f1_std),
                 "status": f"fold-local PCA/scaling; shortlisted {len(shortlist)} configs; C={C:g}"})
    diag_path = tabs / "noisy_kernel_diagnostics.json"
    if diag_path.exists():
        try:
            diag = json.loads(diag_path.read_text())
            noisy_ba, noisy_f1 = diag.get("noisy_balanced_accuracy"), diag.get("noisy_f1")
            status = ("measured noisy-kernel CV; see noisy_kernel_diagnostics.json" if noisy_ba is not None and noisy_f1 is not None
                      else "diagnostics found, but noisy classification metrics are absent; rerun noisy-kernel experiment")
        except Exception:
            noisy_ba = noisy_f1 = np.nan
            status = "noisy-kernel diagnostics unreadable; rerun noisy-kernel experiment"
    else:
        noisy_ba = noisy_f1 = np.nan
        status = "NOT RUN: no noisy-kernel diagnostics file found"
    rows.append({"model": "Noisy quantum kernel (PSD-clipped)", "balanced_accuracy_cv": noisy_ba,
                 "balanced_accuracy_std": np.nan, "f1_cv": noisy_f1, "f1_std": np.nan, "status": status})
    pd.DataFrame(rows).to_csv(tabs / "final_comparison.csv", index=False)
    selection = {"selection_rule": "highest fold-local CV balanced accuracy among top ablation shortlist; tie-break lower fold SD",
                 "shortlist_size": len(shortlist), "validation_protocol": "PCA and MinMaxScaler fit only on each CV training fold",
                 "entanglement": ent, "reps": reps, "n_qubits": n, "bandwidth_s": bandwidth, "C": C,
                 "cv_balanced_accuracy": float(best.balanced_accuracy_mean), "cv_balanced_accuracy_std": float(best.balanced_accuracy_std),
                 "cv_f1": float(best.f1_mean), "cv_f1_std": float(best.f1_std),
                 "train_rows": len(y), "test_predictions": len(pred), "seed": SEED,
                 "caveat": "CV score is used for model selection from a shortlist; it is not an independent final test estimate."}
    (root / "results" / "final_model_selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    print("Selected leakage-aware ideal kernel:", ent, "reps", reps, "k", n, "s", bandwidth, "C", C)
    print("CV BA", round(float(best.balanced_accuracy_mean), 4), "CV F1", round(float(best.f1_mean), 4), "predictions", len(pred))
    print(pd.DataFrame(rows).to_string(index=False))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    run()
