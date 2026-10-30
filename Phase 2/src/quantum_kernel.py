"""Ideal fidelity-kernel ablation for QFF2026 PS-Q2.

The feature map adapts QuantumMedx Phase 1's H-P-ZZ map. Ideal kernels use
statevectors (no circuit executions). If Qiskit is installed, logical circuits
are also passed through the official prepare_for_noisy_run helper to report
actual prepared depth / two-qubit counts. In environments without Qiskit those
transpiled metrics are deliberately left blank, never estimated as measured.
"""
from __future__ import annotations

import itertools
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

from src.data import SEED, load_q2_data

ENTANGLEMENTS = ("none", "linear", "ring", "full")
REPS_VALUES = (1, 2)
N_QUBITS_VALUES = (4, 5, 6)
# Phase 1 bandwidth grid, retained for comparability.
BANDWIDTH_VALUES = (0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 3.14159)
C_VALUES = (0.1, 1.0, 10.0)


def entangling_pairs(n: int, entanglement: str) -> list[tuple[int, int]]:
    """Return logical interaction pairs for the requested topology."""
    if entanglement == "none":
        return []
    if entanglement == "linear":
        return [(i, i + 1) for i in range(n - 1)]
    if entanglement == "ring":
        return [(i, i + 1) for i in range(n - 1)] + ([(n - 1, 0)] if n > 2 else [])
    if entanglement == "full":
        return list(itertools.combinations(range(n), 2))
    raise ValueError(f"Unknown entanglement: {entanglement}")


def _apply_single_qubit(state: np.ndarray, n: int, q: int, gate: np.ndarray) -> np.ndarray:
    """Apply a 2x2 gate using Qiskit's little-endian qubit convention."""
    stride = 1 << q
    shaped = state.reshape(-1, 2 * stride)
    low = shaped[:, :stride].copy()
    high = shaped[:, stride:].copy()
    shaped[:, :stride] = gate[0, 0] * low + gate[0, 1] * high
    shaped[:, stride:] = gate[1, 0] * low + gate[1, 1] * high
    return shaped.reshape(1 << n)


def _apply_cx(state: np.ndarray, n: int, control: int, target: int) -> np.ndarray:
    """Apply CX exactly to a statevector; qubit indices are little endian."""
    indices = np.arange(1 << n)
    mask = (1 << control) | (1 << target)
    swap_idx = indices ^ (1 << target)
    condition = ((indices >> control) & 1).astype(bool) & (((indices >> target) & 1) == 0)
    out = state.copy()
    out[indices[condition]] = state[swap_idx[condition]]
    out[swap_idx[condition]] = state[indices[condition]]
    return out


def _statevectors_batch(X, reps: int = 1, entanglement: str = "full") -> np.ndarray:
    """Vectorized pure-NumPy statevectors for a batch of feature vectors."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2D feature matrix")
    n_samples, n = X.shape
    state = np.zeros((n_samples, 1 << n), dtype=complex)
    state[:, 0] = 1.0
    h = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2.0)
    pairs = entangling_pairs(n, entanglement)

    def single(q: int, gate: np.ndarray) -> None:
        stride = 1 << q
        view = state.reshape(n_samples, -1, 2 * stride)
        low = view[:, :, :stride].copy()
        high = view[:, :, stride:].copy()
        view[:, :, :stride] = gate[0, 0] * low + gate[0, 1] * high
        view[:, :, stride:] = gate[1, 0] * low + gate[1, 1] * high

    def phase(q: int, angles: np.ndarray) -> None:
        mask = np.array([(idx >> q) & 1 for idx in range(1 << n)], dtype=bool)
        state[:, mask] *= np.exp(1j * angles[:, None])

    def cx(control: int, target: int) -> None:
        idx = np.arange(1 << n)
        select = (((idx >> control) & 1) == 1) & (((idx >> target) & 1) == 0)
        low_idx = idx[select]
        high_idx = low_idx ^ (1 << target)
        temp = state[:, low_idx].copy()
        state[:, low_idx] = state[:, high_idx]
        state[:, high_idx] = temp

    for _ in range(reps):
        for q in range(n):
            single(q, h)
            phase(q, 2.0 * X[:, q])
        for i, j in pairs:
            cx(i, j)
            phase(j, 2.0 * (np.pi - X[:, i]) * (np.pi - X[:, j]))
            cx(i, j)
    return state


def statevector(x, reps: int = 1, entanglement: str = "full") -> np.ndarray:
    """Pure-NumPy statevector for Phase 1's H-P-ZZ feature map."""
    return _statevectors_batch(np.asarray(x, dtype=float)[None, :], reps, entanglement)[0]


def ideal_states(X, reps: int, entanglement: str) -> np.ndarray:
    """Create one exact statevector per row; no shot noise or circuits executed."""
    return _statevectors_batch(np.asarray(X, dtype=float), reps, entanglement)

def ideal_kernel(A, B=None, reps: int = 1, entanglement: str = "full") -> np.ndarray:
    """Fidelity kernel |<phi(x)|phi(y)>|^2 computed from statevectors."""
    A = np.asarray(A, dtype=float)
    B = A if B is None else np.asarray(B, dtype=float)
    SA = ideal_states(A, reps, entanglement)
    SB = SA if B is A else ideal_states(B, reps, entanglement)
    return np.abs(SA.conj() @ SB.T) ** 2


def make_logical_circuit(n: int, reps: int, entanglement: str):
    """Build the parameterized Phase 1 feature-map circuit when Qiskit is available."""
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector
    params = ParameterVector("x", n)
    qc = QuantumCircuit(n)
    for _ in range(reps):
        for i in range(n):
            qc.h(i)
            qc.p(2 * params[i], i)
        for i, j in entangling_pairs(n, entanglement):
            qc.cx(i, j)
            qc.p(2 * (np.pi - params[i]) * (np.pi - params[j]), j)
            qc.cx(i, j)
    return qc


def _logical_depth(n: int, reps: int, entanglement: str) -> int:
    """Compute the logical circuit's longest qubit-dependency path."""
    last = [0] * n
    for _ in range(reps):
        for q in range(n):
            last[q] += 2  # H then P
        for i, j in entangling_pairs(n, entanglement):
            d = max(last[i], last[j]) + 1
            last[i] = last[j] = d  # CX
            last[j] += 1            # P on target
            d = max(last[i], last[j]) + 1
            last[i] = last[j] = d  # CX
    return max(last, default=0)


def _circuit_metrics(n: int, reps: int, entanglement: str) -> dict:
    """Logical metrics plus real metrics after official noisy-run preparation."""
    pairs = entangling_pairs(n, entanglement)
    logical_2q = 2 * len(pairs) * reps
    result = {
        "logical_depth": _logical_depth(n, reps, entanglement),
        "logical_two_qubit_gates": logical_2q,
        "prepared_depth": np.nan,
        "prepared_two_qubit_gates": np.nan,
        "prepared_metric_status": "unavailable: Qiskit not installed",
    }
    try:
        from noise_model.benchmark_noise_model import prepare_for_noisy_run
        qc = make_logical_circuit(n, reps, entanglement)
        prepared = prepare_for_noisy_run(qc)
        ops = prepared.count_ops()
        result.update({
            "prepared_depth": int(prepared.depth()),
            "prepared_two_qubit_gates": int(sum(count for gate, count in ops.items()
                                                   if gate in {"cx", "cz", "swap"})),
            "prepared_metric_status": "measured by prepare_for_noisy_run",
        })
    except ImportError as exc:
        result["prepared_metric_status"] = f"unavailable: {exc.__class__.__name__} ({exc})"
    return result


def kernel_diagnostics(K: np.ndarray, y: np.ndarray) -> dict:
    """Off-diagonal concentration, centered kernel-target alignment and spectrum summary."""
    off = K[~np.eye(len(K), dtype=bool)]
    yy = np.where(np.asarray(y) == 1, 1.0, -1.0)
    target = np.outer(yy, yy)
    # Equivalent to H K H, computed in O(n^2) without dense matrix products.
    row_mean = K.mean(axis=1, keepdims=True)
    col_mean = K.mean(axis=0, keepdims=True)
    Kc = K - row_mean - col_mean + K.mean()
    alignment = float(np.sum(Kc * target) /
                       (np.linalg.norm(Kc) * np.linalg.norm(target) + 1e-12))
    eig = np.linalg.eigvalsh((K + K.T) / 2.0)
    return {
        "offdiag_mean": float(off.mean()),
        "offdiag_std": float(off.std()),
        "kernel_target_alignment": alignment,
        "eigenvalue_min": float(eig.min()),
        "eigenvalue_max": float(eig.max()),
        "eigenvalue_negative_count": int((eig < -1e-8).sum()),
        "eigenvalue_spectrum": eig.tolist(),
    }


def _rebuild_complete_spectra(X_raw: np.ndarray, df: pd.DataFrame, root: Path) -> list[dict]:
    """Recompute and save complete 800-eigenvalue spectra for every saved configuration.

    This also repairs a partial spectrum checkpoint if a long run was interrupted
    while writing the eigenvalue table.
    """
    pca_cache = {n: PCA(n_components=int(n), random_state=SEED).fit(X_raw)
                 for n in sorted(df["n_qubits"].astype(int).unique())}
    scaled_cache: dict[tuple[int, float], np.ndarray] = {}
    eigen_rows: list[dict] = []
    spectrum_rows: list[dict] = []
    for row in df.to_dict("records"):
        ent, reps, n, s = str(row["entanglement"]), int(row["reps"]), int(row["n_qubits"]), float(row["bandwidth_s"])
        key = (n, s)
        if key not in scaled_cache:
            Z = pca_cache[n].transform(X_raw)
            scaled_cache[key] = MinMaxScaler(feature_range=(0.0, s)).fit_transform(Z)
        states = ideal_states(scaled_cache[key], reps=reps, entanglement=ent)
        K = np.abs(states.conj() @ states.T) ** 2
        K = (K + K.T) / 2.0
        eig = np.linalg.eigvalsh(K)
        full = {**row, "eigenvalue_spectrum": eig.tolist()}
        eigen_rows.append(full)
        for rank, value in enumerate(eig, start=1):
            spectrum_rows.append({"entanglement": ent, "reps": reps, "n_qubits": n,
                                  "bandwidth_s": s, "eigenvalue_rank_ascending": rank,
                                  "eigenvalue": float(value)})
    (root / "results" / "tables" / "ideal_kernel_eigenvalues.csv").write_text("")
    pd.DataFrame(spectrum_rows).to_csv(root / "results" / "tables" / "ideal_kernel_eigenvalues.csv", index=False)
    return eigen_rows


def _plot_heatmaps(df: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    for n in N_QUBITS_VALUES:
        subset = df[df["n_qubits"] == n]
        fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
        for ax, ent in zip(axes.flat, ENTANGLEMENTS):
            part = subset[subset["entanglement"] == ent]
            pivot = part.pivot(index="reps", columns="bandwidth_s", values="cv_balanced_accuracy")
            im = ax.imshow(pivot.values, aspect="auto", origin="lower", vmin=0.0, vmax=1.0,
                           cmap="viridis")
            ax.set_title(f"{ent} entanglement")
            ax.set_xticks(range(len(pivot.columns)), [f"{v:g}" for v in pivot.columns])
            ax.set_yticks(range(len(pivot.index)), [str(v) for v in pivot.index])
            ax.set_xlabel("MinMax bandwidth s")
            ax.set_ylabel("reps")
            for r in range(pivot.shape[0]):
                for c in range(pivot.shape[1]):
                    val = pivot.values[r, c]
                    if np.isfinite(val):
                        ax.text(c, r, f"{val:.3f}", ha="center", va="center", color="white", fontsize=8)
        fig.colorbar(im, ax=axes.ravel().tolist(), label="5-fold CV balanced accuracy", shrink=0.8)
        fig.suptitle(f"Ideal quantum-kernel ablation (n_qubits={n})")
        fig.savefig(output_dir / f"ideal_kernel_heatmap_k{n}.png", dpi=160)
        plt.close(fig)


def _plot_eigenvalue_spectrum(eigen_rows: list[dict], output_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    # Draw representative spectrum for the top-CV configuration, as well as a broad baseline.
    if not eigen_rows:
        plt.close(fig)
        return
    ordered = sorted(eigen_rows, key=lambda r: r["cv_balanced_accuracy"], reverse=True)
    chosen = ordered[:5]
    for row in chosen:
        eig = np.asarray(row["eigenvalue_spectrum"], dtype=float)
        ax.plot(np.arange(1, len(eig) + 1), np.sort(eig)[::-1], linewidth=1.2,
                label=f'{row["entanglement"]}, reps={row["reps"]}, n={row["n_qubits"]}, s={row["bandwidth_s"]:g}')
    ax.set_xlabel("Eigenvalue rank (descending)")
    ax.set_ylabel("Kernel Gram-matrix eigenvalue")
    ax.set_title("Ideal kernel eigenvalue spectra: top five CV configurations")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def run_ideal_kernel_ablation(root: str | Path | None = None, force_recompute: bool = False) -> pd.DataFrame:
    """Run the ideal-kernel grid with fold-local PCA/scaling for CV metrics.

    Full-data kernels are still computed for unsupervised kernel diagnostics only.
    """
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    X_raw, y, _, _, _, _ = load_q2_data(root)
    X_raw = np.asarray(X_raw, dtype=float)
    y = np.asarray(y, dtype=int)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    # PCA is fitted on the Q2 training set only. No test rows enter this computation.
    rows: list[dict] = []
    eigen_rows: list[dict] = []
    cache: dict[tuple[int, float], np.ndarray] = {}
    total = len(ENTANGLEMENTS) * len(REPS_VALUES) * len(N_QUBITS_VALUES) * len(BANDWIDTH_VALUES)
    completed = 0
    circuit_metric_cache: dict[tuple[int, int, str], dict] = {}
    results_dir = root / "results" / "tables"
    figures_dir = root / "results" / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    result_path = results_dir / "ideal_kernel_ablation.csv"
    spectrum_path = results_dir / "ideal_kernel_eigenvalues.csv"
    if result_path.exists() and spectrum_path.exists() and not force_recompute:
        old_df = pd.read_csv(result_path)
        rows = old_df.to_dict("records")
        old_spectrum = pd.read_csv(spectrum_path)
        for (ent0, reps0, n0, s0), part in old_spectrum.groupby(
                ["entanglement", "reps", "n_qubits", "bandwidth_s"]):
            keyrow = next((r for r in rows if r["entanglement"] == ent0 and int(r["reps"]) == int(reps0)
                           and int(r["n_qubits"]) == int(n0) and abs(float(r["bandwidth_s"]) - float(s0)) < 1e-8), None)
            if keyrow is not None:
                eigen_rows.append({**keyrow, "eigenvalue_spectrum":
                    part.sort_values("eigenvalue_rank_ascending")["eigenvalue"].tolist()})

    for n in N_QUBITS_VALUES:
        pca = PCA(n_components=n, random_state=SEED).fit(X_raw)
        Z = pca.transform(X_raw)
        for s in BANDWIDTH_VALUES:
            key = (n, float(s))
            scaler = MinMaxScaler(feature_range=(0.0, float(s))).fit(Z)
            cache[key] = scaler.transform(Z)

    for ent in ENTANGLEMENTS:
        for reps in REPS_VALUES:
            for n in N_QUBITS_VALUES:
                for s in BANDWIDTH_VALUES:
                    config_exists = any(r["entanglement"] == ent and int(r["reps"]) == reps
                                         and int(r["n_qubits"]) == n
                                         and abs(float(r["bandwidth_s"]) - float(s)) < 1e-8
                                         for r in rows)
                    if config_exists:
                        completed += 1
                        continue
                    Z = cache[(n, float(s))]
                    # Build all 800 statevectors, then the exact full Gram matrix.
                    states = ideal_states(Z, reps=reps, entanglement=ent)
                    K = np.abs(states.conj() @ states.T) ** 2
                    K = (K + K.T) / 2.0
                    np.fill_diagonal(K, 1.0)
                    # Leakage-safe CV: PCA and MinMax scaling are fitted only on each
                    # fold's training partition. The held-out fold is transformed with
                    # those fitted objects before its cross-kernel is computed.
                    fold_scores_by_c = {float(c): [] for c in C_VALUES}
                    for tr_idx, va_idx in cv.split(X_raw, y):
                        pca_fold = PCA(n_components=n, random_state=SEED).fit(X_raw[tr_idx])
                        ztr0 = pca_fold.transform(X_raw[tr_idx])
                        zva0 = pca_fold.transform(X_raw[va_idx])
                        scaler_fold = MinMaxScaler(feature_range=(0.0, float(s))).fit(ztr0)
                        ztr = scaler_fold.transform(ztr0)
                        zva = scaler_fold.transform(zva0)
                        ktr = ideal_kernel(ztr, reps=reps, entanglement=ent)
                        kva = ideal_kernel(zva, ztr, reps=reps, entanglement=ent)
                        for c_value in C_VALUES:
                            clf = SVC(kernel="precomputed", C=float(c_value), class_weight="balanced")
                            clf.fit(ktr, y[tr_idx])
                            fold_scores_by_c[float(c_value)].append(
                                float(balanced_accuracy_score(y[va_idx], clf.predict(kva))))
                    cv_summary = []
                    for c_value, scores in fold_scores_by_c.items():
                        cv_summary.append({"C": c_value, "mean": float(np.mean(scores)),
                                           "std": float(np.std(scores, ddof=1)), "scores": scores})
                    chosen = sorted(cv_summary, key=lambda z: (z["mean"], -z["std"]), reverse=True)[0]
                    best_c = float(chosen["C"])
                    fold_scores = [float(v) for v in chosen["scores"]]
                    diagnostics = kernel_diagnostics(K, y)
                    metric_key = (n, reps, ent)
                    if metric_key not in circuit_metric_cache:
                        circuit_metric_cache[metric_key] = _circuit_metrics(n, reps, ent)
                    metrics = circuit_metric_cache[metric_key]
                    row = {
                        "entanglement": ent,
                        "reps": reps,
                        "n_qubits": n,
                        "bandwidth_s": float(s),
                        "cv_balanced_accuracy": float(np.mean(fold_scores)),
                        "cv_balanced_accuracy_std": float(np.std(fold_scores, ddof=1)),
                        "best_C": best_c,
                        "grid_search_best_cv_balanced_accuracy": float(chosen["mean"]),
                        "offdiag_mean": diagnostics["offdiag_mean"],
                        "offdiag_std": diagnostics["offdiag_std"],
                        "kernel_target_alignment": diagnostics["kernel_target_alignment"],
                        "eigenvalue_min": diagnostics["eigenvalue_min"],
                        "eigenvalue_max": diagnostics["eigenvalue_max"],
                        "eigenvalue_negative_count": diagnostics["eigenvalue_negative_count"],
                        # Nominal compute-uncompute circuit evaluation counts for a hardware run.
                        "nominal_train_pair_evaluations": len(y) * (len(y) - 1) // 2,
                        "nominal_test_vs_train_evaluations": 200 * len(y),
                        "ideal_statevectors_evaluated": len(y),
                        **metrics,
                    }
                    rows.append(row)
                    completed += 1
                    full_eigen_row = {**row, "eigenvalue_spectrum": diagnostics["eigenvalue_spectrum"]}
                    eigen_rows.append(full_eigen_row)
                    # Checkpoint after each configuration so long grids can resume after interruption.
                    pd.DataFrame(rows).to_csv(result_path, index=False)
                    spectrum_rows_checkpoint = []
                    for erow in eigen_rows:
                        for rank, value in enumerate(erow["eigenvalue_spectrum"], start=1):
                            spectrum_rows_checkpoint.append({"entanglement": erow["entanglement"], "reps": erow["reps"],
                                "n_qubits": erow["n_qubits"], "bandwidth_s": erow["bandwidth_s"],
                                "eigenvalue_rank_ascending": rank, "eigenvalue": value})
                    pd.DataFrame(spectrum_rows_checkpoint).to_csv(spectrum_path, index=False)
                    print(f"[{completed}/{total}] {ent}, reps={reps}, n={n}, s={s:g}: "
                          f"BA={row['cv_balanced_accuracy']:.4f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(result_path, index=False)
    # Reuse complete saved spectra when available. Only recompute expensive 800x800
    # eigenspectra if the checkpoint is missing or incomplete.
    eigen_rows = []
    spectra_complete = False
    if spectrum_path.exists():
        saved_spectrum = pd.read_csv(spectrum_path)
        group_cols = ["entanglement", "reps", "n_qubits", "bandwidth_s"]
        group_sizes = saved_spectrum.groupby(group_cols).size()
        expected_keys = set(tuple(r) for r in df[group_cols].itertuples(index=False, name=None))
        actual_keys = set(tuple(r) for r in group_sizes.index.tolist())
        spectra_complete = (actual_keys == expected_keys and len(group_sizes) == len(df)
                            and bool((group_sizes == len(y)).all()))
        if spectra_complete:
            for key, part in saved_spectrum.groupby(group_cols):
                match = df[(df.entanglement == key[0]) & (df.reps == key[1]) &
                           (df.n_qubits == key[2]) & np.isclose(df.bandwidth_s, key[3])].iloc[0].to_dict()
                match["eigenvalue_spectrum"] = part.sort_values("eigenvalue_rank_ascending")["eigenvalue"].tolist()
                eigen_rows.append(match)
    if not spectra_complete:
        eigen_rows = _rebuild_complete_spectra(X_raw, df, root)
    _plot_heatmaps(df, figures_dir)
    _plot_eigenvalue_spectrum(eigen_rows, figures_dir / "ideal_kernel_eigenvalue_spectrum.png")
    return df


def print_top_configs(df: pd.DataFrame, root: str | Path | None = None) -> None:
    """Print top five quantum configs beside the best Prompt 2 classical baseline."""
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    baseline_path = root / "results" / "tables" / "classical_baselines.csv"
    if baseline_path.exists():
        baselines = pd.read_csv(baseline_path)
        best = baselines.loc[baselines["balanced_accuracy_mean"].idxmax()]
        baseline_label = str(best["model"])
        baseline_ba = float(best["balanced_accuracy_mean"])
    else:
        baseline_label, baseline_ba = "classical baseline unavailable", np.nan
    top = df.sort_values(["cv_balanced_accuracy", "cv_balanced_accuracy_std"],
                         ascending=[False, True]).head(5).copy()
    top["classical_baseline"] = baseline_label
    top["classical_balanced_accuracy"] = baseline_ba
    columns = ["entanglement", "reps", "n_qubits", "bandwidth_s", "cv_balanced_accuracy",
               "cv_balanced_accuracy_std", "best_C", "classical_baseline", "classical_balanced_accuracy",
               "logical_depth", "logical_two_qubit_gates", "prepared_depth",
               "prepared_two_qubit_gates", "nominal_train_pair_evaluations",
               "nominal_test_vs_train_evaluations"]
    print("\nTop 5 ideal quantum-kernel configurations beside the best Prompt 2 classical baseline:")
    print(top[columns].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    top[columns].to_csv(root / "results" / "tables" / "ideal_kernel_top5.csv", index=False)
    print(f"\nBest Prompt 2 baseline: {baseline_label} (balanced accuracy={baseline_ba:.4f})")


if __name__ == "__main__":
    frame = run_ideal_kernel_ablation(force_recompute="--recompute" in sys.argv)
    print_top_configs(frame)
