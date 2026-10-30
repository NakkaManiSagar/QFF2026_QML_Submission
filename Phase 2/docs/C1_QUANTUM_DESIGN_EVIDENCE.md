# C1 – Quantum Design Quality Evidence

**Evaluation criterion:** C1 – Quantum Design Quality  
**Maximum marks:** 30  
**Project:** QFF 2026 – PS-Q2 High-Dimensional Quantum-Kernel Classification

## 1. Data Encoding and Dimensionality Reduction (8 Marks)

### Dataset
- Dataset: QML-HIGHD-KER
- Training samples: 800 labelled samples
- Test samples: 200 unlabelled samples
- Original feature dimensions: 50 (`x01`–`x50`)
- Target classes: A and B
- `sample_id` is treated as an identifier, not a model feature.

### Implemented work
- Data preparation and quantum-kernel experiments are implemented in `src/data.py` and `src/quantum_kernel.py`.
- Experiments evaluated 4, 5 and 6 qubit configurations.
- Feature-map bandwidths were evaluated to study their effect on kernel behaviour.

### Verified encoding and preprocessing

The final pipeline uses PCA followed by MinMax scaling.

- PCA reduces the original 50-dimensional feature space to the selected number of components.
- The selected six-qubit configuration uses six PCA components.
- MinMaxScaler maps the transformed features to [0, 0.8].
- During five-fold cross-validation, PCA and MinMaxScaler are fitted exclusively on each training fold.
- Validation data is transformed using the corresponding fitted training-fold transformations.
- For the final model, PCA and MinMaxScaler are fitted on all labelled training samples. Test samples are transformed using these fitted objects.
- Test labels are never loaded or used.

This prevents validation and test data from influencing fitted preprocessing transformations.

## 2. Circuit and Feature-Map Design Justification (8 Marks)

### Implemented experiments
The project evaluated:
- Entanglement: none, linear, ring and full.
- Repetitions: 1 and 2.
- Qubit configurations: 4, 5 and 6.
- Bandwidths: 0.1, 0.2, 0.3, 0.5, 0.8, 1.2 and 3.14159.
- SVM regularization values: 0.1, 1 and 10.

A total of 168 ideal quantum-kernel configurations were evaluated.

### Selected configuration
| Parameter | Selected value |
|---|---|
| Qubits | 6 |
| Entanglement | None |
| Repetitions | 2 |
| Bandwidth | 0.8 |
| SVM C | 1 |

The selected configuration avoids unnecessary entangling gates in the quantum-kernel circuit. The selection was based on experimental model evaluation rather than adopting a tutorial configuration.

## 3. Quantum Resource Reporting (7 Marks)

### Measured execution time

The final leakage-aware model pipeline was executed successfully.

| Measurement | Result |
|---|---:|
| Shortlisted configurations evaluated | 12 |
| End-to-end execution time | 4.61 seconds |
| Final predictions generated | 200 |
| Selected model | Ideal quantum kernel |
| Selected configuration | 6 qubits, 2 repetitions, no entanglement |
| Cross-validation balanced accuracy | 0.7362 |
| Cross-validation F1 | 0.7327 |

The runtime includes shortlist cross-validation, final preprocessing,
kernel construction, SVM fitting and prediction generation.

This is a software execution measurement on the development machine.
It is not a physical quantum hardware execution time.

### Available resource evidence

| Resource | Reported result |
|---|---:|
| Qubits | 6 |
| Repetitions | 2 |
| Entanglement | None |
| Logical circuit depth | 4 |
| Prepared circuit depth | 6 |
| Logical two-qubit gates | 0 |
| Prepared two-qubit gates | 0 |
| Training subset | 100 samples |
| Evaluation subset | 50 samples |
| Shots | 1024 |
| Bandwidth | 0.8 |
| SVM C | 1 |

Prepared circuit metrics were measured using the official
`prepare_for_noisy_run` function. These are simulator-prepared circuit
metrics, not measurements from physical quantum hardware.

### Resource evaluation and execution cost

The implementation records nominal compute-uncompute circuit evaluation
estimates for a hardware execution.

| Resource | Value |
|---|---:|
| Training samples | 800 |
| Test samples | 200 |
| Nominal training pair evaluations | 319,600 |
| Nominal test-versus-training evaluations | 160,000 |
| Total nominal pair evaluations | 479,600 |
| Ideal statevectors evaluated | 800 |
| Qubits in selected configuration | 6 |
| Repetitions | 2 |
| Bandwidth | 0.8 |

**Important:** These circuit evaluation counts are theoretical hardware
execution estimates. The ideal kernel experiments use NumPy statevector
simulation; they are not actual quantum hardware executions.

Training time has not yet been explicitly recorded in the experiment
results and must be measured separately.
## 4. Evidence of Model Behaviour (7 Marks)

### Ideal quantum-kernel evaluation
- Five-fold CV balanced accuracy: 0.7362.
- Fold standard deviation: 0.0233.
- F1 score: 0.7327.
- F1 fold standard deviation: 0.0274.

These are model-selection cross-validation results, not hidden-test performance.

### Noisy quantum-kernel experiment
- Training subset: 100 samples.
- Evaluation subset: 50 samples.
- Balanced accuracy: 0.747596.
- F1 score: 0.770642.
- Mean absolute ideal/noisy kernel error: 0.134636.
- Ideal off-diagonal mean: 0.756275.
- Noisy off-diagonal mean: 0.620279.
- Ideal kernel alignment: 0.116607.
- Noisy kernel alignment: 0.129392.
- Negative eigenvalues before PSD clipping: 7.
- Linear-fit R-squared: 0.946.

The noisy subset experiment uses a different evaluation setup from the ideal five-fold cross-validation experiment. Their accuracy values should not be treated as a direct comparison.

### Ablation and diagnostic evidence
The project includes:
- Entanglement configuration comparisons.
- Ideal kernel ablation experiments.
- Ideal versus noisy kernel matrix comparisons.
- Kernel eigenvalue analysis.
- Kernel alignment and concentration diagnostics.

## 5. Evidence File References

| File | Purpose |
|---|---|
| `src/data.py` | Data preparation |
| `src/quantum_kernel.py` | Quantum-kernel implementation |
| `src/noisy_kernel.py` | Noisy-kernel evaluation |
| `src/classical.py` | Classical baseline |
| `results/tables/ideal_kernel_ablation.csv` | Configuration ablation |
| `results/tables/noisy_kernel_diagnostics.json` | Noisy-kernel diagnostics |
| `results/tables/ideal_kernel_top5.csv` | Top model configurations |
| `results/figures/` | Experimental visualizations |
| `docs/EXPERIMENT_METHODOLOGY.md` | Experimental methodology |
| `docs/PROJECT_ARCHITECTURE.md` | Project architecture |

## 6. C1 Completion Checklist

- [x] Multiple quantum configurations evaluated.
- [x] Entanglement strategies compared.
- [x] Qubit counts and repetitions evaluated.
- [x] Quantum-kernel model selected using cross-validation.
- [x] Ideal and noisy kernel behaviour evaluated.
- [x] Kernel diagnostics and ablation evidence generated.
- [x] Exact final feature encoding documented.
- [x] Train-only dimensionality-reduction procedure verified and documented.
- [x] Nominal circuit evaluation estimates reported.
- [x] Kernel execution cost estimated.
- [ ] Actual training time measured and reported.
- [x] Circuit depth reported.
- [x] Two-qubit gate counts reported.
- [x] Nominal circuit evaluation estimates reported.
- [x] Training and prediction pipeline execution time measured.
- [ ] Actual training time measured and reported.
- [ ] Exact final feature encoding documented.
- [ ] Train-only dimensionality-reduction procedure verified and documented.
- [ ] Training time reported.
- [ ] Total circuit evaluations reported.
- [ ] Complete kernel cost reported.

## 7. Conclusion

The project implements a quantum-kernel classification workflow with systematic configuration search, ideal and noisy kernel experiments, classical comparisons and kernel diagnostics.

The experimental results provide evidence for the selected six-qubit configuration and its behaviour under the evaluated conditions.

The outstanding checklist items must be verified against the implementation and execution logs before claiming complete compliance with all 30 C1 marks.
