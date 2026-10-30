# C1 – Quantum Design Quality Evidence

**Evaluation criterion:** C1 – Quantum Design Quality  
**Maximum marks:** 30  
**Project:** QFF 2026 – PS-Q2 High-Dimensional Quantum-Kernel Classification

---

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

The final pipeline uses Principal Component Analysis (PCA) followed by MinMax scaling.

- PCA reduces the original 50-dimensional feature space to the selected number of components.
- The selected six-qubit configuration uses six PCA components.
- MinMaxScaler maps the transformed features to the range [0, 0.8].
- During five-fold cross-validation, PCA and MinMaxScaler are fitted exclusively on each training fold.
- Validation data is transformed using the corresponding fitted training-fold transformations.
- For the final model, PCA and MinMaxScaler are fitted on all labelled training samples. Test samples are transformed using these fitted objects.
- Test labels are never loaded or used.

This prevents validation and test data from influencing fitted preprocessing transformations and helps avoid data leakage.

---

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

### Quantum feature-map implementation

The project implements a custom Hadamard–Phase–ZZ (H-P-ZZ) feature map using a logical quantum-circuit construction.

The encoding process consists of:

1. **Superposition preparation:** Hadamard gates are applied to initialize the qubits in superposition.
2. **Feature-dependent phase encoding:** Phase gates encode the input feature values into quantum-state phases.
3. **Optional feature interactions:** Entangling interactions are implemented using CX–Phase–CX gate sequences. The phase angle depends on pairs of encoded feature values.
4. **Repetitions:** The feature-map structure is repeated according to the selected repetition parameter.

### Design selection rationale

The experiments compared none, linear, ring and full entanglement strategies.

The selected configuration uses six qubits, two repetitions and no entanglement. This configuration was selected through the ideal quantum-kernel cross-validation search rather than copied from a standard tutorial.

The no-entanglement configuration avoids additional two-qubit interaction gates in the selected logical feature map. Its selection is supported by the measured cross-validation results:

- Balanced accuracy: 0.7362
- Balanced accuracy fold standard deviation: 0.0233
- F1 score: 0.7327
- F1 fold standard deviation: 0.0274

The quantum kernel is evaluated using state fidelity:

\[
K(x_i,x_j)=|\langle\phi(x_i)|\phi(x_j)\rangle|^2
\]

Here, \(\phi(x)\) represents the quantum state produced by encoding input \(x\). The kernel measures similarity between the quantum states corresponding to two input samples.

The implementation and circuit construction are available in `src/quantum_kernel.py`.

---

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

The runtime includes shortlist cross-validation, final preprocessing, kernel construction, SVM fitting and prediction generation.

This is a software execution measurement on the development machine. It is not a physical quantum hardware execution time.

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

Prepared circuit metrics were measured using the official `prepare_for_noisy_run` function.

These are simulator-prepared circuit metrics, not measurements from physical quantum hardware.

### Resource evaluation and execution cost

The implementation records nominal compute-uncompute circuit evaluation estimates for a potential hardware execution.

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

**Important:** These circuit evaluation counts are theoretical hardware execution estimates. The ideal kernel experiments use NumPy statevector simulation. They are not actual quantum hardware executions.

### Training-time clarification

Isolated model training time has not been measured separately.

The reported 4.61-second measurement represents the complete software pipeline runtime, including:

- Shortlist configuration evaluation.
- Cross-validation.
- Final preprocessing.
- Kernel construction.
- SVM fitting.
- Prediction generation.

Therefore, the 4.61-second measurement should be interpreted as end-to-end pipeline execution time rather than isolated model training time.

---

## 4. Evidence of Model Behaviour (7 Marks)

### Ideal quantum-kernel evaluation

The selected ideal quantum-kernel configuration achieved the following five-fold cross-validation results:

| Metric | Result |
|---|---:|
| Balanced accuracy | 0.7362 |
| Balanced accuracy fold standard deviation | 0.0233 |
| F1 score | 0.7327 |
| F1 fold standard deviation | 0.0274 |

These are model-selection cross-validation results, not hidden-test performance.

### Noisy quantum-kernel experiment

The noisy quantum-kernel experiment was conducted using a smaller subset.

| Metric | Result |
|---|---:|
| Training subset | 100 samples |
| Evaluation subset | 50 samples |
| Shots | 1024 |
| Balanced accuracy | 0.747596 |
| F1 score | 0.770642 |
| Mean absolute ideal/noisy kernel error | 0.134636 |
| Ideal off-diagonal mean | 0.756275 |
| Noisy off-diagonal mean | 0.620279 |
| Ideal kernel alignment | 0.116607 |
| Noisy kernel alignment | 0.129392 |
| Negative eigenvalues before PSD clipping | 7 |
| Linear-fit R-squared | 0.946 |

The noisy subset experiment uses a different evaluation setup from the ideal five-fold cross-validation experiment. Their accuracy values should not be treated as a direct comparison.

### Ablation and diagnostic evidence

The project includes:

- Entanglement configuration comparisons.
- Ideal kernel ablation experiments.
- Ideal versus noisy kernel matrix comparisons.
- Kernel eigenvalue analysis.
- Kernel alignment and concentration diagnostics.

These experiments provide evidence about the effect of circuit configuration, kernel behaviour and noise on the classification workflow.

---

## 5. Evidence File References

| File | Purpose |
|---|---|
| `src/data.py` | Data preparation and preprocessing |
| `src/quantum_kernel.py` | Quantum-kernel and feature-map implementation |
| `src/noisy_kernel.py` | Noisy-kernel evaluation |
| `src/classical.py` | Classical baseline implementation |
| `src/final_model.py` | Final model selection and prediction pipeline |
| `results/tables/ideal_kernel_ablation.csv` | Configuration ablation results |
| `results/tables/noisy_kernel_diagnostics.json` | Noisy-kernel diagnostics |
| `results/tables/ideal_kernel_top5.csv` | Top model configurations |
| `results/figures/` | Experimental visualizations |
| `results/final_model_timing.log` | End-to-end execution timing |
| `docs/EXPERIMENT_METHODOLOGY.md` | Experimental methodology |
| `docs/PROJECT_ARCHITECTURE.md` | Project architecture |

---

## 6. C1 Completion Checklist

- [x] Multiple quantum configurations evaluated.
- [x] Entanglement strategies compared.
- [x] Qubit counts and repetitions evaluated.
- [x] Quantum-kernel model selected using cross-validation.
- [x] Ideal and noisy kernel behaviour evaluated.
- [x] Kernel diagnostics and ablation evidence generated.
- [x] Final PCA and MinMax feature preprocessing documented.
- [x] Train-only dimensionality-reduction procedure verified.
- [x] Nominal circuit evaluation estimates reported.
- [x] Kernel execution cost estimates reported.
- [x] Circuit depth reported.
- [x] Two-qubit gate counts reported.
- [x] Training and prediction pipeline execution time measured.
- [ ] Isolated model training time measured.
- [ ] Physical quantum hardware execution performed.

The reported 4.61-second runtime represents software execution of the end-to-end pipeline on the development machine. Isolated model training time and physical quantum hardware execution were not separately measured.

The nominal circuit evaluation counts are theoretical estimates for a compute-uncompute hardware execution. They are not actual hardware executions.

---

## 7. Conclusion

The project implements a quantum-kernel classification workflow with systematic configuration search, dimensionality reduction, ideal and noisy kernel experiments, classical comparisons and kernel diagnostics.

The experimental results provide evidence for the selected six-qubit configuration and its behaviour under the evaluated conditions.

The documented evidence supports the implemented quantum design, experimental evaluation and resource analysis. Isolated model training time and physical quantum hardware execution were not measured and are explicitly identified as limitations.

All reported performance metrics represent the stated experimental evaluation conditions. Hidden-test performance and physical quantum hardware performance have not been established.