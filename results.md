# Qiskit Fall Fest 2026 — QML Track Results

**Selected problem:** PS-Q2 — High-Dimensional Quantum-Kernel Classification  
**Dataset:** `QML-HIGHD-KER`  
**Random seed:** `42`  
**Official noise model:** `QFF2026-v2`  
**Official evaluation metric:** Balanced accuracy

## 1. Problem and Methodology

The objective is to investigate whether a quantum-kernel approach can capture useful classification structure in high-dimensional data and compare it with classical kernel methods.

The dataset contains 800 labelled training samples and 200 unlabelled test samples, with 50 features (`x01`–`x50`). The target labels are `A` and `B`.

The implementation compares a quantum-kernel classifier with classical RBF-SVM baselines. The quantum-kernel search explores different feature-map configurations, qubit counts, repetitions, bandwidths, and SVM regularization values. Dimensionality reduction and scaling are fitted within cross-validation training folds to reduce preprocessing leakage.

The `sample_id` column is excluded from the model features. Test labels are unavailable and are not used for training or model selection.

## 2. Ideal Quantum Kernel and Classical Baselines

### Ideal quantum-kernel search

The full search evaluated 168 configurations across the following choices:

- Entanglement patterns: `none`, `linear`, `ring`, and `full`
- Repetitions: 1 and 2
- Qubit counts: 4, 5, and 6
- Bandwidths: 0.1, 0.2, 0.3, 0.5, 0.8, 1.2, and 3.14159
- SVM regularization values: `C = 0.1`, `1`, and `10`

The selected configuration was:

| Parameter | Selected value |
|---|---|
| Entanglement pattern | `none` |
| Repetitions | 2 |
| Qubits | 6 |
| Bandwidth | 0.8 |
| SVM `C` | 1 |

The reported five-fold cross-validation results were:

| Metric | Result |
|---|---:|
| Balanced accuracy | 0.7362 |
| Balanced accuracy fold standard deviation | 0.0233 |
| F1 score | 0.7327 |
| F1 fold standard deviation | 0.0274 |

These are model-selection cross-validation results, not an independent estimate of hidden-test performance.

### Classical baselines

Previously recorded classical results are:

| Model | Balanced accuracy | F1 score |
|---|---:|---:|
| RBF-SVM using all 50 features | 0.7185 | 0.7267 |
| PCA-RBF-SVM using 8 components | 0.7242 | 0.7254 |

These classical results were recorded under earlier evaluation runs. They should not be interpreted as a definitive, perfectly controlled comparison with the updated quantum-kernel result unless the models are rerun with the same folds and preprocessing protocol.

The results do not establish that the quantum model is superior to classical methods. They show that the selected quantum-kernel configuration achieved a balanced accuracy of approximately 0.74 under its recorded cross-validation protocol.

## 3. Task A — Bell-State Experiment

The Bell-state experiment was run with 8,192 shots using the organizer's `QFF2026-v2` noise model.

| Condition | XX correlation | YY correlation | ZZ correlation | Fidelity | Fidelity SE |
|---|---:|---:|---:|---:|---:|
| Ideal | 1.000000 | -1.000000 | 1.000000 | 1.000000 | 0.000000 |
| Noisy | 0.819092 | -0.827881 | 0.873535 | 0.880127 | 0.002592 |

The recorded noisy expectation standard errors were:

- XX: `0.006338`
- YY: `0.006197`
- ZZ: `0.005378`

Under ideal simulation, the measured correlations and fidelity match the expected Bell-state results. Under the organizer's noise model, the correlations move away from their ideal values and the fidelity decreases to approximately 0.88. This illustrates that noise affects the quality of the prepared and measured entangled state.

## 4. Task B — Processor D Benchmark

The Processor D sweep compared eight candidates using line, ring, star, and all-to-all topologies with the gate sets `rx,ry,rz,cx` and `rz,sx,cx`.

The recorded best QFT-3 candidate was a six-qubit all-to-all topology using `rx,ry,rz,cx`.

| Benchmark measurement | Recorded result |
|---|---:|
| Circuit depth before transpilation | 6 |
| Circuit depth after transpilation | 18 |
| Two-qubit gate count | 9 |
| SWAP overhead | 0 |
| Ideal fidelity | 1.000000 |
| Noisy fidelity under `QFF2026-v2` | 0.668377 |

The `processor_d_config.json` configuration was checked with the official benchmark command:

```powershell
python benchmark/qff_qml_benchmark.py processor_d_config.json
```

The reported run completed without a mismatch warning. These Processor D measurements describe the QFT-3 benchmark circuit; they are not the depth and gate count of the selected classification kernel.

## 5. Noisy Quantum-Kernel Experiment

The noisy-kernel experiment used the official `QFF2026-v2` noise model through the organizer-provided noisy execution helpers.

The experiment used:

| Setting | Value |
|---|---:|
| Shots | 1,024 |
| Qubits | 6 |
| Entanglement pattern | None |
| Repetitions | 2 |
| Bandwidth | 0.8 |
| Training subset | 100 samples |
| Evaluation subset | 50 samples |

### Recorded subset classification metrics

| Metric | Noisy subset result |
|---|---:|
| Balanced accuracy | 0.747596 |
| F1 score | 0.770642 |

These scores are from a limited subset experiment. They are not directly comparable to the ideal-kernel five-fold cross-validation scores because the sample sizes and evaluation protocols differ.

### Kernel diagnostics

| Diagnostic | Recorded result |
|---|---:|
| Mean absolute ideal/noisy kernel error | 0.134636 |
| Ideal off-diagonal kernel mean | 0.756275 |
| Ideal off-diagonal kernel standard deviation | 0.116767 |
| Noisy off-diagonal kernel mean | 0.620279 |
| Noisy off-diagonal kernel standard deviation | 0.093603 |
| Ideal kernel alignment | 0.116607 |
| Noisy kernel alignment | 0.129392 |
| Negative eigenvalues before PSD clipping | 7 |
| Linear-fit R² | 0.946000 |
| Linear hypothesis check | True |

The recorded mean circuit depth for this non-entangling kernel configuration was 4, with zero two-qubit gates. This is consistent with the selected `none` entanglement setting and should not be interpreted as a measurement of an entangling circuit.

The diagnostic values show that the noise model changes the kernel matrix. The noisy off-diagonal mean is lower than the ideal mean, and the mean absolute kernel error is approximately 0.135. The alignment value changes slightly, but this alone does not demonstrate improved generalization or better hidden-test performance.

The noisy subset metrics should be interpreted as a limited experiment rather than evidence that noise improves the classifier.

## 6. Prediction File and Reproducibility Checks

The final prediction file is `predictions_q2.csv`.

The following validation command completed successfully:

```powershell
python tools/validate_predictions.py q2 predictions_q2.csv
```

Output:

```text
OK - predictions file is valid.
```

The file contains 200 predictions and passes the official format validator.

Other checks reported as successful in the working environment:

- The environment checker matched all 15 required dependency versions.
- The Processor D configuration passed the official benchmark without a mismatch warning.
- Task A completed under ideal and noisy simulation.
- The Processor D candidate sweep completed.
- The noisy-kernel subset experiment completed.
- The final model generated the prediction file.
- `solution.ipynb` executed successfully and produced `solution.executed.ipynb`.
- The final ZIP archive was rebuilt and checked for the key deliverables.

These checks establish that the reported commands ran successfully and the prediction file passed format validation. They do not establish hidden-test accuracy.

## 7. Limitations and Conclusions

The selected quantum-kernel configuration achieved a five-fold cross-validation balanced accuracy of `0.7362`. The recorded classical baselines achieved balanced accuracies of `0.7185` and `0.7242` under previously recorded evaluation runs. A definitive comparison requires consistent evaluation protocols.

The noisy-kernel experiment indicates that the organizer's noise model changes kernel values. Its subset metrics cannot be compared directly with the ideal model's cross-validation results.

The selected feature map uses no entanglement, so the experiment does not establish that entanglement improves classification performance. Further controlled experiments would be required to assess the contribution of circuit architecture, feature-map expressiveness, and noise.

The hidden test labels are unavailable. Therefore, the hidden-test balanced accuracy and final competition ranking remain unknown. No claim of guaranteed performance or quantum advantage is made.

## 8. Final Deliverables

The project package contains the principal deliverables:

- `solution.ipynb`
- `solution.executed.ipynb`
- `processor_d_config.json`
- `predictions_q2.csv`
- `results.md`
- `README.md`
- `requirements.txt`
- `pip_freeze.txt`
- `AI_DISCLOSURE.md`
- Supporting source code, experiment outputs, benchmark assets, and requirement audit

The project should be submitted through the official hackathon process using the public GitHub repository URL, following the organizer's instructions.
