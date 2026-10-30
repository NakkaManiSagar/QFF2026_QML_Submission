# QFF 2026 QML — Experiment Methodology

## 1. Objective

The objective is to evaluate quantum-kernel classification for PS-Q2 and compare it with classical kernel baselines.

The project also completes the common Bell-state and Processor D benchmark tasks.

## 2. Dataset and Evaluation

Dataset: QML-HIGHD-KER

- 800 labelled training samples
- 200 unlabelled test samples
- 50 input features
- Binary labels A and B
- Evaluation metric: balanced accuracy

The sample_id field is excluded from model features.

Test labels are unavailable and are not used for training or model selection.

## 3. Data Preprocessing

Preprocessing is performed using training data only.

Dimensionality reduction and scaling are fitted within cross-validation training folds to reduce information leakage.

Random seed: 42.

## 4. Classical Baselines

The recorded baseline models are:

| Model | Balanced accuracy | F1 |
|---|---:|---:|
| RBF-SVM, 50 features | 0.7185 | 0.7267 |
| PCA-RBF-SVM, 8 components | 0.7242 | 0.7254 |

These results were recorded under earlier evaluation runs. A controlled comparison requires rerunning the baselines using the same folds and preprocessing protocol as the selected quantum model.

## 5. Ideal Quantum-Kernel Search

The full search evaluated 168 configurations.

Search dimensions included:

- Entanglement patterns: none, linear, ring and full
- Repetitions: 1 and 2
- Qubit counts: 4, 5 and 6
- Bandwidth values: 0.1, 0.2, 0.3, 0.5, 0.8, 1.2 and 3.14159
- SVM regularization: C = 0.1, 1 and 10

Selected configuration:

| Parameter | Value |
|---|---|
| Entanglement | none |
| Repetitions | 2 |
| Qubits | 6 |
| Bandwidth | 0.8 |
| SVM C | 1 |

Recorded five-fold cross-validation:

- Balanced accuracy: 0.7362
- Balanced accuracy standard deviation: 0.0233
- F1 score: 0.7327
- F1 standard deviation: 0.0274

These are model-selection cross-validation results, not hidden-test results.

## 6. Task A — Bell-State Fidelity

Target state:

|Phi+> = (|00> + |11>) / sqrt(2)

The official benchmark computes fidelity from Pauli correlations:

F_Bell = (1 + <XX> - <YY> + <ZZ>) / 4

The reported standard error is:

SE(F) = sqrt(SE_XX^2 + SE_YY^2 + SE_ZZ^2) / 4

Each Pauli expectation is estimated from measurement parity:

<P> = (N_even - N_odd) / N

with standard error:

SE(P) = sqrt((1 - <P>^2) / N)

The benchmark uses 8192 shots.

The quantity P(00) + P(11) is a population overlap and must not be described as the complete Bell-state fidelity.

Recorded values:

| Condition | Fidelity |
|---|---:|
| Ideal | 1.000000 |
| Noisy | 0.880127 |

## 7. Task B — Processor D QFT-3 Fidelity

For each of the eight computational-basis inputs |j>, the ideal output is QFT|j>.

The benchmark calculates:

F_j = <psi_j| rho_j |psi_j>

where psi_j is the ideal output state and rho_j is the simulated density matrix of the three logical qubits.

The reported result is:

F_QFT3 = (1/8) * sum(F_j), for j = 0,...,7.

The input is loaded noise-free. The transpiled circuit is simulated with the benchmark noise model using density-matrix simulation.

The fidelity calculation includes gate errors but excludes measurement and readout errors.

Verified Processor D results:

| Measurement | Value |
|---|---:|
| Depth before transpilation | 6 |
| Depth after transpilation | 18 |
| Two-qubit gate count | 9 |
| SWAP overhead | 0 |
| Ideal fidelity | 1.000000 |
| Noisy fidelity | 0.668377 |

Noise model: QFF2026-v2.

## 8. Noisy Quantum-Kernel Experiment

The noisy-kernel experiment uses the official QFF2026-v2 noise model.

Recorded configuration:

- Shots: 1024
- Qubits: 6
- Entanglement: none
- Repetitions: 2
- Bandwidth: 0.8
- Training subset: 100 samples
- Evaluation subset: 50 samples

Recorded subset metrics:

- Balanced accuracy: 0.747596
- F1 score: 0.770642

These metrics come from a limited subset experiment and are not directly comparable with the ideal five-fold cross-validation results.

The experiment also records ideal/noisy kernel differences, kernel alignment, eigenvalue diagnostics and other matrix statistics.

## 9. Prediction Validation

The final prediction file is predictions_q2.csv.

Validate it using:

python tools/validate_predictions.py q2 predictions_q2.csv

The recorded successful output is:

OK - predictions file is valid.

The format validation does not establish hidden-test predictive accuracy.

## 10. Reproducibility

Use Python 3.11, 3.12 or 3.13 and install the pinned dependencies.

Run commands from the repository root.

The official noise model and benchmark implementation must remain unchanged.

Save experiment outputs, logs, configurations and tables under results/.

## 11. Limitations

- Hidden test labels are unavailable.
- The classical baseline results were recorded under earlier evaluation runs.
- The noisy-kernel metrics use a subset protocol.
- The ideal and noisy evaluation protocols differ.
- The selected feature map uses no entanglement.
- Simulator results are not physical quantum hardware results.
- No quantum advantage or guaranteed competition ranking is established.
