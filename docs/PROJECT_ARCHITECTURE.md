# QFF 2026 QML — Project Architecture

## 1. Overview

This project implements PS-Q2: High-Dimensional Quantum-Kernel Classification for the Qiskit Fall Fest 2026 Quantum Machine Learning Track.

The system compares quantum-kernel classification with classical kernel baselines using the QML-HIGHD-KER dataset. It also implements the common Bell-state experiment and Processor D QFT-3 benchmark.

## 2. High-Level Architecture

The project follows this workflow:

1. Load and validate the training and test datasets.
2. Separate identifiers, features and training labels.
3. Apply preprocessing using training data only.
4. Evaluate classical classification baselines.
5. Construct and evaluate ideal quantum kernels.
6. Run selected experiments using the official noise model.
7. Execute Task A and Task B benchmarks.
8. Select the final model and generate Q2 predictions.
9. Validate the prediction file and preserve experiment evidence.

## 3. Source Code Organization

| Module | Responsibility |
|---|---|
| src/data.py | Dataset loading, feature handling and preprocessing |
| src/classical.py | Classical baseline experiments |
| src/quantum_kernel.py | Ideal quantum-kernel experiments |
| src/noisy_kernel.py | Noisy quantum-kernel experiments |
| src/final_model.py | Final model execution and prediction generation |
| src/processor_d.py | Processor D design and candidate evaluation |
| src/task_a.py | Bell-state experiment |
| benchmark/qff_qml_benchmark.py | Official benchmark implementation |
| noise_model/benchmark_noise_model.py | Organizer-provided noise model |
| tools/validate_predictions.py | Prediction-file validation |
| tools/check_environment.py | Dependency environment verification |

## 4. Dataset

The selected problem uses QML-HIGHD-KER.

- Training samples: 800
- Test samples: 200
- Input features: 50
- Feature names: x01 through x50
- Classes: A and B
- Identifier: sample_id

The sample_id field is not a model feature. Test labels are unavailable and are not used for model selection.

## 5. Model Pipeline

### Classical pipeline

Classical RBF-SVM baselines provide reference performance for evaluating the quantum-kernel approach.

### Quantum pipeline

The quantum-kernel implementation evaluates feature-map configurations using different qubit counts, repetitions, entanglement patterns and bandwidth values.

The selected configuration is:

- Entanglement: none
- Repetitions: 2
- Qubits: 6
- Bandwidth: 0.8
- SVM C: 1

### Noisy execution

The noisy-kernel experiment uses the organizer-provided QFF2026-v2 noise model and the supplied noisy execution helpers.

The ideal and noisy experiments use different evaluation protocols and must be reported separately.

## 6. Common Tasks

### Task A

Prepare the Bell state:

|Phi+> = (|00> + |11>) / sqrt(2)

Evaluate its correlations and fidelity under ideal and noisy simulation.

### Task B

Evaluate a Processor D configuration using the official QFT-3 benchmark.

The current verified configuration has:

- Six qubits
- All-to-all bidirectional coupling
- Native gates: rx, ry, rz, cx
- Transpiled depth: 18
- Two-qubit gates: 9
- SWAP overhead: 0
- Ideal fidelity: 1.000000
- Noisy fidelity: 0.668377

These are simulator results, not physical quantum-device measurements.

## 7. Reproducibility

The project uses random seed 42 and the pinned dependency versions in requirements.txt.

Run experiments from the repository root. Preserve generated evidence under results/.

## 8. Main Deliverables

- solution.ipynb
- solution.executed.ipynb
- predictions_q2.csv
- processor_d_config.json
- results.md
- README.md
- requirements.txt
- pip_freeze.txt
- AI_DISCLOSURE.md

## 9. Limitations

The hidden test labels are unavailable. Cross-validation and subset experiment results do not establish hidden-test accuracy.

The noisy subset experiment is not directly comparable with the ideal five-fold cross-validation result.

The simulator benchmark does not establish performance on a physical quantum processor.

## 10. Official Benchmark Responsibilities

The official benchmark module implements:

- Task A: Bell-state preparation, Pauli correlations and fidelity estimation.
- Task B: Processor D QFT-3 circuit transpilation, density-matrix simulation and fidelity evaluation.

The organizer-provided noise model is used for noisy benchmark execution. Both benchmark and noise-model implementations are preserved unchanged.
