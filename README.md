# Qiskit Fall Fest 2026 — QML PS-Q2 Submission

## Project Overview

This repository implements **PS-Q2: High-Dimensional Quantum-Kernel Classification** for the Qiskit Fall Fest 2026 Quantum Machine Learning Track.

The project compares a quantum-kernel classifier with classical kernel baselines on the `QML-HIGHD-KER` dataset. It also investigates the effect of the organizer's noise model and includes the required Task A Bell-state experiment and Task B Processor D benchmark.

The official challenge-kit assets, including the benchmark, schemas, tools, templates, dataset, and noise model, are preserved unchanged. The pinned `requirements.txt` is the source of truth for the required package versions.

## Dataset

- Training data: `data/q2/hd_data_train.csv`
- Test features: `data/q2/hd_data_test_features.csv`
- Training samples: 800
- Test samples: 200
- Input features: 50 (`x01`–`x50`)
- Target labels: `A` and `B`
- Official evaluation metric: balanced accuracy

The `sample_id` column is an identifier and is not used as a model feature. Test labels are unavailable in the challenge kit and are not used for training or model selection.

## Environment Setup

Use Python 3.11, 3.12, or 3.13 in a dedicated virtual environment.

### Windows PowerShell

```powershell
python -m venv qml-env
.\qml-env\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python tools/check_environment.py
```

If PowerShell blocks environment activation, use the activation instructions appropriate for your system or invoke the environment's Python executable directly.

## Reproduce the Experiments

Run these commands from the repository root after installing the pinned dependencies.

```powershell
python noise_model/benchmark_noise_model.py
python -m src.classical
python -m src.quantum_kernel --recompute
python -m src.task_a
python -m src.processor_d
python benchmark/qff_qml_benchmark.py processor_d_config.json
python -m src.noisy_kernel --recompute
python -m src.final_model
python tools/validate_predictions.py q2 predictions_q2.csv
jupyter nbconvert --to notebook --execute solution.ipynb --output solution.executed.ipynb
python -m pip freeze > pip_freeze.txt
```

The full ideal-kernel search evaluates 168 configurations. The noisy-kernel experiment uses a documented subset to manage circuit-evaluation cost. Re-running experiments may update result files and can take time.

## Main Deliverables

- `solution.ipynb` — runnable notebook containing the selected problem and common tasks.
- `solution.executed.ipynb` — notebook with recorded execution outputs.
- `processor_d_config.json` — Processor D configuration and benchmark results.
- `predictions_q2.csv` — predictions for the 200 test samples.
- `results.md` — methodology, metrics, noise analysis, and limitations.
- `README.md` — setup and reproduction instructions.
- `pip_freeze.txt` — package-version snapshot from the working environment.
- `AI_DISCLOSURE.md` — disclosure of AI assistance.
- `results/` — experiment tables, diagnostic outputs, logs, and requirement audit.
- `src/` — implementation of data preparation, classical baselines, ideal and noisy quantum kernels, Task A, Processor D, and final prediction generation.

## Verified Checks

The following checks were reported as successful in the working environment:

- All 15 required dependency versions matched the pinned environment.
- `predictions_q2.csv` passed the official prediction-file validator and contains 200 predictions.
- The Processor D configuration passed the official QFT-3 benchmark without a mismatch warning.
- Task A ideal and noisy Bell-state measurements completed.
- The Processor D candidate sweep completed.
- The noisy-kernel subset experiment completed.
- The final model generated the prediction file.
- `solution.ipynb` executed successfully and produced `solution.executed.ipynb`.
- The final ZIP archive was rebuilt and checked for the key submission files.

These checks establish successful execution and file-format validation; they do not establish hidden-test accuracy or guarantee competition ranking.

## Reproducibility and Limitations

The random seed is `42`. Preprocessing must be fitted on training data only, and test data must not be used for model selection. Noisy quantum-circuit experiments use the organizer-provided `noise_model/benchmark_noise_model.py` helpers without modifying the official noise model.

The ideal-kernel cross-validation results and noisy-kernel subset results use different evaluation protocols and should not be compared as if they were equivalent. The hidden-test performance is unknown.

The `requirements.txt` file defines the required versions. `pip_freeze.txt` records the environment snapshot and should be regenerated after installing the exact pinned requirements if the environment changes.

## Submission

The official instructions specify submitting the **public GitHub repository URL** through the hackathon submission portal. Ensure the final required files are committed and pushed to the repository, and follow the portal's instructions for submitting the URL.


## Technical Documentation

The additional project documentation is available here:

- [Project Architecture](docs/PROJECT_ARCHITECTURE.md)
- [Experiment Methodology](docs/EXPERIMENT_METHODOLOGY.md)
- [Results and Analysis](results.md)
- [Requirement Audit](results/REQUIREMENT_AUDIT.md)
- [AI Use Disclosure](AI_DISCLOSURE.md)

## Verification Commands

Run the following commands from the repository root:

```powershell
python tools/check_environment.py
python tools/validate_predictions.py q2 predictions_q2.csv
python benchmark/qff_qml_benchmark.py processor_d_config.json
python noise_model/benchmark_noise_model.py
```

The Processor D benchmark reports QFT-3 circuit depth, gate count, SWAP overhead and ideal/noisy fidelity.

The prediction validator checks the submission file format. It does not measure hidden-test accuracy.

The noise-model self-test confirms that the organizer-provided QFF2026-v2 noise model initializes successfully.
