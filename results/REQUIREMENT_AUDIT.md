# QFF2026 QML requirement audit

This audit reflects the available project files and successful local verification commands. The authoritative participant problem-statement DOCX was not available, so this does not establish that every requirement in that document has been checked.

| Requirement | Current state | Evidence / caveat |
|---|---|---|
| Problem selection and common tasks | PASS | PS-Q2 selected; Q2 data and prediction output included. |
| Test-label separation | PASS by source audit | src/data.py uses x01-x50 as features; hidden test labels must not be used. |
| Leakage-safe preprocessing | PASS | PCA and MinMaxScaler are fitted within each CV training fold. |
| Ideal kernel configuration comparison | PASS | 168 configurations across entanglement, repetitions, qubit count, and bandwidth settings. |
| Prediction schema | PASS | tools/validate_predictions.py confirms predictions_q2.csv is valid. |
| Task A | PASS, recorded local evidence | 8192-shot Bell-state results in results/tables/task_a.csv and results/task_a_run.log. |
| Processor D | PASS | Official benchmark of processor_d_config.json succeeds with no mismatch warning. Depth after transpilation 18; 9 two-qubit gates; 0 SWAP overhead; ideal fidelity 1.0; noisy fidelity 0.668377. |
| Official noise model | PASS | Noise-model benchmark succeeds with QFF2026-v2 parameters. Do not modify the official noise-model file. |
| Noisy-kernel execution | PASS, subset experiment | src/noisy_kernel.py uses prepare_for_noisy_run and noisy_simulator. Diagnostics and run configuration are saved. |
| Noisy-kernel evaluation | RECORDED WITH CAVEAT | 100 training samples, 50 evaluation samples, 1024 shots; balanced accuracy 0.747596 and F1 0.770642. These are not five-fold CV scores. |
| Dependency pins | PASS | tools/check_environment.py confirms all 15 exact versions. |
| Notebook consistency | REVIEW REQUIRED | Confirm the latest notebook execution corresponds to the final code and environment. |
| Final report consistency | REVIEW REQUIRED | Check results.md against the current logs and distinguish carried-forward evidence from newly generated evidence. |
| Authoritative participant DOCX | UNVERIFIED | The DOCX was not included in the available package. |

## Performance and evaluation caveats

The ideal quantum-kernel grid selected none entanglement, reps=2, k=6, bandwidth s=0.8, C=1. Its five-fold CV balanced accuracy is 0.7362 and F1 is 0.7327. These are model-selection scores rather than an independent test estimate. The noisy-kernel metrics come from a separate subset experiment and are not directly comparable to the ideal model's five-fold CV results. No hidden-test score is established by these results.

## Latest Verification Update — 2026-10-30

The following verification commands were rerun successfully in the project virtual environment:

| Check | Current result | Evidence |
|---|---|---|
| Processor D official benchmark | PASS | results/processor_d_benchmark_run.log |
| Official noise model initialization | PASS | results/noise_model_self_test.log |
| Q2 prediction validator | PASS | predictions_q2.csv |
| GitHub synchronization at previous check | PASS | Local and remote main commit hashes matched |

Latest Processor D metrics:

- Depth before transpilation: 6
- Depth after transpilation: 18
- Two-qubit gates: 9
- SWAP overhead: 0
- Ideal fidelity: 1.000000
- Noisy fidelity: 0.668377
- Noise model: QFF2026-v2

### Fidelity definitions

Task A Bell-state fidelity:

F = (1 + <XX> - <YY> + <ZZ>) / 4

Task B QFT-3 fidelity:

F_j = <psi_j|rho_j|psi_j>

The QFT-3 result is the average of F_j over the eight computational-basis inputs.

### Remaining review items

| Requirement | State | Next action |
|---|---|---|
| Notebook consistency | REVIEW REQUIRED | Re-execute the final notebook and inspect outputs |
| Classical comparison consistency | REVIEW REQUIRED | Rerun baselines with the same folds and preprocessing protocol |
| Authoritative participant problem statement | UNVERIFIED | Compare against the official participant DOCX |
| Final report consistency | REVIEW REQUIRED | Check all report figures against current experiment evidence |
| Final ZIP archive | REVIEW REQUIRED | Rebuild and inspect after all documentation changes |

This audit records local verification evidence. It does not establish hidden-test accuracy, quantum advantage or competition ranking.
