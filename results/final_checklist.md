# Final checklist — verified local environment

| Requirement | Status | Evidence / remaining action |
|---|---|---|
| One problem statement selected | PASS | PS-Q2 selected; Q2 training/test data included. |
| Test labels not used; sample_id excluded | PASS by implementation audit | src/data.py uses x01-x50 as features; do not use hidden test labels. |
| Full ideal kernel parameter grid | PASS | results/tables/ideal_kernel_ablation.csv contains 168 configurations. |
| Fold-local preprocessing | PASS | PCA and scaling are fitted inside CV training folds. |
| Final model selection | PASS with caveat | Five-fold CV is used for model selection, not as an independent test estimate. |
| Q2 prediction file | PASS | predictions_q2.csv passes tools/validate_predictions.py. |
| Task A Bell-state benchmark | PASS, recorded evidence | results/tables/task_a.csv and results/task_a_run.log; 8192-shot measurements. |
| Processor D configuration | PASS | processor_d_config.json passes the official benchmark with no mismatch warning. |
| Official noise model | PASS | noise_model/benchmark_noise_model.py succeeds; QFF2026-v2 unchanged. |
| Noisy-kernel execution | PASS, subset evaluation | results/tables/noisy_kernel_diagnostics.json and results/noisy_kernel_run.json. Uses 100 training samples, 50 evaluation samples, 1024 shots. |
| Noisy-kernel subset metrics | RECORDED WITH CAVEAT | Balanced accuracy 0.747596; F1 0.770642. Not directly comparable to five-fold CV results. |
| Exact pinned dependencies | PASS | tools/check_environment.py confirms all 15 pinned versions. |
| Notebook end-to-end consistency | REVIEW REQUIRED | Inspect notebook execution log and confirm it reflects the final environment and source. |
| Report and evidence consistency | REVIEW REQUIRED | Reconcile results.md, REQUIREMENT_AUDIT.md, final_checklist.md, and current run logs. |
| Official participant DOCX | NOT AVAILABLE | The authoritative participant problem-statement DOCX was not present in the supplied package. |

## Performance statement

The leakage-safe ideal-kernel search selected none entanglement, reps=2, k=6, bandwidth s=0.8, C=1, with five-fold CV balanced accuracy 0.7362 and F1 0.7327. These are model-selection scores, not independent test estimates. The noisy-kernel balanced accuracy 0.7476 and F1 0.7706 are from a separate 100-training/50-evaluation subset experiment and must not be presented as directly comparable cross-validation results. Do not claim hidden-test performance or 90%+ accuracy without valid supporting evidence.
