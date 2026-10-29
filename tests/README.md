# Tests

The official kit self-tests are:

```bash
python noise_model/benchmark_noise_model.py
python benchmark/qff_qml_benchmark.py submission_templates/processor_d_example.json
python tools/validate_predictions.py q2 predictions_q2.csv
```

Run these inside the dedicated environment created from the unmodified `requirements.txt`.
