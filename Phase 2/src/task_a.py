"""Task A: run the official Bell benchmark unchanged and persist its returned values."""
from __future__ import annotations
import json, sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def run_task_a(root: Path = ROOT) -> dict:
    # Import the organizer's function; intentionally do not reimplement its parity/fidelity formula.
    sys.path.insert(0, str(root / "benchmark"))
    from qff_qml_benchmark import task_a_bell
    result = task_a_bell(shots=8192)
    rows = []
    for condition in ("ideal", "noisy"):
        r = result[condition]
        rows.append({"condition": condition, "shots": result["shots"],
                     "noise_model_version": result["noise_model_version"],
                     "XX": r["XX"], "XX_se": r["XX_se"],
                     "YY": r["YY"], "YY_se": r["YY_se"],
                     "ZZ": r["ZZ"], "ZZ_se": r["ZZ_se"],
                     "fidelity": r["fidelity"], "fidelity_se": r["fidelity_se"]})
    (root / "results").mkdir(exist_ok=True)
    (root / "results" / "task_a_raw.json").write_text(json.dumps(result, indent=2) + "\n")
    (root / "results" / "tables").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(root / "results" / "tables" / "task_a.csv", index=False)
    ideal, noisy = result["ideal"], result["noisy"]
    explanation = (
        f"With 8192 shots, the ideal run measured <XX>={ideal['XX']:.6f}, "
        f"<YY>={ideal['YY']:.6f}, and <ZZ>={ideal['ZZ']:.6f}, giving Bell fidelity "
        f"F={ideal['fidelity']:.6f} (SE {ideal['fidelity_se']:.6f}). The noisy run measured "
        f"<XX>={noisy['XX']:.6f}, <YY>={noisy['YY']:.6f}, and <ZZ>={noisy['ZZ']:.6f}, "
        f"giving F={noisy['fidelity']:.6f} (SE {noisy['fidelity_se']:.6f}). "
        f"The measured fidelity changed by {noisy['fidelity']-ideal['fidelity']:+.6f}; this is the observed "
        f"ideal-versus-noisy difference for these shots, not a claim beyond this measurement. "
        "The organizer model includes two-qubit depolarizing error on CX, single-qubit depolarizing "
        "error on the basis gates, and readout bit-flip error on measurement. These sources all contribute "
        "to the noisy result; this benchmark does not separately isolate their individual contributions.\n"
    )
    (root / "results" / "task_a_explanation.md").write_text(explanation)
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nSaved results/tables/task_a.csv and results/task_a_explanation.md")
    return result

if __name__ == "__main__":
    run_task_a()
