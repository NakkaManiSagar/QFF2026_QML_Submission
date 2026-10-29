"""Processor D topology sweep using the organizer's QFT-3 benchmark and noise helper."""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GATE_SETS = {"rx_ry_rz_cx": ["rx", "ry", "rz", "cx"], "rz_sx_cx": ["rz", "sx", "cx"]}

def edges(n: int, topology: str):
    if topology == "line": base=[(i,i+1) for i in range(n-1)]
    elif topology == "ring": base=[(i,i+1) for i in range(n-1)]+([(n-1,0)] if n>2 else [])
    elif topology == "star": base=[(0,i) for i in range(1,n)]
    elif topology == "all_to_all": base=[(i,j) for i in range(n) for j in range(i+1,n)]
    else: raise ValueError(topology)
    return [list(p) for a,b in base for p in ((a,b),(b,a))]

def candidates():
    for topology in ("line","ring","star","all_to_all"):
        for n in (6,):
            for gate_name, gates in GATE_SETS.items():
                yield {"processor_id":f"procD_{topology}_{n}_{gate_name}","n_qubits":n,
                       "coupling_map":edges(n,topology),"native_gate_set":gates,
                       "qubit_layout": ("linear" if topology == "line" else topology) if topology in {"line","ring","star"} else "custom",
                       "noise_assumptions":{"single_qubit_error_rate":0.01,"two_qubit_error_rate":0.05,
                           "readout_error_rate":0.02,"notes":"Documentary assumptions only; benchmark uses unchanged QFF2026-v2 noise model."},
                       "design_rationale":"Candidate topology assessed by measured QFT-3 compilation/fidelity and transpiled kernel costs; final selection is deferred until benchmark evidence is available.",
                       "benchmark_results":{}}

def _kernel_circuit(n=6, reps=1, entanglement="none"):
    from src.quantum_kernel import make_logical_circuit
    return make_logical_circuit(n,reps,entanglement)

def run_sweep(root: Path=ROOT):
    root=Path(root); outdir=root/"results"/"tables"; figdir=root/"results"/"figures"
    outdir.mkdir(parents=True,exist_ok=True); figdir.mkdir(parents=True,exist_ok=True)
    rows=[]; benchmark_module=None; qiskit_error=None
    try:
        sys.path.insert(0,str(root/"benchmark")); sys.path.insert(0,str(root))
        import qff_qml_benchmark as bench
        from qiskit import transpile
        from qiskit.transpiler import CouplingMap, StagedPassManager
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        from noise_model.benchmark_noise_model import prepare_for_noisy_run
        benchmark_module=bench
    except Exception as e:
        qiskit_error=f"{type(e).__name__}: {e}"
    # best configuration from the saved all-training ideal ablation (selection uses CV only)
    best_path=outdir/"ideal_kernel_ablation.csv"
    best={"entanglement":"none","reps":1,"n_qubits":6,"bandwidth_s":1.2}
    if best_path.exists():
        df=pd.read_csv(best_path)
        r=df.sort_values(["cv_balanced_accuracy","cv_balanced_accuracy_std"],ascending=[False,True]).iloc[0]
        best={"entanglement":str(r.entanglement),"reps":int(r.reps),"n_qubits":int(r.n_qubits),"bandwidth_s":float(r.bandwidth_s)}
    for cfg in candidates():
        row={"processor_id":cfg["processor_id"],"topology":cfg["qubit_layout"],"n_qubits":cfg["n_qubits"],
             "native_gate_set":";".join(cfg["native_gate_set"]),"directed_edges":len(cfg["coupling_map"]),
             "kernel_entanglement":best["entanglement"],"kernel_reps":best["reps"],
             "kernel_n_qubits":best["n_qubits"],"qft3_circuit_depth_before_transpile":np.nan,
             "qft3_circuit_depth_after_transpile":np.nan,"qft3_two_qubit_gate_count":np.nan,"qft3_swap_overhead":np.nan,
             "qft3_fidelity_ideal":np.nan,"qft3_fidelity_noisy":np.nan,"kernel_depth_before":np.nan,
             "kernel_depth_after":np.nan,"kernel_two_qubit_gates_logical":np.nan,
             "kernel_two_qubit_gates_after_prepare":np.nan,"kernel_swap_overhead":np.nan,
             "qft3_status":"not run","kernel_transpile_status":"not run"}
        if benchmark_module is not None:
            try:
                qft=benchmark_module.task_b_qft3(cfg)
                row.update(qft); row["qft3_status"]="measured"
                cm=CouplingMap(cfg["coupling_map"])
                circ=_kernel_circuit(best["n_qubits"],best["reps"],best["entanglement"])
                before=circ.depth(); logical2q=sum(c for g,c in circ.count_ops().items() if g in {"cx","cz","swap"})
                routed=transpile(circ,basis_gates=cfg["native_gate_set"],coupling_map=cm,optimization_level=1,seed_transpiler=42)
                prepared=prepare_for_noisy_run(circ,basis_gates=cfg["native_gate_set"],coupling_map=cm)
                rops=routed.count_ops(); pops=prepared.count_ops()
                row.update({"kernel_depth_before":before,"kernel_depth_after":routed.depth(),
                    "kernel_two_qubit_gates_logical":logical2q,
                    "kernel_two_qubit_gates_after_prepare":sum(v for g,v in pops.items() if g in {"cx","cz","swap"}),
                    "kernel_swap_overhead":max(0,rops.get("swap",0)-circ.count_ops().get("swap",0)),
                    "kernel_transpile_status":"measured"})
            except Exception as e:
                row["qft3_status"]=f"failed: {type(e).__name__}: {e}"
                row["kernel_transpile_status"]=row["qft3_status"]
        else:
            row.update({"qft3_status":f"blocked: {qiskit_error}","kernel_transpile_status":f"blocked: {qiskit_error}"})
        rows.append(row)
    frame=pd.DataFrame(rows); frame.to_csv(outdir/"processor_d_candidates.csv",index=False)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        # Plot actual noisy QFT fidelity when measured; otherwise produce an explicit availability plot.
        fig,ax=plt.subplots(figsize=(10,5))
        if "qft3_fidelity_noisy" in frame and frame["qft3_fidelity_noisy"].notna().any():
            ax.bar(frame.processor_id,frame.qft3_fidelity_noisy); ax.set_ylabel("QFT-3 noisy fidelity")
        else:
            ax.text(.5,.5,"Qiskit/Aer unavailable: benchmark metrics not measured",ha="center",va="center",transform=ax.transAxes)
            ax.set_yticks([])
        ax.tick_params(axis="x",rotation=55); ax.set_title("Processor D candidate comparison"); fig.tight_layout()
        fig.savefig(figdir/"processor_d_candidates.png",dpi=160); plt.close(fig)
    except Exception: pass
    if benchmark_module is not None and frame["qft3_fidelity_noisy"].notna().any():
        # Evidence-first selection: maximize measured noisy QFT fidelity, then minimize mapped kernel depth/gate count.
        measured=frame.dropna(subset=["qft3_fidelity_noisy"]).sort_values(
            ["qft3_fidelity_noisy","kernel_depth_after","kernel_two_qubit_gates_after_prepare"],
            ascending=[False,True,True])
        selected=measured.iloc[0].to_dict()
        config=next(c for c in candidates() if c["processor_id"]==selected["processor_id"])
        rationale=(f"The selected {selected['topology']} topology with native gates {selected['native_gate_set']} was selected from measured candidate evidence. "
          f"Its QFT-3 noisy fidelity was {selected['qft3_fidelity_noisy']}, with transpiled depth {selected['qft3_circuit_depth_after_transpile']} and {selected['qft3_two_qubit_gate_count']} two-qubit gates. "
          f"The mapped best CV-selected kernel had depth {selected['kernel_depth_after']} and {selected['kernel_two_qubit_gates_after_prepare']} two-qubit gates after official preparation. "
          "The directed coupling map is fully bidirectional on each listed link, reducing direction-reversal overhead for CX. "
          "These are simulator measurements under the organizer QFF2026-v2 noise model, not physical-device results.")
        config["design_rationale"]=rationale
        config["benchmark_results"]={k:selected[k] for k in ["qft3_circuit_depth_before_transpile","qft3_circuit_depth_after_transpile","qft3_two_qubit_gate_count","qft3_swap_overhead","qft3_fidelity_ideal","qft3_fidelity_noisy","benchmark_noise_model_version"]}
        status="Processor D candidate sweep measured successfully. Final config selected by noisy QFT-3 fidelity, then mapped kernel depth and two-qubit gate count."
    else:
        config=next(c for c in candidates() if c["processor_id"]=="procD_line_6_rx_ry_rz_cx")
        config["design_rationale"]=("This is a provisional line-topology configuration, not an evidence-based final selection. The candidate sweep includes line, ring, star, and all-to-all topologies with both required native gate sets. "
          "The current runtime cannot import Qiskit/Aer, so QFT-3 fidelity, transpiled depth, two-qubit counts, routing overhead, and mapped-kernel costs are unavailable. "
          "No claim is made that this provisional topology outperforms the alternatives; rerun the candidate sweep in the pinned environment before submission.")
        config["benchmark_results"]={"qft3_circuit_depth_before_transpile":0,"qft3_circuit_depth_after_transpile":0,
           "qft3_two_qubit_gate_count":0,"qft3_swap_overhead":0,"qft3_fidelity_ideal":0.0,
           "qft3_fidelity_noisy":0.0,"benchmark_noise_model_version":"QFF2026-v2"}
        status=("Processor D benchmark status: NOT MEASURED in this runtime. Qiskit/Aer import failed: `"+str(qiskit_error)+"`. "
          "The zero values in processor_d_config.json are schema placeholders, not benchmark results; do not submit them as measurements. "
          "Run `python benchmark/qff_qml_benchmark.py processor_d_config.json` in the pinned environment and replace them with its output.")
    (root/"processor_d_config.json").write_text(json.dumps(config,indent=2)+"\n")
    (root/"results"/"processor_d_sweep_status.md").write_text(status+"\n")
    print(frame.to_string(index=False)); return frame

if __name__=="__main__": run_sweep()
