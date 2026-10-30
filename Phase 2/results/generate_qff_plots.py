"""
Recreates QFF 2026 PS-Q2 figures from the values documented in this file.
Run with: python generate_qff_plots.py
Requires: matplotlib, numpy, pandas
"""
import os
import numpy as np
import matplotlib.pyplot as plt

OUT = "figures"
os.makedirs(OUT, exist_ok=True)

# Values are documented in README.md; dashboard subset and CV protocols stay separate.
models = ["RBF-SVM\n50 features", "PCA-RBF-SVM\n8 components", "Ideal quantum\nkernel"]
ba = [0.7185, 0.7242, 0.7362]
f1 = [0.7267, 0.7254, 0.7327]
x=np.arange(3); w=.34
fig,ax=plt.subplots(figsize=(9,5))
ax.bar(x-w/2,np.array(ba)*100,w,label="Balanced Accuracy")
ax.bar(x+w/2,np.array(f1)*100,w,label="F1 Score")
ax.set_xticks(x,models); ax.set_ylim(0,100); ax.set_ylabel("Score (%)")
ax.set_title("Five-Fold Cross-Validation: Classical vs Ideal Quantum Kernel")
ax.legend(); ax.grid(axis="y",alpha=.25); fig.tight_layout()
fig.savefig(f"{OUT}/01_cv_classical_vs_quantum.png",dpi=220); plt.close(fig)

print("Figures generated in", OUT)
