"""Organizer-noise compute-uncompute kernel experiment with pilot timing and disk caches.

No noisy run bypasses prepare_for_noisy_run. This module never uses test labels or test data
for tuning/scoring; test rows are used only to construct prediction kernels.
"""
from __future__ import annotations
import json, time, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from src.data import load_q2_data, SEED, INT_TO_LABEL
from src.quantum_kernel import ideal_kernel, make_logical_circuit, entangling_pairs

ROOT=Path(__file__).resolve().parents[1]
SHOTS=1024
BUDGET_SECONDS=30*60
TARGET_TRAIN=100
TARGET_TEST=50

def _load_config(root):
    return json.loads((root/'processor_d_config.json').read_text())

def _compute_uncompute_circuit(x, y, n, reps, entanglement):
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector
    # Construct independent parameterized feature maps, bind each sample, then U(y)^dagger U(x).
    base=make_logical_circuit(n,reps,entanglement)
    params=list(base.parameters)
    valsx={p:float(x[i]) for i,p in enumerate(params)}
    valsy={p:float(y[i]) for i,p in enumerate(params)}
    ux=base.assign_parameters(valsx)
    uy=base.assign_parameters(valsy)
    qc=QuantumCircuit(n,n)
    qc.compose(ux,inplace=True)
    qc.compose(uy.inverse(),inplace=True)
    qc.measure(range(n),range(n))
    return qc

def _evaluate_pair(x,y,cfg,shots=SHOTS,seed=SEED):
    from noise_model.benchmark_noise_model import prepare_for_noisy_run, noisy_simulator
    qc=_compute_uncompute_circuit(x,y,len(x),_ACTIVE_REPS,_ACTIVE_ENT)
    prepared=prepare_for_noisy_run(qc,basis_gates=cfg['native_gate_set'],coupling_map=cfg['coupling_map'])
    counts=noisy_simulator(seed=seed).run(prepared,shots=shots).result().get_counts()
    zero='0'*len(x)
    return counts.get(zero,0)/shots, int(prepared.depth()), int(sum(v for g,v in prepared.count_ops().items() if g in {'cx','cz','swap'}))

_ACTIVE_REPS=1
_ACTIVE_ENT='none'

def _matrix_noisy(A,B,cfg,reps,ent,shots,seed, symmetric=False):
    global _ACTIVE_REPS,_ACTIVE_ENT
    _ACTIVE_REPS=int(reps); _ACTIVE_ENT=str(ent)
    K=np.zeros((len(A),len(B)),float); depths=[]; twos=[]; start=time.time(); done=0
    if symmetric:
        np.fill_diagonal(K, 1.0)  # Gram diagonal is fixed to one; no diagonal hardware runs counted.
    for i,a in enumerate(A):
        jstart=(i+1) if symmetric else 0
        for j in range(jstart,len(B)):
            value,depth,two=_evaluate_pair(a,B[j],cfg,shots,seed+done)
            K[i,j]=value
            if symmetric: K[j,i]=value
            depths.append(depth); twos.append(two); done+=1
    return K, {'circuit_evaluations':done,'runtime_seconds':time.time()-start,
               'mean_depth':float(np.mean(depths)) if depths else np.nan,
               'mean_two_qubit_gates':float(np.mean(twos)) if twos else np.nan}

def _psd_clip(K):
    sym=(K+K.T)/2
    vals,vecs=np.linalg.eigh(sym); clipped=np.maximum(vals,0.0)
    return (vecs*clipped)@vecs.T, int((vals<0).sum()), vals

def run_noisy_kernel(root=ROOT, recompute=False):
    root=Path(root); tables=root/'results'/'tables'; figs=root/'results'/'figures'; cache=root/'results'/'cache'
    for p in (tables,figs,cache): p.mkdir(parents=True,exist_ok=True)
    train_path=cache/'noisy_subset_train.npy'; ideal_path=cache/'ideal_subset_train.npy'; test_path=cache/'noisy_subset_test_vs_train.npy'; idealtest_path=cache/'ideal_subset_test_vs_train.npy'
    metrics_path=cache/'noisy_kernel_matrix_metrics.json'
    try:
        import qiskit, qiskit_aer
    except Exception as e:
        msg=f"Noisy-kernel experiment NOT RUN: Qiskit/Aer unavailable ({type(e).__name__}: {e}). No noisy metrics fabricated."
        (root/'results'/'noisy_kernel_status.md').write_text(msg+'\n')
        print(msg); return None
    X,y,Xtest,sample_ids,l2i,i2l=load_q2_data(root)
    # Fixed best config from CV-only ideal ablation; test features are only used for predictions.
    best=pd.read_csv(tables/'ideal_kernel_ablation.csv').sort_values(['cv_balanced_accuracy','cv_balanced_accuracy_std'],ascending=[False,True]).iloc[0]
    n=int(best.n_qubits); reps=int(best.reps); ent=str(best.entanglement); s=float(best.bandwidth_s)
    # Train-only preprocessing. PCA/scaler are fit on training rows only.
    pca=PCA(n_components=n,random_state=SEED).fit(X)
    scaler=MinMaxScaler(feature_range=(0,s)).fit(pca.transform(X))
    Xtr=scaler.transform(pca.transform(X)); Xte=scaler.transform(pca.transform(Xtest))
    ix, _=train_test_split(np.arange(len(y)),train_size=TARGET_TRAIN,random_state=SEED,stratify=y)
    rng=np.random.RandomState(SEED); test_ix=np.sort(rng.choice(len(Xte),size=min(TARGET_TEST,len(Xte)),replace=False))
    ix=np.sort(ix); Xsub=Xtr[ix]; ysub=np.asarray(y)[ix]; Xtestsub=Xte[test_ix]
    cfg=_load_config(root)
    # Pilot: time 20 distinct compute-uncompute circuits before selecting a feasible subset.
    global _ACTIVE_REPS,_ACTIVE_ENT
    _ACTIVE_REPS=reps; _ACTIVE_ENT=ent
    from noise_model.benchmark_noise_model import prepare_for_noisy_run, noisy_simulator
    pilot_start=time.time(); pilot_done=0
    for t in range(20):
        a=Xsub[t%len(Xsub)]; b=Xsub[(t+1)%len(Xsub)]
        qc=_compute_uncompute_circuit(a,b,n,reps,ent)
        prepared=prepare_for_noisy_run(qc,basis_gates=cfg['native_gate_set'],coupling_map=cfg['coupling_map'])
        noisy_simulator(seed=SEED+t).run(prepared,shots=SHOTS).result().get_counts(); pilot_done+=1
    pilot_seconds=time.time()-pilot_start; sec_per=max(pilot_seconds/pilot_done,0.001)
    # Include train triangle and test-vs-train; cap based on 30 min. Preserve stratification and seed.
    max_evals=max(20,int(BUDGET_SECONDS/sec_per))
    print(f'Pilot: {pilot_done} circuits in {pilot_seconds:.2f}s ({sec_per:.3f}s/circuit); estimated budget capacity ~{max_evals} circuit evaluations.', flush=True)
    chosen_train=min(TARGET_TRAIN,int((1+np.sqrt(1+8*max_evals))/2))
    chosen_train=max(10,chosen_train)
    chosen_test=min(TARGET_TEST,max(0,(max_evals-chosen_train*(chosen_train-1)//2)//chosen_train))
    if chosen_test < 1: chosen_test=1
    if chosen_train < len(ix):
        ix,_=train_test_split(ix,train_size=chosen_train,random_state=SEED,stratify=np.asarray(y)[ix]); ix=np.sort(ix)
        Xsub=Xtr[ix]; ysub=np.asarray(y)[ix]
    Xtestsub=Xtestsub[:chosen_test]
    # Cache full symmetric training kernel and prediction kernel.
    meta={'n_train':len(Xsub),'n_test':len(Xtestsub),'shots':SHOTS,'pilot_circuits':20,'pilot_seconds':pilot_seconds,
          'estimated_seconds_per_circuit':sec_per,'estimated_total_seconds':sec_per*(len(Xsub)*(len(Xsub)-1)//2+len(Xtestsub)*len(Xsub)),
          'entanglement':ent,'reps':reps,'n_qubits':n,'bandwidth_s':s,'processor_id':cfg['processor_id']}
    print(f'Chosen subset: {len(Xsub)} train, {len(Xtestsub)} test; {SHOTS} shots; estimated total kernel time {meta["estimated_total_seconds"]:.1f}s.', flush=True)
    (root/'results'/'noisy_kernel_run.json').write_text(json.dumps(meta,indent=2)+'\n')
    if recompute or not train_path.exists() or not test_path.exists() or not metrics_path.exists():
        Ki=ideal_kernel(Xsub,reps=reps,entanglement=ent); Kt=ideal_kernel(Xtestsub,Xsub,reps=reps,entanglement=ent)
        Kn,mt=_matrix_noisy(Xsub,Xsub,cfg,reps,ent,SHOTS,SEED,True)
        Ktn,mtest=_matrix_noisy(Xtestsub,Xsub,cfg,reps,ent,SHOTS,SEED+100000,False)
        np.save(ideal_path,Ki); np.save(idealtest_path,Kt); np.save(train_path,Kn); np.save(test_path,Ktn)
        metrics_path.write_text(json.dumps({'train':mt,'test':mtest},indent=2)+'\n')
    else:
        Ki=np.load(ideal_path); Kt=np.load(idealtest_path); Kn=np.load(train_path); Ktn=np.load(test_path)
        matrix_metrics=json.loads(metrics_path.read_text()); mt=matrix_metrics['train']; mtest=matrix_metrics['test']
    Kn_psd,neg_count,eig_before=_psd_clip(Kn)
    np.save(cache/'noisy_subset_train_psd.npy',Kn_psd)
    off=Kn[~np.eye(len(Kn),dtype=bool)]; offi=Ki[~np.eye(len(Ki),dtype=bool)]
    delta=(Kn-Ki).ravel(); idealflat=Ki.ravel(); noisyflat=Kn.ravel()
    slope,intercept=np.polyfit(idealflat,noisyflat,1); pred=slope*idealflat+intercept
    r2=1-float(np.sum((noisyflat-pred)**2)/(np.sum((noisyflat-noisyflat.mean())**2)+1e-12))
    cv=StratifiedKFold(5,shuffle=True,random_state=SEED); grid={'C':[0.1,1,10]}
    def score(K):
        gs=GridSearchCV(SVC(kernel='precomputed',class_weight='balanced'),grid,scoring='balanced_accuracy',cv=cv,n_jobs=1).fit(K,ysub)
        pred=cross_val_predict(SVC(kernel='precomputed',C=gs.best_params_['C'],class_weight='balanced'),K,ysub,cv=cv)
        return gs.best_params_['C'],balanced_accuracy_score(ysub,pred),f1_score(ysub,pred)
    Ci,bai,f1i=score(Ki); Cn,ban,f1n=score(Kn_psd)
    baseline=pd.read_csv(tables/'classical_baselines.csv'); br=baseline.sort_values('balanced_accuracy_mean',ascending=False).iloc[0]
    pd.DataFrame([{'model':'ideal_kernel','balanced_accuracy':bai,'f1':f1i,'C':Ci},
      {'model':'noisy_kernel_psd_clipped','balanced_accuracy':ban,'f1':f1n,'C':Cn},
      {'model':str(br.model),'balanced_accuracy':br.balanced_accuracy_mean,'f1':br.f1_mean,'C':np.nan}]).to_csv(tables/'noisy_kernel_accuracy_comparison.csv',index=False)
    diagnostics={'mean_abs_kernel_error':float(np.mean(np.abs(Kn-Ki))),'ideal_offdiag_mean':float(offi.mean()),
      'ideal_offdiag_std':float(offi.std()),'noisy_offdiag_mean':float(off.mean()),'noisy_offdiag_std':float(off.std()),
      'ideal_alignment':_alignment(Ki,ysub),'noisy_alignment':_alignment(Kn_psd,ysub),'negative_eigenvalues_before_clip':neg_count,
      'ideal_balanced_accuracy':float(bai),'ideal_f1':float(f1i),'noisy_balanced_accuracy':float(ban),'noisy_f1':float(f1n),
      'ideal_best_C':float(Ci),'noisy_best_C':float(Cn),
      'lambda':float(slope),'c':float(intercept),'linear_fit_R2':r2,'linear_hypothesis_holds':bool(r2>=0.9),
      'ideal_eigenvalues':np.linalg.eigvalsh((Ki+Ki.T)/2).tolist(),'noisy_eigenvalues_before_clip':eig_before.tolist(),
      'shots':SHOTS,'train_subset':len(ysub),'test_subset':len(Xtestsub),'pilot_circuits':20,'pilot_seconds':pilot_seconds,
      'estimated_total_seconds':meta['estimated_total_seconds'],'noisy_train_metrics':mt,'noisy_test_metrics':mtest,
      'circuit_evaluations_train':len(ysub)*(len(ysub)-1)//2,'circuit_evaluations_test':len(Xtestsub)*len(ysub),
      'n_qubits':n,'depth_mean':mt.get('mean_depth',np.nan),'two_qubit_gates_mean':mt.get('mean_two_qubit_gates',np.nan),
      'noise_model':'official QFF2026-v2 via prepare_for_noisy_run'}
    (tables/'noisy_kernel_diagnostics.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
    pd.DataFrame([{'entanglement':ent,'reps':reps,'n_qubits':n,'bandwidth_s':s,'cx_count_logical':2*len(entangling_pairs(n,ent))*reps,
      'kernel_error':diagnostics['mean_abs_kernel_error'],'ideal_balanced_accuracy':bai,'noisy_balanced_accuracy':ban}]).to_csv(tables/'noisy_kernel_entanglement_comparison.csv',index=False)
    _plots(Ki,Kn,figs,idealflat,noisyflat,diagnostics,tables)
    print(json.dumps(diagnostics,indent=2)); return diagnostics

def _alignment(K,y):
    yy=np.where(np.asarray(y)==1,1.,-1.); T=np.outer(yy,yy); Kc=K-K.mean(0)-K.mean(1)[:,None]+K.mean()
    return float(np.sum(Kc*T)/(np.linalg.norm(Kc)*np.linalg.norm(T)+1e-12))

def _plots(Ki,Kn,figs,x,y,d,tables):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(10,4)); axs[0].imshow(Ki,vmin=0,vmax=1); axs[0].set_title('Ideal kernel'); axs[1].imshow(Kn,vmin=0,vmax=1); axs[1].set_title('Noisy kernel'); fig.tight_layout(); fig.savefig(figs/'kernel_matrices_ideal_vs_noisy.png',dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,5)); ax.scatter(x,y,s=5,alpha=.25); xx=np.linspace(float(np.min(x)),float(np.max(x)),100); ax.plot(xx,d['lambda']*xx+d['c']); ax.set_xlabel('K ideal'); ax.set_ylabel('K noisy'); ax.set_title(f"Noisy vs ideal; R²={d['linear_fit_R2']:.3f}"); fig.tight_layout(); fig.savefig(figs/'kernel_noisy_vs_ideal_scatter.png',dpi=160); plt.close(fig)
    comp=pd.read_csv(tables/'noisy_kernel_entanglement_comparison.csv'); fig,ax=plt.subplots(figsize=(6,4)); ax.scatter(comp.cx_count_logical,comp.kernel_error); ax.set_xlabel('Logical CX count'); ax.set_ylabel('Mean absolute kernel error'); fig.tight_layout(); fig.savefig(figs/'accuracy_vs_cx_count.png',dpi=160); plt.close(fig)

if __name__=='__main__': run_noisy_kernel(recompute='--recompute' in sys.argv)
