# === Quantum MMD over selected 150 windows ===

import os, time, math, json
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import pennylane as qml
from pennylane import numpy as pnp
import hashlib

# ---------------- CONFIG ----------------
BASE_CSV = "qml_drift/data/creditcard.csv"
OUT_DIR = "results/creditcard/quantum_150_run"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_CSV = os.path.join(OUT_DIR, "quantum_results_150_windows.csv")

# Quantum / performance params (tweak if slow)
N_QUBITS = 4          
REPS = 2
ANALYTIC = True       
FEATURE_SCALE = 1.0
N_SUB = 200           
B_PERM = 50           
RANDOM_SEED = 42
PROFILE = False       
# ----------------------------------------

# Device helper (use lightning.qubit if available)
def _make_device(n_qubits:int = N_QUBITS, analytic:bool = ANALYTIC):
    if analytic:
        # lightning.qubit supports statevector when shots=None
        try:
            return qml.device("lightning.qubit", wires=n_qubits, shots=None)
        except Exception:
            # fallback
            return qml.device("default.qubit", wires=n_qubits, shots=None)
    else:
        return qml.device("lightning.qubit", wires=n_qubits, shots=1024)

# feature map / qnode factory (angle encoding + simple entanglement)
def _feature_map(x: np.ndarray, n_qubits:int = N_QUBITS, reps:int = REPS, feature_scale:float = FEATURE_SCALE):
    for i in range(n_qubits):
        qml.RY(feature_scale * float(x[i]) * math.pi, wires=i)
    for r in range(reps):
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i+1])
        for i in range(n_qubits):
            qml.RZ(0.5 * math.pi * float(x[i]), wires=i)

def _make_qnode(n_qubits:int = N_QUBITS, reps:int = REPS, analytic:bool = ANALYTIC):
    dev = _make_device(n_qubits=n_qubits, analytic=analytic)
    @qml.qnode(dev, interface="autograd")
    def circuit(x):
        _feature_map(x, n_qubits=n_qubits, reps=reps, feature_scale=FEATURE_SCALE)
        return qml.state()
    return circuit

# small in-memory statevector cache keyed by hashed vector
_state_cache = {}
def _hash_vector(v: np.ndarray) -> str:
    a = np.asarray(v, dtype=float).copy()
    a = np.round(a, decimals=12)
    return hashlib.sha256(a.tobytes()).hexdigest()

def _compute_state(x: np.ndarray, n_qubits:int = N_QUBITS, reps:int = REPS, analytic:bool = ANALYTIC):
    key = (n_qubits, reps, analytic, FEATURE_SCALE, _hash_vector(np.asarray(x)))
    if key in _state_cache:
        return _state_cache[key]
    qnode = _make_qnode(n_qubits=n_qubits, reps=reps, analytic=analytic)
    st = np.asarray(qnode(pnp.array(x, dtype=float)))
    _state_cache[key] = st
    return st

def compute_statevectors(Z: np.ndarray, n_qubits:int = N_QUBITS, reps:int = REPS, analytic:bool = ANALYTIC, profile:bool = False):
    Z = np.asarray(Z, dtype=float)
    states = []
    t0 = time.time()
    for i in range(Z.shape[0]):
        states.append(_compute_state(Z[i], n_qubits=n_qubits, reps=reps, analytic=analytic))
    states = np.stack(states, axis=0)
    if profile:
        print(f"[quantum] computed {Z.shape[0]} statevectors in {time.time()-t0:.2f}s")
    return states

def fidelity_kernel_from_states(states):
    return np.abs(states @ states.conj().T) ** 2

def compute_quantum_kernel_matrix(X, Y, n_qubits:int = N_QUBITS, reps:int = REPS, analytic:bool = ANALYTIC, n_sub:int = N_SUB, random_seed:int = RANDOM_SEED, profile:bool = PROFILE):
    np.random.seed(random_seed)
    X = np.asarray(X, dtype=float); Y = np.asarray(Y, dtype=float)
    X = np.round(X, decimals=12); Y = np.round(Y, decimals=12)
    nx = X.shape[0]; ny = Y.shape[0]
    if n_sub is None:
        X_sel = X; Y_sel = Y
        idx_X = np.arange(0, X_sel.shape[0])
        idx_Y = np.arange(X_sel.shape[0], X_sel.shape[0] + Y_sel.shape[0])
    else:
        frac = nx / (nx + ny)
        n_sub_x = max(2, int(round(n_sub * frac)))
        n_sub_y = max(2, n_sub - n_sub_x)
        n_sub_x = min(n_sub_x, nx); n_sub_y = min(n_sub_y, ny)
        ix = np.random.choice(np.arange(nx), size=n_sub_x, replace=False)
        iy = np.random.choice(np.arange(ny), size=n_sub_y, replace=False)
        X_sel = X[ix]; Y_sel = Y[iy]
        idx_X = np.arange(0, X_sel.shape[0])
        idx_Y = np.arange(X_sel.shape[0], X_sel.shape[0] + Y_sel.shape[0])
    Z = np.vstack([X_sel, Y_sel])
    states = compute_statevectors(Z, n_qubits=n_qubits, reps=reps, analytic=analytic, profile=profile)
    K = fidelity_kernel_from_states(states, )
    return K, idx_X, idx_Y

def mmd_unbiased_from_kernel(K, idx_X, idx_Y):
    n = len(idx_X); m = len(idx_Y)
    Kxx = K[np.ix_(idx_X, idx_X)]
    Kyy = K[np.ix_(idx_Y, idx_Y)]
    Kxy = K[np.ix_(idx_X, idx_Y)]
    sum_xx = (np.sum(Kxx) - np.trace(Kxx)) / (n * (n - 1) + 1e-12)
    sum_yy = (np.sum(Kyy) - np.trace(Kyy)) / (m * (m - 1) + 1e-12)
    sum_xy = np.sum(Kxy) / (n * m + 1e-12)
    return float(sum_xx + sum_yy - 2.0 * sum_xy)

def permutation_p_value_from_kernel(K, idx_X, idx_Y, B:int = B_PERM, seed:int = RANDOM_SEED):
    np.random.seed(seed)
    Ntot = K.shape[0]
    n = len(idx_X); m = len(idx_Y)
    obs = mmd_unbiased_from_kernel(K, idx_X, idx_Y)
    cnt = 0
    stats = []
    for b in range(B):
        perm = np.random.permutation(Ntot)
        px = perm[:n]; py = perm[n:n+m]
        s = mmd_unbiased_from_kernel(K, px, py)
        stats.append(float(s))
        if s >= obs:
            cnt += 1
    pval = (cnt + 1) / (B + 1)
    return float(obs), float(pval), np.array(stats, dtype=float)

def quantum_mmd_and_pvalue(X, Y, n_qubits:int = N_QUBITS, reps:int = REPS, analytic:bool = ANALYTIC, n_sub:int = N_SUB, B:int = B_PERM, feature_scale:float = FEATURE_SCALE, random_seed:int = RANDOM_SEED, profile:bool = PROFILE):
    K, idx_X, idx_Y = compute_quantum_kernel_matrix(X, Y, n_qubits=n_qubits, reps=reps, analytic=analytic, n_sub=n_sub, random_seed=random_seed, profile=profile)
    t0 = time.time()
    mmd_stat = mmd_unbiased_from_kernel(K, idx_X, idx_Y)
    obs, pval, perm_stats = permutation_p_value_from_kernel(K, idx_X, idx_Y, B=B, seed=random_seed)
    meta = {
        "n_sub_total": int(K.shape[0]),
        "n_sub_X": int(len(idx_X)),
        "n_sub_Y": int(len(idx_Y)),
        "kernel_var": float(np.var(K[np.ix_(idx_X, idx_Y)].flatten())) if len(idx_X)>0 and len(idx_Y)>0 else float("nan"),
        "elapsed_kernel_s": time.time() - t0
    }
    return float(mmd_stat), float(pval), meta

# ---------------- Prepare windows selection ----------------
print("Loading dataset to determine windows...")
df_all = pd.read_csv(BASE_CSV)
n_total = len(df_all)

def sliding_windows_indices_local(n_total, n_ref=4000, n_test=1000, stride=500):
    indices = []
    max_start = n_total - (n_ref + n_test)
    start = 0; win_id = 0
    while start <= max_start:
        indices.append((win_id, start, start + n_ref, start + n_ref + n_test))
        start += stride; win_id += 1
    return indices

try:
    indices = sliding_windows_indices(n_total)   
except Exception:
    indices = sliding_windows_indices_local(n_total)

all_win_ids = [t[0] for t in indices]
total_windows = len(all_win_ids)
print(f"Total rows: {n_total} -> Total windows available: {total_windows}")


sel_file = "selected_windows.txt"
if os.path.exists(sel_file):
    sel = np.loadtxt(sel_file, dtype=int).tolist()
    selected_win_ids = [int(x) for x in sel if int(x) in all_win_ids]
    print("Using selected windows from", sel_file, "->", len(selected_win_ids), "windows")
else:
    
    N_SELECT = 150
    if N_SELECT > total_windows:
        N_SELECT = total_windows
    idxs = np.linspace(0, total_windows - 1, N_SELECT, dtype=int)
    selected_win_ids = [int(all_win_ids[i]) for i in idxs]
    print(f"No selected_windows.txt found — selecting {len(selected_win_ids)} evenly spaced windows")


win_map = {t[0]: t[1:] for t in indices}
selected_windows = []
for wid in selected_win_ids:
    sref, stest, etest = win_map[wid]
    selected_windows.append((wid, sref, stest, etest))

print(f"Processing {len(selected_windows)} windows (quantum).")

# ---------------- Run quantum MMD on selected windows ----------------
results = []
_seed_base = RANDOM_SEED
for (wid, sref, stest, etest) in tqdm(selected_windows, desc="quantum-windows"):
    rec = {
        "window_id": int(wid),
        "start_ref": int(sref),
        "start_test": int(stest),
        "end_test": int(etest),
        "mmd_stat_q": np.nan,
        "p_q": np.nan,
        "kernel_var": np.nan,
        "n_sub_total": np.nan,
        "n_sub_X": np.nan,
        "n_sub_Y": np.nan,
        "perm_time_s": np.nan,
        "total_time_s": np.nan,
        "error": ""
    }
    t0_win = time.time()
    try:
        R_df = df_all.iloc[sref:stest].reset_index(drop=True)
        N_df = df_all.iloc[stest:etest].reset_index(drop=True)
        # preprocess_window must return 'R_q' and 'N_q' (quantum embeddings) of shape (n_samples, N_QUBITS)
        pre = preprocess_window(R_df, N_df)
        Xq = np.asarray(pre['R_q'], dtype=float)
        Yq = np.asarray(pre['N_q'], dtype=float)
        # enforce dims
        if Xq.shape[1] != N_QUBITS or Yq.shape[1] != N_QUBITS:
            raise ValueError(f"Quantum feature dim mismatch: expected {N_QUBITS}, got {Xq.shape[1]} and {Yq.shape[1]}")

        # per-window seed for reproducibility (mix with window id)
        seed = int(_seed_base + int(wid))
        tq0 = time.time()
        q_mmd, q_p, qmeta = quantum_mmd_and_pvalue(Xq, Yq, n_qubits=N_QUBITS, reps=REPS, analytic=ANALYTIC,
                                                  n_sub=N_SUB, B=B_PERM, feature_scale=FEATURE_SCALE,
                                                  random_seed=seed, profile=PROFILE)
        tq1 = time.time()
        rec["mmd_stat_q"] = float(q_mmd)
        rec["p_q"] = float(q_p)
        rec["kernel_var"] = float(qmeta.get("kernel_var", np.nan))
        rec["n_sub_total"] = int(qmeta.get("n_sub_total", np.nan))
        rec["n_sub_X"] = int(qmeta.get("n_sub_X", np.nan))
        rec["n_sub_Y"] = int(qmeta.get("n_sub_Y", np.nan))
        rec["perm_time_s"] = float(qmeta.get("elapsed_kernel_s", tq1 - tq0))
        rec["total_time_s"] = float(time.time() - t0_win)
    except Exception as e:
        rec["error"] = str(e)
        rec["total_time_s"] = float(time.time() - t0_win)
    results.append(rec)
    # save intermediate results every 10 windows to avoid losing progress
    if len(results) % 10 == 0:
        pd.DataFrame(results).to_csv(OUT_CSV, index=False)

# final save
df_out = pd.DataFrame(results)
df_out.to_csv(OUT_CSV, index=False)
print("Saved quantum results to:", OUT_CSV)
print("Done. Summary:")
print("  windows processed:", len(df_out))
print("  errors:", int(df_out['error'].astype(bool).sum()) if 'error' in df_out.columns else 0)
