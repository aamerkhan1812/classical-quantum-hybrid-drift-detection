# === Classical drift detection on 150 windows ===
import os, time, math
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.spatial.distance import pdist
from scipy.stats import ks_2samp
from tqdm.auto import tqdm

# ---------------- CONFIG ----------------
BASE_CSV = "qml_drift/data/creditcard.csv"
OUT_DIR = "results/creditcard/classical_150_windows"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_CSV = os.path.join(OUT_DIR, "classical_150_windows.csv")

N_REF = 4000
N_TEST = 1000
STRIDE = 500
WINDOW_COUNT = 150            
WINDOW_SELECTION = "first"    
# Performance / statistical tradeoffs
N_JOBS = max(1, (os.cpu_count() or 2) - 1)
B_PERM = 100                  # permutations per window (reduce to 50 if too slow)
SAMPLE_FOR_GAMMA = 2000
SUBSAMPLE_FOR_KERNEL = 1000   # set to integer to reduce kernel O(n^2); None to use full R/N
SEED = 42
# ---------------------------------------

np.random.seed(SEED)

# ---------- helper fns  ----------
def median_heuristic_gamma_sample(X, max_sample=SAMPLE_FOR_GAMMA):
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    if n > max_sample:
        idx = np.random.choice(n, max_sample, replace=False)
        sub = X[idx]
    else:
        sub = X
    if sub.shape[0] < 2:
        return 1.0
    dists = pdist(sub, metric='euclidean')
    med = np.median(dists) if len(dists) > 0 else 1.0
    if med <= 0:
        return 1.0
    return 1.0 / (2.0 * (med**2))

def rbf_kernel(X, Y, gamma):
    X = np.asarray(X, dtype=float); Y = np.asarray(Y, dtype=float)
    XX = np.sum(X * X, axis=1)[:, None]
    YY = np.sum(Y * Y, axis=1)[None, :]
    D2 = XX + YY - 2.0 * X.dot(Y.T)
    K = np.exp(-gamma * D2)
    return K

def mmd_unbiased_from_kernel(Kxx, Kyy, Kxy):
    n = Kxx.shape[0]; m = Kyy.shape[0]
    if n < 2 or m < 2:
        return 0.0
    sum_xx = (np.sum(Kxx) - np.trace(Kxx)) / (n * (n - 1) + 1e-12)
    sum_yy = (np.sum(Kyy) - np.trace(Kyy)) / (m * (m - 1) + 1e-12)
    sum_xy = np.sum(Kxy) / (n * m + 1e-12)
    return float(sum_xx + sum_yy - 2 * sum_xy)

def mmd_stat_from_data(X, Y, gamma=None, sample_for_gamma=SAMPLE_FOR_GAMMA):
    if gamma is None:
        gamma = median_heuristic_gamma_sample(np.vstack([X, Y]), max_sample=sample_for_gamma)
    Kxx = rbf_kernel(X, X, gamma)
    Kyy = rbf_kernel(Y, Y, gamma)
    Kxy = rbf_kernel(X, Y, gamma)
    return mmd_unbiased_from_kernel(Kxx, Kyy, Kxy), float(gamma)

def permutation_p_value(X, Y, stat_func, B=B_PERM, seed=SEED):
    np.random.seed(seed)
    Z = np.vstack([X, Y])
    n = X.shape[0]; m = Y.shape[0]
    if n == 0 or m == 0:
        return 0.0, 1.0, np.array([])
    obs = stat_func(X, Y)
    cnt = 0
    stats = []
    for b in range(B):
        idx = np.random.permutation(n + m)
        Xp = Z[idx[:n]]
        Yp = Z[idx[n:]]
        s = stat_func(Xp, Yp)
        stats.append(float(s))
        if s >= obs:
            cnt += 1
    pval = (cnt + 1) / (B + 1)
    return float(obs), float(pval), np.array(stats, dtype=float)

def psi_score(a, b, n_bins=10):
    a = np.asarray(a); b = np.asarray(b)
    try:
        bins = np.quantile(a, q=np.linspace(0, 1, n_bins + 1))
    except Exception:
        bins = np.linspace(min(a.min(), b.min()), max(a.max(), b.max()), n_bins + 1)
    bins = np.unique(bins)
    if len(bins) <= 1:
        return 0.0
    a_counts, _ = np.histogram(a, bins=bins)
    b_counts, _ = np.histogram(b, bins=bins)
    pa = a_counts / (np.sum(a_counts) + 1e-12)
    pb = b_counts / (np.sum(b_counts) + 1e-12)
    pa = np.where(pa == 0, 1e-8, pa)
    pb = np.where(pb == 0, 1e-8, pb)
    return float(np.sum((pa - pb) * np.log(pa / pb)))

# ---------- sliding windows helper ----------
def sliding_windows_indices(n_total, n_ref=N_REF, n_test=N_TEST, stride=STRIDE):
    indices = []
    max_start = n_total - (n_ref + n_test)
    start = 0
    win_id = 0
    while start <= max_start:
        indices.append((win_id, start, start + n_ref, start + n_ref + n_test))
        start += stride
        win_id += 1
    return indices

# ---------- worker that uses preprocess_window (or fallback to user-defined process_window) ----------
def _worker_from_preprocess(window_tuple):
    win_id, sref, stest, etest = window_tuple
    try:
        R_df = df_all.iloc[sref:stest].reset_index(drop=True)
        N_df = df_all.iloc[stest:etest].reset_index(drop=True)
        pre = preprocess_window(R_df, N_df)   # assumes preprocess_window exists
        R_cls = np.asarray(pre['R_cls'], dtype=float)
        N_cls = np.asarray(pre['N_cls'], dtype=float)
        R_scaled = pre['R_scaled_df']; N_scaled = pre['N_scaled_df']
        # optional subsample for kernel speed
        if SUBSAMPLE_FOR_KERNEL is not None:
            nx = R_cls.shape[0]; ny = N_cls.shape[0]
            ix = np.random.choice(np.arange(nx), size=min(SUBSAMPLE_FOR_KERNEL, nx), replace=False)
            iy = np.random.choice(np.arange(ny), size=min(SUBSAMPLE_FOR_KERNEL, ny), replace=False)
            Rm = R_cls[ix]; Nm = N_cls[iy]
        else:
            Rm = R_cls; Nm = N_cls
        mmd_val, gamma_used = mmd_stat_from_data(Rm, Nm, gamma=None)
        obs, pval, perm_stats = permutation_p_value(Rm, Nm, lambda a,b: mmd_stat_from_data(a,b,gamma_used)[0],
                                                   B=B_PERM, seed=(SEED + int(win_id)))
        ks_list = []; psi_list = []
        for c in pre.get('feat_cols', []):
            try:
                ks = ks_2samp(R_scaled[c].values, N_scaled[c].values)
                ks_list.append((c, float(ks.statistic), float(ks.pvalue)))
            except Exception:
                ks_list.append((c, 0.0, 1.0))
            try:
                psi_v = psi_score(R_scaled[c].values, N_scaled[c].values)
                psi_list.append((c, float(psi_v)))
            except Exception:
                psi_list.append((c, 0.0))
        ks_sorted = sorted(ks_list, key=lambda x: -x[1])[:3]
        psi_sorted = sorted(psi_list, key=lambda x: -x[1])[:3]
        return {
            "window_id": int(win_id),
            "start_ref": int(sref),
            "start_test": int(stest),
            "end_test": int(etest),
            "mmd_stat": float(mmd_val),
            "gamma": float(gamma_used),
            "p_cls": float(pval),
            "perm_mean": float(np.mean(perm_stats)) if perm_stats.size>0 else float("nan"),
            "perm_std": float(np.std(perm_stats)) if perm_stats.size>0 else float("nan"),
            "top_ks": ";".join([f"{k}:{stat:.4f}:{pv:.1e}" for (k,stat,pv) in ks_sorted]),
            "top_psi": ";".join([f"{k}:{v:.4f}" for (k,v) in psi_sorted])
        }
    except Exception as e:
        return {"window_id": int(win_id), "start_ref": int(sref), "start_test": int(stest),
                "end_test": int(etest), "error": str(e)}

# ---------- choose worker: if process_window exists, use it; else use local preprocess worker ----------
_worker = None
try:
    # prefer user-defined process_window if available
    if 'process_window' in globals() and callable(process_window):
        _worker = process_window
        print("Using existing user-defined process_window()")
    else:
        raise NameError
except Exception:
    _worker = _worker_from_preprocess
    print("Using internal worker that calls preprocess_window()")

# ---------- load data and pick windows ----------
print("Loading dataset ...")
df_all = pd.read_csv(BASE_CSV)
n_total = len(df_all)
all_indices = sliding_windows_indices(n_total, N_REF, N_TEST, STRIDE)
n_all_windows = len(all_indices)
if WINDOW_COUNT <= 0 or WINDOW_COUNT > n_all_windows:
    WINDOW_COUNT = n_all_windows

# select windows
if WINDOW_SELECTION == "first":
    sel_indices = all_indices[:WINDOW_COUNT]
elif WINDOW_SELECTION == "even":
    # sample WINDOW_COUNT indices evenly across entire range
    idxs = np.linspace(0, n_all_windows-1, WINDOW_COUNT, dtype=int)
    sel_indices = [all_indices[i] for i in idxs]
else:
    sel_indices = all_indices[:WINDOW_COUNT]

print(f"Total rows: {n_total}  → All windows: {n_all_windows}  → Selected windows: {len(sel_indices)}")
print(f"Running with N_jobs={N_JOBS}, B_permutations={B_PERM}, SUBSAMPLE_FOR_KERNEL={SUBSAMPLE_FOR_KERNEL}")

# ---------- run in parallel ----------
t0 = time.time()
results = Parallel(n_jobs=N_JOBS, prefer="threads")(
    delayed(_worker)(idx_tuple) for idx_tuple in tqdm(sel_indices, desc="Processing windows")
)
elapsed = time.time() - t0
print(f"\nDone classical stage. Time taken: {elapsed/60:.2f} minutes (elapsed {elapsed:.1f}s)")

# ---------- save ----------
df_out = pd.DataFrame(results)
df_out.to_csv(OUT_CSV, index=False)
print("Saved classical results to:", OUT_CSV)

# diagnostics
n_err = df_out['error'].notna().sum() if 'error' in df_out.columns else 0
print(f"Windows processed: {len(df_out)}; windows with error: {n_err}")
if n_err > 0:
    print("Example errors (first 5):")
    print(df_out[df_out['error'].notna()].head(5)[['window_id','error']])
else:
    print("No errors detected in result rows.")
