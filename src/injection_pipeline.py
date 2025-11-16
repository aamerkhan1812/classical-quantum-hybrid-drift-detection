# ===== Injection & Detection =====
import os, time, random, traceback
os.environ["OMP_NUM_THREADS"] = "8"          
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["VECLIB_MAXIMUM_THREADS"] = "8"
os.environ["NUMEXPR_NUM_THREADS"] = "8"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import inspect

# ---------- CONFIG ----------
BASE_CSV_FUSION = "results/creditcard/fusion_adaptive/fusion_adaptive.csv" 
OUT_CSV = "results/creditcard/dataset_with_drift/dataset_with_drift.csv"
DF_ALL_PATH = "qml_drift/data/creditcard.csv"
RANDOM_SEED = 42

INJECT_START = 100
INJECT_COUNT = 15

FRACTION = 0.15
MEAN_SHIFT_MULT = 1.5
B_CLS = 300
N_SUB_CLS = 1200
B_Q   = 120
N_SUB_Q = 200
ROLLING_K = 25
ALPHA = 0.9
TAU = 0.01
TAU_GATING = 0.015
N_REF = 4000
N_TEST = 1000
STRIDE = 500



# Quantum runtime knobs 
N_QUBITS = globals().get("N_QUBITS", 4)
REPS = globals().get("REPS", 2)
ANALYTIC = globals().get("ANALYTIC", True)
FEATURE_SCALE = globals().get("FEATURE_SCALE", 1.0)
# ------------------------------------

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ---------- Basic checks ----------
if not os.path.exists(BASE_CSV_FUSION):
    raise FileNotFoundError(f"Baseline fusion CSV not found at: {BASE_CSV_FUSION}")
if not os.path.exists(DF_ALL_PATH):
    raise FileNotFoundError(f"Main dataset not found at: {DF_ALL_PATH}")

# load baseline + dataset
baseline_df = pd.read_csv(BASE_CSV_FUSION)
# --- detect duplicates 
dupes = baseline_df[baseline_df.duplicated(subset="window_id", keep=False)]
if not dupes.empty:
    print(f"[baseline] found {dupes['window_id'].nunique()} duplicate window_id(s). Showing sample rows:")
    display(dupes.sort_values('window_id').head(20))

# --- dedupe baseline, keep last occurrence 
baseline_df = baseline_df.drop_duplicates(subset="window_id", keep="last").reset_index(drop=True)

# confirm
assert not baseline_df['window_id'].duplicated().any(), "Still have duplicate window_id after drop_duplicates!"
print("[baseline] deduped — unique window count:", baseline_df['window_id'].nunique())

df_all = pd.read_csv(DF_ALL_PATH)
n_total = len(df_all)

print(f"Loaded baseline rows: {len(baseline_df)}")
print(f"Loaded main dataset rows: {n_total}")

def window_start_from_index(win_idx, stride=STRIDE):
    return int(win_idx * stride)

def sanitize_df(df):
    df = df.copy()
    df.columns = [str(c) for c in df.columns]
    return df.reset_index(drop=True)

df_all = sanitize_df(df_all)

# ensure preprocess_window exists
if 'preprocess_window' not in globals() or not callable(globals()['preprocess_window']):
    raise RuntimeError("preprocess_window function is not defined in this notebook. Define it before running this cell.")

# prefer existing quantum function(s) if available
compute_statevectors_fn = globals().get("compute_statevectors", None)
quantum_mmd_existing = globals().get("quantum_mmd_and_pvalue", None)

# ===== safe classical helpers =====
def median_heuristic_gamma_sample(X, max_sample=2000, rng=None):
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    if n > max_sample:
        idx = (rng.choice(n, size=max_sample, replace=False) if rng is not None else np.random.choice(n, size=max_sample, replace=False))
        sub = X[idx]
    else:
        sub = X
    if sub.shape[0] < 2:
        return 1.0
    # pairwise pdist via efficient approach
    d2 = np.sum((sub[:,None,:] - sub[None,:,:])**2, axis=-1)
    d = np.sqrt(d2)
    d_flat = d[np.triu_indices_from(d, k=1)]
    if d_flat.size == 0:
        return 1.0
    med = np.median(d_flat)
    if med <= 0:
        return 1.0
    return 1.0 / (2.0 * (med**2))

def rbf_kernel(X, Y, gamma):
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    XX = np.sum(X*X, axis=1)[:,None]
    YY = np.sum(Y*Y, axis=1)[None,:]
    D2 = XX + YY - 2.0 * X.dot(Y.T)
    np.maximum(D2, 0, out=D2)
    return np.exp(-gamma * D2)

def mmd_unbiased_safe_from_arrays(X, Y, gamma=None, sample_for_gamma=2000, subsample=None, rng=None):
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if X.size == 0 or Y.size == 0:
        return 0.0, float(gamma if gamma is not None else 1.0)
    if subsample is not None and subsample > 0:
        # balanced subsample
        total = subsample
        nx, ny = X.shape[0], Y.shape[0]
        frac = float(nx) / max(1, nx+ny)
        nx_s = max(1, int(round(total*frac)))
        ny_s = max(1, total - nx_s)
        nx_s = min(nx_s, nx); ny_s = min(ny_s, ny)
        ix = (rng.choice(nx, size=nx_s, replace=False) if rng is not None else np.random.choice(nx, size=nx_s, replace=False))
        iy = (rng.choice(ny, size=ny_s, replace=False) if rng is not None else np.random.choice(ny, size=ny_s, replace=False))
        Xs = X[ix]
        Ys = Y[iy]
    else:
        Xs = X; Ys = Y
    if gamma is None:
        gamma = median_heuristic_gamma_sample(np.vstack([Xs, Ys]), max_sample=sample_for_gamma, rng=rng)
    Kxx = rbf_kernel(Xs, Xs, gamma)
    Kyy = rbf_kernel(Ys, Ys, gamma)
    Kxy = rbf_kernel(Xs, Ys, gamma)
    n = Kxx.shape[0]; m = Kyy.shape[0]
    if n < 2 and m < 2:
        return 0.0, float(gamma)
    sum_xx = (np.sum(Kxx) - np.trace(Kxx)) / (n * (n - 1) + 1e-12) if n > 1 else 0.0
    sum_yy = (np.sum(Kyy) - np.trace(Kyy)) / (m * (m - 1) + 1e-12) if m > 1 else 0.0
    sum_xy = np.sum(Kxy) / (n * m + 1e-12)
    return float(sum_xx + sum_yy - 2.0 * sum_xy), float(gamma)

def permutation_p_value_safe(X, Y, stat_func, B=100, seed=42, rng=None):
    rng = (np.random.RandomState(seed) if rng is None else rng)
    Z = np.vstack([X, Y])
    n = X.shape[0]; m = Y.shape[0]
    if n == 0 or m == 0:
        return 0.0, 1.0, np.array([])
    obs = float(stat_func(X, Y))
    cnt = 0
    stats = []
    for b in range(B):
        idx = rng.permutation(n+m)
        Xp = Z[idx[:n]]
        Yp = Z[idx[n:]]
        s = float(stat_func(Xp, Yp))
        stats.append(s)
        if s >= obs:
            cnt += 1
    pval = (cnt + 1) / (B + 1)
    return float(obs), float(pval), np.array(stats, dtype=float)

# ===== safe quantum wrapper that prefers existing functions =====

def quantum_wrapper_safe(R_q, N_q, n_sub=N_SUB_Q, B=B_Q, seed=RANDOM_SEED, profile=False):
    """
    Robust quantum wrapper:
    - If existing high-level quantum_mmd_and_pvalue works and input sizes >=2, use it (guarded).
    - Else if compute_statevectors is available, call it with compatible args (inspect signature),
      build kernel from states, run kernel-level permutation safely.
    - Otherwise return conservative output (mmd=0, p=1).
    """
    # ensure arrays
    try:
        R_q = np.atleast_2d(np.asarray(R_q, dtype=float))
        N_q = np.atleast_2d(np.asarray(N_q, dtype=float))
    except Exception:
        return 0.0, 1.0, {"n_sub_total":0,"n_sub_X":0,"n_sub_Y":0,"kernel_var":np.nan}

    nx, dx = (R_q.shape[0], R_q.shape[1]) if R_q.size>0 else (0, R_q.shape[1] if R_q.ndim>1 else 0)
    ny = N_q.shape[0] if N_q.size>0 else 0

    # If either side has <2 samples, calling kernel-based permutation/statistics is unstable:
    if nx < 2 or ny < 2:
        
        if callable(quantum_mmd_existing):
            try:
                return quantum_mmd_existing(R_q, N_q, n_qubits=dx, reps=REPS, analytic=ANALYTIC,
                                             n_sub=n_sub, B=B, random_seed=seed, profile=profile)
            except Exception as e:
                
                return 0.0, 1.0, {"n_sub_total":0,"n_sub_X":nx,"n_sub_Y":ny,"kernel_var":np.nan}
        else:
            return 0.0, 1.0, {"n_sub_total":0,"n_sub_X":nx,"n_sub_Y":ny,"kernel_var":np.nan}

    
    if callable(quantum_mmd_existing):
        try:
            return quantum_mmd_existing(R_q, N_q, n_qubits=dx, reps=REPS, analytic=ANALYTIC,
                                         n_sub=n_sub, B=B, random_seed=seed, profile=profile)
        except Exception as e:
            
            print("[quantum_wrapper_safe] existing quantum_mmd failed:", e)
            

    
    if callable(compute_statevectors_fn):
        try:
            
            total_sub = int(n_sub) if n_sub is not None and n_sub>0 else (nx + ny)
            frac = float(nx) / (nx + ny)
            nx_s = max(2, min(nx, int(round(total_sub * frac)))) 
            ny_s = max(2, min(ny, int(round(total_sub - nx_s))))
            rng = np.random.RandomState(seed)
            ix = rng.choice(nx, size=nx_s, replace=False)
            iy = rng.choice(ny, size=ny_s, replace=False)
            Xs = R_q[ix]
            Ys = N_q[iy]
            if Xs.shape[0] < 2 or Ys.shape[0] < 2:
                return 0.0, 1.0, {"n_sub_total":0,"n_sub_X":len(ix),"n_sub_Y":len(iy),"kernel_var":np.nan}

            Z = np.vstack([Xs, Ys])

            
            sig = None
            try:
                sig = inspect.signature(compute_statevectors_fn)
            except Exception:
                sig = None

            
            call_kwargs = {}
            
            possible_kwargs = {
                "n_qubits": dx,
                "reps": REPS,
                "analytic": ANALYTIC,
                "feature_scale": FEATURE_SCALE,
                "profile": profile
            }
            if sig is not None:
                for name in possible_kwargs:
                    if name in sig.parameters:
                        call_kwargs[name] = possible_kwargs[name]

            
            try:
                states = compute_statevectors_fn(Z, **call_kwargs) if len(call_kwargs)>0 else compute_statevectors_fn(Z)
            except TypeError:
                
                try:
                    states = compute_statevectors_fn(Z, dx, REPS, ANALYTIC)
                except Exception as e2:
                    print("[quantum_wrapper_safe] compute_statevectors failed with positional fallback:", e2)
                    raise

            states = np.asarray(states)
           
            if states.ndim == 1:
                states = states.reshape(1, -1)
            if states.ndim == 2 and states.shape[0] != Z.shape[0]:
                
                raise ValueError("compute_statevectors returned unexpected shape")

            
            K = np.abs(states @ states.conj().T) ** 2
            idx_X = np.arange(0, Xs.shape[0])
            idx_Y = np.arange(Xs.shape[0], Xs.shape[0] + Ys.shape[0])

            
            n = len(idx_X); m = len(idx_Y)
            if n < 2 or m < 2:
                return 0.0, 1.0, {"n_sub_total":K.shape[0],"n_sub_X":n,"n_sub_Y":m,"kernel_var":np.nan}

            Kxx = K[idx_X][:, idx_X]
            Kyy = K[idx_Y][:, idx_Y]
            Kxy = K[idx_X][:, idx_Y]
            sum_xx = (Kxx.sum() - np.trace(Kxx)) / (n * (n - 1) + 1e-12)
            sum_yy = (Kyy.sum() - np.trace(Kyy)) / (m * (m - 1) + 1e-12)
            sum_xy = Kxy.sum() / (n * m + 1e-12)
            obs = float(sum_xx + sum_yy - 2.0*sum_xy)

            
            cnt = 0; stats=[]
            Ntot = K.shape[0]
            for b in range(min(B, 500)):
                perm = rng.permutation(Ntot)
                px = perm[:n]; py = perm[n:n+m]
                Kxx_p = K[px][:, px]; Kyy_p = K[py][:, py]; Kxy_p = K[px][:, py]
                s = float((Kxx_p.sum() - np.trace(Kxx_p)) / (n*(n-1)+1e-12) + (Kyy_p.sum() - np.trace(Kyy_p)) / (m*(m-1)+1e-12) - 2.0*(Kxy_p.sum()/(n*m+1e-12)))
                stats.append(s)
                if s >= obs: cnt += 1
            pval = (cnt + 1) / (len(stats) + 1) if len(stats)>0 else 1.0

            Kxy_flat = K[np.ix_(idx_X, idx_Y)].flatten()
            meta = {"n_sub_total": int(K.shape[0]), "n_sub_X": int(n), "n_sub_Y": int(m), "kernel_var": float(np.var(Kxy_flat)) if Kxy_flat.size>0 else float(np.nan)}
            return float(obs), float(pval), meta

        except Exception as e:
            print("[quantum_wrapper_safe] compute_statevectors path failed:", e)
            
            return 0.0, 1.0, {"n_sub_total":0,"n_sub_X":0,"n_sub_Y":0,"kernel_var":np.nan}

    # final fallback
    return 0.0, 1.0, {"n_sub_total":0,"n_sub_X":nx,"n_sub_Y":ny,"kernel_var":np.nan}
neg_records = []
new_records = []
t0_all = time.time()

# ---- negatives ----
for win in tqdm(range(INJECT_START, INJECT_START + INJECT_COUNT), desc="negatives"):
    try:
        start_ref = window_start_from_index(win)
        start_test = start_ref + N_REF
        end_test = start_test + N_TEST
        if end_test > n_total:
            break

        R_df = sanitize_df(df_all.iloc[start_ref:start_test].reset_index(drop=True))
        N_df = sanitize_df(df_all.iloc[start_test:end_test].reset_index(drop=True))
        for c in ['payment_type','employment_status','housing_status','source','device_os']:
            if c not in R_df.columns: R_df[c] = ""
            if c not in N_df.columns: N_df[c] = ""

        pre = preprocess_window(R_df, N_df)
        if 'R_cls' not in pre or 'N_cls' not in pre:
            print(f"  Negative window {win}: preprocess_window missing keys, skipping")
            continue

        R_cls = np.asarray(pre['R_cls'], dtype=float)
        N_cls = np.asarray(pre['N_cls'], dtype=float)

        rng_local = np.random.RandomState(RANDOM_SEED + win)
        mmd_cls, gamma_used = mmd_unbiased_safe_from_arrays(R_cls, N_cls, gamma=None, sample_for_gamma=2000, subsample=N_SUB_CLS, rng=rng_local)
        obs_cls, p_cls, stats_cls = permutation_p_value_safe(R_cls, N_cls, lambda a,b: mmd_unbiased_safe_from_arrays(a,b,gamma=gamma_used, subsample=None, rng=rng_local)[0], B=min(B_CLS,200), seed=RANDOM_SEED+win, rng=rng_local)

        R_q = pre.get('R_q', np.empty((0, N_QUBITS)))
        N_q = pre.get('N_q', np.empty((0, N_QUBITS)))
        q_mmd, q_p, qmeta = quantum_wrapper_safe(R_q, N_q, n_sub=N_SUB_Q, B=min(B_Q,200), seed=RANDOM_SEED+win, profile=False)

        neg_records.append({
            "window_id": int(win),
            "start_ref": int(start_ref),
            "start_test": int(start_test),
            "end_test": int(end_test),
            "drift_injected": 0,
            "drift_feat": "",
            "drift_fraction": 0.0,
            "drift_delta": 0.0,
            "mmd_stat_cls": float(mmd_cls),
            "p_cls": float(p_cls),
            "mmd_stat_q": float(q_mmd),
            "p_q": float(q_p),
            "kernel_var": float(qmeta.get("kernel_var", np.nan))
        })
        print(f"  Negative window {win}: p_cls={p_cls:.4f} p_q={q_p:.4f}")

    except Exception as e:
        print(f"  Error (neg) at window {win}: {e}")
        traceback.print_exc()
        continue

# ---- injections ----
# This block injects mean-shift drift into the test window (N_df) for INJECT_COUNT windows.
for win in tqdm(range(INJECT_START, INJECT_START + INJECT_COUNT), desc="injects"):
    try:
        start_ref = window_start_from_index(win)
        start_test = start_ref + N_REF
        end_test = start_test + N_TEST
        if end_test > n_total:
            break


        R_df = sanitize_df(df_all.iloc[start_ref:start_test].reset_index(drop=True))
        N_df_orig = sanitize_df(df_all.iloc[start_test:end_test].reset_index(drop=True))

        
        for c in ['payment_type','employment_status','housing_status','source','device_os']:
            if c not in R_df.columns: R_df[c] = ""
            if c not in N_df_orig.columns: N_df_orig[c] = ""

        
        N_df = N_df_orig.copy()

        
        num_cols = N_df.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) > 0 and FRACTION > 0:
            rng_local = np.random.RandomState(RANDOM_SEED + win)
            n_rows = N_df.shape[0]
            n_shift = max(1, int(round(FRACTION * n_rows)))
            idx_shift = rng_local.choice(n_rows, size=n_shift, replace=False)
            
            for col in num_cols:
                col_mean = R_df[col].mean() if col in R_df.columns else N_df[col].mean()
                N_df.loc[idx_shift, col] = N_df.loc[idx_shift, col] + (MEAN_SHIFT_MULT * col_mean)

        
        pre = preprocess_window(R_df, N_df)
        if 'R_cls' not in pre or 'N_cls' not in pre:
            print(f"  Inject window {win}: preprocess_window missing keys, skipping")
            continue

        R_cls = np.asarray(pre['R_cls'], dtype=float)
        N_cls = np.asarray(pre['N_cls'], dtype=float)

        rng_local = np.random.RandomState(RANDOM_SEED + win)
        mmd_cls, gamma_used = mmd_unbiased_safe_from_arrays(R_cls, N_cls, gamma=None,
                                                            sample_for_gamma=2000, subsample=N_SUB_CLS, rng=rng_local)
        obs_cls, p_cls, stats_cls = permutation_p_value_safe(R_cls, N_cls,
            lambda a,b: mmd_unbiased_safe_from_arrays(a,b,gamma=gamma_used,subsample=None,rng=rng_local)[0],
            B=min(B_CLS,200), seed=RANDOM_SEED+win, rng=rng_local)

        R_q = pre.get('R_q', np.empty((0, N_QUBITS)))
        N_q = pre.get('N_q', np.empty((0, N_QUBITS)))
        q_mmd, q_p, qmeta = quantum_wrapper_safe(R_q, N_q, n_sub=N_SUB_Q, B=min(B_Q,200), seed=RANDOM_SEED+win, profile=False)

        new_records.append({
            "window_id": int(win),
            "start_ref": int(start_ref),
            "start_test": int(start_test),
            "end_test": int(end_test),
            "drift_injected": 1,
            "drift_feat": ",".join(num_cols) if num_cols else "",
            "drift_fraction": float(FRACTION),
            "drift_delta": float(MEAN_SHIFT_MULT),
            "mmd_stat_cls": float(mmd_cls),
            "p_cls": float(p_cls),
            "mmd_stat_q": float(q_mmd),
            "p_q": float(q_p),
            "kernel_var": float(qmeta.get("kernel_var", np.nan))
        })
        print(f"  Injected window {win}: p_cls={p_cls:.4f} p_q={q_p:.4f}")

    except Exception as e:
        print(f"  Error (inject) at window {win}: {e}")
        traceback.print_exc()
        continue


t_elapsed = time.time() - t0_all
print(f"\nDone evaluation. Injected evaluated: {len(new_records)} negatives: {len(neg_records)}  time: {t_elapsed:.1f}s")

# ---- Combine with baseline fusion and recompute adaptive fusion scores ----
new_df = pd.DataFrame(new_records)
neg_df = pd.DataFrame(neg_records)


append_list = []
for df_src in [neg_df, new_df]:
    if not df_src.empty:
        append_list.append(df_src[['window_id','p_cls','p_q','kernel_var']])
append_df = pd.concat(append_list, ignore_index=True).sort_values('window_id').reset_index(drop=True) if append_list else pd.DataFrame(columns=['window_id','p_cls','p_q','kernel_var'])

# Use a copy of baseline (ensure deduped)
hist_df = baseline_df.copy()
if hist_df['window_id'].duplicated(keep=False).any():
    print("[combine] WARNING: baseline still has duplicate window_id values; deduping by keeping last.")
    hist_df = hist_df.drop_duplicates(subset='window_id', keep='last').reset_index(drop=True)

hist_df = hist_df.set_index('window_id')

if not append_df.empty:
    append_df = append_df.set_index('window_id')
    upd = append_df[['p_cls','p_q','kernel_var']].copy()
    # restrict to window_ids present in baseline
    upd = upd.loc[upd.index.intersection(hist_df.index)]
    # assign per-column which avoids update() reindex internals
    for col in ['p_cls','p_q','kernel_var']:
        if col in upd.columns:
            hist_df.loc[upd.index, col] = upd[col]

# ensure drift_injected exists and mark injected windows
if 'drift_injected' not in hist_df.columns:
    hist_df['drift_injected'] = 0
else:
    hist_df['drift_injected'] = hist_df['drift_injected'].fillna(0).astype(int)

injected_window_ids = list(new_df['window_id'].astype(int).values) if not new_df.empty else []
existing = [wid for wid in injected_window_ids if wid in hist_df.index]
if existing:
    hist_df.loc[existing, 'drift_injected'] = 1


hist_combined = hist_df.reset_index().sort_values('window_id').reset_index(drop=True)



hist_combined['p_cls'] = hist_combined['p_cls'].fillna(1.0)
hist_combined['p_q'] = hist_combined['p_q'].fillna(1.0)
hist_combined['kernel_var'] = hist_combined['kernel_var'].fillna(hist_combined['kernel_var'].median() if not hist_combined['kernel_var'].isna().all() else 0.0)


hist_combined['C_cls'] = 1.0 - hist_combined['p_cls']
hist_combined['C_q'] = 1.0 - hist_combined['p_q']
hist_combined['w_q_prior'] = 1.0/(1.0 + (hist_combined['kernel_var']/(TAU+1e-12)))
hist_combined['w_q_prior'] = hist_combined['w_q_prior'].clip(0,1)
hist_combined['gating'] = hist_combined['kernel_var'].apply(lambda kv: 1.0/(1.0 + (kv/(TAU_GATING+1e-12))))
hist_combined['w_q_gated'] = (hist_combined['w_q_prior'] * hist_combined['gating']).clip(0,1)
hist_combined['w_cls_gated'] = 1.0 - hist_combined['w_q_gated']
hist_combined['S_hybrid_robust'] = hist_combined['w_cls_gated'] * hist_combined['C_cls'] + hist_combined['w_q_gated'] * hist_combined['C_q']

hist_combined['S_roll_mean'] = hist_combined['S_hybrid_robust'].rolling(window=ROLLING_K, min_periods=1).mean()
hist_combined['S_roll_std'] = hist_combined['S_hybrid_robust'].rolling(window=ROLLING_K, min_periods=1).std(ddof=0).fillna(0.0)
hist_combined['T_rolling'] = hist_combined['S_roll_mean'] + ALPHA * hist_combined['S_roll_std']
gm = hist_combined['S_hybrid_robust'].mean(); gs = hist_combined['S_hybrid_robust'].std(ddof=0) if hist_combined['S_hybrid_robust'].std(ddof=0)>0 else 1.0
hist_combined.loc[hist_combined.index < MIN_WINDOWS_FOR_ROLL, 'T_rolling'] = gm + ALPHA * gs
hist_combined['drift_flag_rolling'] = (hist_combined['S_hybrid_robust'] > hist_combined['T_rolling']).astype(int)
hist_combined['drift_flag_fixed'] = (hist_combined['S_hybrid'] > 0.6).astype(int) if 'S_hybrid' in hist_combined.columns else (hist_combined['S_hybrid_robust'] > 0.6).astype(int)


# Save
os.makedirs(os.path.dirname(OUT_CSV) or ".", exist_ok=True)
if 'drift_injected' not in hist_combined.columns:
    hist_combined['drift_injected'] = 0
else:
    hist_combined['drift_injected'] = hist_combined['drift_injected'].fillna(0).astype(int)

hist_combined.to_csv(OUT_CSV, index=False)
print(f"\nSaved combined history to: {OUT_CSV}")

# Summaries
injected_window_ids = list(new_df['window_id'].astype(int).values) if not new_df.empty else []
detected_injected = hist_combined[hist_combined['window_id'].isin(injected_window_ids) & (hist_combined['drift_flag_rolling']==1)]
print(f"\nInjected windows: {len(injected_window_ids)}")
print(f"Detected injected windows by rolling fusion: {len(detected_injected)}")
print("Detected window ids:", list(detected_injected['window_id'].astype(int).values))

# Quick plot 
plt.figure(figsize=(10,4))
plt.plot(hist_combined['window_id'], hist_combined['S_hybrid_robust'], marker='o', ms=3, label='S_hybrid_robust')
plt.plot(hist_combined['window_id'], hist_combined['T_rolling'], linestyle='--', color='r', label='T_rolling')
if injected_window_ids:
    plt.scatter(hist_combined[hist_combined['window_id'].isin(injected_window_ids)]['window_id'],
                hist_combined[hist_combined['window_id'].isin(injected_window_ids)]['S_hybrid_robust'],
                color='orange', s=80, label='injected')
plt.scatter(hist_combined[hist_combined['drift_flag_rolling']==1]['window_id'],
            hist_combined[hist_combined['drift_flag_rolling']==1]['S_hybrid_robust'],
            color='red', s=50, label='detections')
plt.legend(); plt.xlabel('window_id'); plt.ylabel('S_hybrid_robust'); plt.title('Adaptive fusion (post-injection)')
plt.grid(alpha=0.2); plt.show()

print("\nComplete.")
