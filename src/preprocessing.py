# PREPROCESSING

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD

os.environ["OMP_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["NUMEXPR_NUM_THREADS"] = "8"


D_CLS = 12
D_Q   = 4
WINSOR_PCTS = (0.01, 0.99)
TARGET_COL = "isFraud"


def preprocess_window(R_df, N_df,
                      winsorize_pct=WINSOR_PCTS,
                      d_cls=D_CLS,
                      d_q=D_Q):
    """
    UNIVERSAL preprocessing:
    - Works for ANY dataset shape
    - Detects numeric + categorical automatically
    - Median impute, winsorize, freq-encode, scale
    - TruncatedSVD Used
    """


    R = R_df.copy()
    N = N_df.copy()


    if TARGET_COL in R.columns:
        R = R.drop(columns=[TARGET_COL])
    if TARGET_COL in N.columns:
        N = N.drop(columns=[TARGET_COL])


    numeric_cols = R.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols     = R.select_dtypes(exclude=[np.number]).columns.tolist()


    for c in numeric_cols:
        med = R[c].median()
        R[c] = R[c].fillna(med)
        N[c] = N[c].fillna(med)


    lo, hi = winsorize_pct
    for c in numeric_cols:
        lo_v = R[c].quantile(lo)
        hi_v = R[c].quantile(hi)
        R[c] = R[c].clip(lo_v, hi_v)
        N[c] = N[c].clip(lo_v, hi_v)


    freq_maps = {}
    R_freq = {}
    N_freq = {}

    for c in cat_cols:
        freq = R[c].value_counts(normalize=True)
        freq_maps[c] = freq
        
        R_freq[c+"_freq"] = R[c].map(freq).fillna(0.0)
        N_freq[c+"_freq"] = N[c].map(freq).fillna(0.0)

    
    R = pd.concat([R, pd.DataFrame(R_freq)], axis=1)
    N = pd.concat([N, pd.DataFrame(N_freq)], axis=1)

    
    feat_cols = numeric_cols + [c+"_freq" for c in cat_cols]

   
    for c in feat_cols:
        if c not in R:
            R[c] = 0.0
        if c not in N:
            N[c] = 0.0


    scaler = StandardScaler().fit(R[feat_cols].values)
    R_scaled = scaler.transform(R[feat_cols].values)
    N_scaled = scaler.transform(N[feat_cols].values)

    R_mat = np.nan_to_num(R_scaled, nan=0.0, posinf=1e6, neginf=-1e6)
    N_mat = np.nan_to_num(N_scaled, nan=0.0, posinf=1e6, neginf=-1e6)

    d1 = min(d_cls, R_mat.shape[0], R_mat.shape[1])
    d2 = min(d_q,   R_mat.shape[0], R_mat.shape[1])

    svd_cls = TruncatedSVD(n_components=d1, random_state=42)
    R_cls = svd_cls.fit_transform(R_mat)
    N_cls = svd_cls.transform(N_mat)

    svd_q = TruncatedSVD(n_components=d2, random_state=42)
    R_q = svd_q.fit_transform(R_mat)
    N_q = svd_q.transform(N_mat)

    return {
        "R_scaled_df": pd.DataFrame(R_scaled, columns=feat_cols),
        "N_scaled_df": pd.DataFrame(N_scaled, columns=feat_cols),
        "R_cls": R_cls,
        "N_cls": N_cls,
        "R_q": R_q,
        "N_q": N_q,
        "scaler": scaler,
        "svd_cls": svd_cls,
        "svd_q": svd_q,
        "feat_cols": feat_cols,
    }


df = pd.read_csv("qml_drift/data/creditcard.csv")
N_REF = 4000
N_TEST = 1000

R0 = df.iloc[:N_REF].reset_index(drop=True)
N0 = df.iloc[N_REF:N_REF+N_TEST].reset_index(drop=True)

print("Running preprocess_window()...")
res = preprocess_window(R0, N0)

print("R_cls:", res["R_cls"].shape)
print("N_cls:", res["N_cls"].shape)
print("R_q:", res["R_q"].shape)
print("N_q:", res["N_q"].shape)
print("Feature count:", len(res["feat_cols"]))
