# === Adaptive Threshold (rolling) fusion ===
import os, math, pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ------- Paths -------
IN_FUSION_CSV = "results/creditcard/fusion/fusion_results_fixed.csv"
OUT_DIR = "results/creditcard/fusion_adaptive"
OUT_CSV = os.path.join(OUT_DIR, "fusion_adaptive.csv")
os.makedirs(OUT_DIR, exist_ok=True)

# ------- Parameters (tweakable) -------
ROLLING_K = 25         # number of windows in rolling buffer
ALPHA = 0.9            # threshold in std devs above rolling mean
TAU_GATING = 0.015      # kernel_var scale for reliability gating
MIN_WINDOWS_FOR_ROLL = 5  # minimum windows before using rolling stats
EPS = 1e-12

# ------- Load fusion file -------
if not os.path.exists(IN_FUSION_CSV):
    raise FileNotFoundError(f"Input fusion CSV not found: {IN_FUSION_CSV}")

df = pd.read_csv(IN_FUSION_CSV)
print("Loaded fusion file:", IN_FUSION_CSV, "rows:", len(df))

# ------- Defensive column checks / fill -------
# Required-ish columns: window_id, S_hybrid, kernel_var, p_cls, p_q.
# w_q/w_cls may or may not exist; compute if missing using kernel_var as prior.
if "window_id" not in df.columns:
    raise ValueError("Input fusion CSV must contain 'window_id' column.")

# Fill missing p-values with 1.0 (no confidence)
if "p_cls" not in df.columns:
    print("Warning: 'p_cls' missing, creating default (1.0) column.")
    df["p_cls"] = 1.0
else:
    df["p_cls"] = df["p_cls"].fillna(1.0)

if "p_q" not in df.columns:
    print("Warning: 'p_q' missing, creating default (1.0) column.")
    df["p_q"] = 1.0
else:
    df["p_q"] = df["p_q"].fillna(1.0)

# kernel_var fallback: fill NaN with median (or 0)
if "kernel_var" not in df.columns:
    print("Warning: 'kernel_var' missing, creating default (0.0) column.")
    df["kernel_var"] = 0.0
else:
    df["kernel_var"] = df["kernel_var"].fillna(df["kernel_var"].median() if not df["kernel_var"].isna().all() else 0.0)

# S_hybrid fallback: if missing, try compute from available w_q/w_cls and p-values
if "S_hybrid" not in df.columns:
    print("Warning: 'S_hybrid' missing — attempting to compute using p-values and w_q/w_cls or kernel_var.")
    # compute confidences
    df["C_cls"] = 1.0 - df["p_cls"]
    df["C_q"] = 1.0 - df["p_q"]
    # if w_q present use it, else derive prior weight from kernel_var
    if "w_q" in df.columns:
        df["w_q"] = df["w_q"].fillna(1.0/(1.0 + df["kernel_var"]/(0.01 + EPS)))
    else:
        # prior w_q from kernel_var (same as earlier logic)
        TAU_PRIOR = 0.01
        df["w_q"] = 1.0 / (1.0 + (df["kernel_var"] / (TAU_PRIOR + EPS)))
    df["w_q"] = df["w_q"].clip(0.0, 1.0)
    df["w_cls"] = 1.0 - df["w_q"]
    df["S_hybrid"] = df["w_cls"] * df["C_cls"] + df["w_q"] * df["C_q"]
else:
    # ensure S_hybrid numeric and fill NaN with 0
    df["S_hybrid"] = pd.to_numeric(df["S_hybrid"], errors="coerce").fillna(0.0)

# Ensure w_q/w_cls exist (if not, compute prior from kernel_var)
if "w_q" not in df.columns:
    TAU_PRIOR = 0.01
    df["w_q"] = 1.0 / (1.0 + (df["kernel_var"] / (TAU_PRIOR + EPS)))
    df["w_q"] = df["w_q"].clip(0.0, 1.0)
if "w_cls" not in df.columns:
    df["w_cls"] = 1.0 - df["w_q"]

# Copy frame to avoid accidental mutation upstream
df2 = df.copy().reset_index(drop=True)

# ------- Robust hybrid recomputation with reliability gating -------
def compute_robust_hybrid_row(row, tau=TAU_GATING, eps=EPS):
    kv = float(row["kernel_var"]) if not pd.isna(row["kernel_var"]) else 0.0
    # gating factor in (0,1]
    gating = 1.0 / (1.0 + (kv / (tau + eps)))
    C_cls = 1.0 - float(row.get("p_cls", 1.0))
    C_q = 1.0 - float(row.get("p_q", 1.0))
    prior_wq = float(row.get("w_q", 0.5))
    w_q_new = prior_wq * gating
    w_q_new = min(max(w_q_new, 0.0), 1.0)
    w_cls_new = 1.0 - w_q_new
    S_robust = w_cls_new * C_cls + w_q_new * C_q
    return pd.Series({
        "w_q_gated": w_q_new,
        "w_cls_gated": w_cls_new,
        "S_hybrid_robust": S_robust,
        "gating": gating
    })

# apply row-wise (vectorized apply is fine here; df likely small ~150-1991 windows)
gated_df = df2.apply(lambda r: compute_robust_hybrid_row(r, tau=TAU_GATING), axis=1)
df2 = pd.concat([df2.reset_index(drop=True), gated_df.reset_index(drop=True)], axis=1)

# ------- Rolling statistics and adaptive thresholding -------
df2["S_roll_mean"] = df2["S_hybrid_robust"].rolling(window=ROLLING_K, min_periods=1).mean()
df2["S_roll_std"] = df2["S_hybrid_robust"].rolling(window=ROLLING_K, min_periods=1).std(ddof=0).fillna(0.0)
df2["T_rolling"] = df2["S_roll_mean"] + ALPHA * df2["S_roll_std"]

# fallback for very early windows
global_mu = df2["S_hybrid_robust"].mean()
global_sigma = df2["S_hybrid_robust"].std(ddof=0) if df2["S_hybrid_robust"].std(ddof=0) > 0 else 1.0
df2.loc[df2.index < MIN_WINDOWS_FOR_ROLL, "T_rolling"] = global_mu + ALPHA * global_sigma

# Decision flags
df2["drift_flag_rolling"] = (df2["S_hybrid_robust"] > df2["T_rolling"]).astype(int)
if "drift_flag" not in df2.columns:
    # keep your original fixed threshold comparison for compatibility
    df2["drift_flag_fixed"] = (df2["S_hybrid"] > 0.6).astype(int)
else:
    df2["drift_flag_fixed"] = df2["drift_flag"].astype(int)

# ------- Summaries -------
n_total = len(df2)
n_fixed = int(df2["drift_flag_fixed"].sum())
n_rolling = int(df2["drift_flag_rolling"].sum())

print(f"Total windows: {n_total}")
print(f"Fixed-threshold detections (threshold=0.6): {n_fixed}")
print(f"Rolling adaptive detections (alpha={ALPHA}, K={ROLLING_K}): {n_rolling}")

# Show first rows for inspection
cols_show = ["window_id","S_hybrid","S_hybrid_robust","S_roll_mean","S_roll_std","T_rolling","drift_flag_fixed","drift_flag_rolling","kernel_var","w_q","w_q_gated"]
existing_show = [c for c in cols_show if c in df2.columns]
display(df2[existing_show].head(20))

# ------- Save results -------
df2.to_csv(OUT_CSV, index=False)
print("Saved adaptive fusion CSV:", OUT_CSV)

# ------- Plots for inspection -------
plt.figure(figsize=(12,5))
plt.plot(df2["window_id"], df2["S_hybrid_robust"], marker='o', ms=4, label="S_hybrid_robust")
plt.plot(df2["window_id"], df2["S_roll_mean"], linestyle='-', label="rolling mean")
plt.plot(df2["window_id"], df2["T_rolling"], linestyle='--', color='r', label=f"rolling thresh (alpha={ALPHA})")
plt.scatter(df2[df2["drift_flag_rolling"]==1]["window_id"], df2[df2["drift_flag_rolling"]==1]["S_hybrid_robust"], color='red', s=60, label="rolling detections")
plt.legend()
plt.xlabel("window_id"); plt.ylabel("hybrid score"); plt.title("Adaptive Hybrid Score and Rolling Threshold")
plt.grid(alpha=0.2)
plt.show()

plt.figure(figsize=(8,4))
if "kernel_var" in df2.columns:
    plt.plot(df2["window_id"], df2["kernel_var"], marker='.', ms=4)
    plt.title("kernel_var across windows (for gating reference)")
    plt.xlabel("window_id"); plt.ylabel("kernel_var")
    plt.grid(alpha=0.2)
    plt.show()
else:
    print("No kernel_var column to plot.")

# ------- Quick diagnostics: gating effects -------
if ("w_q" in df2.columns) and ("w_q_gated" in df2.columns):
    df2["wq_diff"] = (df2["w_q"] - df2["w_q_gated"]).abs()
    significant_gates = df2[df2["wq_diff"] > 0.05].sort_values("wq_diff", ascending=False).head(10)
    if not significant_gates.empty:
        print("Top windows where gating reduced quantum weight significantly (showing up to 10):")
        display(significant_gates[["window_id","w_q","w_q_gated","gating","kernel_var","S_hybrid","S_hybrid_robust"]])
    else:
        print("No windows with significant gating adjustments (quantum weight stable).")

print("\nAdaptive fusion complete. If you want to change behavior, adjust ROLLING_K, ALPHA, or TAU_GATING and re-run this cell.")
