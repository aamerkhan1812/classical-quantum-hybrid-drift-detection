# === Hybrid Fusion ===
import os, math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Paths 
CLASSICAL_CSV = "results/creditcard/classical_150_windows/classical_150_windows.csv"
QUANTUM_CSV   = "results/creditcard/quantum_150_run/quantum_results_150_windows.csv"
OUT_DIR = "results/creditcard/fusion"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_CSV = os.path.join(OUT_DIR, "fusion_results_fixed.csv")

# Fusion hyperparams 
TAU = 0.01          # kernel_var scale for reliability weighting
THRESHOLD = 0.6     # decision threshold on S_hybrid
EPS = 1e-12

# Load both CSVs
cls = pd.read_csv(CLASSICAL_CSV)
qnt = pd.read_csv(QUANTUM_CSV)
print("Loaded classical:", CLASSICAL_CSV, "rows:", len(cls))
print("Loaded quantum:  ", QUANTUM_CSV, "rows:", len(qnt))

# Use classical windows as canonical set 
# This ensures we only produce fused rows for windows that classical actually processed.
# Quantum columns will be attached where a matching window_id exists, otherwise NaN.
merged = cls.merge(qnt, how="left", on=["window_id"], suffixes=("_cls","_q"))

# If you want stricter alignment you could verify start_ref/start_test equality and warn if mismatches:
if ("start_ref_cls" in merged.columns) and ("start_ref_q" in merged.columns):
    mism = merged[ (merged["start_ref_cls"].notna()) & (merged["start_ref_q"].notna()) & (merged["start_ref_cls"] != merged["start_ref_q"]) ]
    if len(mism) > 0:
        print(f"Warning: {len(mism)} fused rows have mismatched start_ref between classical and quantum (kept classical values).")

# Fill missing p-values (NaN -> 1.0) and kernel_var (median fallback)
merged["p_cls"] = merged.get("p_cls", merged.get("p_cls_cls", pd.Series(np.nan,index=merged.index))).fillna(1.0)
merged["p_q"]   = merged.get("p_q", merged.get("p_q_q", pd.Series(np.nan,index=merged.index))).fillna(1.0)
# kernel_var may be in quantum file; fall back to median or 0
if "kernel_var" in merged.columns:
    merged["kernel_var"] = merged["kernel_var"].fillna( merged["kernel_var"].median() )
else:
    merged["kernel_var"] = 0.0

# Keep original fusion logic unchanged: confidences, w_q weighting, hybrid score, z-score, decision
merged["C_cls"] = 1.0 - merged["p_cls"]
merged["C_q"] = 1.0 - merged["p_q"]

merged["w_q_raw"] = 1.0 / (1.0 + (merged["kernel_var"] / (TAU + EPS)))
merged["w_q"] = merged["w_q_raw"].clip(0.0, 1.0)
merged["w_cls"] = 1.0 - merged["w_q"]

merged["S_hybrid"] = merged["w_cls"] * merged["C_cls"] + merged["w_q"] * merged["C_q"]

mu = merged["S_hybrid"].mean()
sigma = merged["S_hybrid"].std(ddof=0) if merged["S_hybrid"].std(ddof=0) > 0 else 1.0
merged["S_hybrid_z"] = (merged["S_hybrid"] - mu) / (sigma + EPS)

merged["drift_flag"] = (merged["S_hybrid"] > THRESHOLD).astype(int)

# Save results
merged.to_csv(OUT_CSV, index=False)
print("Saved fusion results to:", OUT_CSV)

# Summaries 
n_total = len(merged)
n_cls = int((merged["p_cls"] < 0.05).sum())
n_q = int((merged["p_q"] < 0.05).sum())
n_hybrid = int(merged["drift_flag"].sum())
print(f"Total windows (classical canonical): {n_total}")
print(f"Classical detections (p_cls < 0.05): {n_cls}")
print(f"Quantum detections (p_q < 0.05): {n_q}")
print(f"Hybrid detections (S_hybrid > {THRESHOLD}): {n_hybrid}")

# Disagreements 
cls_only = merged[(merged["p_cls"] < 0.05) & (merged["drift_flag"] == 0)]
q_only = merged[(merged["p_q"] < 0.05) & (merged["drift_flag"] == 0)]
hybrid_only = merged[(merged["drift_flag"] == 1) & (merged["p_cls"] >= 0.05) & (merged["p_q"] >= 0.05)]

print("\nDisagreements summary:")
print(f"  Classical-only (cls drift & hybrid no): {len(cls_only)}")
print(f"  Quantum-only (q drift & hybrid no): {len(q_only)}")
print(f"  Hybrid-only (hybrid drift but both single detectors no): {len(hybrid_only)}")

# Show samples (up to 5) — identical display fields as your original cell
def show_sample(df_sub, name):
    if len(df_sub) == 0:
        print(f"\n{name}: None")
    else:
        print(f"\n{name}: showing up to 5 rows")
        display(df_sub.sort_values("S_hybrid", ascending=False).head(5)[[
            "window_id","start_ref_cls","start_test_cls","p_cls","p_q","kernel_var","w_q","S_hybrid","S_hybrid_z","drift_flag"
        ]])

show_sample(cls_only, "Classical-only windows")
show_sample(q_only, "Quantum-only windows")
show_sample(hybrid_only, "Hybrid-only windows")

# Reproduce the same plots you had (p_cls, p_q, S_hybrid)
plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.plot(merged["window_id"], merged["p_cls"], marker='o', linestyle='-', markersize=4)
plt.title("p_cls across windows")
plt.xlabel("window_id"); plt.ylabel("p_cls"); plt.ylim(-0.05,1.05)

plt.subplot(1,3,2)
plt.plot(merged["window_id"], merged["p_q"], marker='o', linestyle='-', markersize=4)
plt.title("p_q across windows")
plt.xlabel("window_id"); plt.ylabel("p_q"); plt.ylim(-0.05,1.05)

plt.subplot(1,3,3)
plt.plot(merged["window_id"], merged["S_hybrid"], marker='o', linestyle='-', markersize=4)
plt.axhline(THRESHOLD, color='r', linestyle='--', label=f"threshold={THRESHOLD}")
plt.title("Hybrid score S_hybrid")
plt.xlabel("window_id"); plt.ylabel("S_hybrid")
plt.legend()
plt.tight_layout()
plt.show()

# Scatter: classical vs quantum confidence colored by S_hybrid
plt.figure(figsize=(6,5))
sc = plt.scatter(1.0-merged["p_cls"], 1.0-merged["p_q"], c=merged["S_hybrid"], cmap="viridis", s=40)
plt.colorbar(sc, label="S_hybrid")
plt.xlabel("C_cls = 1 - p_cls")
plt.ylabel("C_q = 1 - p_q")
plt.title("Classical vs Quantum confidence (colored by hybrid)")
plt.grid(alpha=0.2)
plt.show()

print("\nDone. Fusion merge fixed: classical windows used as canonical set.")
