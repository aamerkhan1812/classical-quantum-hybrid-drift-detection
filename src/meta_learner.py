# FP-Constrained Meta-Learner 
import os, pickle, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_auc_score, average_precision_score

# --- CONFIG ---
CSV_PATH = "results/creditcard/dataset_with_drift/dataset_with_drift.csv"
OUT_DIR = "results/creditcard/results/full_run_fp_constrained"
os.makedirs(OUT_DIR, exist_ok=True)

N_SPLITS = 5
SEED = 42
FP_RATE_CAP = 0.05   
# ----------------

# 1) load
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"CSV not found: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)
print("Loaded:", CSV_PATH, "shape:", df.shape)

# 2) build features 
def safe_neglog(p):
    p = np.asarray(p, dtype=float)
    p = np.clip(p, 1e-12, 1.0)
    return -np.log(p)

feat_candidates = []

for col in ["S_hybrid_robust", "S_roll_mean", "S_roll_std", "T_rolling", "w_q_gated", "kernel_var", "p_cls", "p_q"]:
    if col in df.columns:
        feat_candidates.append(col)

if "p_cls" in df.columns:
    df["neglog_p_cls"] = safe_neglog(df["p_cls"])
    feat_candidates.append("neglog_p_cls")
if "p_q" in df.columns:
    df["neglog_p_q"] = safe_neglog(df["p_q"])
    feat_candidates.append("neglog_p_q")

features = []
for f in feat_candidates:
    if f in df.columns and f not in features:
        features.append(f)

if len(features) == 0:
    raise ValueError("No features available. Need at least p_cls or S_hybrid_robust in CSV.")

print("Using features:", features)

# 3) prepare X, y
if "drift_injected" in df.columns:
    y = df["drift_injected"].astype(int).values
else:
    
    for lbl in ["drift_label", "drift_flag_rolling", "drift_flag", "drift", "label"]:
        if lbl in df.columns:
            y = df[lbl].astype(int).values
            print("Using alternative label:", lbl)
            break
    else:
        raise ValueError("No label column found (drift_injected/drift_label/drift_flag).")

X = df[features].copy().fillna(df[features].median())

# standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4) OOF calibrated probabilities via cross_val_predict
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
base_clf = LogisticRegression(class_weight='balanced', C=0.5, max_iter=2000, random_state=SEED)

calibrator = CalibratedClassifierCV(base_clf, cv=skf, method='sigmoid')
print("Computing OOF calibrated probabilities (this may take a moment)...")

y_proba_oof = cross_val_predict(calibrator, X_scaled, y, cv=skf, method='predict_proba', n_jobs=1)[:,1]


# 5) choose threshold subject to FP cap
n_total = len(y)
n_neg = int((y==0).sum())
max_fp = int(np.floor(FP_RATE_CAP * int((y==0).sum())))


print(f"Total windows: {n_total}  FP cap (<= {FP_RATE_CAP*100:.1f}%): {max_fp}")

# iterate thresholds and pick highest recall (TPR) subject to FP <= max_fp
best_thr = None
best_recall = -1.0
best_prec = 0.0
best_stats = None

# we examine candidate thresholds from unique probs plus fine grid
cands = np.unique(np.concatenate([np.linspace(0,1,1001), y_proba_oof, np.array([0.001,0.005,0.01,0.02,0.05,0.1])]))
cands = np.sort(cands)[::-1]  # high to low to favor higher recall early

for t in cands:
    preds = (y_proba_oof >= t).astype(int)
    FP = int(((preds==1) & (y==0)).sum())
    TP = int(((preds==1) & (y==1)).sum())
    FN = int(((preds==0) & (y==1)).sum())
    TN = int(((preds==0) & (y==0)).sum())
    recall = TP / (TP + FN + 1e-12)
    if FP <= max_fp:
        # pick threshold that maximizes recall; if tie, maximize precision
        prec = TP / (TP + FP + 1e-12)
        if recall > best_recall or (np.isclose(recall, best_recall) and prec > best_prec):
            best_recall = recall
            best_prec = prec
            best_thr = float(t)
            best_stats = {"TP":TP,"FP":FP,"TN":TN,"FN":FN,"precision":prec,"recall":recall}

if best_thr is None:
    # no threshold satisfied FP cap — fallback: choose threshold with smallest FP
    print("No threshold met FP cap. Falling back to threshold that minimizes FP.")
    unique_ts = np.unique(y_proba_oof)
    best_thr = float(np.max(unique_ts))  # most conservative
    preds = (y_proba_oof >= best_thr).astype(int)
    FP = int(((preds==1) & (y==0)).sum())
    TP = int(((preds==1) & (y==1)).sum())
    FN = int(((preds==0) & (y==1)).sum())
    TN = int(((preds==0) & (y==0)).sum())
    best_stats = {"TP":TP,"FP":FP,"TN":TN,"FN":FN,"precision":TP/(TP+FP+1e-12),"recall":TP/(TP+FN+1e-12)}

print("Chosen threshold:", best_thr)
print("OOF stats at chosen threshold:", best_stats)
print("OOF ROC-AUC:", roc_auc_score(y, y_proba_oof))
print("OOF PR-AUC:", average_precision_score(y, y_proba_oof))

# 6) Fit final calibrated classifier on full data 
final_cal = CalibratedClassifierCV(base_clf, cv=skf, method='sigmoid')
final_cal.fit(X_scaled, y)
# get final probabilities on full set
final_proba = final_cal.predict_proba(X_scaled)[:,1]
final_preds = (final_proba >= best_thr).astype(int)

# final confusion matrix & detection ids
cm = confusion_matrix(y, final_preds)
TP = int(cm[1,1]); FP = int(cm[0,1]); TN = int(cm[0,0]); FN = int(cm[1,0])
print("\n--- Final (full-data) confusion matrix ---")
print(cm)
print(f"TP={TP} FP={FP} TN={TN} FN={FN}  FP_rate={FP/n_total:.4f}")

detected_windows = df.loc[final_preds==1, "window_id"].tolist() if "window_id" in df.columns else np.where(final_preds==1)[0].tolist()
print("Detected window ids (meta_pred==1):", detected_windows)

# 7) Save model, scaler, threshold, and predictions
model_artifact = {"model": final_cal, "scaler": scaler, "features": features, "threshold": best_thr}
with open(os.path.join(OUT_DIR, "meta_model_fp_constrained.pkl"), "wb") as f:
    pickle.dump(model_artifact, f)
print("Saved model artifact to:", os.path.join(OUT_DIR, "meta_model_fp_constrained.pkl"))

df_out = df.copy().reset_index(drop=True)
df_out["meta_proba_oof"] = y_proba_oof
df_out["meta_proba"] = final_proba
df_out["meta_pred"] = final_preds
df_out.to_csv(os.path.join(OUT_DIR, "full_run_meta_predictions_fp_constrained.csv"), index=False)
print("Saved predictions to:", os.path.join(OUT_DIR, "full_run_meta_predictions_fp_constrained.csv"))

# 8) Detailed report: TP/FP lists and per-window info
report = {
    "n_total": n_total,
    "n_pos": int(y.sum()),
    "n_neg": int(n_total - y.sum()),
    "chosen_threshold": best_thr,
    "TP": TP, "FP": FP, "TN": TN, "FN": FN,
    "FP_rate": FP / n_total
}
print("\nReport:", report)

topk = df_out.sort_values("meta_proba", ascending=False).head(30)[["window_id","meta_proba","meta_pred"] + features]
print("\nTop-30 windows by meta_proba:")
print(topk.to_string(index=False))
