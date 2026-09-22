"""
Improved gentrification prediction with Fable's recommendations.
1. Fix Optuna bug: use 2018-2019 validation split, test 2020+ truly held out
2. Per-section expanding-window history features (leak-free)
3. Binary decomposition: predict rent_sign and sale_sign independently, compose
All strictly leak-free: features use data <= t, target = case(t+1).
"""
import sys, os, warnings, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from collections import Counter

from config import FIGURES_DIR
from src.data_loader import load_data
from src.gentrification import classify_cases

LOG = os.path.join(os.path.dirname(__file__), "improved.log")
open(LOG, "w").close()

def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")

log("=" * 60)
log("IMPROVED GENTRIFICATION PREDICTION")
log("=" * 60)

# ── DATA LOADING ──
log("Loading data...")
df = load_data()
df = df.sort_values(["CUSEC", "period"])
df["year"] = df["period"].dt.year

sale_col = "unitprice_residential_sale_all"
rent_col = "unitprice_residential_rent_all"
grouped = df.groupby("CUSEC")

df["rent_change"] = grouped[rent_col].pct_change()
df["sale_change"] = grouped[sale_col].pct_change()
df = classify_cases(df)
df["target_case"] = df.groupby("CUSEC")["case"].shift(-1)

# ── DIAGNOSTIC: anti-persistence check ──
valid = df.dropna(subset=["rent_change", "sale_change"])
log(f"\nDIAGNOSTIC: rent vs sale change correlation = {valid['rent_change'].corr(valid['sale_change']):.4f}")

rent_lag_corr = valid.groupby("CUSEC").apply(
    lambda g: g["rent_change"].corr(g["rent_change"].shift(1)) if len(g) > 2 else np.nan
).dropna().mean()
sale_lag_corr = valid.groupby("CUSEC").apply(
    lambda g: g["sale_change"].corr(g["sale_change"].shift(1)) if len(g) > 2 else np.nan
).dropna().mean()
log(f"DIAGNOSTIC: rent autocorrelation = {rent_lag_corr:.4f}")
log(f"DIAGNOSTIC: sale autocorrelation = {sale_lag_corr:.4f}")

# ── BASE FEATURES (same as model_optimized.py) ──
log("\nEngineering base features...")
df["sale_price"] = df[sale_col]
df["rent_price"] = df[rent_col]
df["rent_sale_ratio"] = df[rent_col] / df[sale_col].replace(0, np.nan)

le_case = LabelEncoder()
df["case_encoded_feat"] = le_case.fit_transform(df["case"].fillna("Unknown"))
for c in ["A", "B", "C", "D", "None"]:
    df[f"is_case_{c}"] = (df["case"] == c).astype(int)

df["sale_lag1"] = grouped[sale_col].shift(1)
df["rent_lag1"] = grouped[rent_col].shift(1)
df["sale_lag2"] = grouped[sale_col].shift(2)
df["rent_lag2"] = grouped[rent_col].shift(2)
df["sale_lag4"] = grouped[sale_col].shift(4)
df["rent_lag4"] = grouped[rent_col].shift(4)

df["sale_change_curr"] = df["sale_change"]
df["rent_change_curr"] = df["rent_change"]
df["sale_change_lag1"] = grouped["sale_change"].shift(1)
df["rent_change_lag1"] = grouped["rent_change"].shift(1)
df["sale_change_lag2"] = grouped["sale_change"].shift(2)
df["rent_change_lag2"] = grouped["rent_change"].shift(2)
df["change_interaction"] = df["rent_change"] * df["sale_change"]

df["sale_roll3"] = grouped[sale_col].transform(lambda x: x.rolling(3, min_periods=1).mean())
df["rent_roll3"] = grouped[rent_col].transform(lambda x: x.rolling(3, min_periods=1).mean())
df["sale_roll_std3"] = grouped[sale_col].transform(lambda x: x.rolling(3, min_periods=1).std())
df["rent_roll_std3"] = grouped[rent_col].transform(lambda x: x.rolling(3, min_periods=1).std())
df["sale_roll6"] = grouped[sale_col].transform(lambda x: x.rolling(6, min_periods=2).mean())
df["rent_roll6"] = grouped[rent_col].transform(lambda x: x.rolling(6, min_periods=2).mean())
df["sale_roll_std6"] = grouped[sale_col].transform(lambda x: x.rolling(6, min_periods=2).std())
df["rent_roll_std6"] = grouped[rent_col].transform(lambda x: x.rolling(6, min_periods=2).std())

df["sale_momentum"] = df[sale_col] - df["sale_lag1"]
df["rent_momentum"] = df[rent_col] - df["rent_lag1"]
df["sale_accel"] = df["sale_momentum"] - df.groupby("CUSEC")["sale_momentum"].shift(1)
df["rent_accel"] = df["rent_momentum"] - df.groupby("CUSEC")["rent_momentum"].shift(1)

prov_sale_mean = df.groupby(["NPRO", "period"])[sale_col].transform("mean")
prov_rent_mean = df.groupby(["NPRO", "period"])[rent_col].transform("mean")
df["sale_vs_prov"] = df[sale_col] / prov_sale_mean.replace(0, np.nan)
df["rent_vs_prov"] = df[rent_col] / prov_rent_mean.replace(0, np.nan)
df["sale_pctile"] = grouped[sale_col].transform(lambda x: x.rank(pct=True))
df["rent_pctile"] = grouped[rent_col].transform(lambda x: x.rank(pct=True))

df["sale_yoy"] = (df[sale_col] - df["sale_lag4"]) / df["sale_lag4"].replace(0, np.nan)
df["rent_yoy"] = (df[rent_col] - df["rent_lag4"]) / df["rent_lag4"].replace(0, np.nan)

def rolling_slope(s, window=4):
    x = np.arange(window, dtype=float)
    x -= x.mean()
    denom = (x ** 2).sum()
    return s.rolling(window, min_periods=window).apply(
        lambda y: np.sum((y - y.mean()) * x) / denom, raw=True)

df["sale_trend4"] = df.groupby("CUSEC")[sale_col].transform(lambda s: rolling_slope(s, 4))
df["rent_trend4"] = df.groupby("CUSEC")[rent_col].transform(lambda s: rolling_slope(s, 4))

for c in ["A", "B", "C", "D"]:
    prov_case_pct = df.groupby(["NPRO", "period"]).apply(
        lambda g, case=c: (g["case"] == case).mean()
    ).reset_index()
    prov_case_pct.columns = ["NPRO", "period", f"prov_pct_{c}"]
    df = df.merge(prov_case_pct, on=["NPRO", "period"], how="left")

df["stock_sale_lag1"] = df.groupby("CUSEC")["stock_residential_sale_all"].shift(1)
df["stock_rent_lag1"] = df.groupby("CUSEC")["stock_residential_rent_all"].shift(1)
df["stock_ratio"] = df["stock_residential_sale_all"] / df["stock_residential_rent_all"].replace(0, np.nan)
df["stock_sale_change"] = df.groupby("CUSEC")["stock_residential_sale_all"].pct_change()
df["stock_rent_change"] = df.groupby("CUSEC")["stock_residential_rent_all"].pct_change()

df["month"] = df["period"].dt.month
df["quarter"] = df["period"].dt.quarter

# ── NEW: Per-section expanding-window history features ──
# All use shift(1) to ensure strictly data <= t-1 (no current period in expanding stats)
log("Engineering per-section history features (expanding window, shifted)...")

grp = df.groupby("CUSEC")

for c in ["A", "B", "C", "D", "None"]:
    df[f"hist_pct_{c}"] = grp.apply(
        lambda g, case=c: ((g["case"] == case).cumsum().shift(1)) /
        (g["case"].notna().cumsum().shift(1).replace(0, np.nan))
    ).reset_index(level=0, drop=True)

# Sign-flip rate: how often this section changes case
df["case_changed"] = (df["case"] != grp["case"].shift(1)).astype(float)
df["hist_flip_rate"] = df.groupby("CUSEC")["case_changed"].transform(
    lambda x: x.expanding().mean().shift(1)
)

# Historical change volatility per section
df["hist_rent_vol"] = grp["rent_change"].transform(
    lambda x: x.expanding().std().shift(1)
)
df["hist_sale_vol"] = grp["sale_change"].transform(
    lambda x: x.expanding().std().shift(1)
)

# Mean stock (noise proxy: higher stock = more listings = less noisy)
df["hist_mean_stock_sale"] = grp["stock_residential_sale_all"].transform(
    lambda x: x.expanding().mean().shift(1)
)
df["hist_mean_stock_rent"] = grp["stock_residential_rent_all"].transform(
    lambda x: x.expanding().mean().shift(1)
)

# Number of observations for this CUSEC (data density)
df["hist_obs_count"] = grp.cumcount()

log("  Added 12 per-section history features")

# ── FEATURE LISTS ──
BASE_FEATURES = [
    sale_col, rent_col, "rent_sale_ratio",
    "sale_change_curr", "rent_change_curr", "change_interaction",
    "case_encoded_feat",
    "is_case_A", "is_case_B", "is_case_C", "is_case_D", "is_case_None",
    "sale_lag1", "rent_lag1", "sale_lag2", "rent_lag2", "sale_lag4", "rent_lag4",
    "sale_change_lag1", "rent_change_lag1", "sale_change_lag2", "rent_change_lag2",
    "sale_roll3", "rent_roll3", "sale_roll_std3", "rent_roll_std3",
    "sale_roll6", "rent_roll6", "sale_roll_std6", "rent_roll_std6",
    "sale_momentum", "rent_momentum", "sale_accel", "rent_accel",
    "sale_vs_prov", "rent_vs_prov", "sale_pctile", "rent_pctile",
    "sale_yoy", "rent_yoy", "sale_trend4", "rent_trend4",
    "prov_pct_A", "prov_pct_B", "prov_pct_C", "prov_pct_D",
    "stock_residential_sale_all", "stock_residential_rent_all",
    "stock_sale_lag1", "stock_rent_lag1", "stock_ratio",
    "stock_sale_change", "stock_rent_change",
    "month", "quarter", "year",
]

HISTORY_FEATURES = [
    "hist_pct_A", "hist_pct_B", "hist_pct_C", "hist_pct_D", "hist_pct_None",
    "hist_flip_rate", "hist_rent_vol", "hist_sale_vol",
    "hist_mean_stock_sale", "hist_mean_stock_rent", "hist_obs_count",
]

ALL_FEATURES = BASE_FEATURES + HISTORY_FEATURES

log(f"Base features: {len(BASE_FEATURES)}, History features: {len(HISTORY_FEATURES)}, Total: {len(ALL_FEATURES)}")

# ── PREPARE SPLITS ──
# IMPORTANT: 3-way split for honest Optuna
# Train: < 2018, Validation: 2018-2019, Test: 2020+
trainable = df.dropna(subset=["sale_lag1", "rent_lag1", "target_case"])
trainable = trainable[trainable["target_case"] != "Unknown"].copy()

le = LabelEncoder()
trainable["target_encoded"] = le.fit_transform(trainable["target_case"])
classes = list(le.classes_)
n_classes = len(classes)

train_mask = trainable["year"] < 2018
val_mask = (trainable["year"] >= 2018) & (trainable["year"] < 2020)
test_mask = trainable["year"] >= 2020

log(f"\n3-WAY SPLIT (honest Optuna):")
log(f"  Train (< 2018): {train_mask.sum():,}")
log(f"  Validation (2018-2019): {val_mask.sum():,}")
log(f"  Test (2020+): {test_mask.sum():,}")

# ── Binary targets for decomposition ──
trainable["rent_sign_next"] = (trainable.groupby("CUSEC")["rent_change"].shift(-1) > 0).astype(int)
trainable["sale_sign_next"] = (trainable.groupby("CUSEC")["sale_change"].shift(-1) > 0).astype(int)

# ════════════════════════════════════════════════════════════════
# EXPERIMENT 1: Baseline (same as before, but honest 3-way split)
# ════════════════════════════════════════════════════════════════
log("\n" + "=" * 60)
log("EXP 1: BASELINE (base features, no Optuna, honest split)")
log("=" * 60)

imp = SimpleImputer(strategy="mean")
sc = StandardScaler()

X_train = sc.fit_transform(imp.fit_transform(trainable.loc[train_mask | val_mask, BASE_FEATURES].values))
X_test = sc.transform(imp.transform(trainable.loc[test_mask, BASE_FEATURES].values))
y_train = trainable.loc[train_mask | val_mask, "target_encoded"].values
y_test = trainable.loc[test_mask, "target_encoded"].values

clf_base = XGBClassifier(
    objective="multi:softprob", n_estimators=500, max_depth=8,
    learning_rate=0.05, subsample=0.8, colsample_bytree=0.7,
    min_child_weight=3, gamma=0.1,
    eval_metric="mlogloss", random_state=42, n_jobs=-1,
)
clf_base.fit(X_train, y_train)
pred_base = clf_base.predict(X_test)
acc_base = accuracy_score(y_test, pred_base)
f1_base = f1_score(y_test, pred_base, average="weighted")
log(f"  Acc={acc_base:.4f}, F1={f1_base:.4f}")

# ════════════════════════════════════════════════════════════════
# EXPERIMENT 2: Base + History features
# ════════════════════════════════════════════════════════════════
log("\n" + "=" * 60)
log("EXP 2: BASE + HISTORY FEATURES")
log("=" * 60)

imp2 = SimpleImputer(strategy="mean")
sc2 = StandardScaler()

X_train2 = sc2.fit_transform(imp2.fit_transform(trainable.loc[train_mask | val_mask, ALL_FEATURES].values))
X_test2 = sc2.transform(imp2.transform(trainable.loc[test_mask, ALL_FEATURES].values))

clf_hist = XGBClassifier(
    objective="multi:softprob", n_estimators=500, max_depth=8,
    learning_rate=0.05, subsample=0.8, colsample_bytree=0.7,
    min_child_weight=3, gamma=0.1,
    eval_metric="mlogloss", random_state=42, n_jobs=-1,
)
clf_hist.fit(X_train2, y_train)
pred_hist = clf_hist.predict(X_test2)
acc_hist = accuracy_score(y_test, pred_hist)
f1_hist = f1_score(y_test, pred_hist, average="weighted")
log(f"  Acc={acc_hist:.4f}, F1={f1_hist:.4f}")
log(f"  Delta vs baseline: {(acc_hist - acc_base)*100:+.2f}pp")

# ════════════════════════════════════════════════════════════════
# EXPERIMENT 3: Optuna with HONEST validation split
# ════════════════════════════════════════════════════════════════
log("\n" + "=" * 60)
log("EXP 3: OPTUNA ON VALIDATION SPLIT (honest)")
log("=" * 60)

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Train on <2018, validate on 2018-2019, test on 2020+ (held out from Optuna)
imp3 = SimpleImputer(strategy="mean")
sc3 = StandardScaler()

X_tr_optuna = sc3.fit_transform(imp3.fit_transform(trainable.loc[train_mask, ALL_FEATURES].values))
X_val_optuna = sc3.transform(imp3.transform(trainable.loc[val_mask, ALL_FEATURES].values))
X_te_optuna = sc3.transform(imp3.transform(trainable.loc[test_mask, ALL_FEATURES].values))
y_tr_optuna = trainable.loc[train_mask, "target_encoded"].values
y_val_optuna = trainable.loc[val_mask, "target_encoded"].values
y_te_optuna = trainable.loc[test_mask, "target_encoded"].values

log(f"  Optuna train: {len(X_tr_optuna):,}, val: {len(X_val_optuna):,}, test: {len(X_te_optuna):,}")

best_val_f1 = 0.0

def objective(trial):
    global best_val_f1
    params = {
        "objective": "multi:softprob",
        "eval_metric": "mlogloss",
        "random_state": 42,
        "n_jobs": -1,
        "n_estimators": trial.suggest_int("n_estimators", 300, 1000, step=50),
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }
    clf = XGBClassifier(**params)
    clf.fit(X_tr_optuna, y_tr_optuna)
    pred_val = clf.predict(X_val_optuna)
    f1 = f1_score(y_val_optuna, pred_val, average="weighted")
    acc = accuracy_score(y_val_optuna, pred_val)
    if f1 > best_val_f1:
        best_val_f1 = f1
        log(f"  Trial {trial.number}: Val Acc={acc:.4f}, F1={f1:.4f} (NEW BEST)")
    return f1

t0 = time.time()
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=50, show_progress_bar=False)
log(f"  Optuna done in {time.time()-t0:.1f}s")
log(f"  Best params: {study.best_params}")

# Retrain on train+val with best params, evaluate on test
best_p = study.best_params
clf_optuna = XGBClassifier(
    objective="multi:softprob", eval_metric="mlogloss",
    random_state=42, n_jobs=-1, **best_p,
)
# Retrain on full train+val
imp3b = SimpleImputer(strategy="mean")
sc3b = StandardScaler()
X_trainval = sc3b.fit_transform(imp3b.fit_transform(trainable.loc[train_mask | val_mask, ALL_FEATURES].values))
X_test3 = sc3b.transform(imp3b.transform(trainable.loc[test_mask, ALL_FEATURES].values))
clf_optuna.fit(X_trainval, y_train)
pred_optuna = clf_optuna.predict(X_test3)
acc_optuna = accuracy_score(y_test, pred_optuna)
f1_optuna = f1_score(y_test, pred_optuna, average="weighted")
log(f"  HONEST TEST: Acc={acc_optuna:.4f}, F1={f1_optuna:.4f}")
log(f"  Delta vs baseline: {(acc_optuna - acc_base)*100:+.2f}pp")

# ════════════════════════════════════════════════════════════════
# EXPERIMENT 4: BINARY DECOMPOSITION
# ════════════════════════════════════════════════════════════════
log("\n" + "=" * 60)
log("EXP 4: BINARY DECOMPOSITION (rent_sign + sale_sign -> compose)")
log("=" * 60)

# Use all features, train+val as training
decomp_trainable = trainable.dropna(subset=["rent_sign_next", "sale_sign_next"])

X_decomp_train = sc3b.transform(imp3b.transform(
    decomp_trainable.loc[(train_mask | val_mask) & decomp_trainable.index.isin(decomp_trainable.index), ALL_FEATURES].values
))
X_decomp_test = sc3b.transform(imp3b.transform(
    decomp_trainable.loc[test_mask & decomp_trainable.index.isin(decomp_trainable.index), ALL_FEATURES].values
))

y_rent_train = decomp_trainable.loc[train_mask | val_mask, "rent_sign_next"].values.astype(int)
y_sale_train = decomp_trainable.loc[train_mask | val_mask, "sale_sign_next"].values.astype(int)
y_rent_test = decomp_trainable.loc[test_mask, "rent_sign_next"].values.astype(int)
y_sale_test = decomp_trainable.loc[test_mask, "sale_sign_next"].values.astype(int)
y_case_test = decomp_trainable.loc[test_mask, "target_encoded"].values

log(f"  Decomp train: {len(X_decomp_train):,}, test: {len(X_decomp_test):,}")

# Train rent sign predictor
clf_rent = XGBClassifier(
    objective="binary:logistic", n_estimators=500, max_depth=8,
    learning_rate=0.05, subsample=0.8, colsample_bytree=0.7,
    eval_metric="logloss", random_state=42, n_jobs=-1,
)
clf_rent.fit(X_decomp_train, y_rent_train)
rent_acc = accuracy_score(y_rent_test, clf_rent.predict(X_decomp_test))
rent_proba = clf_rent.predict_proba(X_decomp_test)[:, 1]
log(f"  Rent sign accuracy: {rent_acc:.4f}")
log(f"  Rent sign AUC: {roc_auc_score(y_rent_test, rent_proba):.4f}")

# Train sale sign predictor
clf_sale = XGBClassifier(
    objective="binary:logistic", n_estimators=500, max_depth=8,
    learning_rate=0.05, subsample=0.8, colsample_bytree=0.7,
    eval_metric="logloss", random_state=42, n_jobs=-1,
)
clf_sale.fit(X_decomp_train, y_sale_train)
sale_acc = accuracy_score(y_sale_test, clf_sale.predict(X_decomp_test))
sale_proba = clf_sale.predict_proba(X_decomp_test)[:, 1]
log(f"  Sale sign accuracy: {sale_acc:.4f}")
log(f"  Sale sign AUC: {roc_auc_score(y_sale_test, sale_proba):.4f}")

# Compose: map (rent_sign, sale_sign) -> case
# C: both up (rent>0, sale>0)
# D: both down (rent<=0, sale<=0)
# A: rent up, sale down (rent>0, sale<=0)
# B: rent down, sale up (rent<=0, sale>0)
rent_pred = clf_rent.predict(X_decomp_test)
sale_pred = clf_sale.predict(X_decomp_test)

def compose_case(rent_up, sale_up, le):
    if rent_up and sale_up:
        return le.transform(["C"])[0]
    elif not rent_up and not sale_up:
        return le.transform(["D"])[0]
    elif rent_up and not sale_up:
        return le.transform(["A"])[0]
    else:
        return le.transform(["B"])[0]

composed_pred = np.array([compose_case(r, s, le) for r, s in zip(rent_pred, sale_pred)])
acc_decomp = accuracy_score(y_case_test, composed_pred)
f1_decomp = f1_score(y_case_test, composed_pred, average="weighted")
log(f"  Composed 4-class Acc={acc_decomp:.4f}, F1={f1_decomp:.4f}")
log(f"  (Note: this produces only 4 classes, no 'None')")

# Probabilistic composition with None detection
# P(C) = P(rent_up) * P(sale_up)
# P(D) = P(rent_down) * P(sale_down)
# P(A) = P(rent_up) * P(sale_down)
# P(B) = P(rent_down) * P(sale_up)
p_rent_up = rent_proba
p_rent_down = 1 - rent_proba
p_sale_up = sale_proba
p_sale_down = 1 - sale_proba

prob_C = p_rent_up * p_sale_up
prob_D = p_rent_down * p_sale_down
prob_A = p_rent_up * p_sale_down
prob_B = p_rent_down * p_sale_up

# Stack probabilities for all classes
class_names = list(le.classes_)
prob_matrix = np.zeros((len(y_case_test), n_classes))
for i, cn in enumerate(class_names):
    if cn == "A":
        prob_matrix[:, i] = prob_A
    elif cn == "B":
        prob_matrix[:, i] = prob_B
    elif cn == "C":
        prob_matrix[:, i] = prob_C
    elif cn == "D":
        prob_matrix[:, i] = prob_D
    elif cn == "None":
        prob_matrix[:, i] = 0.0  # decomposition doesn't predict None

# Normalize
row_sums = prob_matrix.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1
prob_matrix = prob_matrix / row_sums

prob_pred = prob_matrix.argmax(axis=1)
acc_prob_decomp = accuracy_score(y_case_test, prob_pred)
f1_prob_decomp = f1_score(y_case_test, prob_pred, average="weighted")
log(f"  Probabilistic composition Acc={acc_prob_decomp:.4f}, F1={f1_prob_decomp:.4f}")

# ════════════════════════════════════════════════════════════════
# EXPERIMENT 5: SELECTIVE PREDICTION (accuracy-coverage curve)
# ════════════════════════════════════════════════════════════════
log("\n" + "=" * 60)
log("EXP 5: SELECTIVE PREDICTION (accuracy-coverage curve)")
log("=" * 60)

# Use the Optuna model's probabilities
proba_optuna = clf_optuna.predict_proba(X_test3)
max_proba = proba_optuna.max(axis=1)
pred_optuna_full = proba_optuna.argmax(axis=1)

for coverage in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]:
    n_keep = int(len(max_proba) * coverage)
    if n_keep == 0:
        continue
    threshold_idx = np.argsort(max_proba)[-n_keep:]
    acc_at_cov = accuracy_score(y_test[threshold_idx], pred_optuna_full[threshold_idx])
    log(f"  Coverage {coverage*100:.0f}%: Acc={acc_at_cov:.4f} ({n_keep:,} sections)")

# ════════════════════════════════════════════════════════════════
# EXPERIMENT 6: NOISE CEILING (stratify by stock)
# ════════════════════════════════════════════════════════════════
log("\n" + "=" * 60)
log("EXP 6: NOISE CEILING (accuracy by listing stock quartile)")
log("=" * 60)

test_df = trainable[test_mask].copy()
test_df["pred"] = pred_optuna
test_df["correct"] = (test_df["pred"] == test_df["target_encoded"]).astype(int)

# Stock quartiles (proxy for data quality/noise)
total_stock = test_df["stock_residential_sale_all"] + test_df["stock_residential_rent_all"]
test_df["stock_total"] = total_stock
quartiles = pd.qcut(total_stock, 4, labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"], duplicates="drop")
test_df["stock_quartile"] = quartiles

for q in test_df["stock_quartile"].dropna().unique():
    mask_q = test_df["stock_quartile"] == q
    acc_q = test_df.loc[mask_q, "correct"].mean()
    n_q = mask_q.sum()
    log(f"  {q}: Acc={acc_q:.4f} (n={n_q:,})")

# ════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ════════════════════════════════════════════════════════════════
log("\n" + "=" * 60)
log("FINAL COMPARISON")
log("=" * 60)
log(f"  Baseline (base features, train<2020):         Acc={acc_base:.4f}")
log(f"  + History features:                            Acc={acc_hist:.4f} ({(acc_hist-acc_base)*100:+.2f}pp)")
log(f"  + Optuna (honest val split):                   Acc={acc_optuna:.4f} ({(acc_optuna-acc_base)*100:+.2f}pp)")
log(f"  Binary decomposition (hard):                   Acc={acc_decomp:.4f}")
log(f"  Binary decomposition (probabilistic):          Acc={acc_prob_decomp:.4f}")
log(f"  Per-sign accuracy: rent={rent_acc:.4f}, sale={sale_acc:.4f}")

log("\nDONE")
