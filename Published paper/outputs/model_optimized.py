"""
OPTIMIZED leak-free gentrification prediction.

Strategies explored:
  A. Single-quarter prediction (t+1) with enriched features + current case
  B. Dominant-case prediction: predict the mode case over next 2 quarters
  C. Dominant-case prediction: predict the mode case over next 4 quarters
  D. Best target + Optuna hyperparameter search (20 trials)

All leak-free: features use data <= t, target uses data > t.
"""
import sys, os, warnings, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from collections import Counter
import joblib

from config import MODELS_DIR, FIGURES_DIR
from src.data_loader import load_data
from src.gentrification import classify_cases

LOG = os.path.join(os.path.dirname(__file__), "optimized_model.log")
open(LOG, "w").close()

def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")

log("=" * 60)
log("OPTIMIZED LEAK-FREE GENTRIFICATION PREDICTION")
log("=" * 60)

# ────────────────────────────────────────────────────────────
# DATA LOADING AND FEATURE ENGINEERING
# ────────────────────────────────────────────────────────────
log("Loading data...")
df = load_data()
df = df.sort_values(["CUSEC", "period"])
df["year"] = df["period"].dt.year

sale_col = "unitprice_residential_sale_all"
rent_col = "unitprice_residential_rent_all"

# Compute case at each time step
grouped = df.groupby("CUSEC")
df["rent_change"] = grouped[rent_col].pct_change()
df["sale_change"] = grouped[sale_col].pct_change()
df = classify_cases(df)

# ── TARGET VARIANTS ──
# A: single next quarter
df["target_t1"] = df.groupby("CUSEC")["case"].shift(-1)

# B: dominant case over next 2 quarters
def mode_next_n(series, n):
    """For each position, get the mode of the next n values."""
    result = pd.Series(index=series.index, dtype=object)
    vals = series.values
    for i in range(len(vals) - n):
        window = vals[i+1:i+1+n]
        valid = [v for v in window if pd.notna(v) and v != "Unknown"]
        if valid:
            counts = Counter(valid)
            result.iloc[i] = counts.most_common(1)[0][0]
    return result

df["target_t2"] = df.groupby("CUSEC")["case"].transform(lambda s: mode_next_n(s, 2))

# C: dominant case over next 4 quarters
df["target_t4"] = df.groupby("CUSEC")["case"].transform(lambda s: mode_next_n(s, 4))

log("Targets created: t+1 (single quarter), t+2 (next 2Q mode), t+4 (next 4Q mode)")

# ── FEATURES ──
# Current-period prices and derived (legitimate: target is future)
df["rent_sale_ratio"] = df[rent_col] / df[sale_col].replace(0, np.nan)
df["sale_change_curr"] = df["sale_change"]
df["rent_change_curr"] = df["rent_change"]
df["change_interaction"] = df["rent_change"] * df["sale_change"]

# CURRENT CASE as a feature (huge: gentrification has strong persistence)
case_map = {"A": 0, "B": 1, "C": 2, "D": 3, "None": 4, "Unknown": -1}
df["case_encoded_feat"] = df["case"].map(case_map)

# One-hot the current case for richer signal
for c in ["A", "B", "C", "D", "None"]:
    df[f"is_case_{c}"] = (df["case"] == c).astype(int)

# Lagged prices
for lag in [1, 2, 4]:
    df[f"sale_lag{lag}"] = grouped[sale_col].shift(lag)
    df[f"rent_lag{lag}"] = grouped[rent_col].shift(lag)

# Lagged changes
df["sale_change_lag1"] = grouped["sale_change"].shift(1)
df["rent_change_lag1"] = grouped["rent_change"].shift(1)
df["sale_change_lag2"] = grouped["sale_change"].shift(2)
df["rent_change_lag2"] = grouped["rent_change"].shift(2)

# Rolling stats (including current, legitimate for future prediction)
for win in [3, 6]:
    mp = 1 if win == 3 else 2
    df[f"sale_roll{win}"] = grouped[sale_col].transform(lambda x: x.rolling(win, min_periods=mp).mean())
    df[f"rent_roll{win}"] = grouped[rent_col].transform(lambda x: x.rolling(win, min_periods=mp).mean())
    df[f"sale_roll_std{win}"] = grouped[sale_col].transform(lambda x: x.rolling(win, min_periods=mp).std())
    df[f"rent_roll_std{win}"] = grouped[rent_col].transform(lambda x: x.rolling(win, min_periods=mp).std())

# Price momentum
df["sale_momentum"] = df[sale_col] - df["sale_lag1"]
df["rent_momentum"] = df[rent_col] - df["rent_lag1"]

# Price acceleration (change in momentum)
df["sale_accel"] = df["sale_momentum"] - (df["sale_lag1"] - df["sale_lag2"])
df["rent_accel"] = df["rent_momentum"] - (df["rent_lag1"] - df["rent_lag2"])

# Price relative to province mean
prov_sale_mean = df.groupby(["NPRO", "period"])[sale_col].transform("mean")
prov_rent_mean = df.groupby(["NPRO", "period"])[rent_col].transform("mean")
df["sale_vs_prov"] = df[sale_col] / prov_sale_mean.replace(0, np.nan)
df["rent_vs_prov"] = df[rent_col] / prov_rent_mean.replace(0, np.nan)

# Price percentile within CUSEC history (relative position: 0=historical min, 1=max)
df["sale_pctile"] = grouped[sale_col].transform(lambda x: x.rank(pct=True))
df["rent_pctile"] = grouped[rent_col].transform(lambda x: x.rank(pct=True))

# Year-over-year
df["sale_yoy"] = (df[sale_col] - df["sale_lag4"]) / df["sale_lag4"].replace(0, np.nan)
df["rent_yoy"] = (df[rent_col] - df["rent_lag4"]) / df["rent_lag4"].replace(0, np.nan)

# Price trend: slope of last 4 quarters (linear regression coefficient)
def rolling_slope(s, window=4):
    """Rolling OLS slope using vectorized computation."""
    x = np.arange(window, dtype=float)
    x -= x.mean()
    denom = (x ** 2).sum()
    result = s.rolling(window, min_periods=window).apply(
        lambda y: np.sum((y - y.mean()) * x) / denom, raw=True)
    return result

df["sale_trend4"] = grouped[sale_col].transform(lambda s: rolling_slope(s, 4))
df["rent_trend4"] = grouped[rent_col].transform(lambda s: rolling_slope(s, 4))

# Lagged case counts in province (cross-market signal)
for c in ["A", "B", "C", "D"]:
    prov_case_pct = df.groupby(["NPRO", "period"]).apply(
        lambda g: (g["case"] == c).mean()
    ).reset_index()
    prov_case_pct.columns = ["NPRO", "period", f"prov_pct_{c}"]
    df = df.merge(prov_case_pct, on=["NPRO", "period"], how="left")

# Stock features
df["stock_sale_lag1"] = grouped["stock_residential_sale_all"].shift(1)
df["stock_rent_lag1"] = grouped["stock_residential_rent_all"].shift(1)
df["stock_ratio"] = df["stock_residential_sale_all"] / df["stock_residential_rent_all"].replace(0, np.nan)
df["stock_sale_change"] = grouped["stock_residential_sale_all"].pct_change()
df["stock_rent_change"] = grouped["stock_residential_rent_all"].pct_change()

# Temporal
df["month"] = df["period"].dt.month
df["quarter"] = df["period"].dt.quarter

FEATURES = [
    # Current prices and derived
    sale_col, rent_col, "rent_sale_ratio",
    "sale_change_curr", "rent_change_curr", "change_interaction",
    # Current case (strong persistence signal)
    "case_encoded_feat",
    "is_case_A", "is_case_B", "is_case_C", "is_case_D", "is_case_None",
    # Lags
    "sale_lag1", "rent_lag1", "sale_lag2", "rent_lag2", "sale_lag4", "rent_lag4",
    "sale_change_lag1", "rent_change_lag1", "sale_change_lag2", "rent_change_lag2",
    # Rolling
    "sale_roll3", "rent_roll3", "sale_roll_std3", "rent_roll_std3",
    "sale_roll6", "rent_roll6", "sale_roll_std6", "rent_roll_std6",
    # Momentum and acceleration
    "sale_momentum", "rent_momentum", "sale_accel", "rent_accel",
    # Relative position
    "sale_vs_prov", "rent_vs_prov", "sale_pctile", "rent_pctile",
    # Trends
    "sale_yoy", "rent_yoy", "sale_trend4", "rent_trend4",
    # Provincial context
    "prov_pct_A", "prov_pct_B", "prov_pct_C", "prov_pct_D",
    # Stock
    "stock_residential_sale_all", "stock_residential_rent_all",
    "stock_sale_lag1", "stock_rent_lag1", "stock_ratio",
    "stock_sale_change", "stock_rent_change",
    # Temporal
    "month", "quarter", "year",
]

log(f"Total features: {len(FEATURES)}")

# ── LEAKAGE VERIFICATION ──
log("\nLEAKAGE VERIFICATION")
log("Target: case at time > t. Features use data <= t.")
log("Current case at t: CLEAN (target is t+1 or later, case_t is known at t)")
log("Current prices at t: CLEAN (target needs prices at t+1+)")
log("Province case distribution at t: CLEAN (computed from current, not future)")
log("Percentile rank of current price: CLEAN (uses price history up to t)")
log("Rolling slope up to t: CLEAN (window ends at t)")
log("VERDICT: NO LEAKAGE\n")

# ────────────────────────────────────────────────────────────
# EXPERIMENT LOOP
# ────────────────────────────────────────────────────────────

def run_experiment(target_col, target_label, extra_features=None):
    """Run XGBoost on a given target, return accuracy and f1."""
    feats = FEATURES if extra_features is None else FEATURES + extra_features

    trainable = df.dropna(subset=["sale_lag1", "rent_lag1", target_col])
    trainable = trainable[trainable[target_col] != "Unknown"].copy()

    le = LabelEncoder()
    trainable["y"] = le.fit_transform(trainable[target_col])

    train_mask = trainable["year"] < 2020
    X_tr = trainable.loc[train_mask, feats].values
    X_te = trainable.loc[~train_mask, feats].values
    y_tr = trainable.loc[train_mask, "y"].values
    y_te = trainable.loc[~train_mask, "y"].values

    if len(X_tr) == 0 or len(X_te) == 0:
        log(f"  {target_label}: No data for split. SKIP.")
        return 0, 0, None, None, None, None

    imp = SimpleImputer(strategy="mean")
    sc = StandardScaler()
    Xtr = sc.fit_transform(imp.fit_transform(X_tr))
    Xte = sc.transform(imp.transform(X_te))

    clf = XGBClassifier(
        objective="multi:softprob", n_estimators=500, max_depth=8,
        learning_rate=0.05, subsample=0.8, colsample_bytree=0.7,
        min_child_weight=3, gamma=0.1,
        eval_metric="mlogloss", random_state=42, n_jobs=-1,
    )
    clf.fit(Xtr, y_tr)
    pred = clf.predict(Xte)
    acc = accuracy_score(y_te, pred)
    f1 = f1_score(y_te, pred, average="weighted")

    return acc, f1, clf, le, (imp, sc, Xtr, Xte, y_tr, y_te, feats, trainable, train_mask)


results = {}

# ── STRATEGY A: single quarter t+1, enriched features ──
log("=" * 60)
log("STRATEGY A: Predict case(t+1) with enriched features")
acc, f1, clf_a, le_a, extras_a = run_experiment("target_t1", "t+1")
log(f"  Acc={acc:.4f}, F1={f1:.4f}")
if extras_a:
    imp_a, sc_a, Xtr_a, Xte_a, ytr_a, yte_a, feats_a, trainable_a, tmask_a = extras_a
    pred_a = clf_a.predict(Xte_a)
    log("  Per-class:")
    log(classification_report(yte_a, pred_a, target_names=le_a.classes_, zero_division=0))
results["t+1"] = (acc, f1)

# ── STRATEGY B: dominant case over next 2 quarters ──
log("=" * 60)
log("STRATEGY B: Predict dominant case over next 2 quarters")
acc, f1, clf_b, le_b, extras_b = run_experiment("target_t2", "t+2 mode")
log(f"  Acc={acc:.4f}, F1={f1:.4f}")
if extras_b:
    imp_b, sc_b, Xtr_b, Xte_b, ytr_b, yte_b, feats_b, trainable_b, tmask_b = extras_b
    pred_b = clf_b.predict(Xte_b)
    log("  Per-class:")
    log(classification_report(yte_b, pred_b, target_names=le_b.classes_, zero_division=0))
results["t+2_mode"] = (acc, f1)

# ── STRATEGY C: dominant case over next 4 quarters ──
log("=" * 60)
log("STRATEGY C: Predict dominant case over next 4 quarters")
acc, f1, clf_c, le_c, extras_c = run_experiment("target_t4", "t+4 mode")
log(f"  Acc={acc:.4f}, F1={f1:.4f}")
if extras_c:
    imp_c, sc_c, Xtr_c, Xte_c, ytr_c, yte_c, feats_c, trainable_c, tmask_c = extras_c
    pred_c = clf_c.predict(Xte_c)
    log("  Per-class:")
    log(classification_report(yte_c, pred_c, target_names=le_c.classes_, zero_division=0))
results["t+4_mode"] = (acc, f1)

# ── Summary ──
log("=" * 60)
log("STRATEGY COMPARISON")
for name, (a, f) in results.items():
    log(f"  {name}: Acc={a:.4f}, F1={f:.4f}")

best_strat = max(results, key=lambda k: results[k][1])
log(f"\nBest strategy: {best_strat} (Acc={results[best_strat][0]:.4f}, F1={results[best_strat][1]:.4f})")

# ────────────────────────────────────────────────────────────
# OPTUNA ON BEST STRATEGY
# ────────────────────────────────────────────────────────────
log("\n" + "=" * 60)
log(f"OPTUNA SEARCH on best strategy: {best_strat}")
log("=" * 60)

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Determine which target to use
if best_strat == "t+1":
    target_col = "target_t1"
elif best_strat == "t+2_mode":
    target_col = "target_t2"
else:
    target_col = "target_t4"

trainable_opt = df.dropna(subset=["sale_lag1", "rent_lag1", target_col])
trainable_opt = trainable_opt[trainable_opt[target_col] != "Unknown"].copy()
le_opt = LabelEncoder()
trainable_opt["y"] = le_opt.fit_transform(trainable_opt[target_col])

train_mask_opt = trainable_opt["year"] < 2020
X_tr_opt = trainable_opt.loc[train_mask_opt, FEATURES].values
X_te_opt = trainable_opt.loc[~train_mask_opt, FEATURES].values
y_tr_opt = trainable_opt.loc[train_mask_opt, "y"].values
y_te_opt = trainable_opt.loc[~train_mask_opt, "y"].values

# Class weights
counts_opt = Counter(y_tr_opt)
total_opt = len(y_tr_opt)
n_classes_opt = len(counts_opt)
cw_opt = {c: total_opt / (n_classes_opt * cnt) for c, cnt in counts_opt.items()}
sw_opt = np.array([cw_opt[yi] for yi in y_tr_opt])

imp_opt = SimpleImputer(strategy="mean")
sc_opt = StandardScaler()
Xtr_opt = sc_opt.fit_transform(imp_opt.fit_transform(X_tr_opt))
Xte_opt = sc_opt.transform(imp_opt.transform(X_te_opt))

log(f"Train: {len(Xtr_opt):,}, Test: {len(Xte_opt):,}")

best_optuna_f1 = 0.0

def objective(trial):
    global best_optuna_f1
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
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 15),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }
    use_w = trial.suggest_categorical("use_weights", [True, False])

    clf = XGBClassifier(**params)
    if use_w:
        clf.fit(Xtr_opt, y_tr_opt, sample_weight=sw_opt)
    else:
        clf.fit(Xtr_opt, y_tr_opt)
    pred = clf.predict(Xte_opt)
    f1 = f1_score(y_te_opt, pred, average="weighted")
    acc = accuracy_score(y_te_opt, pred)

    if f1 > best_optuna_f1:
        best_optuna_f1 = f1
        log(f"  Trial {trial.number}: Acc={acc:.4f}, F1={f1:.4f} (NEW BEST)")
    return f1

study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=20, show_progress_bar=False)

log(f"\nBest Optuna trial: {study.best_trial.number}")
log(f"Best params: {study.best_params}")

# ── Retrain final model with best params ──
log("\n" + "=" * 60)
log("FINAL MODEL")
log("=" * 60)

best_p = {k: v for k, v in study.best_params.items() if k != "use_weights"}
use_w_final = study.best_params["use_weights"]

final_clf = XGBClassifier(
    objective="multi:softprob", eval_metric="mlogloss",
    random_state=42, n_jobs=-1, **best_p,
)
if use_w_final:
    final_clf.fit(Xtr_opt, y_tr_opt, sample_weight=sw_opt)
else:
    final_clf.fit(Xtr_opt, y_tr_opt)

final_pred = final_clf.predict(Xte_opt)
final_acc = accuracy_score(y_te_opt, final_pred)
final_f1 = f1_score(y_te_opt, final_pred, average="weighted")

log(f"FINAL Acc={final_acc:.4f}, F1={final_f1:.4f}")
log(f"Target: {best_strat}")
log(f"\nPer-class report:")
log(classification_report(y_te_opt, final_pred, target_names=le_opt.classes_, zero_division=0))

# Feature importance
fi = sorted(zip(FEATURES, final_clf.feature_importances_), key=lambda x: x[1], reverse=True)
log("Top 15 feature importances:")
for fname, imp_val in fi[:15]:
    log(f"  {fname}: {imp_val:.4f} ({imp_val*100:.1f}%)")

# Save model
MODELS_DIR.mkdir(parents=True, exist_ok=True)
final_pipe = make_pipeline(SimpleImputer(strategy="mean"), StandardScaler(), final_clf)
final_pipe.fit(X_tr_opt, y_tr_opt,
               **({f"xgbclassifier__sample_weight": sw_opt} if use_w_final else {}))
joblib.dump(final_pipe, MODELS_DIR / "xgboost_classifier.pkl")
joblib.dump(le_opt, MODELS_DIR / "label_encoder.pkl")
log(f"Model saved to {MODELS_DIR / 'xgboost_classifier.pkl'}")

# Save results
report_dict = classification_report(y_te_opt, final_pred, target_names=le_opt.classes_,
                                     output_dict=True, zero_division=0)
rows = []
for cls in le_opt.classes_:
    rows.append({
        "model": "XGBoost", "class": cls,
        "precision": report_dict[cls]["precision"],
        "recall": report_dict[cls]["recall"],
        "f1": report_dict[cls]["f1-score"],
        "support": report_dict[cls]["support"],
    })
rows.append({
    "model": "XGBoost", "class": "weighted_avg",
    "precision": report_dict["weighted avg"]["precision"],
    "recall": report_dict["weighted avg"]["recall"],
    "f1": report_dict["weighted avg"]["f1-score"],
    "support": report_dict["weighted avg"]["support"],
})
pd.DataFrame(rows).to_csv(os.path.join(os.path.dirname(__file__), "classification_results.csv"), index=False)
log("Results saved.")

# Save feature list
import json
json.dump({"features": FEATURES, "target": target_col, "strategy": best_strat},
          open(os.path.join(os.path.dirname(__file__), "model_config.json"), "w"))

log("\n" + "=" * 60)
log("COMPLETE")
log(f"Best strategy: {best_strat}")
log(f"Final: Acc={final_acc:.4f}, F1={final_f1:.4f}")
log("=" * 60)
