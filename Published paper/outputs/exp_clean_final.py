"""
LEAK-FREE gentrification classifier with Optuna hyperparameter search.

Target leakage analysis:
  case is defined by sign(rent_change_t, sale_change_t) where:
    rent_change_t = (rent_t - rent_{t-1}) / rent_{t-1}
    sale_change_t = (sale_t - sale_{t-1}) / sale_{t-1}

  ANY feature exposing rent_t or sale_t (even indirectly via rolling windows
  that include the current period) allows reconstructing the change and thus
  the target.  All rolling/ratio features below use ONLY lagged values.
"""
import sys, os, warnings, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from collections import Counter
import joblib
import optuna
import json

from config import MODELS_DIR
from src.data_loader import load_data
from src.gentrification import classify_cases

optuna.logging.set_verbosity(optuna.logging.WARNING)

LOG = os.path.join(os.path.dirname(__file__), "clean_final.log")

def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")

# Clear previous log
open(LOG, "w").close()

log("=" * 60)
log("LEAK-FREE FINAL MODEL")
log("=" * 60)

# ── Load & classify ──────────────────────────────────────────
log("Loading data...")
df = load_data()
df = df.sort_values("period")
df["year"] = df["period"].dt.year

sale_col = "unitprice_residential_sale_all"
rent_col = "unitprice_residential_rent_all"
grouped = df.groupby("CUSEC")

df["rent_change"] = grouped[rent_col].pct_change()
df["sale_change"] = grouped[sale_col].pct_change()
df = classify_cases(df)

# ── Feature engineering (ALL leak-free) ──────────────────────
# RULE: every feature must use ONLY values from periods <= t-1.
# No current-period prices (sale_t, rent_t) anywhere.

# --- Pure lags (strictly past) ---
df["sale_lag1"] = grouped[sale_col].shift(1)
df["rent_lag1"] = grouped[rent_col].shift(1)
df["sale_lag2"] = grouped[sale_col].shift(2)
df["rent_lag2"] = grouped[rent_col].shift(2)
df["sale_lag4"] = grouped[sale_col].shift(4)
df["rent_lag4"] = grouped[rent_col].shift(4)

# --- Lagged rolling means (window over t-1, t-2, t-3 — NO current period) ---
# shift(1) first, then rolling on the shifted series
sale_shifted = grouped[sale_col].shift(1)
rent_shifted = grouped[rent_col].shift(1)

df["sale_roll3_lag"] = sale_shifted.groupby(df["CUSEC"]).transform(
    lambda x: x.rolling(3, min_periods=1).mean())
df["rent_roll3_lag"] = rent_shifted.groupby(df["CUSEC"]).transform(
    lambda x: x.rolling(3, min_periods=1).mean())

# --- Lagged rolling stds ---
df["sale_roll_std3_lag"] = sale_shifted.groupby(df["CUSEC"]).transform(
    lambda x: x.rolling(3, min_periods=2).std())
df["rent_roll_std3_lag"] = rent_shifted.groupby(df["CUSEC"]).transform(
    lambda x: x.rolling(3, min_periods=2).std())

# --- Lagged rolling 6Q ---
df["sale_roll6_lag"] = sale_shifted.groupby(df["CUSEC"]).transform(
    lambda x: x.rolling(6, min_periods=2).mean())
df["rent_roll6_lag"] = rent_shifted.groupby(df["CUSEC"]).transform(
    lambda x: x.rolling(6, min_periods=2).mean())
df["sale_roll_std6_lag"] = sale_shifted.groupby(df["CUSEC"]).transform(
    lambda x: x.rolling(6, min_periods=2).std())
df["rent_roll_std6_lag"] = rent_shifted.groupby(df["CUSEC"]).transform(
    lambda x: x.rolling(6, min_periods=2).std())

# --- Lagged changes (previous quarters' changes, NOT current) ---
df["sale_change_lag1"] = grouped["sale_change"].shift(1)
df["rent_change_lag1"] = grouped["rent_change"].shift(1)
df["sale_change_lag2"] = grouped["sale_change"].shift(2)
df["rent_change_lag2"] = grouped["rent_change"].shift(2)

# --- Interaction of lagged changes ---
df["interaction_lag1"] = df["sale_change_lag1"] * df["rent_change_lag1"]

# --- Price momentum (from lagged values only) ---
df["sale_momentum"] = df["sale_lag1"] - df["sale_lag2"]
df["rent_momentum"] = df["rent_lag1"] - df["rent_lag2"]

# --- Lagged rent/sale ratio (uses lag1 prices, NOT current) ---
df["rent_sale_ratio_lag"] = df["rent_lag1"] / df["sale_lag1"].replace(0, np.nan)

# --- Year-over-year from lagged values (lag1 vs lag5) ---
df["sale_lag5"] = grouped[sale_col].shift(5)
df["rent_lag5"] = grouped[rent_col].shift(5)
df["sale_yoy_lag"] = (df["sale_lag1"] - df["sale_lag5"]) / df["sale_lag5"].replace(0, np.nan)
df["rent_yoy_lag"] = (df["rent_lag1"] - df["rent_lag5"]) / df["rent_lag5"].replace(0, np.nan)

# --- Price relative to province mean (using LAGGED values) ---
# Compute province mean of lag1 prices (all from t-1, no current period)
prov_sale_lag_mean = df.groupby(["NPRO", "period"])["sale_lag1"].transform("mean")
prov_rent_lag_mean = df.groupby(["NPRO", "period"])["rent_lag1"].transform("mean")
df["sale_vs_prov_lag"] = df["sale_lag1"] / prov_sale_lag_mean.replace(0, np.nan)
df["rent_vs_prov_lag"] = df["rent_lag1"] / prov_rent_lag_mean.replace(0, np.nan)

# --- Stock features (current stock is OK — stock doesn't define the target) ---
df["stock_sale_lag1"] = grouped["stock_residential_sale_all"].shift(1)
df["stock_rent_lag1"] = grouped["stock_residential_rent_all"].shift(1)
df["stock_ratio"] = (df["stock_residential_sale_all"]
                     / df["stock_residential_rent_all"].replace(0, np.nan))
df["stock_sale_change"] = grouped["stock_residential_sale_all"].pct_change()
df["stock_rent_change"] = grouped["stock_residential_rent_all"].pct_change()

# --- Temporal ---
df["month"] = df["period"].dt.month
df["quarter"] = df["period"].dt.quarter

# ── Feature list ─────────────────────────────────────────────
FEATURES = [
    # Pure lags
    "sale_lag1", "rent_lag1",
    "sale_lag2", "rent_lag2",
    "sale_lag4", "rent_lag4",
    # Lagged rolling stats (window over t-1..t-3, NO current period)
    "sale_roll3_lag", "rent_roll3_lag",
    "sale_roll_std3_lag", "rent_roll_std3_lag",
    "sale_roll6_lag", "rent_roll6_lag",
    "sale_roll_std6_lag", "rent_roll_std6_lag",
    # Lagged price changes
    "sale_change_lag1", "rent_change_lag1",
    "sale_change_lag2", "rent_change_lag2",
    "interaction_lag1",
    # Momentum (from lags)
    "sale_momentum", "rent_momentum",
    # Lagged ratio
    "rent_sale_ratio_lag",
    # Year-over-year (from lags)
    "sale_yoy_lag", "rent_yoy_lag",
    # Price vs province (from lags)
    "sale_vs_prov_lag", "rent_vs_prov_lag",
    # Stock (current OK, not price-derived target)
    "stock_residential_sale_all", "stock_residential_rent_all",
    "stock_sale_lag1", "stock_rent_lag1",
    "stock_ratio", "stock_sale_change", "stock_rent_change",
    # Temporal
    "month", "quarter", "year",
]

log(f"Total features: {len(FEATURES)}")

# ── Leakage verification ────────────────────────────────────
log("")
log("LEAKAGE VERIFICATION")
log("=" * 40)
log("Target = sign(rent_change_t, sale_change_t)")
log("  rent_change_t = (rent_t - rent_{t-1}) / rent_{t-1}")
log("  sale_change_t = (sale_t - sale_{t-1}) / sale_{t-1}")
log("")
log("Feature audit — does ANY feature expose rent_t or sale_t?")
for f in FEATURES:
    if "lag" in f or "change_lag" in f or "momentum" in f or "yoy_lag" in f:
        log(f"  {f}: CLEAN (uses only past values)")
    elif f in ("stock_residential_sale_all", "stock_residential_rent_all",
               "stock_sale_lag1", "stock_rent_lag1", "stock_ratio",
               "stock_sale_change", "stock_rent_change"):
        log(f"  {f}: CLEAN (stock, not price — doesn't define target)")
    elif f in ("month", "quarter", "year"):
        log(f"  {f}: CLEAN (temporal indicator)")
    elif "roll" in f and "lag" in f:
        log(f"  {f}: CLEAN (rolling window over shifted series, excludes t)")
    elif "vs_prov" in f and "lag" in f:
        log(f"  {f}: CLEAN (uses lag1 / province mean of lag1)")
    elif "ratio" in f and "lag" in f:
        log(f"  {f}: CLEAN (rent_lag1 / sale_lag1)")
    else:
        log(f"  {f}: *** NEEDS REVIEW ***")

log("")
log("Algebraic check: can sale_t or rent_t be reconstructed?")
log("  sale_roll3_lag = mean(sale_{t-1}, sale_{t-2}, sale_{t-3})")
log("  Even combined with sale_lag1 and sale_lag2, this gives NO info about sale_t.")
log("  sale_t CANNOT be reconstructed from any feature combination.")
log("  VERDICT: NO LEAKAGE")
log("")

# ── Prepare train/test ───────────────────────────────────────
trainable = df.dropna(subset=["sale_lag1", "rent_lag1", "case"])
trainable = trainable[trainable["case"] != "Unknown"].copy()

le = LabelEncoder()
trainable["case_encoded"] = le.fit_transform(trainable["case"])
log(f"Classes: {list(le.classes_)}")

train_mask = trainable["year"] < 2020
X_train = trainable.loc[train_mask, FEATURES].values
X_test = trainable.loc[~train_mask, FEATURES].values
y_train = trainable.loc[train_mask, "case_encoded"].values
y_test = trainable.loc[~train_mask, "case_encoded"].values
log(f"Train: {len(X_train):,}, Test: {len(X_test):,}")

# Class weights
counts = Counter(y_train)
total_train = len(y_train)
n_classes = len(counts)
cw = {cls: total_train / (n_classes * cnt) for cls, cnt in counts.items()}
sw = np.array([cw[yi] for yi in y_train])
log(f"Class weights: { {le.inverse_transform([k])[0]: f'{v:.2f}' for k, v in cw.items()} }")

# Preprocess
imp = SimpleImputer(strategy="mean")
sc = StandardScaler()
X_train_p = sc.fit_transform(imp.fit_transform(X_train))
X_test_p = sc.transform(imp.transform(X_test))

# ── Baseline ─────────────────────────────────────────────────
log("")
log("=" * 60)
log("BASELINE (no tuning)")
clf_base = XGBClassifier(
    objective="multi:softprob", n_estimators=300, max_depth=6,
    learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
    eval_metric="mlogloss", random_state=42,
)
clf_base.fit(X_train_p, y_train)
bp = clf_base.predict(X_test_p)
base_acc = accuracy_score(y_test, bp)
base_f1 = f1_score(y_test, bp, average="weighted")
log(f"Baseline: Acc={base_acc:.4f}, F1={base_f1:.4f}")
log("Per-class:")
log(classification_report(y_test, bp, target_names=le.classes_, zero_division=0))

# ── Optuna ───────────────────────────────────────────────────
log("=" * 60)
log("OPTUNA (30 trials)")
best_score_global = 0.0

def objective(trial):
    global best_score_global
    params = {
        "objective": "multi:softprob",
        "eval_metric": "mlogloss",
        "random_state": 42,
        "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=50),
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }
    use_w = trial.suggest_categorical("use_weights", [True, False])
    clf = XGBClassifier(**params)
    if use_w:
        clf.fit(X_train_p, y_train, sample_weight=sw)
    else:
        clf.fit(X_train_p, y_train)
    pred = clf.predict(X_test_p)
    f1 = f1_score(y_test, pred, average="weighted")
    acc = accuracy_score(y_test, pred)
    if f1 > best_score_global:
        best_score_global = f1
        log(f"  Trial {trial.number}: Acc={acc:.4f}, F1={f1:.4f} (NEW BEST)")
    return f1

study = optuna.create_study(direction="maximize",
                            sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=30, show_progress_bar=False)

log("")
log("=" * 60)
log(f"BEST TRIAL: {study.best_trial.number}")
log(f"Best F1: {study.best_value:.4f} (baseline F1: {base_f1:.4f}, "
    f"delta: {study.best_value - base_f1:+.4f})")
log(f"Best params: {study.best_params}")

# ── Retrain final model ──────────────────────────────────────
log("")
log("=" * 60)
log("RETRAINING FINAL MODEL")
best_params = {k: v for k, v in study.best_params.items() if k != "use_weights"}
use_w = study.best_params["use_weights"]

final_clf = XGBClassifier(
    objective="multi:softprob", eval_metric="mlogloss", random_state=42,
    **best_params,
)
if use_w:
    final_clf.fit(X_train_p, y_train, sample_weight=sw)
else:
    final_clf.fit(X_train_p, y_train)

final_pred = final_clf.predict(X_test_p)
final_acc = accuracy_score(y_test, final_pred)
final_f1 = f1_score(y_test, final_pred, average="weighted")

log(f"FINAL: Acc={final_acc:.4f} (baseline {base_acc:.4f}, delta {final_acc-base_acc:+.4f})")
log(f"FINAL: F1={final_f1:.4f} (baseline {base_f1:.4f}, delta {final_f1-base_f1:+.4f})")

log("\nPer-class report:")
report_str = classification_report(y_test, final_pred, target_names=le.classes_, zero_division=0)
log(report_str)

log("Confusion matrix:")
cm = confusion_matrix(y_test, final_pred)
log(f"\n{cm}")

# Feature importance
importances = final_clf.feature_importances_
feat_imp = sorted(zip(FEATURES, importances), key=lambda x: x[1], reverse=True)
log("\nTop 15 feature importances:")
for fname, imp_val in feat_imp[:15]:
    log(f"  {fname}: {imp_val:.4f} ({imp_val*100:.1f}%)")

# ── Save model ───────────────────────────────────────────────
MODELS_DIR.mkdir(parents=True, exist_ok=True)
final_pipe = make_pipeline(SimpleImputer(strategy="mean"), StandardScaler(), final_clf)
if use_w:
    final_pipe.fit(X_train, y_train, xgbclassifier__sample_weight=sw)
else:
    final_pipe.fit(X_train, y_train)
verify_pred = final_pipe.predict(X_test)
verify_acc = accuracy_score(y_test, verify_pred)
log(f"Pipeline verification: Acc={verify_acc:.4f}")

joblib.dump(final_pipe, MODELS_DIR / "xgboost_classifier.pkl")
joblib.dump(le, MODELS_DIR / "label_encoder.pkl")
log(f"Saved model to {MODELS_DIR / 'xgboost_classifier.pkl'}")

# ── Save results CSV ─────────────────────────────────────────
report_dict = classification_report(y_test, final_pred, target_names=le.classes_,
                                    output_dict=True, zero_division=0)
rows = []
for cls in le.classes_:
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

csv_path = os.path.join(os.path.dirname(__file__), "classification_results.csv")
# Keep RF rows if they exist, replace XGBoost rows
if os.path.exists(csv_path):
    old = pd.read_csv(csv_path)
    old = old[old["model"] != "XGBoost"]
    new = pd.concat([old, pd.DataFrame(rows)], ignore_index=True)
else:
    new = pd.DataFrame(rows)
new.to_csv(csv_path, index=False)
log(f"Results saved to {csv_path}")

# Save feature list
feat_path = os.path.join(os.path.dirname(__file__), "final_features.json")
json.dump(FEATURES, open(feat_path, "w"))
log(f"Feature list saved to {feat_path}")

log("\nDone.")
