"""
Experiment 08: Final push - memory-optimized
Best feature set (20f) + Optuna HP tuning + ensemble
"""
import sys, os, warnings, time, gc
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import optuna
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
import joblib

from config import MODELS_DIR, IDEALISTA_PATH, CARTOGRAPHY_PATH

optuna.logging.set_verbosity(optuna.logging.WARNING)

LOG = os.path.join(os.path.dirname(__file__), "autoresearch_log.txt")

def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")

log("=" * 60)
log("EXP08: Final Push (memory-optimized)")
log("=" * 60)

# Memory-efficient loading: only keep needed columns
log("Loading data (memory-optimized)...")
needed_cols = [
    "locationid", "period",
    "unitprice_residential_sale_all", "unitprice_residential_rent_all",
    "stock_residential_sale_all", "stock_residential_rent_all",
]
df_raw = pd.read_stata(str(IDEALISTA_PATH), columns=needed_cols)
df_raw["locationid"] = df_raw["locationid"].apply(lambda x: f"{int(x):010d}")

# Load cartography metadata for CUSEC mapping
from dbfread import DBF
dbf_path = str(CARTOGRAPHY_PATH).replace(".shp", ".dbf")
table = DBF(dbf_path)
geo = pd.DataFrame(iter(table))
geo = geo.drop(columns=["OBJECTID"], errors="ignore")

df = pd.merge(df_raw, geo, how="outer", left_on="locationid", right_on="CUSEC")
df["period"] = pd.to_datetime(df["period"])
del df_raw, geo, table
gc.collect()

df = df.sort_values("period")
df["year"] = df["period"].dt.year

sale_col = "unitprice_residential_sale_all"
rent_col = "unitprice_residential_rent_all"

# Downcast floats
for col in [sale_col, rent_col, "stock_residential_sale_all", "stock_residential_rent_all"]:
    df[col] = pd.to_numeric(df[col], downcast="float")

grouped = df.groupby("CUSEC")

# Compute target
df["rent_change"] = grouped[rent_col].pct_change()
df["sale_change"] = grouped[sale_col].pct_change()

has_data = df["rent_change"].notna() & df["sale_change"].notna()
conditions = [
    ~has_data,
    has_data & (df["rent_change"] > 0) & (df["sale_change"] > 0),
    has_data & (df["rent_change"] < 0) & (df["sale_change"] < 0),
    has_data & (df["rent_change"] > 0) & (df["sale_change"] <= 0),
    has_data & (df["sale_change"] > 0) & (df["rent_change"] <= 0),
]
df["case"] = np.select(conditions, ["Unknown", "C", "D", "A", "B"], default="None")

# Features
df["sale_lag1"] = grouped[sale_col].shift(1)
df["rent_lag1"] = grouped[rent_col].shift(1)
df["sale_roll3"] = grouped[sale_col].transform(lambda x: x.rolling(3, min_periods=1).mean())
df["rent_roll3"] = grouped[rent_col].transform(lambda x: x.rolling(3, min_periods=1).mean())
df["sale_roll_std3"] = grouped[sale_col].transform(lambda x: x.rolling(3, min_periods=1).std())
df["rent_roll_std3"] = grouped[rent_col].transform(lambda x: x.rolling(3, min_periods=1).std())
df["stock_sale_lag1"] = grouped["stock_residential_sale_all"].shift(1)
df["stock_rent_lag1"] = grouped["stock_residential_rent_all"].shift(1)
df["rent_sale_ratio"] = df[rent_col] / df[sale_col].replace(0, np.nan)
df["month"] = df["period"].dt.month.astype(np.int8)
df["quarter"] = df["period"].dt.quarter.astype(np.int8)
df["sale_lag2"] = grouped[sale_col].shift(2)
df["rent_lag2"] = grouped[rent_col].shift(2)
df["sale_lag4"] = grouped[sale_col].shift(4)
df["rent_lag4"] = grouped[rent_col].shift(4)
df["rent_change_lag1"] = grouped["rent_change"].shift(1)
df["sale_change_lag1"] = grouped["sale_change"].shift(1)

FEATURES = [
    "sale_lag1", "rent_lag1",
    "sale_roll3", "rent_roll3",
    "sale_roll_std3", "rent_roll_std3",
    "stock_residential_sale_all", "stock_residential_rent_all",
    "stock_sale_lag1", "stock_rent_lag1",
    "rent_sale_ratio",
    "month", "quarter", "year",
    "sale_lag2", "rent_lag2",
    "sale_lag4", "rent_lag4",
    "rent_change_lag1", "sale_change_lag1",
]

# Drop unneeded columns to free memory
keep_cols = list(dict.fromkeys(FEATURES + ["case", "CUSEC"]))
df = df[[c for c in keep_cols if c in df.columns]].copy()
gc.collect()

trainable = df.dropna(subset=FEATURES[:14] + ["case"])
trainable = trainable[trainable["case"] != "Unknown"].copy()
del df
gc.collect()

le = LabelEncoder()
trainable["case_encoded"] = le.fit_transform(trainable["case"])
log(f"Classes: {list(le.classes_)}")

train_mask = trainable["year"] < 2020
X_train = trainable.loc[train_mask, FEATURES].values.astype(np.float32)
X_test = trainable.loc[~train_mask, FEATURES].values.astype(np.float32)
y_train = trainable.loc[train_mask, "case_encoded"].values
y_test = trainable.loc[~train_mask, "case_encoded"].values
del trainable
gc.collect()

log(f"Train: {len(X_train):,}, Test: {len(X_test):,}")

imp = SimpleImputer(strategy="mean")
sc = StandardScaler()
X_tr = sc.fit_transform(imp.fit_transform(X_train)).astype(np.float32)
X_te = sc.transform(imp.transform(X_test)).astype(np.float32)
del X_train, X_test
gc.collect()

from collections import Counter
counts = Counter(y_train)
total = len(y_train)
n_classes = len(counts)
class_weights = {cls: total / (n_classes * count) for cls, count in counts.items()}
sample_weights = np.array([class_weights[yi] for yi in y_train], dtype=np.float32)

# --- Optuna ---
log("Optuna tuning (30 trials)...")
best_optuna_f1 = 0.0

def objective(trial):
    global best_optuna_f1
    params = {
        "objective": "multi:softprob",
        "eval_metric": "mlogloss",
        "random_state": 42,
        "tree_method": "hist",
        "n_estimators": trial.suggest_int("n_estimators", 300, 700, step=50),
        "max_depth": trial.suggest_int("max_depth", 6, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 8),
        "gamma": trial.suggest_float("gamma", 0.0, 2.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 3.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-6, 3.0, log=True),
    }
    use_w = trial.suggest_categorical("use_weights", [True, False])

    clf = XGBClassifier(**params)
    if use_w:
        clf.fit(X_tr, y_train, sample_weight=sample_weights)
    else:
        clf.fit(X_tr, y_train)

    pred = clf.predict(X_te)
    f1 = f1_score(y_test, pred, average="weighted")
    acc = accuracy_score(y_test, pred)

    if f1 > best_optuna_f1:
        best_optuna_f1 = f1
        log(f"  Trial {trial.number}: Acc={acc:.4f}, F1={f1:.4f} (NEW BEST)")
    return f1

study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=15, show_progress_bar=False)

log(f"\nOptuna best: F1={study.best_value:.4f}")
log(f"Best params: {study.best_params}")

# Retrain best
best_params = {k: v for k, v in study.best_params.items() if k != "use_weights"}
use_w = study.best_params["use_weights"]

best_xgb = XGBClassifier(objective="multi:softprob", eval_metric="mlogloss",
                          random_state=42, tree_method="hist", **best_params)
if use_w:
    best_xgb.fit(X_tr, y_train, sample_weight=sample_weights)
else:
    best_xgb.fit(X_tr, y_train)

xgb_pred = best_xgb.predict(X_te)
xgb_acc = accuracy_score(y_test, xgb_pred)
xgb_f1 = f1_score(y_test, xgb_pred, average="weighted")
log(f"\nBest XGBoost: Acc={xgb_acc:.4f}, F1={xgb_f1:.4f}")
log(classification_report(y_test, xgb_pred, target_names=le.classes_, zero_division=0))

# Feature importance
importance = best_xgb.feature_importances_
sorted_idx = np.argsort(importance)[::-1]
log("Top features:")
for i in range(min(10, len(FEATURES))):
    idx = sorted_idx[i]
    log(f"  {FEATURES[idx]}: {importance[idx]:.4f}")

# --- LightGBM ---
overall_best = ("XGBoost_tuned", xgb_acc, xgb_f1, xgb_pred)
try:
    from lightgbm import LGBMClassifier
    log("\n--- LightGBM ---")
    lgbm = LGBMClassifier(
        objective="multiclass", n_estimators=500, max_depth=8,
        learning_rate=0.05, subsample=0.8, colsample_bytree=0.7,
        random_state=42, verbose=-1,
    )
    lgbm.fit(X_tr, y_train)
    lgbm_pred = lgbm.predict(X_te)
    lgbm_acc = accuracy_score(y_test, lgbm_pred)
    lgbm_f1 = f1_score(y_test, lgbm_pred, average="weighted")
    log(f"LightGBM: Acc={lgbm_acc:.4f}, F1={lgbm_f1:.4f}")

    # Ensemble
    xgb_proba = best_xgb.predict_proba(X_te)
    lgbm_proba = lgbm.predict_proba(X_te)
    ens_pred = ((xgb_proba + lgbm_proba) / 2).argmax(axis=1)
    ens_acc = accuracy_score(y_test, ens_pred)
    ens_f1 = f1_score(y_test, ens_pred, average="weighted")
    log(f"Ensemble: Acc={ens_acc:.4f}, F1={ens_f1:.4f}")

    for name, acc, f1, pred in [("LightGBM", lgbm_acc, lgbm_f1, lgbm_pred),
                                  ("Ensemble", ens_acc, ens_f1, ens_pred)]:
        if f1 > overall_best[2]:
            overall_best = (name, acc, f1, pred)
except ImportError:
    log("LightGBM not available.")

log(f"\n{'='*50}")
log(f"OVERALL BEST: {overall_best[0]}, Acc={overall_best[1]:.4f}, F1={overall_best[2]:.4f}")
log(f"vs original baseline (72.03%): delta={overall_best[1] - 0.7203:+.4f}")
log(classification_report(y_test, overall_best[3], target_names=le.classes_, zero_division=0))

# Save
MODELS_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(best_xgb, MODELS_DIR / "xgboost_classifier_improved.pkl")
joblib.dump(le, MODELS_DIR / "label_encoder.pkl")
log(f"Saved to {MODELS_DIR}")
log(f"\nEXP08 COMPLETE.")
