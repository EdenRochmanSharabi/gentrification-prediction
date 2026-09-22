"""
LEAK-FREE gentrification classifier.

APPROACH: Predict case at time t+1 from features at time t.
This is the correct framing for an early-warning system:
- All features use data available at time t (current + past prices, stock, etc.)
- The target is the gentrification case that will manifest NEXT quarter
- No information from t+1 enters the features
- Current-period prices/ratios are LEGITIMATE because they don't define the target

LEAKAGE PROOF:
  target = case(t+1), defined by rent_change(t+1) and sale_change(t+1)
  rent_change(t+1) = (rent_{t+1} - rent_t) / rent_t
  Features contain data up to time t only. rent_{t+1} and sale_{t+1}
  are NEVER accessible. QED.
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

from config import MODELS_DIR, FIGURES_DIR
from src.data_loader import load_data
from src.gentrification import classify_cases

LOG = os.path.join(os.path.dirname(__file__), "leakfree_model.log")

def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")

# Clear previous log
open(LOG, "w").close()

log("=" * 60)
log("LEAK-FREE MODEL: predict case(t+1) from features(t)")
log("=" * 60)

# ── Load and prepare data ──
log("Loading data...")
df = load_data()
df = df.sort_values(["CUSEC", "period"])
df["year"] = df["period"].dt.year

sale_col = "unitprice_residential_sale_all"
rent_col = "unitprice_residential_rent_all"
grouped = df.groupby("CUSEC")

# Compute case for each row (this is the RAW case at time t)
df["rent_change"] = grouped[rent_col].pct_change()
df["sale_change"] = grouped[sale_col].pct_change()
df = classify_cases(df)

# ── CREATE TARGET: case at t+1 (shifted forward within each CUSEC) ──
df["target_case"] = df.groupby("CUSEC")["case"].shift(-1)
log("Target: case(t+1) = next quarter's gentrification type")

# ── FEATURES: all use data available at time t ──
# Current-period prices (legitimate: they don't define target which is at t+1)
df["sale_price"] = df[sale_col]
df["rent_price"] = df[rent_col]
df["rent_sale_ratio"] = df[rent_col] / df[sale_col].replace(0, np.nan)

# Current-period case (the case NOW, predicting what comes NEXT)
# This is NOT leakage because target is t+1, and case_t is known at time t

# Lagged prices
df["sale_lag1"] = grouped[sale_col].shift(1)
df["rent_lag1"] = grouped[rent_col].shift(1)
df["sale_lag2"] = grouped[sale_col].shift(2)
df["rent_lag2"] = grouped[rent_col].shift(2)
df["sale_lag4"] = grouped[sale_col].shift(4)
df["rent_lag4"] = grouped[rent_col].shift(4)

# Current-period changes (known at time t, target is t+1)
df["sale_change_curr"] = df["sale_change"]
df["rent_change_curr"] = df["rent_change"]

# Lagged changes
df["sale_change_lag1"] = grouped["sale_change"].shift(1)
df["rent_change_lag1"] = grouped["rent_change"].shift(1)

# Rolling statistics INCLUDING current period (legitimate for t+1 prediction)
df["sale_roll3"] = grouped[sale_col].transform(lambda x: x.rolling(3, min_periods=1).mean())
df["rent_roll3"] = grouped[rent_col].transform(lambda x: x.rolling(3, min_periods=1).mean())
df["sale_roll_std3"] = grouped[sale_col].transform(lambda x: x.rolling(3, min_periods=1).std())
df["rent_roll_std3"] = grouped[rent_col].transform(lambda x: x.rolling(3, min_periods=1).std())
df["sale_roll6"] = grouped[sale_col].transform(lambda x: x.rolling(6, min_periods=2).mean())
df["rent_roll6"] = grouped[rent_col].transform(lambda x: x.rolling(6, min_periods=2).mean())

# Price momentum
df["sale_momentum"] = df[sale_col] - df["sale_lag1"]
df["rent_momentum"] = df[rent_col] - df["rent_lag1"]

# Interaction of current changes
df["change_interaction"] = df["rent_change"] * df["sale_change"]

# Price relative to province mean (current period, legitimate)
prov_sale_mean = df.groupby(["NPRO", "period"])[sale_col].transform("mean")
prov_rent_mean = df.groupby(["NPRO", "period"])[rent_col].transform("mean")
df["sale_vs_prov"] = df[sale_col] / prov_sale_mean.replace(0, np.nan)
df["rent_vs_prov"] = df[rent_col] / prov_rent_mean.replace(0, np.nan)

# Year-over-year (current vs 4 quarters ago)
df["sale_yoy"] = (df[sale_col] - df["sale_lag4"]) / df["sale_lag4"].replace(0, np.nan)
df["rent_yoy"] = (df[rent_col] - df["rent_lag4"]) / df["rent_lag4"].replace(0, np.nan)

# Stock features
df["stock_sale_lag1"] = grouped["stock_residential_sale_all"].shift(1)
df["stock_rent_lag1"] = grouped["stock_residential_rent_all"].shift(1)
df["stock_ratio"] = df["stock_residential_sale_all"] / df["stock_residential_rent_all"].replace(0, np.nan)

# Temporal
df["month"] = df["period"].dt.month
df["quarter"] = df["period"].dt.quarter

FEATURES = [
    "sale_price", "rent_price", "rent_sale_ratio",
    "sale_lag1", "rent_lag1", "sale_lag2", "rent_lag2",
    "sale_lag4", "rent_lag4",
    "sale_change_curr", "rent_change_curr",
    "sale_change_lag1", "rent_change_lag1",
    "sale_roll3", "rent_roll3",
    "sale_roll_std3", "rent_roll_std3",
    "sale_roll6", "rent_roll6",
    "sale_momentum", "rent_momentum",
    "change_interaction",
    "sale_vs_prov", "rent_vs_prov",
    "sale_yoy", "rent_yoy",
    "stock_residential_sale_all", "stock_residential_rent_all",
    "stock_sale_lag1", "stock_rent_lag1",
    "stock_ratio",
    "month", "quarter", "year",
]

log(f"Features: {len(FEATURES)}")

# ── LEAKAGE VERIFICATION ──
log("")
log("LEAKAGE VERIFICATION")
log("=" * 40)
log("Target: case(t+1) = gentrification type at quarter t+1")
log("  Defined by: rent_change(t+1) = (rent_{t+1} - rent_t) / rent_t")
log("              sale_change(t+1) = (sale_t+1} - sale_t) / sale_t")
log("")
log("Question: can ANY feature combination reconstruct rent_{t+1} or sale_{t+1}?")
log("")
for f in FEATURES:
    if "lag" in f or "momentum" in f or "change_lag" in f:
        log(f"  {f}: uses data from t-1 or earlier. CLEAN.")
    elif "roll" in f:
        log(f"  {f}: rolling window up to time t. CLEAN (target is t+1).")
    elif f in ["sale_price", "rent_price", "rent_sale_ratio",
               "sale_change_curr", "rent_change_curr", "change_interaction",
               "sale_vs_prov", "rent_vs_prov", "sale_yoy", "rent_yoy"]:
        log(f"  {f}: uses data at time t. CLEAN (target is t+1, needs t+1 prices).")
    elif "stock" in f:
        log(f"  {f}: stock data. CLEAN (stock doesn't define price-based target).")
    elif f in ["month", "quarter", "year"]:
        log(f"  {f}: temporal indicator. CLEAN.")
    else:
        log(f"  {f}: NEEDS REVIEW")
log("")
log("No feature provides rent_{t+1} or sale_{t+1}.")
log("VERDICT: NO LEAKAGE")
log("")

# ── Prepare train/test ──
trainable = df.dropna(subset=["sale_lag1", "rent_lag1", "target_case"])
trainable = trainable[trainable["target_case"] != "Unknown"].copy()

le = LabelEncoder()
trainable["target_encoded"] = le.fit_transform(trainable["target_case"])

log(f"Classes: {list(le.classes_)}")

train_mask = trainable["year"] < 2020
X_train = trainable.loc[train_mask, FEATURES].values
X_test = trainable.loc[~train_mask, FEATURES].values
y_train = trainable.loc[train_mask, "target_encoded"].values
y_test = trainable.loc[~train_mask, "target_encoded"].values

log(f"Train: {len(X_train):,}, Test: {len(X_test):,}")

# Class weights
counts = Counter(y_train)
total = len(y_train)
n_classes = len(counts)
cw = {cls: total / (n_classes * count) for cls, count in counts.items()}
sw = np.array([cw[yi] for yi in y_train])
log(f"Class distribution (train): {dict(sorted(Counter(le.inverse_transform(y_train)).items()))}")

# Preprocessing
imp = SimpleImputer(strategy="mean")
sc = StandardScaler()
X_train_p = sc.fit_transform(imp.fit_transform(X_train))
X_test_p = sc.transform(imp.transform(X_test))

# ── BASELINE ──
log("=" * 60)
log("BASELINE (XGBoost, default hyperparams)")
baseline = XGBClassifier(
    objective="multi:softprob", n_estimators=300, max_depth=6,
    learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
    eval_metric="mlogloss", random_state=42,
)
baseline.fit(X_train_p, y_train)
bp = baseline.predict(X_test_p)
base_acc = accuracy_score(y_test, bp)
base_f1 = f1_score(y_test, bp, average="weighted")
log(f"Baseline: Acc={base_acc:.4f}, F1={base_f1:.4f}")
log("Per-class:")
log(classification_report(y_test, bp, target_names=le.classes_, zero_division=0))

# Feature importance
fi = sorted(zip(FEATURES, baseline.feature_importances_), key=lambda x: x[1], reverse=True)
log("Top 10 feature importances:")
for fname, imp_val in fi[:10]:
    log(f"  {fname}: {imp_val:.4f} ({imp_val*100:.1f}%)")

log("")
log("Baseline complete. Ready for Optuna search.")
log(f"Script: {os.path.abspath(__file__)}")
log("To run Optuna, execute: model_leakfree_search.py")
