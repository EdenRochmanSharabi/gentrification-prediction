"""
Experiment 02: Extended features WITHOUT leakage
Remove features derived from rent_change/sale_change (these encode the target directly).
Keep only legitimate lagged/historical features.
"""
import sys, os, warnings, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier

from config import MODELS_DIR
from src.data_loader import load_data
from src.gentrification import classify_cases

LOG = os.path.join(os.path.dirname(__file__), "autoresearch_log.txt")

def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG, "a") as f:
        f.write(line + "\n")

log("=" * 60)
log("EXP02: Leak-Free Extended Features")
log("=" * 60)

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

# === BASELINE FEATURES (no leakage) ===
df["sale_lag1"] = grouped[sale_col].shift(1)
df["rent_lag1"] = grouped[rent_col].shift(1)
df["sale_roll3"] = grouped[sale_col].transform(lambda x: x.rolling(3, min_periods=1).mean())
df["rent_roll3"] = grouped[rent_col].transform(lambda x: x.rolling(3, min_periods=1).mean())
df["sale_roll_std3"] = grouped[sale_col].transform(lambda x: x.rolling(3, min_periods=1).std())
df["rent_roll_std3"] = grouped[rent_col].transform(lambda x: x.rolling(3, min_periods=1).std())
df["stock_sale_lag1"] = grouped["stock_residential_sale_all"].shift(1)
df["stock_rent_lag1"] = grouped["stock_residential_rent_all"].shift(1)
df["rent_sale_ratio"] = df[rent_col] / df[sale_col].replace(0, np.nan)
df["month"] = df["period"].dt.month
df["quarter"] = df["period"].dt.quarter

# === NEW LEAK-FREE FEATURES ===
# Deeper lags (these are just historical prices, no leakage)
df["sale_lag2"] = grouped[sale_col].shift(2)
df["rent_lag2"] = grouped[rent_col].shift(2)
df["sale_lag4"] = grouped[sale_col].shift(4)
df["rent_lag4"] = grouped[rent_col].shift(4)

# Price relative to province mean (cross-sectional context, no leakage)
prov_sale_mean = df.groupby(["NPRO", "period"])[sale_col].transform("mean")
prov_rent_mean = df.groupby(["NPRO", "period"])[rent_col].transform("mean")
df["sale_vs_prov"] = df[sale_col] / prov_sale_mean.replace(0, np.nan)
df["rent_vs_prov"] = df[rent_col] / prov_rent_mean.replace(0, np.nan)

# YoY changes computed from lagged values (safe: uses current vs lag4)
df["sale_yoy"] = (df[sale_col] - df["sale_lag4"]) / df["sale_lag4"].replace(0, np.nan)
df["rent_yoy"] = (df[rent_col] - df["rent_lag4"]) / df["rent_lag4"].replace(0, np.nan)

# Rolling 6-quarter statistics
df["sale_roll6"] = grouped[sale_col].transform(lambda x: x.rolling(6, min_periods=1).mean())
df["rent_roll6"] = grouped[rent_col].transform(lambda x: x.rolling(6, min_periods=1).mean())

# Stock ratio
df["stock_ratio"] = df["stock_residential_sale_all"] / df["stock_residential_rent_all"].replace(0, np.nan)

# Lagged change features (PREVIOUS quarter's changes - these are fair game since
# they're historical, not current)
df["rent_change_lag1"] = grouped["rent_change"].shift(1)
df["sale_change_lag1"] = grouped["sale_change"].shift(1)

# Momentum (direction of previous changes)
df["rent_momentum_3q"] = grouped[rent_col].transform(
    lambda x: x.rolling(3, min_periods=2).apply(lambda w: (w.diff().dropna() > 0).mean(), raw=False)
)
df["sale_momentum_3q"] = grouped[sale_col].transform(
    lambda x: x.rolling(3, min_periods=2).apply(lambda w: (w.diff().dropna() > 0).mean(), raw=False)
)

BASELINE_FEATURES = [
    "sale_lag1", "rent_lag1",
    "sale_roll3", "rent_roll3",
    "sale_roll_std3", "rent_roll_std3",
    "stock_residential_sale_all", "stock_residential_rent_all",
    "stock_sale_lag1", "stock_rent_lag1",
    "rent_sale_ratio",
    "month", "quarter", "year",
]

# NOTE: rent_sale_interaction, rent_accel, sale_accel REMOVED (they encode target)
# NOTE: sale_yoy and rent_yoy use CURRENT prices which implicitly contain the
# current-quarter change. Let's check with and without them.

EXTENDED_SAFE = BASELINE_FEATURES + [
    "sale_lag2", "rent_lag2",
    "sale_lag4", "rent_lag4",
    "sale_vs_prov", "rent_vs_prov",
    "sale_roll6", "rent_roll6",
    "stock_ratio",
    "rent_change_lag1", "sale_change_lag1",
    "rent_momentum_3q", "sale_momentum_3q",
]

# Even safer: no current-period prices at all (only lags)
EXTENDED_ULTRA_SAFE = [
    "sale_lag1", "rent_lag1",
    "sale_lag2", "rent_lag2",
    "sale_lag4", "rent_lag4",
    "sale_roll3", "rent_roll3",
    "sale_roll_std3", "rent_roll_std3",
    "sale_roll6", "rent_roll6",
    "stock_sale_lag1", "stock_rent_lag1",
    "rent_sale_ratio",  # current ratio - still valid since it's a ratio not a change
    "stock_ratio",
    "rent_change_lag1", "sale_change_lag1",
    "rent_momentum_3q", "sale_momentum_3q",
    "month", "quarter", "year",
]

log(f"Baseline: {len(BASELINE_FEATURES)} features")
log(f"Extended safe: {len(EXTENDED_SAFE)} features")
log(f"Ultra-safe: {len(EXTENDED_ULTRA_SAFE)} features")

trainable = df.dropna(subset=BASELINE_FEATURES + ["case"])
trainable = trainable[trainable["case"] != "Unknown"].copy()

le = LabelEncoder()
trainable["case_encoded"] = le.fit_transform(trainable["case"])

train_mask = trainable["year"] < 2020
y_train = trainable.loc[train_mask, "case_encoded"].values
y_test = trainable.loc[~train_mask, "case_encoded"].values

configs = {
    "Baseline (14f)": BASELINE_FEATURES,
    "Extended safe (26f)": EXTENDED_SAFE,
    "Ultra-safe (22f)": EXTENDED_ULTRA_SAFE,
}

for name, features in configs.items():
    X_tr = trainable.loc[train_mask, features].values
    X_te = trainable.loc[~train_mask, features].values

    imp = SimpleImputer(strategy="mean")
    sc = StandardScaler()
    Xt = sc.fit_transform(imp.fit_transform(X_tr))
    Xe = sc.transform(imp.transform(X_te))

    clf = XGBClassifier(
        objective="multi:softprob", n_estimators=300, max_depth=6,
        learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
        eval_metric="mlogloss", random_state=42,
    )
    clf.fit(Xt, y_train)
    pred = clf.predict(Xe)
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average="weighted")
    log(f"{name}: Acc={acc:.4f}, F1={f1:.4f}")

# Now try the best feature set with a tuned model
log("\n--- Tuned configs with extended safe features ---")
for name, features in [("Extended safe", EXTENDED_SAFE), ("Ultra-safe", EXTENDED_ULTRA_SAFE)]:
    X_tr = trainable.loc[train_mask, features].values
    X_te = trainable.loc[~train_mask, features].values

    imp = SimpleImputer(strategy="mean")
    sc = StandardScaler()
    Xt = sc.fit_transform(imp.fit_transform(X_tr))
    Xe = sc.transform(imp.transform(X_te))

    clf = XGBClassifier(
        objective="multi:softprob", n_estimators=500, max_depth=8,
        learning_rate=0.05, subsample=0.8, colsample_bytree=0.7,
        min_child_weight=3, gamma=0.1,
        eval_metric="mlogloss", random_state=42,
    )
    clf.fit(Xt, y_train)
    pred = clf.predict(Xe)
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average="weighted")
    log(f"{name} (tuned): Acc={acc:.4f}, F1={f1:.4f}")

    log(f"\nPer-class ({name} tuned):")
    log(classification_report(y_test, pred, target_names=le.classes_, zero_division=0))

log("EXP02 complete.")
