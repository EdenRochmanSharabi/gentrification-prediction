"""
Experiment 05: Truly clean features - NO current-period prices
The target (case) is derived from current rent_change and sale_change.
rent_change = (rent_t - rent_{t-1}) / rent_{t-1}
So ANY feature containing rent_t or sale_t (current period) combined with lag1
allows the model to reconstruct the target.

This experiment uses ONLY:
- Lagged prices (t-1, t-2, t-4) -- NO current prices
- Rolling stats computed from LAGGED values only
- Lagged stock
- Temporal indicators
- Province-level historical statistics

The CURRENT values of stock_residential_sale_all and stock_residential_rent_all
are acceptable since they measure listing count, not price levels, and don't
encode rent_change/sale_change.
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
import joblib

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
log("EXP05: Truly Clean Features (NO current-period prices)")
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

# === STRICTLY LAGGED FEATURES (no current prices) ===
# Price lags
df["sale_lag1"] = grouped[sale_col].shift(1)
df["rent_lag1"] = grouped[rent_col].shift(1)
df["sale_lag2"] = grouped[sale_col].shift(2)
df["rent_lag2"] = grouped[rent_col].shift(2)
df["sale_lag4"] = grouped[sale_col].shift(4)
df["rent_lag4"] = grouped[rent_col].shift(4)

# Rolling stats from LAGGED series (shift first, then roll)
grouped2 = df.groupby("CUSEC")
df["sale_lag1_roll3"] = grouped2["sale_lag1"].transform(lambda x: x.rolling(3, min_periods=1).mean())
df["rent_lag1_roll3"] = grouped2["rent_lag1"].transform(lambda x: x.rolling(3, min_periods=1).mean())
df["sale_lag1_roll_std3"] = grouped2["sale_lag1"].transform(lambda x: x.rolling(3, min_periods=1).std())
df["rent_lag1_roll_std3"] = grouped2["rent_lag1"].transform(lambda x: x.rolling(3, min_periods=1).std())

# Stock (current is OK - measures listing volume not prices)
# But stock lag is safer
df["stock_sale_lag1"] = grouped["stock_residential_sale_all"].shift(1)
df["stock_rent_lag1"] = grouped["stock_residential_rent_all"].shift(1)

# Rent-to-sale ratio from LAGGED prices
df["rent_sale_ratio_lag1"] = df["rent_lag1"] / df["sale_lag1"].replace(0, np.nan)

# Historical change rates (from even earlier)
df["sale_change_lag1"] = grouped["sale_change"].shift(1)  # change from t-2 to t-1
df["rent_change_lag1"] = grouped["rent_change"].shift(1)
df["sale_change_lag2"] = grouped["sale_change"].shift(2)  # change from t-3 to t-2
df["rent_change_lag2"] = grouped["rent_change"].shift(2)

# YoY from lagged values: (lag1 - lag4+1) / lag4+1 = recent historical trend
df["sale_lag1_yoy"] = (df["sale_lag1"] - grouped[sale_col].shift(5)) / grouped[sale_col].shift(5).replace(0, np.nan)
df["rent_lag1_yoy"] = (df["rent_lag1"] - grouped[rent_col].shift(5)) / grouped[rent_col].shift(5).replace(0, np.nan)

# Momentum: fraction of positive changes in last 3 historical periods
grouped3 = df.groupby("CUSEC")
df["sale_momentum"] = grouped3["sale_change_lag1"].transform(
    lambda x: x.rolling(3, min_periods=1).apply(lambda w: (w > 0).mean(), raw=True)
)
df["rent_momentum"] = grouped3["rent_change_lag1"].transform(
    lambda x: x.rolling(3, min_periods=1).apply(lambda w: (w > 0).mean(), raw=True)
)

# Province-level stats from lagged prices
prov_lag_sale_mean = df.groupby(["NPRO", "period"])["sale_lag1"].transform("mean")
prov_lag_rent_mean = df.groupby(["NPRO", "period"])["rent_lag1"].transform("mean")
df["sale_vs_prov_lag"] = df["sale_lag1"] / prov_lag_sale_mean.replace(0, np.nan)
df["rent_vs_prov_lag"] = df["rent_lag1"] / prov_lag_rent_mean.replace(0, np.nan)

# Temporal
df["month"] = df["period"].dt.month
df["quarter"] = df["period"].dt.quarter

# Stock current values (not price-based, no leakage)
FEATURES_TRULY_CLEAN = [
    "sale_lag1", "rent_lag1",
    "sale_lag2", "rent_lag2",
    "sale_lag4", "rent_lag4",
    "sale_lag1_roll3", "rent_lag1_roll3",
    "sale_lag1_roll_std3", "rent_lag1_roll_std3",
    "stock_residential_sale_all", "stock_residential_rent_all",
    "stock_sale_lag1", "stock_rent_lag1",
    "rent_sale_ratio_lag1",
    "sale_change_lag1", "rent_change_lag1",
    "sale_change_lag2", "rent_change_lag2",
    "sale_lag1_yoy", "rent_lag1_yoy",
    "sale_momentum", "rent_momentum",
    "sale_vs_prov_lag", "rent_vs_prov_lag",
    "month", "quarter", "year",
]

# Also check: is the BASELINE leaking?
# Baseline uses: rent_sale_ratio = df[rent_col] / df[sale_col]  → CURRENT prices
# And sale_roll3/rent_roll3 include CURRENT value in the rolling window
# So the baseline features have partial leakage too but it's diluted
# which is why baseline gets 72% and not 99%

BASELINE_FEATURES = [
    "sale_lag1", "rent_lag1",
    "sale_lag1_roll3", "rent_lag1_roll3",  # using lagged rolling instead
    "sale_lag1_roll_std3", "rent_lag1_roll_std3",
    "stock_residential_sale_all", "stock_residential_rent_all",
    "stock_sale_lag1", "stock_rent_lag1",
    "rent_sale_ratio_lag1",  # using lagged ratio
    "month", "quarter", "year",
]

log(f"Truly clean features: {len(FEATURES_TRULY_CLEAN)}")
log(f"Clean baseline features: {len(BASELINE_FEATURES)}")

trainable = df.dropna(subset=["sale_lag1", "rent_lag1", "case"])
trainable = trainable[trainable["case"] != "Unknown"].copy()

le = LabelEncoder()
trainable["case_encoded"] = le.fit_transform(trainable["case"])

train_mask = trainable["year"] < 2020
y_train = trainable.loc[train_mask, "case_encoded"].values
y_test = trainable.loc[~train_mask, "case_encoded"].values

configs = [
    ("Clean baseline (14f)", BASELINE_FEATURES, dict(n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8)),
    ("Clean extended (28f)", FEATURES_TRULY_CLEAN, dict(n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8)),
    ("Clean extended deep (28f)", FEATURES_TRULY_CLEAN, dict(n_estimators=500, max_depth=8, learning_rate=0.05, subsample=0.8, colsample_bytree=0.7, min_child_weight=3, gamma=0.1)),
    ("Clean extended deeper (28f)", FEATURES_TRULY_CLEAN, dict(n_estimators=700, max_depth=10, learning_rate=0.03, subsample=0.8, colsample_bytree=0.7, min_child_weight=2, gamma=0.05)),
]

best_acc = 0
best_f1 = 0
best_name = ""
best_pred = None

for name, features, hp in configs:
    X_tr = trainable.loc[train_mask, features].values
    X_te = trainable.loc[~train_mask, features].values

    imp = SimpleImputer(strategy="mean")
    sc = StandardScaler()
    Xt = sc.fit_transform(imp.fit_transform(X_tr))
    Xe = sc.transform(imp.transform(X_te))

    clf = XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss", random_state=42,
        **hp,
    )
    clf.fit(Xt, y_train)
    pred = clf.predict(Xe)
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average="weighted")
    log(f"{name}: Acc={acc:.4f}, F1={f1:.4f}")

    if f1 > best_f1:
        best_f1 = f1
        best_acc = acc
        best_name = name
        best_pred = pred
        best_clf = clf
        best_features = features

log(f"\nBest: {best_name}, Acc={best_acc:.4f}, F1={best_f1:.4f}")
log(f"\nPer-class ({best_name}):")
log(classification_report(y_test, best_pred, target_names=le.classes_, zero_division=0))

# Feature importance
importance = best_clf.feature_importances_
sorted_idx = np.argsort(importance)[::-1]
log("\nTop features:")
for i in range(min(15, len(best_features))):
    idx = sorted_idx[i]
    log(f"  {best_features[idx]}: {importance[idx]:.4f}")

log(f"\nEXP05 RESULT: {best_name}, Acc={best_acc:.4f}, F1={best_f1:.4f}")
