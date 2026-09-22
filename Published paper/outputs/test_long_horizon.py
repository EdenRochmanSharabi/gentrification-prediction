"""Test long-horizon prediction: t+4, t+8, t+12 (1, 2, 3 years ahead)."""
import sys, os, warnings, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from collections import Counter

from src.data_loader import load_data
from src.gentrification import classify_cases

LOG = os.path.join(os.path.dirname(__file__), "long_horizon.log")
open(LOG, "w").close()

def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")

log("=" * 60)
log("LONG-HORIZON PREDICTION TEST")
log("=" * 60)

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

# Features (same as model_leakfree.py baseline)
df["sale_price"] = df[sale_col]
df["rent_price"] = df[rent_col]
df["rent_sale_ratio"] = df[rent_col] / df[sale_col].replace(0, np.nan)
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
df["sale_roll3"] = grouped[sale_col].transform(lambda x: x.rolling(3, min_periods=1).mean())
df["rent_roll3"] = grouped[rent_col].transform(lambda x: x.rolling(3, min_periods=1).mean())
df["sale_roll_std3"] = grouped[sale_col].transform(lambda x: x.rolling(3, min_periods=1).std())
df["rent_roll_std3"] = grouped[rent_col].transform(lambda x: x.rolling(3, min_periods=1).std())
df["sale_roll6"] = grouped[sale_col].transform(lambda x: x.rolling(6, min_periods=2).mean())
df["rent_roll6"] = grouped[rent_col].transform(lambda x: x.rolling(6, min_periods=2).mean())
df["sale_momentum"] = df[sale_col] - df["sale_lag1"]
df["rent_momentum"] = df[rent_col] - df["rent_lag1"]
df["change_interaction"] = df["rent_change"] * df["sale_change"]
prov_sale_mean = df.groupby(["NPRO", "period"])[sale_col].transform("mean")
prov_rent_mean = df.groupby(["NPRO", "period"])[rent_col].transform("mean")
df["sale_vs_prov"] = df[sale_col] / prov_sale_mean.replace(0, np.nan)
df["rent_vs_prov"] = df[rent_col] / prov_rent_mean.replace(0, np.nan)
df["sale_yoy"] = (df[sale_col] - df["sale_lag4"]) / df["sale_lag4"].replace(0, np.nan)
df["rent_yoy"] = (df[rent_col] - df["rent_lag4"]) / df["rent_lag4"].replace(0, np.nan)
df["stock_sale_lag1"] = grouped["stock_residential_sale_all"].shift(1)
df["stock_rent_lag1"] = grouped["stock_residential_rent_all"].shift(1)
df["stock_ratio"] = df["stock_residential_sale_all"] / df["stock_residential_rent_all"].replace(0, np.nan)
df["month"] = df["period"].dt.month
df["quarter"] = df["period"].dt.quarter

# Current case as feature
le_case = LabelEncoder()
df["case_encoded_feat"] = le_case.fit_transform(df["case"].fillna("Unknown"))

FEATURES = [
    "sale_price", "rent_price", "rent_sale_ratio",
    "case_encoded_feat",
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

HORIZONS = [1, 2, 4, 8, 12]

for h in HORIZONS:
    log("")
    log(f"{'=' * 60}")
    log(f"HORIZON: t+{h} ({h*3} meses / {h/4:.1f} años)")
    log(f"{'=' * 60}")

    df[f"target_{h}"] = df.groupby("CUSEC")["case"].shift(-h)

    trainable = df.dropna(subset=["sale_lag1", "rent_lag1", f"target_{h}"])
    trainable = trainable[trainable[f"target_{h}"] != "Unknown"].copy()

    le = LabelEncoder()
    trainable["target_enc"] = le.fit_transform(trainable[f"target_{h}"])

    train_mask = trainable["year"] < 2020
    X_train = trainable.loc[train_mask, FEATURES].values
    X_test = trainable.loc[~train_mask, FEATURES].values
    y_train = trainable.loc[train_mask, "target_enc"].values
    y_test = trainable.loc[~train_mask, "target_enc"].values

    if len(X_test) < 100:
        log(f"  Not enough test data ({len(X_test)} rows). Skipping.")
        continue

    log(f"  Train: {len(X_train):,}, Test: {len(X_test):,}")

    imp = SimpleImputer(strategy="mean")
    sc = StandardScaler()
    X_train_p = sc.fit_transform(imp.fit_transform(X_train))
    X_test_p = sc.transform(imp.transform(X_test))

    clf = XGBClassifier(
        objective="multi:softprob", n_estimators=500, max_depth=8,
        learning_rate=0.05, subsample=0.8, colsample_bytree=0.7,
        eval_metric="mlogloss", random_state=42,
    )
    clf.fit(X_train_p, y_train)
    pred = clf.predict(X_test_p)
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average="weighted")

    # Binary: displacement risk (A or C)
    classes = list(le.classes_)
    if "A" in classes and "C" in classes:
        y_bin = np.isin(trainable.loc[~train_mask, f"target_{h}"], ["A", "C"]).astype(int)
        proba = clf.predict_proba(X_test_p)
        a_idx = classes.index("A")
        c_idx = classes.index("C")
        displacement_prob = proba[:, a_idx] + proba[:, c_idx]
        auc = roc_auc_score(y_bin, displacement_prob)
        log(f"  Multiclass: Acc={acc:.4f}, F1={f1:.4f}")
        log(f"  Binary displacement AUC: {auc:.4f}")
    else:
        log(f"  Multiclass: Acc={acc:.4f}, F1={f1:.4f}")

log("")
log("=" * 60)
log("DONE")
log("=" * 60)
