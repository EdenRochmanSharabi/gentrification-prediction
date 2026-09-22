"""
Binary gentrification prediction: gentrifying (C) vs not-gentrifying (A/B/D/None).
Same leak-free framework: predict state at t+1 from features at t.
Also tries: any-displacement (A or C) vs stable.
"""
import sys, os, warnings, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from collections import Counter

from config import MODELS_DIR
from src.data_loader import load_data
from src.gentrification import classify_cases

LOG = os.path.join(os.path.dirname(__file__), "binary_model.log")
open(LOG, "w").close()

def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")

log("=" * 60)
log("BINARY GENTRIFICATION PREDICTION (leak-free)")
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

# Target: next quarter's case
df["target_case"] = df.groupby("CUSEC")["case"].shift(-1)

# Binary targets
df["target_gentrifying"] = (df["target_case"] == "C").astype(int)
df["target_displacement"] = df["target_case"].isin(["A", "C"]).astype(int)
df["target_investment"] = df["target_case"].isin(["B", "C"]).astype(int)

# Current case as feature
case_map = {"A": 0, "B": 1, "C": 2, "D": 3, "None": 4, "Unknown": -1}
df["case_encoded_feat"] = df["case"].map(case_map)
for c in ["A", "B", "C", "D", "None"]:
    df[f"is_case_{c}"] = (df["case"] == c).astype(int)

# Features (same as optimized model)
for lag in [1, 2, 4]:
    df[f"sale_lag{lag}"] = grouped[sale_col].shift(lag)
    df[f"rent_lag{lag}"] = grouped[rent_col].shift(lag)

df["sale_change_curr"] = df["sale_change"]
df["rent_change_curr"] = df["rent_change"]
df["rent_sale_ratio"] = df[rent_col] / df[sale_col].replace(0, np.nan)
df["change_interaction"] = df["rent_change"] * df["sale_change"]

df["sale_change_lag1"] = grouped["sale_change"].shift(1)
df["rent_change_lag1"] = grouped["rent_change"].shift(1)

for win in [3, 6]:
    mp = 1 if win == 3 else 2
    df[f"sale_roll{win}"] = grouped[sale_col].transform(lambda x: x.rolling(win, min_periods=mp).mean())
    df[f"rent_roll{win}"] = grouped[rent_col].transform(lambda x: x.rolling(win, min_periods=mp).mean())
    df[f"sale_roll_std{win}"] = grouped[sale_col].transform(lambda x: x.rolling(win, min_periods=mp).std())
    df[f"rent_roll_std{win}"] = grouped[rent_col].transform(lambda x: x.rolling(win, min_periods=mp).std())

df["sale_momentum"] = df[sale_col] - df["sale_lag1"]
df["rent_momentum"] = df[rent_col] - df["rent_lag1"]

prov_sale_mean = df.groupby(["NPRO", "period"])[sale_col].transform("mean")
prov_rent_mean = df.groupby(["NPRO", "period"])[rent_col].transform("mean")
df["sale_vs_prov"] = df[sale_col] / prov_sale_mean.replace(0, np.nan)
df["rent_vs_prov"] = df[rent_col] / prov_rent_mean.replace(0, np.nan)

df["sale_pctile"] = grouped[sale_col].transform(lambda x: x.rank(pct=True))
df["rent_pctile"] = grouped[rent_col].transform(lambda x: x.rank(pct=True))

df["sale_yoy"] = (df[sale_col] - df["sale_lag4"]) / df["sale_lag4"].replace(0, np.nan)
df["rent_yoy"] = (df[rent_col] - df["rent_lag4"]) / df["rent_lag4"].replace(0, np.nan)

df["stock_sale_lag1"] = grouped["stock_residential_sale_all"].shift(1)
df["stock_rent_lag1"] = grouped["stock_residential_rent_all"].shift(1)
df["stock_ratio"] = df["stock_residential_sale_all"] / df["stock_residential_rent_all"].replace(0, np.nan)

df["month"] = df["period"].dt.month
df["quarter"] = df["period"].dt.quarter

FEATURES = [
    sale_col, rent_col, "rent_sale_ratio",
    "sale_change_curr", "rent_change_curr", "change_interaction",
    "case_encoded_feat",
    "is_case_A", "is_case_B", "is_case_C", "is_case_D", "is_case_None",
    "sale_lag1", "rent_lag1", "sale_lag2", "rent_lag2", "sale_lag4", "rent_lag4",
    "sale_change_lag1", "rent_change_lag1",
    "sale_roll3", "rent_roll3", "sale_roll_std3", "rent_roll_std3",
    "sale_roll6", "rent_roll6", "sale_roll_std6", "rent_roll_std6",
    "sale_momentum", "rent_momentum",
    "sale_vs_prov", "rent_vs_prov", "sale_pctile", "rent_pctile",
    "sale_yoy", "rent_yoy",
    "stock_residential_sale_all", "stock_residential_rent_all",
    "stock_sale_lag1", "stock_rent_lag1", "stock_ratio",
    "month", "quarter", "year",
]

log(f"Features: {len(FEATURES)}")

trainable = df.dropna(subset=["sale_lag1", "rent_lag1", "target_case"])
trainable = trainable[trainable["target_case"] != "Unknown"].copy()

train_mask = trainable["year"] < 2020

TARGETS = {
    "gentrifying (C vs rest)": "target_gentrifying",
    "displacement (A|C vs rest)": "target_displacement",
    "investment (B|C vs rest)": "target_investment",
}

for tname, tcol in TARGETS.items():
    log(f"\n{'='*60}")
    log(f"TARGET: {tname}")
    log(f"{'='*60}")

    X_tr = trainable.loc[train_mask, FEATURES].values
    X_te = trainable.loc[~train_mask, FEATURES].values
    y_tr = trainable.loc[train_mask, tcol].values
    y_te = trainable.loc[~train_mask, tcol].values

    imp = SimpleImputer(strategy="mean")
    sc = StandardScaler()
    Xtr = sc.fit_transform(imp.fit_transform(X_tr))
    Xte = sc.transform(imp.transform(X_te))

    pos_rate_tr = y_tr.mean()
    pos_rate_te = y_te.mean()
    log(f"Positive rate: train={pos_rate_tr:.3f}, test={pos_rate_te:.3f}")

    scale = (len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1)

    clf = XGBClassifier(
        objective="binary:logistic", n_estimators=500, max_depth=8,
        learning_rate=0.05, subsample=0.8, colsample_bytree=0.7,
        scale_pos_weight=scale,
        eval_metric="logloss", random_state=42, n_jobs=-1,
    )
    clf.fit(Xtr, y_tr)
    pred = clf.predict(Xte)
    proba = clf.predict_proba(Xte)[:, 1]
    acc = accuracy_score(y_te, pred)
    f1 = f1_score(y_te, pred, average="weighted")
    auc = roc_auc_score(y_te, proba)
    log(f"Acc={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}")
    log(classification_report(y_te, pred, target_names=["No", "Yes"], zero_division=0))

log("\n" + "=" * 60)
log("BINARY MODELS COMPLETE")
log("=" * 60)
