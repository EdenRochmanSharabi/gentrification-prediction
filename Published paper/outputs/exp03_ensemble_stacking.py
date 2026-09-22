"""
Experiment 03: Stacking ensemble (XGBoost + LightGBM + extra XGBoost variant)
+ class weight balancing
Uses the baseline 14 features to isolate model improvement from feature engineering.
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
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
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
log("EXP03: Ensemble Stacking + Class Balancing")
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

FEATURES = [
    "sale_lag1", "rent_lag1",
    "sale_roll3", "rent_roll3",
    "sale_roll_std3", "rent_roll_std3",
    "stock_residential_sale_all", "stock_residential_rent_all",
    "stock_sale_lag1", "stock_rent_lag1",
    "rent_sale_ratio",
    "month", "quarter", "year",
]

trainable = df.dropna(subset=FEATURES + ["case"])
trainable = trainable[trainable["case"] != "Unknown"].copy()

le = LabelEncoder()
trainable["case_encoded"] = le.fit_transform(trainable["case"])

train_mask = trainable["year"] < 2020
X_train = trainable.loc[train_mask, FEATURES].values
X_test = trainable.loc[~train_mask, FEATURES].values
y_train = trainable.loc[train_mask, "case_encoded"].values
y_test = trainable.loc[~train_mask, "case_encoded"].values

imp = SimpleImputer(strategy="mean")
sc = StandardScaler()
X_tr = sc.fit_transform(imp.fit_transform(X_train))
X_te = sc.transform(imp.transform(X_test))

# Class weights
from collections import Counter
counts = Counter(y_train)
total = len(y_train)
n_classes = len(counts)
class_weights = {cls: total / (n_classes * count) for cls, count in counts.items()}
sample_weights = np.array([class_weights[yi] for yi in y_train])

# Baseline
log("--- Baseline XGBoost ---")
clf_base = XGBClassifier(
    objective="multi:softprob", n_estimators=300, max_depth=6,
    learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
    eval_metric="mlogloss", random_state=42,
)
clf_base.fit(X_tr, y_train)
bp = clf_base.predict(X_te)
base_acc = accuracy_score(y_test, bp)
base_f1 = f1_score(y_test, bp, average="weighted")
log(f"Baseline: Acc={base_acc:.4f}, F1={base_f1:.4f}")

# XGBoost with class weights
log("\n--- XGBoost + class weights ---")
clf_w = XGBClassifier(
    objective="multi:softprob", n_estimators=300, max_depth=6,
    learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
    eval_metric="mlogloss", random_state=42,
)
clf_w.fit(X_tr, y_train, sample_weight=sample_weights)
wp = clf_w.predict(X_te)
w_acc = accuracy_score(y_test, wp)
w_f1 = f1_score(y_test, wp, average="weighted")
log(f"Weighted: Acc={w_acc:.4f}, F1={w_f1:.4f}")

# Try LightGBM if available
try:
    from lightgbm import LGBMClassifier
    has_lgbm = True
    log("\nLightGBM available!")
except ImportError:
    has_lgbm = False
    log("\nLightGBM NOT available, skipping.")

if has_lgbm:
    log("\n--- LightGBM ---")
    clf_lgbm = LGBMClassifier(
        objective="multiclass", n_estimators=300, max_depth=6,
        learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbose=-1,
    )
    clf_lgbm.fit(X_tr, y_train)
    lp = clf_lgbm.predict(X_te)
    l_acc = accuracy_score(y_test, lp)
    l_f1 = f1_score(y_test, lp, average="weighted")
    log(f"LightGBM: Acc={l_acc:.4f}, F1={l_f1:.4f}")

    # Soft voting ensemble
    log("\n--- Soft Voting (XGBoost + LightGBM) ---")
    # Manual soft voting via predict_proba
    xgb_proba = clf_base.predict_proba(X_te)
    lgbm_proba = clf_lgbm.predict_proba(X_te)
    avg_proba = (xgb_proba + lgbm_proba) / 2
    vote_pred = avg_proba.argmax(axis=1)
    v_acc = accuracy_score(y_test, vote_pred)
    v_f1 = f1_score(y_test, vote_pred, average="weighted")
    log(f"Soft vote: Acc={v_acc:.4f}, F1={v_f1:.4f}")

    # 3-model ensemble: 2 XGBoost variants + LightGBM
    log("\n--- 3-Model Soft Voting ---")
    clf_xgb2 = XGBClassifier(
        objective="multi:softprob", n_estimators=500, max_depth=8,
        learning_rate=0.05, subsample=0.7, colsample_bytree=0.7,
        min_child_weight=3,
        eval_metric="mlogloss", random_state=123,
    )
    clf_xgb2.fit(X_tr, y_train)
    xgb2_proba = clf_xgb2.predict_proba(X_te)
    avg3_proba = (xgb_proba + lgbm_proba + xgb2_proba) / 3
    vote3_pred = avg3_proba.argmax(axis=1)
    v3_acc = accuracy_score(y_test, vote3_pred)
    v3_f1 = f1_score(y_test, vote3_pred, average="weighted")
    log(f"3-model vote: Acc={v3_acc:.4f}, F1={v3_f1:.4f}")

# Try CatBoost if available
try:
    from catboost import CatBoostClassifier
    has_cat = True
    log("\nCatBoost available!")
except ImportError:
    has_cat = False
    log("\nCatBoost NOT available.")

if has_cat:
    log("\n--- CatBoost ---")
    clf_cat = CatBoostClassifier(
        iterations=300, depth=6, learning_rate=0.1,
        random_state=42, verbose=0,
    )
    clf_cat.fit(X_tr, y_train)
    cp = clf_cat.predict(X_te).astype(int)
    c_acc = accuracy_score(y_test, cp)
    c_f1 = f1_score(y_test, cp, average="weighted")
    log(f"CatBoost: Acc={c_acc:.4f}, F1={c_f1:.4f}")

# Print best per-class report
log("\nPer-class report for best model:")
if has_lgbm:
    log(classification_report(y_test, vote3_pred if has_lgbm else bp, target_names=le.classes_, zero_division=0))
else:
    log(classification_report(y_test, bp, target_names=le.classes_, zero_division=0))

log("EXP03 complete.")
