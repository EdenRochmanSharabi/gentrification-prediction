"""
Test multiple ML algorithms on the leak-free gentrification prediction task.
Same features and temporal split as model_leakfree.py.
Target: case(t+1) from features(t). NO LEAKAGE.
"""
import sys, os, warnings, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    StackingClassifier, VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier
from collections import Counter

from src.data_loader import load_data
from src.gentrification import classify_cases

LOG = os.path.join(os.path.dirname(__file__), "algorithms.log")
open(LOG, "w").close()

def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")

log("=" * 60)
log("ALGORITHM COMPARISON (leak-free, predict t+1)")
log("=" * 60)

# ── Data loading and feature engineering (identical to model_leakfree.py) ──
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

# ── Prepare train/test ──
trainable = df.dropna(subset=["sale_lag1", "rent_lag1", "target_case"])
trainable = trainable[trainable["target_case"] != "Unknown"].copy()

le = LabelEncoder()
trainable["target_encoded"] = le.fit_transform(trainable["target_case"])
classes = list(le.classes_)

train_mask = trainable["year"] < 2020
X_train = trainable.loc[train_mask, FEATURES].values
X_test = trainable.loc[~train_mask, FEATURES].values
y_train = trainable.loc[train_mask, "target_encoded"].values
y_test = trainable.loc[~train_mask, "target_encoded"].values

log(f"Train: {len(X_train):,}, Test: {len(X_test):,}")
log(f"Classes: {classes}")

# Binary target for displacement AUC
y_test_bin = np.isin(trainable.loc[~train_mask, "target_case"], ["A", "C"]).astype(int)
a_idx = classes.index("A")
c_idx = classes.index("C")

imp = SimpleImputer(strategy="mean")
sc = StandardScaler()
X_train_p = sc.fit_transform(imp.fit_transform(X_train))
X_test_p = sc.transform(imp.transform(X_test))

def evaluate(name, clf, fit_kwargs=None):
    log(f"\n{'=' * 60}")
    log(f"MODEL: {name}")
    log(f"{'=' * 60}")
    t0 = time.time()
    if fit_kwargs:
        clf.fit(X_train_p, y_train, **fit_kwargs)
    else:
        clf.fit(X_train_p, y_train)
    train_time = time.time() - t0

    pred = clf.predict(X_test_p)
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average="weighted")

    proba = clf.predict_proba(X_test_p)
    disp_prob = proba[:, a_idx] + proba[:, c_idx]
    auc = roc_auc_score(y_test_bin, disp_prob)

    log(f"  Accuracy:  {acc:.4f}")
    log(f"  F1 (wt):   {f1:.4f}")
    log(f"  Disp AUC:  {auc:.4f}")
    log(f"  Time:      {train_time:.1f}s")

    return {"name": name, "acc": acc, "f1": f1, "auc": auc, "time": train_time}


results = []

# 1. XGBoost (baseline reference)
r = evaluate("XGBoost", XGBClassifier(
    objective="multi:softprob", n_estimators=500, max_depth=8,
    learning_rate=0.05, subsample=0.8, colsample_bytree=0.7,
    eval_metric="mlogloss", random_state=42,
))
results.append(r)

# 2. LightGBM
r = evaluate("LightGBM", lgb.LGBMClassifier(
    objective="multiclass", n_estimators=500, max_depth=8,
    learning_rate=0.05, subsample=0.8, colsample_bytree=0.7,
    num_leaves=127, min_child_samples=20, random_state=42, verbose=-1,
))
results.append(r)

# 3. CatBoost
r = evaluate("CatBoost", CatBoostClassifier(
    iterations=500, depth=8, learning_rate=0.05,
    random_seed=42, verbose=0, task_type="CPU",
))
results.append(r)

# 4. Random Forest
r = evaluate("Random Forest", RandomForestClassifier(
    n_estimators=500, max_depth=20, min_samples_leaf=5,
    n_jobs=-1, random_state=42,
))
results.append(r)

# 5. Extra Trees
r = evaluate("Extra Trees", ExtraTreesClassifier(
    n_estimators=500, max_depth=20, min_samples_leaf=5,
    n_jobs=-1, random_state=42,
))
results.append(r)

# 6. Soft Voting Ensemble (XGBoost + LightGBM + CatBoost)
log(f"\n{'=' * 60}")
log("MODEL: Voting Ensemble (XGB + LGBM + CatBoost)")
log(f"{'=' * 60}")
t0 = time.time()
voting = VotingClassifier(
    estimators=[
        ("xgb", XGBClassifier(
            objective="multi:softprob", n_estimators=500, max_depth=8,
            learning_rate=0.05, subsample=0.8, colsample_bytree=0.7,
            eval_metric="mlogloss", random_state=42,
        )),
        ("lgbm", lgb.LGBMClassifier(
            objective="multiclass", n_estimators=500, max_depth=8,
            learning_rate=0.05, subsample=0.8, colsample_bytree=0.7,
            num_leaves=127, min_child_samples=20, random_state=42, verbose=-1,
        )),
        ("cat", CatBoostClassifier(
            iterations=500, depth=8, learning_rate=0.05,
            random_seed=42, verbose=0, task_type="CPU",
        )),
    ],
    voting="soft",
    n_jobs=1,
)
voting.fit(X_train_p, y_train)
train_time = time.time() - t0
pred = voting.predict(X_test_p)
acc = accuracy_score(y_test, pred)
f1v = f1_score(y_test, pred, average="weighted")
proba = voting.predict_proba(X_test_p)
disp_prob = proba[:, a_idx] + proba[:, c_idx]
auc = roc_auc_score(y_test_bin, disp_prob)
log(f"  Accuracy:  {acc:.4f}")
log(f"  F1 (wt):   {f1v:.4f}")
log(f"  Disp AUC:  {auc:.4f}")
log(f"  Time:      {train_time:.1f}s")
results.append({"name": "Voting Ensemble", "acc": acc, "f1": f1v, "auc": auc, "time": train_time})

# 7. Stacking Ensemble (XGBoost + LightGBM + CatBoost → LogReg)
log(f"\n{'=' * 60}")
log("MODEL: Stacking Ensemble (XGB + LGBM + CatBoost → LogReg)")
log(f"{'=' * 60}")
t0 = time.time()
stacking = StackingClassifier(
    estimators=[
        ("xgb", XGBClassifier(
            objective="multi:softprob", n_estimators=300, max_depth=6,
            learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
            eval_metric="mlogloss", random_state=42,
        )),
        ("lgbm", lgb.LGBMClassifier(
            objective="multiclass", n_estimators=300, max_depth=6,
            learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
            num_leaves=63, random_state=42, verbose=-1,
        )),
        ("cat", CatBoostClassifier(
            iterations=300, depth=6, learning_rate=0.1,
            random_seed=42, verbose=0, task_type="CPU",
        )),
    ],
    final_estimator=LogisticRegression(max_iter=1000, random_state=42),
    cv=3,
    n_jobs=1,
    passthrough=False,
)
stacking.fit(X_train_p, y_train)
train_time = time.time() - t0
pred = stacking.predict(X_test_p)
acc = accuracy_score(y_test, pred)
f1s = f1_score(y_test, pred, average="weighted")
proba = stacking.predict_proba(X_test_p)
disp_prob = proba[:, a_idx] + proba[:, c_idx]
auc = roc_auc_score(y_test_bin, disp_prob)
log(f"  Accuracy:  {acc:.4f}")
log(f"  F1 (wt):   {f1s:.4f}")
log(f"  Disp AUC:  {auc:.4f}")
log(f"  Time:      {train_time:.1f}s")
results.append({"name": "Stacking Ensemble", "acc": acc, "f1": f1s, "auc": auc, "time": train_time})

# ── Summary ──
log(f"\n{'=' * 60}")
log("SUMMARY")
log(f"{'=' * 60}")
log(f"{'Model':<30} {'Acc':>8} {'F1':>8} {'AUC':>8} {'Time':>8}")
log("-" * 65)
for r in sorted(results, key=lambda x: x["acc"], reverse=True):
    log(f"{r['name']:<30} {r['acc']:>8.4f} {r['f1']:>8.4f} {r['auc']:>8.4f} {r['time']:>7.1f}s")

best = max(results, key=lambda x: x["acc"])
log(f"\nBEST: {best['name']} (Acc={best['acc']:.4f}, AUC={best['auc']:.4f})")
log("DONE")
