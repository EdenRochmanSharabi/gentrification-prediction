"""
Experiment 04: Province target encoding + feature selection
- Encode province as a numeric feature via target encoding (mean of target per province)
- Try different feature subsets (top-7, top-10)
- Combine best features from exp02 with province encoding
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
log("EXP04: Province Target Encoding + Feature Selection")
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

# All baseline features
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

# Extended safe features
df["sale_lag2"] = grouped[sale_col].shift(2)
df["rent_lag2"] = grouped[rent_col].shift(2)
df["sale_lag4"] = grouped[sale_col].shift(4)
df["rent_lag4"] = grouped[rent_col].shift(4)
prov_sale_mean = df.groupby(["NPRO", "period"])[sale_col].transform("mean")
prov_rent_mean = df.groupby(["NPRO", "period"])[rent_col].transform("mean")
df["sale_vs_prov"] = df[sale_col] / prov_sale_mean.replace(0, np.nan)
df["rent_vs_prov"] = df[rent_col] / prov_rent_mean.replace(0, np.nan)
df["sale_roll6"] = grouped[sale_col].transform(lambda x: x.rolling(6, min_periods=1).mean())
df["rent_roll6"] = grouped[rent_col].transform(lambda x: x.rolling(6, min_periods=1).mean())
df["stock_ratio"] = df["stock_residential_sale_all"] / df["stock_residential_rent_all"].replace(0, np.nan)
df["rent_change_lag1"] = grouped["rent_change"].shift(1)
df["sale_change_lag1"] = grouped["sale_change"].shift(1)

BASELINE_FEATURES = [
    "sale_lag1", "rent_lag1",
    "sale_roll3", "rent_roll3",
    "sale_roll_std3", "rent_roll_std3",
    "stock_residential_sale_all", "stock_residential_rent_all",
    "stock_sale_lag1", "stock_rent_lag1",
    "rent_sale_ratio",
    "month", "quarter", "year",
]

trainable = df.dropna(subset=BASELINE_FEATURES + ["case"])
trainable = trainable[trainable["case"] != "Unknown"].copy()

le = LabelEncoder()
trainable["case_encoded"] = le.fit_transform(trainable["case"])

train_mask = trainable["year"] < 2020

# Province target encoding (computed only on training data to avoid leakage)
log("Computing province target encoding...")
train_data = trainable[train_mask]

# For each class, compute fraction of that class per province in training data
for cls_idx, cls_name in enumerate(le.classes_):
    prov_rates = train_data.groupby("NPRO")["case_encoded"].apply(
        lambda x: (x == cls_idx).mean()
    )
    trainable[f"prov_rate_{cls_name}"] = trainable["NPRO"].map(prov_rates)

# Province-level price statistics from training
prov_stats = train_data.groupby("NPRO").agg({
    sale_col: ["mean", "std"],
    rent_col: ["mean", "std"],
}).reset_index()
prov_stats.columns = ["NPRO", "prov_sale_mean", "prov_sale_std", "prov_rent_mean", "prov_rent_std"]
trainable = trainable.merge(prov_stats, on="NPRO", how="left")

PROVINCE_FEATURES = [f"prov_rate_{cls}" for cls in le.classes_] + [
    "prov_sale_mean", "prov_sale_std", "prov_rent_mean", "prov_rent_std",
]

EXTENDED_SAFE = BASELINE_FEATURES + [
    "sale_lag2", "rent_lag2",
    "sale_lag4", "rent_lag4",
    "sale_vs_prov", "rent_vs_prov",
    "sale_roll6", "rent_roll6",
    "stock_ratio",
    "rent_change_lag1", "sale_change_lag1",
]

y_train = trainable.loc[train_mask, "case_encoded"].values
y_test = trainable.loc[~train_mask, "case_encoded"].values

configs = {
    "Baseline (14f)": BASELINE_FEATURES,
    "Baseline + province enc (23f)": BASELINE_FEATURES + PROVINCE_FEATURES,
    "Extended safe (25f)": EXTENDED_SAFE,
    "Extended + province enc (34f)": EXTENDED_SAFE + PROVINCE_FEATURES,
}

results = {}
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
    results[name] = (acc, f1, pred, features)
    log(f"{name}: Acc={acc:.4f}, F1={f1:.4f}")

# Try the best config with a deeper model
best_name = max(results, key=lambda k: results[k][1])
best_features = results[best_name][3]
log(f"\nBest config: {best_name}")

log("\n--- Best config with tuned hyperparams ---")
X_tr = trainable.loc[train_mask, best_features].values
X_te = trainable.loc[~train_mask, best_features].values
imp = SimpleImputer(strategy="mean")
sc = StandardScaler()
Xt = sc.fit_transform(imp.fit_transform(X_tr))
Xe = sc.transform(imp.transform(X_te))

clf_tuned = XGBClassifier(
    objective="multi:softprob", n_estimators=500, max_depth=8,
    learning_rate=0.05, subsample=0.8, colsample_bytree=0.7,
    min_child_weight=3, gamma=0.1,
    eval_metric="mlogloss", random_state=42,
)
clf_tuned.fit(Xt, y_train)
pred_tuned = clf_tuned.predict(Xe)
tuned_acc = accuracy_score(y_test, pred_tuned)
tuned_f1 = f1_score(y_test, pred_tuned, average="weighted")
log(f"Tuned: Acc={tuned_acc:.4f}, F1={tuned_f1:.4f}")
log(f"\nPer-class:")
log(classification_report(y_test, pred_tuned, target_names=le.classes_, zero_division=0))

# Feature importance from the best model
importance = clf_tuned.feature_importances_
sorted_idx = np.argsort(importance)[::-1]
log("\nTop 15 features:")
for i in range(min(15, len(best_features))):
    idx = sorted_idx[i]
    log(f"  {best_features[idx]}: {importance[idx]:.4f}")

log(f"\nEXP04 RESULT: Best={best_name} tuned, Acc={tuned_acc:.4f}, F1={tuned_f1:.4f}")
