"""
Experiment 01: Extended feature engineering
- Interaction features (rent_change * sale_change)
- Price acceleration (diff of diffs)
- Price relative to province mean
- Lag-2 and lag-4 features
- Year-over-year changes
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
log("EXP01: Extended Feature Engineering")
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

# === BASELINE FEATURES ===
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

# === NEW FEATURES ===
# Lag-2 and lag-4
df["sale_lag2"] = grouped[sale_col].shift(2)
df["rent_lag2"] = grouped[rent_col].shift(2)
df["sale_lag4"] = grouped[sale_col].shift(4)
df["rent_lag4"] = grouped[rent_col].shift(4)

# Interaction: rent_change * sale_change
df["rent_sale_interaction"] = df["rent_change"] * df["sale_change"]

# Price acceleration (change in change)
df["rent_accel"] = grouped["rent_change"].diff()
df["sale_accel"] = grouped["sale_change"].diff()

# Price relative to province mean
prov_sale_mean = df.groupby(["NPRO", "period"])[sale_col].transform("mean")
prov_rent_mean = df.groupby(["NPRO", "period"])[rent_col].transform("mean")
df["sale_vs_prov"] = df[sale_col] / prov_sale_mean.replace(0, np.nan)
df["rent_vs_prov"] = df[rent_col] / prov_rent_mean.replace(0, np.nan)

# Year-over-year changes (lag 4 quarters)
df["sale_yoy"] = (df[sale_col] - df["sale_lag4"]) / df["sale_lag4"].replace(0, np.nan)
df["rent_yoy"] = (df[rent_col] - df["rent_lag4"]) / df["rent_lag4"].replace(0, np.nan)

# Rolling 6-quarter mean and std
df["sale_roll6"] = grouped[sale_col].transform(lambda x: x.rolling(6, min_periods=1).mean())
df["rent_roll6"] = grouped[rent_col].transform(lambda x: x.rolling(6, min_periods=1).mean())

# Stock ratio
df["stock_ratio"] = df["stock_residential_sale_all"] / df["stock_residential_rent_all"].replace(0, np.nan)

BASELINE_FEATURES = [
    "sale_lag1", "rent_lag1",
    "sale_roll3", "rent_roll3",
    "sale_roll_std3", "rent_roll_std3",
    "stock_residential_sale_all", "stock_residential_rent_all",
    "stock_sale_lag1", "stock_rent_lag1",
    "rent_sale_ratio",
    "month", "quarter", "year",
]

EXTENDED_FEATURES = BASELINE_FEATURES + [
    "sale_lag2", "rent_lag2",
    "sale_lag4", "rent_lag4",
    "rent_sale_interaction",
    "rent_accel", "sale_accel",
    "sale_vs_prov", "rent_vs_prov",
    "sale_yoy", "rent_yoy",
    "sale_roll6", "rent_roll6",
    "stock_ratio",
]

log(f"Baseline features: {len(BASELINE_FEATURES)}")
log(f"Extended features: {len(EXTENDED_FEATURES)}")

# Prepare data
trainable = df.dropna(subset=BASELINE_FEATURES + ["case"])
trainable = trainable[trainable["case"] != "Unknown"].copy()

le = LabelEncoder()
trainable["case_encoded"] = le.fit_transform(trainable["case"])

train_mask = trainable["year"] < 2020

# --- Baseline run ---
X_base_train = trainable.loc[train_mask, BASELINE_FEATURES].values
X_base_test = trainable.loc[~train_mask, BASELINE_FEATURES].values
y_train = trainable.loc[train_mask, "case_encoded"].values
y_test = trainable.loc[~train_mask, "case_encoded"].values

imp = SimpleImputer(strategy="mean")
sc = StandardScaler()
Xbt = sc.fit_transform(imp.fit_transform(X_base_train))
Xbe = sc.transform(imp.transform(X_base_test))

clf_base = XGBClassifier(
    objective="multi:softprob", n_estimators=300, max_depth=6,
    learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
    eval_metric="mlogloss", random_state=42,
)
clf_base.fit(Xbt, y_train)
bp = clf_base.predict(Xbe)
base_acc = accuracy_score(y_test, bp)
base_f1 = f1_score(y_test, bp, average="weighted")
log(f"Baseline: Acc={base_acc:.4f}, F1={base_f1:.4f}")

# --- Extended features run ---
X_ext_train = trainable.loc[train_mask, EXTENDED_FEATURES].values
X_ext_test = trainable.loc[~train_mask, EXTENDED_FEATURES].values

imp2 = SimpleImputer(strategy="mean")
sc2 = StandardScaler()
Xet = sc2.fit_transform(imp2.fit_transform(X_ext_train))
Xee = sc2.transform(imp2.transform(X_ext_test))

clf_ext = XGBClassifier(
    objective="multi:softprob", n_estimators=300, max_depth=6,
    learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
    eval_metric="mlogloss", random_state=42,
)
clf_ext.fit(Xet, y_train)
ep = clf_ext.predict(Xee)
ext_acc = accuracy_score(y_test, ep)
ext_f1 = f1_score(y_test, ep, average="weighted")
log(f"Extended features: Acc={ext_acc:.4f}, F1={ext_f1:.4f}")
log(f"Delta: Acc={ext_acc - base_acc:+.4f}, F1={ext_f1 - base_f1:+.4f}")

# --- Extended + deeper trees ---
clf_deep = XGBClassifier(
    objective="multi:softprob", n_estimators=500, max_depth=8,
    learning_rate=0.05, subsample=0.8, colsample_bytree=0.7,
    min_child_weight=3, gamma=0.1,
    eval_metric="mlogloss", random_state=42,
)
clf_deep.fit(Xet, y_train)
dp = clf_deep.predict(Xee)
deep_acc = accuracy_score(y_test, dp)
deep_f1 = f1_score(y_test, dp, average="weighted")
log(f"Extended+deep: Acc={deep_acc:.4f}, F1={deep_f1:.4f}")
log(f"Delta vs base: Acc={deep_acc - base_acc:+.4f}, F1={deep_f1 - base_f1:+.4f}")

log("\nPer-class (best extended model):")
best_pred = dp if deep_f1 > ext_f1 else ep
best_name = "Extended+deep" if deep_f1 > ext_f1 else "Extended"
log(classification_report(y_test, best_pred, target_names=le.classes_, zero_division=0))

log(f"\nEXP01 RESULT: Best={best_name}, Acc={max(ext_acc,deep_acc):.4f}, F1={max(ext_f1,deep_f1):.4f}")
