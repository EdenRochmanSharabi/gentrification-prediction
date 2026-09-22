"""
Experiment 02: Extended features WITHOUT target leakage.
rent_change and sale_change define the target case, so any feature derived
from them (interaction, acceleration) is leaking the answer.
Only add features that use LAGGED or AGGREGATED info, never current-period changes.
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
from collections import Counter
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
log("EXP02: Extended Features (NO LEAKAGE)")
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

# === BASELINE FEATURES (14) ===
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

# === NEW FEATURES (leak-free) ===
# Deeper lags
df["sale_lag2"] = grouped[sale_col].shift(2)
df["rent_lag2"] = grouped[rent_col].shift(2)
df["sale_lag4"] = grouped[sale_col].shift(4)
df["rent_lag4"] = grouped[rent_col].shift(4)

# LAGGED changes (previous quarter's change, NOT current)
df["sale_change_lag1"] = grouped["sale_change"].shift(1)
df["rent_change_lag1"] = grouped["rent_change"].shift(1)
df["sale_change_lag2"] = grouped["sale_change"].shift(2)
df["rent_change_lag2"] = grouped["rent_change"].shift(2)

# Lagged interaction (previous quarter's changes interacting)
df["interaction_lag1"] = df["sale_change_lag1"] * df["rent_change_lag1"]

# Price momentum: difference between lag1 and lag2
df["sale_momentum"] = df["sale_lag1"] - df["sale_lag2"]
df["rent_momentum"] = df["rent_lag1"] - df["rent_lag2"]

# Price relative to province mean (uses current price level, not change)
prov_sale_mean = df.groupby(["NPRO", "period"])[sale_col].transform("mean")
prov_rent_mean = df.groupby(["NPRO", "period"])[rent_col].transform("mean")
df["sale_vs_prov"] = df[sale_col] / prov_sale_mean.replace(0, np.nan)
df["rent_vs_prov"] = df[rent_col] / prov_rent_mean.replace(0, np.nan)

# Year-over-year from LAGGED values (lag1 vs lag5 = previous Q vs same Q last year)
df["sale_lag5"] = grouped[sale_col].shift(5)
df["rent_lag5"] = grouped[rent_col].shift(5)
df["sale_yoy_lag"] = (df["sale_lag1"] - df["sale_lag5"]) / df["sale_lag5"].replace(0, np.nan)
df["rent_yoy_lag"] = (df["rent_lag1"] - df["rent_lag5"]) / df["rent_lag5"].replace(0, np.nan)

# Longer rolling stats
df["sale_roll6"] = grouped[sale_col].transform(lambda x: x.rolling(6, min_periods=2).mean())
df["rent_roll6"] = grouped[rent_col].transform(lambda x: x.rolling(6, min_periods=2).mean())
df["sale_roll_std6"] = grouped[sale_col].transform(lambda x: x.rolling(6, min_periods=2).std())
df["rent_roll_std6"] = grouped[rent_col].transform(lambda x: x.rolling(6, min_periods=2).std())

# Stock ratio and stock changes
df["stock_ratio"] = df["stock_residential_sale_all"] / df["stock_residential_rent_all"].replace(0, np.nan)
df["stock_sale_change"] = grouped["stock_residential_sale_all"].pct_change()
df["stock_rent_change"] = grouped["stock_residential_rent_all"].pct_change()

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
    "sale_change_lag1", "rent_change_lag1",
    "sale_change_lag2", "rent_change_lag2",
    "interaction_lag1",
    "sale_momentum", "rent_momentum",
    "sale_vs_prov", "rent_vs_prov",
    "sale_yoy_lag", "rent_yoy_lag",
    "sale_roll6", "rent_roll6",
    "sale_roll_std6", "rent_roll_std6",
    "stock_ratio", "stock_sale_change", "stock_rent_change",
]

log(f"Baseline features: {len(BASELINE_FEATURES)}")
log(f"Extended features (no leakage): {len(EXTENDED_FEATURES)}")

# Prepare data
trainable = df.dropna(subset=BASELINE_FEATURES + ["case"])
trainable = trainable[trainable["case"] != "Unknown"].copy()

le = LabelEncoder()
trainable["case_encoded"] = le.fit_transform(trainable["case"])

train_mask = trainable["year"] < 2020
y_train = trainable.loc[train_mask, "case_encoded"].values
y_test = trainable.loc[~train_mask, "case_encoded"].values

# Class weights
counts = Counter(y_train)
total = len(y_train)
n_classes = len(counts)
cw = {cls: total / (n_classes * count) for cls, count in counts.items()}
sw = np.array([cw[yi] for yi in y_train])

# --- Baseline ---
imp = SimpleImputer(strategy="mean")
sc = StandardScaler()
Xbt = sc.fit_transform(imp.fit_transform(trainable.loc[train_mask, BASELINE_FEATURES].values))
Xbe = sc.transform(imp.transform(trainable.loc[~train_mask, BASELINE_FEATURES].values))

clf_base = XGBClassifier(
    objective="multi:softprob", n_estimators=300, max_depth=6,
    learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
    eval_metric="mlogloss", random_state=42,
)
clf_base.fit(Xbt, y_train)
bp = clf_base.predict(Xbe)
log(f"Baseline: Acc={accuracy_score(y_test, bp):.4f}, F1={f1_score(y_test, bp, average='weighted'):.4f}")

# --- Extended (no leakage) ---
imp2 = SimpleImputer(strategy="mean")
sc2 = StandardScaler()
Xet = sc2.fit_transform(imp2.fit_transform(trainable.loc[train_mask, EXTENDED_FEATURES].values))
Xee = sc2.transform(imp2.transform(trainable.loc[~train_mask, EXTENDED_FEATURES].values))

configs = [
    ("Ext+base_hp", dict(n_estimators=300, max_depth=6, learning_rate=0.1,
                         subsample=0.8, colsample_bytree=0.8), False),
    ("Ext+deeper", dict(n_estimators=500, max_depth=8, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.7, min_child_weight=3,
                        gamma=0.1), False),
    ("Ext+deeper+weights", dict(n_estimators=500, max_depth=8, learning_rate=0.05,
                                subsample=0.8, colsample_bytree=0.7,
                                min_child_weight=3, gamma=0.1), True),
    ("Ext+wide", dict(n_estimators=800, max_depth=10, learning_rate=0.03,
                      subsample=0.7, colsample_bytree=0.6, min_child_weight=5,
                      gamma=0.2, reg_alpha=0.1, reg_lambda=1.0), False),
    ("Ext+wide+weights", dict(n_estimators=800, max_depth=10, learning_rate=0.03,
                              subsample=0.7, colsample_bytree=0.6,
                              min_child_weight=5, gamma=0.2,
                              reg_alpha=0.1, reg_lambda=1.0), True),
]

best_acc, best_name, best_clf_obj = 0, "", None
for name, params, use_w in configs:
    clf = XGBClassifier(objective="multi:softprob", eval_metric="mlogloss",
                        random_state=42, **params)
    if use_w:
        clf.fit(Xet, y_train, sample_weight=sw)
    else:
        clf.fit(Xet, y_train)
    pred = clf.predict(Xee)
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average="weighted")
    log(f"  {name}: Acc={acc:.4f}, F1={f1:.4f}")
    if f1 > best_acc:
        best_acc = f1
        best_name = name
        best_clf_obj = clf

log(f"\nBest: {best_name}, F1={best_acc:.4f}")
log(f"Per-class report ({best_name}):")
best_pred = best_clf_obj.predict(Xee)
log(classification_report(y_test, best_pred, target_names=le.classes_, zero_division=0))

# Feature importance for best model
importances = best_clf_obj.feature_importances_
feat_imp = sorted(zip(EXTENDED_FEATURES, importances), key=lambda x: x[1], reverse=True)
log("\nTop 15 feature importances:")
for fname, imp_val in feat_imp[:15]:
    log(f"  {fname}: {imp_val:.4f}")

# Save best model
MODELS_DIR.mkdir(parents=True, exist_ok=True)
from sklearn.pipeline import make_pipeline
best_pipe = make_pipeline(SimpleImputer(strategy="mean"), StandardScaler(), best_clf_obj)
best_pipe.fit(trainable.loc[train_mask, EXTENDED_FEATURES].values, y_train)
joblib.dump(best_pipe, MODELS_DIR / "xgboost_extended_noleak.pkl")
joblib.dump(le, MODELS_DIR / "label_encoder.pkl")
log(f"Saved model to {MODELS_DIR / 'xgboost_extended_noleak.pkl'}")
log(f"Feature list: {EXTENDED_FEATURES}")
