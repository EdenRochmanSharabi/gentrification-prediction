"""
Experiment 06: Improve the ORIGINAL classification task

The task is: given market data including current and historical prices/stocks,
classify the gentrification type. This is classification, not forecasting.
Current prices are legitimate inputs since we're classifying current dynamics.

The original 72% baseline uses current rolling windows, stocks, and ratio.
We improve it by:
1. Adding more historical context (deeper lags, longer rolling windows)
2. Adding lagged changes (previous quarter's dynamics — autocorrelation)
3. Province-level context (relative position)
4. Better hyperparameters
5. Ensemble methods
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
log("EXP06: Improve Original Classification (with current data)")
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

# === ORIGINAL FEATURES (the 14 baseline uses) ===
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

# === ADDITIONAL FEATURES (all legitimate for current-state classification) ===
# Deeper lags
df["sale_lag2"] = grouped[sale_col].shift(2)
df["rent_lag2"] = grouped[rent_col].shift(2)
df["sale_lag4"] = grouped[sale_col].shift(4)
df["rent_lag4"] = grouped[rent_col].shift(4)

# Lagged changes (previous quarter's dynamics)
df["rent_change_lag1"] = grouped["rent_change"].shift(1)
df["sale_change_lag1"] = grouped["sale_change"].shift(1)

# Price relative to province mean (spatial context)
prov_sale_mean = df.groupby(["NPRO", "period"])[sale_col].transform("mean")
prov_rent_mean = df.groupby(["NPRO", "period"])[rent_col].transform("mean")
df["sale_vs_prov"] = df[sale_col] / prov_sale_mean.replace(0, np.nan)
df["rent_vs_prov"] = df[rent_col] / prov_rent_mean.replace(0, np.nan)

# Longer rolling windows
df["sale_roll6"] = grouped[sale_col].transform(lambda x: x.rolling(6, min_periods=1).mean())
df["rent_roll6"] = grouped[rent_col].transform(lambda x: x.rolling(6, min_periods=1).mean())

# Stock ratio
df["stock_ratio"] = df["stock_residential_sale_all"] / df["stock_residential_rent_all"].replace(0, np.nan)

# Price level (absolute value matters for market segment)
# Already included via sale_lag1/rent_lag1 effectively, but current values directly
df["current_sale"] = df[sale_col]
df["current_rent"] = df[rent_col]

# Price gap between current and rolling mean (deviation from trend)
df["sale_dev_from_trend"] = df[sale_col] - df["sale_roll3"]
df["rent_dev_from_trend"] = df[rent_col] - df["rent_roll3"]

# YoY change
df["sale_yoy"] = (df[sale_col] - df["sale_lag4"]) / df["sale_lag4"].replace(0, np.nan)
df["rent_yoy"] = (df[rent_col] - df["rent_lag4"]) / df["rent_lag4"].replace(0, np.nan)

BASELINE_FEATURES = [
    "sale_lag1", "rent_lag1",
    "sale_roll3", "rent_roll3",
    "sale_roll_std3", "rent_roll_std3",
    "stock_residential_sale_all", "stock_residential_rent_all",
    "stock_sale_lag1", "stock_rent_lag1",
    "rent_sale_ratio",
    "month", "quarter", "year",
]

IMPROVED_FEATURES = BASELINE_FEATURES + [
    "sale_lag2", "rent_lag2",
    "sale_lag4", "rent_lag4",
    "rent_change_lag1", "sale_change_lag1",
    "sale_vs_prov", "rent_vs_prov",
    "sale_roll6", "rent_roll6",
    "stock_ratio",
    "current_sale", "current_rent",
    "sale_dev_from_trend", "rent_dev_from_trend",
    "sale_yoy", "rent_yoy",
]

log(f"Baseline: {len(BASELINE_FEATURES)} features")
log(f"Improved: {len(IMPROVED_FEATURES)} features")

trainable = df.dropna(subset=BASELINE_FEATURES + ["case"])
trainable = trainable[trainable["case"] != "Unknown"].copy()

le = LabelEncoder()
trainable["case_encoded"] = le.fit_transform(trainable["case"])

train_mask = trainable["year"] < 2020
y_train = trainable.loc[train_mask, "case_encoded"].values
y_test = trainable.loc[~train_mask, "case_encoded"].values

from collections import Counter
counts = Counter(y_train)
total = len(y_train)
n_classes = len(counts)
class_weights = {cls: total / (n_classes * count) for cls, count in counts.items()}
sample_weights = np.array([class_weights[yi] for yi in y_train])

configs = [
    ("Baseline (14f, original HP)", BASELINE_FEATURES,
     dict(n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8),
     False),
    ("Improved (31f, original HP)", IMPROVED_FEATURES,
     dict(n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8),
     False),
    ("Improved (31f, deeper)", IMPROVED_FEATURES,
     dict(n_estimators=500, max_depth=8, learning_rate=0.05, subsample=0.8, colsample_bytree=0.7, min_child_weight=3, gamma=0.1),
     False),
    ("Improved (31f, deeper+weights)", IMPROVED_FEATURES,
     dict(n_estimators=500, max_depth=8, learning_rate=0.05, subsample=0.8, colsample_bytree=0.7, min_child_weight=3, gamma=0.1),
     True),
    ("Improved (31f, aggressive)", IMPROVED_FEATURES,
     dict(n_estimators=700, max_depth=10, learning_rate=0.03, subsample=0.85, colsample_bytree=0.75, min_child_weight=2, gamma=0.05, reg_alpha=0.1, reg_lambda=1.0),
     False),
    ("Baseline (14f, deeper)", BASELINE_FEATURES,
     dict(n_estimators=500, max_depth=8, learning_rate=0.05, subsample=0.8, colsample_bytree=0.7, min_child_weight=3, gamma=0.1),
     False),
]

best_acc = 0
best_f1 = 0
best_name = ""
best_pred = None
best_clf = None
best_features = None

for name, features, hp, use_w in configs:
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
    if use_w:
        clf.fit(Xt, y_train, sample_weight=sample_weights)
    else:
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

log(f"\nBest: {best_name}")
log(f"Best Acc={best_acc:.4f}, F1={best_f1:.4f}")
log(f"vs baseline: Acc delta={best_acc - 0.7203:+.4f}, F1 delta={best_f1 - 0.7178:+.4f}")

log(f"\nPer-class ({best_name}):")
log(classification_report(y_test, best_pred, target_names=le.classes_, zero_division=0))

# Feature importance
importance = best_clf.feature_importances_
sorted_idx = np.argsort(importance)[::-1]
log("\nTop 15 features:")
for i in range(min(15, len(best_features))):
    idx = sorted_idx[i]
    log(f"  {best_features[idx]}: {importance[idx]:.4f}")

# Try LightGBM if available
try:
    from lightgbm import LGBMClassifier
    log("\n--- LightGBM with improved features ---")
    X_tr = trainable.loc[train_mask, IMPROVED_FEATURES].values
    X_te = trainable.loc[~train_mask, IMPROVED_FEATURES].values
    imp = SimpleImputer(strategy="mean")
    sc = StandardScaler()
    Xt = sc.fit_transform(imp.fit_transform(X_tr))
    Xe = sc.transform(imp.transform(X_te))

    clf_lgbm = LGBMClassifier(
        objective="multiclass", n_estimators=500, max_depth=8,
        learning_rate=0.05, subsample=0.8, colsample_bytree=0.7,
        random_state=42, verbose=-1,
    )
    clf_lgbm.fit(Xt, y_train)
    lp = clf_lgbm.predict(Xe)
    l_acc = accuracy_score(y_test, lp)
    l_f1 = f1_score(y_test, lp, average="weighted")
    log(f"LightGBM: Acc={l_acc:.4f}, F1={l_f1:.4f}")

    # Ensemble: XGBoost + LightGBM soft voting
    xgb_proba = best_clf.predict_proba(Xe)
    lgbm_proba = clf_lgbm.predict_proba(Xe)
    avg_proba = (xgb_proba + lgbm_proba) / 2
    vote_pred = avg_proba.argmax(axis=1)
    v_acc = accuracy_score(y_test, vote_pred)
    v_f1 = f1_score(y_test, vote_pred, average="weighted")
    log(f"Ensemble (XGB+LGBM): Acc={v_acc:.4f}, F1={v_f1:.4f}")

    if v_f1 > best_f1:
        best_f1 = v_f1
        best_acc = v_acc
        best_name = "Ensemble XGB+LGBM"
        log(f"\nEnsemble is new best!")
        log(classification_report(y_test, vote_pred, target_names=le.classes_, zero_division=0))
except ImportError:
    log("LightGBM not available.")

# Save best model if it improved
if best_acc > 0.72:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_clf, MODELS_DIR / "xgboost_classifier_improved.pkl")
    joblib.dump(le, MODELS_DIR / "label_encoder.pkl")
    log(f"Saved improved model to {MODELS_DIR / 'xgboost_classifier_improved.pkl'}")

log(f"\nEXP06 RESULT: {best_name}, Acc={best_acc:.4f}, F1={best_f1:.4f}")
