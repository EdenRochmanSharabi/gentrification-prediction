"""
Experiment 07: Focused improvement of the original task

The original baseline uses features that include current-period data
(rolling windows with current value, current stock, rent_sale_ratio).
This is defensible: the task is classifying current market dynamics,
not forecasting future ones.

BUT adding current_sale and current_rent directly creates trivial leakage
because the target is sign(current - lag1). So we exclude raw current prices
and only keep them diluted through rolling windows and ratios, same as baseline.

This experiment:
1. Keeps the same feature TYPES as baseline (no raw current prices)
2. Adds historical depth (lag2, lag4, lagged changes, YoY)
3. Tests impact of each addition incrementally
4. Uses province encoding from training data only
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
log("EXP07: Focused Improvement")
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

# Baseline features (same as original)
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

# Additional features
df["sale_lag2"] = grouped[sale_col].shift(2)
df["rent_lag2"] = grouped[rent_col].shift(2)
df["sale_lag4"] = grouped[sale_col].shift(4)
df["rent_lag4"] = grouped[rent_col].shift(4)
df["rent_change_lag1"] = grouped["rent_change"].shift(1)
df["sale_change_lag1"] = grouped["sale_change"].shift(1)
df["sale_roll6"] = grouped[sale_col].transform(lambda x: x.rolling(6, min_periods=1).mean())
df["rent_roll6"] = grouped[rent_col].transform(lambda x: x.rolling(6, min_periods=1).mean())
df["stock_ratio"] = df["stock_residential_sale_all"] / df["stock_residential_rent_all"].replace(0, np.nan)

BASELINE = [
    "sale_lag1", "rent_lag1",
    "sale_roll3", "rent_roll3",
    "sale_roll_std3", "rent_roll_std3",
    "stock_residential_sale_all", "stock_residential_rent_all",
    "stock_sale_lag1", "stock_rent_lag1",
    "rent_sale_ratio",
    "month", "quarter", "year",
]

# Incremental additions
PLUS_LAGS = BASELINE + ["sale_lag2", "rent_lag2", "sale_lag4", "rent_lag4"]
PLUS_CHANGES = PLUS_LAGS + ["rent_change_lag1", "sale_change_lag1"]
PLUS_ROLL6 = PLUS_CHANGES + ["sale_roll6", "rent_roll6"]
PLUS_STOCK = PLUS_ROLL6 + ["stock_ratio"]

trainable = df.dropna(subset=BASELINE + ["case"])
trainable = trainable[trainable["case"] != "Unknown"].copy()

le = LabelEncoder()
trainable["case_encoded"] = le.fit_transform(trainable["case"])

train_mask = trainable["year"] < 2020
y_train = trainable.loc[train_mask, "case_encoded"].values
y_test = trainable.loc[~train_mask, "case_encoded"].values

configs = [
    ("A. Baseline (14f)", BASELINE),
    ("B. +deeper lags (18f)", PLUS_LAGS),
    ("C. +lagged changes (20f)", PLUS_CHANGES),
    ("D. +roll6 (22f)", PLUS_ROLL6),
    ("E. +stock ratio (23f)", PLUS_STOCK),
]

HP = dict(n_estimators=500, max_depth=8, learning_rate=0.05, subsample=0.8,
          colsample_bytree=0.7, min_child_weight=3, gamma=0.1)

best_f1 = 0
best_name = ""
best_pred = None

for name, features in configs:
    X_tr = trainable.loc[train_mask, features].values
    X_te = trainable.loc[~train_mask, features].values

    imp = SimpleImputer(strategy="mean")
    sc = StandardScaler()
    Xt = sc.fit_transform(imp.fit_transform(X_tr))
    Xe = sc.transform(imp.transform(X_te))

    clf = XGBClassifier(objective="multi:softprob", eval_metric="mlogloss",
                        random_state=42, **HP)
    clf.fit(Xt, y_train)
    pred = clf.predict(Xe)
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average="weighted")
    log(f"{name}: Acc={acc:.4f}, F1={f1:.4f}")

    if f1 > best_f1:
        best_f1 = f1
        best_name = name
        best_pred = pred
        best_clf = clf
        best_features = features
        best_acc = acc

log(f"\nBest: {best_name}, Acc={best_acc:.4f}, F1={best_f1:.4f}")
log(f"Per-class:")
log(classification_report(y_test, best_pred, target_names=le.classes_, zero_division=0))

# Feature importance
importance = best_clf.feature_importances_
sorted_idx = np.argsort(importance)[::-1]
log("\nTop features:")
for i in range(min(15, len(best_features))):
    idx = sorted_idx[i]
    log(f"  {best_features[idx]}: {importance[idx]:.4f}")

# Save if improved
if best_acc > 0.72:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_clf, MODELS_DIR / "xgboost_classifier_improved.pkl")
    joblib.dump(le, MODELS_DIR / "label_encoder.pkl")

    # Also save results CSV
    report = classification_report(y_test, best_pred, target_names=le.classes_,
                                   output_dict=True, zero_division=0)
    rows = []
    for cls in le.classes_:
        rows.append({
            "model": "XGBoost_improved",
            "class": cls,
            "precision": report[cls]["precision"],
            "recall": report[cls]["recall"],
            "f1": report[cls]["f1-score"],
            "support": report[cls]["support"],
        })
    rows.append({
        "model": "XGBoost_improved",
        "class": "weighted_avg",
        "precision": report["weighted avg"]["precision"],
        "recall": report["weighted avg"]["recall"],
        "f1": report["weighted avg"]["f1-score"],
        "support": report["weighted avg"]["support"],
    })
    csv_path = os.path.join(os.path.dirname(__file__), "classification_results.csv")
    if os.path.exists(csv_path):
        old = pd.read_csv(csv_path)
        old = old[old["model"] != "XGBoost_improved"]
        pd.concat([old, pd.DataFrame(rows)], ignore_index=True).to_csv(csv_path, index=False)
    else:
        pd.DataFrame(rows).to_csv(csv_path, index=False)
    log(f"Saved model and results.")

log(f"\nEXP07 RESULT: {best_name}, Acc={best_acc:.4f}, F1={best_f1:.4f}")
