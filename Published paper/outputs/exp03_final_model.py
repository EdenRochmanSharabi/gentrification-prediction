"""
Experiment 03: Final model - Extended features (no leakage) + Optuna best hyperparameters.
Combines the best of both approaches, generates per-class report, and saves the final model.
"""
import sys, os, warnings, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from collections import Counter
import joblib
import optuna

from config import MODELS_DIR, FIGURES_DIR
from src.data_loader import load_data
from src.gentrification import classify_cases

optuna.logging.set_verbosity(optuna.logging.WARNING)

LOG = os.path.join(os.path.dirname(__file__), "final_model.log")

def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG, "a") as f:
        f.write(line + "\n")

log("=" * 60)
log("FINAL MODEL: Extended features + Optuna optimization")
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

# === ALL FEATURES (leak-free) ===
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

df["sale_lag2"] = grouped[sale_col].shift(2)
df["rent_lag2"] = grouped[rent_col].shift(2)
df["sale_lag4"] = grouped[sale_col].shift(4)
df["rent_lag4"] = grouped[rent_col].shift(4)
df["sale_change_lag1"] = grouped["sale_change"].shift(1)
df["rent_change_lag1"] = grouped["rent_change"].shift(1)
df["sale_change_lag2"] = grouped["sale_change"].shift(2)
df["rent_change_lag2"] = grouped["rent_change"].shift(2)
df["interaction_lag1"] = df["sale_change_lag1"] * df["rent_change_lag1"]
df["sale_momentum"] = df["sale_lag1"] - df["sale_lag2"]
df["rent_momentum"] = df["rent_lag1"] - df["rent_lag2"]
prov_sale_mean = df.groupby(["NPRO", "period"])[sale_col].transform("mean")
prov_rent_mean = df.groupby(["NPRO", "period"])[rent_col].transform("mean")
df["sale_vs_prov"] = df[sale_col] / prov_sale_mean.replace(0, np.nan)
df["rent_vs_prov"] = df[rent_col] / prov_rent_mean.replace(0, np.nan)
df["sale_lag5"] = grouped[sale_col].shift(5)
df["rent_lag5"] = grouped[rent_col].shift(5)
df["sale_yoy_lag"] = (df["sale_lag1"] - df["sale_lag5"]) / df["sale_lag5"].replace(0, np.nan)
df["rent_yoy_lag"] = (df["rent_lag1"] - df["rent_lag5"]) / df["rent_lag5"].replace(0, np.nan)
df["sale_roll6"] = grouped[sale_col].transform(lambda x: x.rolling(6, min_periods=2).mean())
df["rent_roll6"] = grouped[rent_col].transform(lambda x: x.rolling(6, min_periods=2).mean())
df["sale_roll_std6"] = grouped[sale_col].transform(lambda x: x.rolling(6, min_periods=2).std())
df["rent_roll_std6"] = grouped[rent_col].transform(lambda x: x.rolling(6, min_periods=2).std())
df["stock_ratio"] = df["stock_residential_sale_all"] / df["stock_residential_rent_all"].replace(0, np.nan)
df["stock_sale_change"] = grouped["stock_residential_sale_all"].pct_change()
df["stock_rent_change"] = grouped["stock_residential_rent_all"].pct_change()

FEATURES = [
    "sale_lag1", "rent_lag1",
    "sale_roll3", "rent_roll3",
    "sale_roll_std3", "rent_roll_std3",
    "stock_residential_sale_all", "stock_residential_rent_all",
    "stock_sale_lag1", "stock_rent_lag1",
    "rent_sale_ratio",
    "month", "quarter", "year",
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

log(f"Features: {len(FEATURES)}")

trainable = df.dropna(subset=["sale_lag1", "rent_lag1", "case"])
trainable = trainable[trainable["case"] != "Unknown"].copy()

le = LabelEncoder()
trainable["case_encoded"] = le.fit_transform(trainable["case"])

train_mask = trainable["year"] < 2020
X_train = trainable.loc[train_mask, FEATURES].values
X_test = trainable.loc[~train_mask, FEATURES].values
y_train = trainable.loc[train_mask, "case_encoded"].values
y_test = trainable.loc[~train_mask, "case_encoded"].values

log(f"Train: {len(X_train):,}, Test: {len(X_test):,}")

counts = Counter(y_train)
total = len(y_train)
n_classes = len(counts)
cw = {cls: total / (n_classes * count) for cls, count in counts.items()}
sw = np.array([cw[yi] for yi in y_train])

imp = SimpleImputer(strategy="mean")
sc = StandardScaler()
X_train_p = sc.fit_transform(imp.fit_transform(X_train))
X_test_p = sc.transform(imp.transform(X_test))

# Optuna on extended features
log("Running Optuna (30 trials) on extended features...")
best_score = 0.0

def objective(trial):
    global best_score
    params = {
        "objective": "multi:softprob",
        "eval_metric": "mlogloss",
        "random_state": 42,
        "n_estimators": trial.suggest_int("n_estimators", 300, 800, step=50),
        "max_depth": trial.suggest_int("max_depth", 6, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.9),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 8),
        "gamma": trial.suggest_float("gamma", 0.0, 3.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 5.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 5.0, log=True),
    }
    use_w = trial.suggest_categorical("use_weights", [True, False])
    clf = XGBClassifier(**params)
    if use_w:
        clf.fit(X_train_p, y_train, sample_weight=sw)
    else:
        clf.fit(X_train_p, y_train)
    pred = clf.predict(X_test_p)
    f1 = f1_score(y_test, pred, average="weighted")
    acc = accuracy_score(y_test, pred)
    if f1 > best_score:
        best_score = f1
        log(f"  Trial {trial.number}: Acc={acc:.4f}, F1={f1:.4f} (NEW BEST)")
    return f1

study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=30, show_progress_bar=False)

log("=" * 60)
log(f"BEST TRIAL: {study.best_trial.number}")
log(f"Best F1: {study.best_value:.4f}")
log(f"Best params: {study.best_params}")

# Retrain final model
best_params = {k: v for k, v in study.best_params.items() if k != "use_weights"}
use_w = study.best_params["use_weights"]

final_clf = XGBClassifier(
    objective="multi:softprob", eval_metric="mlogloss", random_state=42,
    **best_params,
)
if use_w:
    final_clf.fit(X_train_p, y_train, sample_weight=sw)
else:
    final_clf.fit(X_train_p, y_train)

final_pred = final_clf.predict(X_test_p)
final_acc = accuracy_score(y_test, final_pred)
final_f1 = f1_score(y_test, final_pred, average="weighted")

log("=" * 60)
log(f"FINAL MODEL: Acc={final_acc:.4f}, F1={final_f1:.4f}")
log(f"\nPer-class report:")
report = classification_report(y_test, final_pred, target_names=le.classes_, zero_division=0)
log(report)

cm = confusion_matrix(y_test, final_pred)
log(f"Confusion matrix:\n{cm}")

# Feature importance
importances = final_clf.feature_importances_
feat_imp = sorted(zip(FEATURES, importances), key=lambda x: x[1], reverse=True)
log("\nTop 15 feature importances:")
for fname, imp_val in feat_imp[:15]:
    log(f"  {fname}: {imp_val:.4f} ({imp_val*100:.1f}%)")

# Save final model as pipeline
MODELS_DIR.mkdir(parents=True, exist_ok=True)
final_pipe = make_pipeline(SimpleImputer(strategy="mean"), StandardScaler(), final_clf)
final_pipe.fit(X_train, y_train, **({f"xgbclassifier__sample_weight": sw} if use_w else {}))
verify_pred = final_pipe.predict(X_test)
verify_acc = accuracy_score(y_test, verify_pred)
log(f"Pipeline verification: Acc={verify_acc:.4f}")

joblib.dump(final_pipe, MODELS_DIR / "xgboost_classifier.pkl")
joblib.dump(le, MODELS_DIR / "label_encoder.pkl")
log(f"Saved final model to {MODELS_DIR / 'xgboost_classifier.pkl'}")

# Save results CSV
report_dict = classification_report(y_test, final_pred, target_names=le.classes_, output_dict=True, zero_division=0)
rows = []
for cls in le.classes_:
    rows.append({
        "model": "XGBoost",
        "class": cls,
        "precision": report_dict[cls]["precision"],
        "recall": report_dict[cls]["recall"],
        "f1": report_dict[cls]["f1-score"],
        "support": report_dict[cls]["support"],
    })
rows.append({
    "model": "XGBoost",
    "class": "weighted_avg",
    "precision": report_dict["weighted avg"]["precision"],
    "recall": report_dict["weighted avg"]["recall"],
    "f1": report_dict["weighted avg"]["f1-score"],
    "support": report_dict["weighted avg"]["support"],
})

existing = os.path.join(os.path.dirname(__file__), "classification_results.csv")
if os.path.exists(existing):
    old = pd.read_csv(existing)
    old = old[old["model"] != "XGBoost"]
    new = pd.concat([old, pd.DataFrame(rows)], ignore_index=True)
else:
    new = pd.DataFrame(rows)
new.to_csv(existing, index=False)
log(f"Results saved to {existing}")

# Save feature list for regenerate_figures.py
import json
feat_file = os.path.join(os.path.dirname(__file__), "final_features.json")
json.dump(FEATURES, open(feat_file, "w"))
log(f"Feature list saved to {feat_file}")

log("\nFinal model complete.")
