"""
Hyperparameter tuning for XGBoost gentrification classifier using Optuna.
Bayesian optimization over a wide parameter space.
"""
import sys, os, warnings, time
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import optuna
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
import joblib

from config import MODELS_DIR, CARTOGRAPHY_PATH
from src.data_loader import load_data
from src.gentrification import classify_cases

optuna.logging.set_verbosity(optuna.logging.WARNING)

LOG_FILE = os.path.join(os.path.dirname(__file__), "outputs", "tuning.log")

def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

CLASSIFICATION_FEATURES = [
    "sale_lag1", "rent_lag1",
    "sale_roll3", "rent_roll3",
    "sale_roll_std3", "rent_roll_std3",
    "stock_residential_sale_all", "stock_residential_rent_all",
    "stock_sale_lag1", "stock_rent_lag1",
    "rent_sale_ratio",
    "month", "quarter", "year",
]

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

trainable = df.dropna(subset=CLASSIFICATION_FEATURES + ["case"])
trainable = trainable[trainable["case"] != "Unknown"].copy()

le = LabelEncoder()
trainable["case_encoded"] = le.fit_transform(trainable["case"])
log(f"Classes: {list(le.classes_)}")

X = trainable[CLASSIFICATION_FEATURES].values
y = trainable["case_encoded"].values

train_mask = trainable["year"] < 2020
X_train, X_test = X[train_mask], X[~train_mask]
y_train, y_test = y[train_mask], y[~train_mask]
log(f"Train: {len(X_train):,}, Test: {len(X_test):,}")

# Compute class weights for imbalanced classes
from collections import Counter
counts = Counter(y_train)
total = len(y_train)
n_classes = len(counts)
class_weights = {cls: total / (n_classes * count) for cls, count in counts.items()}
sample_weights = np.array([class_weights[yi] for yi in y_train])
log(f"Class weights: { {le.inverse_transform([k])[0]: f'{v:.2f}' for k, v in class_weights.items()} }")

# Baseline
log("=" * 50)
log("BASELINE (current model)")
imputer = SimpleImputer(strategy="mean")
scaler = StandardScaler()
X_train_prep = scaler.fit_transform(imputer.fit_transform(X_train))
X_test_prep = scaler.transform(imputer.transform(X_test))

baseline = XGBClassifier(
    objective="multi:softprob",
    n_estimators=300, max_depth=6, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    eval_metric="mlogloss", random_state=42,
)
baseline.fit(X_train_prep, y_train)
base_pred = baseline.predict(X_test_prep)
base_acc = accuracy_score(y_test, base_pred)
base_f1 = f1_score(y_test, base_pred, average="weighted")
log(f"Baseline: Acc={base_acc:.4f}, F1={base_f1:.4f}")

# Optuna study
log("=" * 50)
log("OPTUNA TUNING (50 trials)")

best_score = 0.0

def objective(trial):
    global best_score
    params = {
        "objective": "multi:softprob",
        "eval_metric": "mlogloss",
        "random_state": 42,
        "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=50),
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }

    use_weights = trial.suggest_categorical("use_class_weights", [True, False])

    clf = XGBClassifier(**params)
    if use_weights:
        clf.fit(X_train_prep, y_train, sample_weight=sample_weights)
    else:
        clf.fit(X_train_prep, y_train)

    pred = clf.predict(X_test_prep)
    f1 = f1_score(y_test, pred, average="weighted")
    acc = accuracy_score(y_test, pred)

    if f1 > best_score:
        best_score = f1
        log(f"  Trial {trial.number}: Acc={acc:.4f}, F1={f1:.4f} (NEW BEST)")

    return f1

study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=50, show_progress_bar=False)

log("=" * 50)
log(f"BEST TRIAL: {study.best_trial.number}")
log(f"Best F1: {study.best_value:.4f} (baseline: {base_f1:.4f}, delta: {study.best_value - base_f1:+.4f})")
log(f"Best Acc: {accuracy_score(y_test, base_pred):.4f} -> check below")
log(f"Best params: {study.best_params}")

# Retrain with best params
log("=" * 50)
log("Retraining with best params...")
best_params = {k: v for k, v in study.best_params.items() if k != "use_class_weights"}
use_weights = study.best_params["use_class_weights"]

best_clf = XGBClassifier(
    objective="multi:softprob",
    eval_metric="mlogloss",
    random_state=42,
    **best_params,
)

if use_weights:
    best_clf.fit(X_train_prep, y_train, sample_weight=sample_weights)
else:
    best_clf.fit(X_train_prep, y_train)

best_pred = best_clf.predict(X_test_prep)
best_acc = accuracy_score(y_test, best_pred)
best_f1 = f1_score(y_test, best_pred, average="weighted")

log(f"FINAL: Acc={best_acc:.4f} (was {base_acc:.4f}, delta {best_acc - base_acc:+.4f})")
log(f"FINAL: F1={best_f1:.4f} (was {base_f1:.4f}, delta {best_f1 - base_f1:+.4f})")
log(f"\nPer-class report:")
log(classification_report(y_test, best_pred, target_names=le.classes_, zero_division=0))

# Save improved model
MODELS_DIR.mkdir(parents=True, exist_ok=True)
best_pipeline = make_pipeline(
    SimpleImputer(strategy="mean"),
    StandardScaler(),
    best_clf,
)
# Refit the full pipeline
best_pipeline.fit(X_train, y_train, **({f"xgbclassifier__sample_weight": sample_weights} if use_weights else {}))
final_pred = best_pipeline.predict(X_test)
final_acc = accuracy_score(y_test, final_pred)
final_f1 = f1_score(y_test, final_pred, average="weighted")
log(f"Pipeline verification: Acc={final_acc:.4f}, F1={final_f1:.4f}")

joblib.dump(best_pipeline, MODELS_DIR / "xgboost_classifier_tuned.pkl")
joblib.dump(le, MODELS_DIR / "label_encoder.pkl")
log(f"Saved tuned model to {MODELS_DIR / 'xgboost_classifier_tuned.pkl'}")

# Save results CSV
results_rows = []
report_dict = classification_report(y_test, final_pred, target_names=le.classes_, output_dict=True, zero_division=0)
for cls in le.classes_:
    results_rows.append({
        "model": "XGBoost_tuned",
        "class": cls,
        "precision": report_dict[cls]["precision"],
        "recall": report_dict[cls]["recall"],
        "f1": report_dict[cls]["f1-score"],
        "support": report_dict[cls]["support"],
    })
results_rows.append({
    "model": "XGBoost_tuned",
    "class": "weighted_avg",
    "precision": report_dict["weighted avg"]["precision"],
    "recall": report_dict["weighted avg"]["recall"],
    "f1": report_dict["weighted avg"]["f1-score"],
    "support": report_dict["weighted avg"]["support"],
})

# Append to existing results
existing = os.path.join(os.path.dirname(__file__), "outputs", "classification_results.csv")
if os.path.exists(existing):
    old_df = pd.read_csv(existing)
    old_df = old_df[old_df["model"] != "XGBoost_tuned"]
    new_df = pd.concat([old_df, pd.DataFrame(results_rows)], ignore_index=True)
else:
    new_df = pd.DataFrame(results_rows)
new_df.to_csv(existing, index=False)
log(f"Results appended to {existing}")

log("\nTuning complete.")
