"""
Full pipeline: load data, classify gentrification, train classifiers, evaluate, generate outputs.
"""
import sys
import os
import warnings
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, accuracy_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
import joblib

from config import MODELS_DIR, FIGURES_DIR, MAPS_DIR, CARTOGRAPHY_PATH
from src.data_loader import load_data, clean_data
from src.gentrification import compute_pct_change, classify_cases
from src.visualization import CASE_COLORS

LOG_FILE = os.path.join(os.path.dirname(__file__), "outputs", "pipeline.log")
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def build_classification_features(df):
    """Build features for gentrification classification at census-section level."""
    df = df.copy()
    df = df.sort_values("period")
    df["year"] = df["period"].dt.year

    sale_col = "unitprice_residential_sale_all"
    rent_col = "unitprice_residential_rent_all"
    sale_stock = "stock_residential_sale_all"
    rent_stock = "stock_residential_rent_all"

    grouped = df.groupby("CUSEC")

    df["sale_lag1"] = grouped[sale_col].shift(1)
    df["rent_lag1"] = grouped[rent_col].shift(1)
    df["sale_roll3"] = grouped[sale_col].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )
    df["rent_roll3"] = grouped[rent_col].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )
    df["sale_roll_std3"] = grouped[sale_col].transform(
        lambda x: x.rolling(3, min_periods=1).std()
    )
    df["rent_roll_std3"] = grouped[rent_col].transform(
        lambda x: x.rolling(3, min_periods=1).std()
    )
    df["sale_pct"] = grouped[sale_col].pct_change()
    df["rent_pct"] = grouped[rent_col].pct_change()
    df["stock_sale_lag1"] = grouped[sale_stock].shift(1)
    df["stock_rent_lag1"] = grouped[rent_stock].shift(1)
    df["rent_sale_ratio"] = df[rent_col] / df[sale_col].replace(0, np.nan)
    df["month"] = df["period"].dt.month
    df["quarter"] = df["period"].dt.quarter

    return df


CLASSIFICATION_FEATURES = [
    "sale_lag1", "rent_lag1",
    "sale_roll3", "rent_roll3",
    "sale_roll_std3", "rent_roll_std3",
    "stock_residential_sale_all", "stock_residential_rent_all",
    "stock_sale_lag1", "stock_rent_lag1",
    "rent_sale_ratio",
    "month", "quarter", "year",
]


def main():
    log("=" * 60)
    log("GENTRIFICATION CLASSIFICATION PIPELINE")
    log("=" * 60)

    has_geometry = CARTOGRAPHY_PATH.exists()
    if not has_geometry:
        log("NOTE: .shp file missing, loading metadata from .dbf (no maps)")

    # ── 1. Load data ──
    log("Loading Idealista + INE cartography data...")
    df = load_data()
    log(f"Loaded {len(df):,} rows, {df['CUSEC'].nunique():,} census sections")

    # ── 2. Build features ──
    log("Building classification features (lags, rolling stats, ratios)...")
    featured = build_classification_features(df)

    featured["rent_change"] = featured.groupby("CUSEC")["unitprice_residential_rent_all"].pct_change()
    featured["sale_change"] = featured.groupby("CUSEC")["unitprice_residential_sale_all"].pct_change()
    featured = classify_cases(featured)

    case_counts = featured["case"].value_counts()
    log(f"Case distribution (all rows):\n{case_counts.to_string()}")

    trainable = featured.dropna(subset=CLASSIFICATION_FEATURES + ["case"])
    trainable = trainable[trainable["case"] != "Unknown"]
    log(f"Trainable rows after dropping NaN: {len(trainable):,}")

    # ── 3. Encode target ──
    le = LabelEncoder()
    trainable = trainable.copy()
    trainable["case_encoded"] = le.fit_transform(trainable["case"])
    log(f"Classes: {list(le.classes_)}")

    X = trainable[CLASSIFICATION_FEATURES].values
    y = trainable["case_encoded"].values

    # Temporal split: train on pre-2020, test on 2020+
    train_mask = trainable["year"] < 2020
    X_train, X_test = X[train_mask], X[~train_mask]
    y_train, y_test = y[train_mask], y[~train_mask]
    log(f"Train: {len(X_train):,} rows (pre-2020), Test: {len(X_test):,} rows (2020+)")

    if len(X_train) == 0 or len(X_test) == 0:
        log("ERROR: empty train or test set, aborting")
        return

    # ── 4. Train classifiers ──
    results = {}

    log("Training XGBoost Classifier...")
    xgb_clf = make_pipeline(
        SimpleImputer(strategy="mean"),
        StandardScaler(),
        XGBClassifier(
            objective="multi:softprob",
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="mlogloss",
            random_state=42,
        ),
    )
    xgb_clf.fit(X_train, y_train)
    xgb_pred = xgb_clf.predict(X_test)
    xgb_acc = accuracy_score(y_test, xgb_pred)
    xgb_f1 = f1_score(y_test, xgb_pred, average="weighted")
    log(f"XGBoost: Accuracy={xgb_acc:.4f}, Weighted F1={xgb_f1:.4f}")
    results["XGBoost"] = {"model": xgb_clf, "preds": xgb_pred, "acc": xgb_acc, "f1": xgb_f1}

    log("Training Random Forest Classifier...")
    rf_clf = make_pipeline(
        SimpleImputer(strategy="mean"),
        StandardScaler(),
        RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        ),
    )
    rf_clf.fit(X_train, y_train)
    rf_pred = rf_clf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)
    rf_f1 = f1_score(y_test, rf_pred, average="weighted")
    log(f"RandomForest: Accuracy={rf_acc:.4f}, Weighted F1={rf_f1:.4f}")
    results["RandomForest"] = {"model": rf_clf, "preds": rf_pred, "acc": rf_acc, "f1": rf_f1}

    # ── 5. Classification reports ──
    log("\n" + "=" * 40)
    log("CLASSIFICATION REPORTS")
    log("=" * 40)

    for name, res in results.items():
        report = classification_report(
            y_test, res["preds"], target_names=le.classes_, zero_division=0
        )
        log(f"\n{name}:\n{report}")

        cm = confusion_matrix(y_test, res["preds"])
        log(f"{name} Confusion Matrix:\n{cm}")

    # ── 6. Feature importance plots ──
    log("Computing feature importances...")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    for name, res in results.items():
        model = res["model"]
        if hasattr(model[-1], "feature_importances_"):
            importances = model[-1].feature_importances_
            idx = np.argsort(importances)[::-1]

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(
                [CLASSIFICATION_FEATURES[i] for i in idx],
                importances[idx],
                color="steelblue",
            )
            ax.set_xlabel("Importance")
            ax.set_title(f"{name} Feature Importance")
            ax.invert_yaxis()
            plt.tight_layout()
            path = FIGURES_DIR / f"{name.lower()}_feature_importance.png"
            fig.savefig(path, dpi=200)
            plt.close(fig)
            log(f"Saved: {path}")

    # ── 7. Confusion matrix plots ──
    for name, res in results.items():
        cm = confusion_matrix(y_test, res["preds"])
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.set_xticks(range(len(le.classes_)))
        ax.set_yticks(range(len(le.classes_)))
        ax.set_xticklabels(le.classes_)
        ax.set_yticklabels(le.classes_)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"{name} Confusion Matrix")
        for i in range(len(le.classes_)):
            for j in range(len(le.classes_)):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        plt.colorbar(im)
        plt.tight_layout()
        path = FIGURES_DIR / f"{name.lower()}_confusion_matrix.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        log(f"Saved: {path}")

    # ── 8. Per-class performance bar chart ──
    for name, res in results.items():
        report_dict = classification_report(
            y_test, res["preds"], target_names=le.classes_,
            output_dict=True, zero_division=0
        )
        classes = [c for c in le.classes_]
        f1s = [report_dict[c]["f1-score"] for c in classes]
        precisions = [report_dict[c]["precision"] for c in classes]
        recalls = [report_dict[c]["recall"] for c in classes]

        x = np.arange(len(classes))
        width = 0.25
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - width, precisions, width, label="Precision", color="steelblue")
        ax.bar(x, recalls, width, label="Recall", color="coral")
        ax.bar(x + width, f1s, width, label="F1", color="seagreen")
        ax.set_xticks(x)
        ax.set_xticklabels(classes)
        ax.set_ylabel("Score")
        ax.set_title(f"{name} Per-Class Performance")
        ax.legend()
        ax.set_ylim(0, 1.05)
        plt.tight_layout()
        path = FIGURES_DIR / f"{name.lower()}_per_class_performance.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        log(f"Saved: {path}")

    # ── 9. Case distribution over time ──
    log("Generating case distribution over time...")
    featured_clean = featured[featured["case"].isin(["A", "B", "C", "D", "None"])]
    if "year" not in featured_clean.columns:
        featured_clean = featured_clean.copy()
        featured_clean["year"] = featured_clean["period"].dt.year
    case_time = featured_clean.groupby(["year", "case"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(12, 6))
    case_time.plot(kind="bar", stacked=True, ax=ax,
                   color=[CASE_COLORS.get(c, "gray") for c in case_time.columns])
    ax.set_title("Gentrification Case Distribution Over Time")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Census Sections")
    ax.legend(title="Case")
    plt.tight_layout()
    path = FIGURES_DIR / "case_distribution_over_time.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    log(f"Saved: {path}")

    # ── 10. Geographic aggregation by province ──
    log("Generating province-level case breakdown...")
    if "NPRO" in featured.columns:
        prov_cases = featured_clean.groupby(["NPRO", "case"]).size().unstack(fill_value=0)
        prov_cases["total"] = prov_cases.sum(axis=1)
        prov_cases = prov_cases.sort_values("total", ascending=True)
        top15 = prov_cases.tail(15).drop(columns=["total"])

        fig, ax = plt.subplots(figsize=(12, 8))
        top15.plot(kind="barh", stacked=True, ax=ax,
                   color=[CASE_COLORS.get(c, "gray") for c in top15.columns])
        ax.set_title("Gentrification Cases by Province (Top 15)")
        ax.set_xlabel("Number of Observations")
        ax.legend(title="Case")
        plt.tight_layout()
        path = FIGURES_DIR / "cases_by_province_top15.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        log(f"Saved: {path}")

    # ── 11. Save models ──
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for name, res in results.items():
        path = MODELS_DIR / f"{name.lower()}_classifier.pkl"
        joblib.dump(res["model"], path)
        log(f"Model saved: {path}")

    joblib.dump(le, MODELS_DIR / "label_encoder.pkl")

    # ── 12. Save results CSV ──
    results_rows = []
    for name, res in results.items():
        report_dict = classification_report(
            y_test, res["preds"], target_names=le.classes_,
            output_dict=True, zero_division=0
        )
        for cls in le.classes_:
            results_rows.append({
                "model": name,
                "class": cls,
                "precision": report_dict[cls]["precision"],
                "recall": report_dict[cls]["recall"],
                "f1": report_dict[cls]["f1-score"],
                "support": report_dict[cls]["support"],
            })
        results_rows.append({
            "model": name,
            "class": "weighted_avg",
            "precision": report_dict["weighted avg"]["precision"],
            "recall": report_dict["weighted avg"]["recall"],
            "f1": report_dict["weighted avg"]["f1-score"],
            "support": report_dict["weighted avg"]["support"],
        })
    results_df = pd.DataFrame(results_rows)
    results_path = os.path.join(os.path.dirname(__file__), "outputs", "classification_results.csv")
    results_df.to_csv(results_path, index=False)
    log(f"Results CSV saved: {results_path}")

    # ── 13. Summary ──
    log("\n" + "=" * 60)
    log("RESULTS SUMMARY")
    log("=" * 60)
    log(f"{'Model':<20} {'Accuracy':>10} {'F1 (weighted)':>15}")
    log("-" * 45)
    for name, res in results.items():
        log(f"{name:<20} {res['acc']:>10.4f} {res['f1']:>15.4f}")

    log(f"\nTotal census sections: {df['CUSEC'].nunique():,}")
    log(f"Date range: {df['period'].min()} to {df['period'].max()}")
    log(f"Trainable observations: {len(trainable):,}")
    log(f"Train set: {len(X_train):,} (pre-2020)")
    log(f"Test set: {len(X_test):,} (2020+)")

    if not has_geometry:
        log("\nNOTE: Maps were skipped (.shp file missing). "
            "Re-run after restoring the shapefile.")

    log("\nPipeline complete.")


if __name__ == "__main__":
    main()
