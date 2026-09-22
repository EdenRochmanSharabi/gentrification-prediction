"""
Annual reformulation: predict gentrification type at year Y+1 from data through year Y.
Non-overlapping annual windows, strictly leak-free.
"""
import sys, os, warnings, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier

from src.data_loader import load_data

LOG = os.path.join(os.path.dirname(__file__), "annual.log")
open(LOG, "w").close()

def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")

log("=" * 60)
log("ANNUAL REFORMULATION")
log("=" * 60)

log("Loading data...")
df = load_data()
df = df.sort_values(["CUSEC", "period"])
df["year"] = df["period"].dt.year

sale_col = "unitprice_residential_sale_all"
rent_col = "unitprice_residential_rent_all"

# ── Compute annual summary per CUSEC per year ──
log("Computing annual summaries per CUSEC...")

annual = df.groupby(["CUSEC", "year"]).agg(
    sale_mean=(sale_col, "mean"),
    sale_std=(sale_col, "std"),
    sale_min=(sale_col, "min"),
    sale_max=(sale_col, "max"),
    sale_last=(sale_col, "last"),
    sale_first=(sale_col, "first"),
    rent_mean=(rent_col, "mean"),
    rent_std=(rent_col, "std"),
    rent_min=(rent_col, "min"),
    rent_max=(rent_col, "max"),
    rent_last=(rent_col, "last"),
    rent_first=(rent_col, "first"),
    stock_sale_mean=("stock_residential_sale_all", "mean"),
    stock_rent_mean=("stock_residential_rent_all", "mean"),
    n_quarters=("period", "count"),
    province=("NPRO", "first"),
).reset_index()

log(f"Annual rows: {len(annual):,}, CUSECs: {annual['CUSEC'].nunique():,}")
log(f"Years: {sorted(annual['year'].unique())}")

# ── Annual price changes (non-overlapping: compare year Y to year Y-1) ──
annual = annual.sort_values(["CUSEC", "year"])
grp = annual.groupby("CUSEC")

annual["sale_change_annual"] = grp["sale_mean"].pct_change()
annual["rent_change_annual"] = grp["rent_mean"].pct_change()

# Within-year dynamics
annual["sale_intra_change"] = (annual["sale_last"] - annual["sale_first"]) / annual["sale_first"].replace(0, np.nan)
annual["rent_intra_change"] = (annual["rent_last"] - annual["rent_first"]) / annual["rent_first"].replace(0, np.nan)

# ── Classify annual gentrification type ──
def classify_annual(row):
    rc = row["rent_change_annual"]
    sc = row["sale_change_annual"]
    if pd.isna(rc) or pd.isna(sc):
        return "Unknown"
    if rc > 0 and sc > 0:
        return "C"
    elif rc <= 0 and sc <= 0:
        return "D"
    elif rc > 0 and sc <= 0:
        return "A"
    elif rc <= 0 and sc > 0:
        return "B"
    return "None"

annual["case_annual"] = annual.apply(classify_annual, axis=1)

# Target: NEXT year's case
annual["target_case"] = grp["case_annual"].shift(-1)

log(f"\nAnnual case distribution:")
for c in ["A", "B", "C", "D", "Unknown"]:
    n = (annual["case_annual"] == c).sum()
    log(f"  {c}: {n:,} ({n/len(annual)*100:.1f}%)")

# ── Feature engineering (all at year Y, predicting Y+1) ──
log("\nEngineering features...")

# Lagged annual prices
annual["sale_mean_lag1"] = grp["sale_mean"].shift(1)
annual["rent_mean_lag1"] = grp["rent_mean"].shift(1)

# Rent-sale ratio
annual["rent_sale_ratio"] = annual["rent_mean"] / annual["sale_mean"].replace(0, np.nan)

# Current case encoded
le_case = LabelEncoder()
annual["case_encoded"] = le_case.fit_transform(annual["case_annual"])

# Year-over-year momentum
annual["sale_momentum"] = annual["sale_mean"] - annual["sale_mean_lag1"]
annual["rent_momentum"] = annual["rent_mean"] - annual["rent_mean_lag1"]

# Volatility within year (std of quarterly prices)
annual["sale_vol"] = annual["sale_std"]
annual["rent_vol"] = annual["rent_std"]

# Price range within year
annual["sale_range"] = (annual["sale_max"] - annual["sale_min"]) / annual["sale_mean"].replace(0, np.nan)
annual["rent_range"] = (annual["rent_max"] - annual["rent_min"]) / annual["rent_mean"].replace(0, np.nan)

# Stock features
annual["stock_ratio"] = annual["stock_sale_mean"] / annual["stock_rent_mean"].replace(0, np.nan)

# Province average prices (context)
prov_sale = annual.groupby(["province", "year"])["sale_mean"].transform("mean")
prov_rent = annual.groupby(["province", "year"])["rent_mean"].transform("mean")
annual["sale_vs_prov"] = annual["sale_mean"] / prov_sale.replace(0, np.nan)
annual["rent_vs_prov"] = annual["rent_mean"] / prov_rent.replace(0, np.nan)

# Historical case frequency (expanding, shifted)
for c in ["A", "B", "C", "D"]:
    annual[f"hist_pct_{c}"] = grp.apply(
        lambda g, case=c: ((g["case_annual"] == case).cumsum().shift(1)) /
        (g["case_annual"].notna().cumsum().shift(1).replace(0, np.nan))
    ).reset_index(level=0, drop=True)

FEATURES = [
    "sale_mean", "rent_mean", "rent_sale_ratio",
    "sale_change_annual", "rent_change_annual",
    "sale_intra_change", "rent_intra_change",
    "case_encoded",
    "sale_mean_lag1", "rent_mean_lag1",
    "sale_momentum", "rent_momentum",
    "sale_vol", "rent_vol",
    "sale_range", "rent_range",
    "stock_sale_mean", "stock_rent_mean", "stock_ratio",
    "sale_vs_prov", "rent_vs_prov",
    "n_quarters",
    "hist_pct_A", "hist_pct_B", "hist_pct_C", "hist_pct_D",
    "year",
]

log(f"Features: {len(FEATURES)}")

# ── TRAIN/TEST ──
trainable = annual.dropna(subset=["sale_mean_lag1", "target_case"])
trainable = trainable[trainable["target_case"] != "Unknown"].copy()

le = LabelEncoder()
trainable["target_encoded"] = le.fit_transform(trainable["target_case"])
classes = list(le.classes_)

train_mask = trainable["year"] < 2020
test_mask = trainable["year"] >= 2020

X_tr = trainable.loc[train_mask, FEATURES].values
X_te = trainable.loc[test_mask, FEATURES].values
y_tr = trainable.loc[train_mask, "target_encoded"].values
y_te = trainable.loc[test_mask, "target_encoded"].values

log(f"Train: {len(X_tr):,}, Test: {len(X_te):,}")
log(f"Classes: {classes}")

if len(X_te) == 0:
    log("ERROR: No test data. Cannot evaluate.")
    log("DONE")
    sys.exit(0)

imp = SimpleImputer(strategy="mean")
sc = StandardScaler()
X_tr_s = sc.fit_transform(imp.fit_transform(X_tr))
X_te_s = sc.transform(imp.transform(X_te))

# ── Model ──
log("\n" + "=" * 60)
log("ANNUAL PREDICTION: case(Y+1) from features(Y)")

clf = XGBClassifier(
    objective="multi:softprob", n_estimators=500, max_depth=6,
    learning_rate=0.05, subsample=0.8, colsample_bytree=0.7,
    min_child_weight=5, gamma=0.1,
    eval_metric="mlogloss", random_state=42, n_jobs=-1,
)
clf.fit(X_tr_s, y_tr)
pred = clf.predict(X_te_s)
proba = clf.predict_proba(X_te_s)

acc = accuracy_score(y_te, pred)
f1 = f1_score(y_te, pred, average="weighted")
log(f"  Acc={acc:.4f}, F1={f1:.4f}")
log(f"  (Chance baseline = {1/len(classes):.1%})")

# Binary: displacement risk (A or C)
if "A" in classes and "C" in classes:
    a_idx = classes.index("A")
    c_idx = classes.index("C")
    y_bin = np.isin(le.inverse_transform(y_te), ["A", "C"]).astype(int)
    prob_disp = proba[:, a_idx] + proba[:, c_idx]
    try:
        auc = roc_auc_score(y_bin, prob_disp)
        log(f"  Displacement AUC: {auc:.4f}")
    except Exception:
        log("  Displacement AUC: N/A")

log(f"\nPer-class report:")
log(classification_report(y_te, pred, target_names=classes, zero_division=0))

# Feature importance
fi = sorted(zip(FEATURES, clf.feature_importances_), key=lambda x: x[1], reverse=True)
log("Top 10 features:")
for name, imp_val in fi[:10]:
    log(f"  {name}: {imp_val:.4f}")

# ── Selective prediction (accuracy-coverage) ──
log("\nSelective prediction (accuracy-coverage):")
max_proba = proba.max(axis=1)
for coverage in [1.0, 0.8, 0.6, 0.5, 0.4]:
    n_keep = int(len(max_proba) * coverage)
    if n_keep == 0:
        continue
    idx = np.argsort(max_proba)[-n_keep:]
    acc_cov = accuracy_score(y_te[idx], pred[idx])
    log(f"  Coverage {coverage*100:.0f}%: Acc={acc_cov:.4f} ({n_keep:,} sections)")

log("\nDONE")
