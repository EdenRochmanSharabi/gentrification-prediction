"""
Spatial neighbor features: use administrative hierarchy as proxy for proximity.
Census sections sharing the same district (first 7 digits of CUSEC) are neighbors.
All features computed at time t (leak-free).
"""
import sys, os, warnings, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier

from src.data_loader import load_data
from src.gentrification import classify_cases

LOG = os.path.join(os.path.dirname(__file__), "spatial.log")
open(LOG, "w").close()

def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")

log("=" * 60)
log("SPATIAL NEIGHBOR FEATURES")
log("=" * 60)

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

# District = first 7 digits of CUSEC (province + municipality + district)
df["district"] = df["CUSEC"].str[:7]

log("Computing spatial features by district...")

# For each period-district, compute neighbor stats (excluding the section itself)
# These are all at time t, so leak-free for predicting t+1

# Mean rent/sale change in same district (excluding self)
for col_name, change_col in [("nb_rent_change", "rent_change"), ("nb_sale_change", "sale_change")]:
    district_sum = df.groupby(["district", "period"])[change_col].transform("sum")
    district_count = df.groupby(["district", "period"])[change_col].transform("count")
    self_val = df[change_col].fillna(0)
    df[col_name] = (district_sum - self_val) / (district_count - 1).replace(0, np.nan)

# Neighbor case distribution (share of each case in same district, excluding self)
for c in ["A", "B", "C", "D"]:
    is_case = (df["case"] == c).astype(float)
    district_sum = df.groupby(["district", "period"]).transform(lambda x: x.sum() if x.name == f"_tmp_{c}" else None)
    # Simpler approach: compute at district level then adjust
    df[f"_is_{c}"] = is_case

for c in ["A", "B", "C", "D"]:
    col = f"_is_{c}"
    dist_sum = df.groupby(["district", "period"])[col].transform("sum")
    dist_count = df.groupby(["district", "period"])[col].transform("count")
    df[f"nb_pct_{c}"] = (dist_sum - df[col]) / (dist_count - 1).replace(0, np.nan)
    df.drop(columns=[col], inplace=True)

# Neighbor mean prices
for col_name, price_col in [("nb_sale_price", sale_col), ("nb_rent_price", rent_col)]:
    dist_sum = df.groupby(["district", "period"])[price_col].transform("sum")
    dist_count = df.groupby(["district", "period"])[price_col].transform("count")
    df[col_name] = (dist_sum - df[price_col].fillna(0)) / (dist_count - 1).replace(0, np.nan)

# Price gap vs neighbors
df["sale_vs_nb"] = df[sale_col] / df["nb_sale_price"].replace(0, np.nan)
df["rent_vs_nb"] = df[rent_col] / df["nb_rent_price"].replace(0, np.nan)

# Number of sections in same district (density proxy)
df["nb_sections_count"] = df.groupby(["district", "period"])["CUSEC"].transform("count")

# Municipality-level features (broader context)
df["municipality"] = df["CUSEC"].str[:5]
for col_name, change_col in [("mun_rent_change", "rent_change"), ("mun_sale_change", "sale_change")]:
    df[col_name] = df.groupby(["municipality", "period"])[change_col].transform("mean")

log("Spatial features computed")

# ── BASE FEATURES (same as improved_model.py) ──
log("Engineering base features...")
df["sale_price"] = df[sale_col]
df["rent_price"] = df[rent_col]
df["rent_sale_ratio"] = df[rent_col] / df[sale_col].replace(0, np.nan)

le_case = LabelEncoder()
df["case_encoded_feat"] = le_case.fit_transform(df["case"].fillna("Unknown"))
for c in ["A", "B", "C", "D", "None"]:
    df[f"is_case_{c}"] = (df["case"] == c).astype(int)

df["sale_lag1"] = df.groupby("CUSEC")[sale_col].shift(1)
df["rent_lag1"] = df.groupby("CUSEC")[rent_col].shift(1)
df["sale_lag2"] = df.groupby("CUSEC")[sale_col].shift(2)
df["rent_lag2"] = df.groupby("CUSEC")[rent_col].shift(2)
df["sale_lag4"] = df.groupby("CUSEC")[sale_col].shift(4)
df["rent_lag4"] = df.groupby("CUSEC")[rent_col].shift(4)

df["sale_change_curr"] = df["sale_change"]
df["rent_change_curr"] = df["rent_change"]
df["sale_change_lag1"] = df.groupby("CUSEC")["sale_change"].shift(1)
df["rent_change_lag1"] = df.groupby("CUSEC")["rent_change"].shift(1)
df["change_interaction"] = df["rent_change"] * df["sale_change"]

df["sale_roll3"] = df.groupby("CUSEC")[sale_col].transform(lambda x: x.rolling(3, min_periods=1).mean())
df["rent_roll3"] = df.groupby("CUSEC")[rent_col].transform(lambda x: x.rolling(3, min_periods=1).mean())
df["sale_roll_std3"] = df.groupby("CUSEC")[sale_col].transform(lambda x: x.rolling(3, min_periods=1).std())
df["rent_roll_std3"] = df.groupby("CUSEC")[rent_col].transform(lambda x: x.rolling(3, min_periods=1).std())

df["sale_momentum"] = df[sale_col] - df["sale_lag1"]
df["rent_momentum"] = df[rent_col] - df["rent_lag1"]

prov_sale_mean = df.groupby(["NPRO", "period"])[sale_col].transform("mean")
prov_rent_mean = df.groupby(["NPRO", "period"])[rent_col].transform("mean")
df["sale_vs_prov"] = df[sale_col] / prov_sale_mean.replace(0, np.nan)
df["rent_vs_prov"] = df[rent_col] / prov_rent_mean.replace(0, np.nan)

df["sale_yoy"] = (df[sale_col] - df["sale_lag4"]) / df["sale_lag4"].replace(0, np.nan)
df["rent_yoy"] = (df[rent_col] - df["rent_lag4"]) / df["rent_lag4"].replace(0, np.nan)

df["stock_sale_lag1"] = df.groupby("CUSEC")["stock_residential_sale_all"].shift(1)
df["stock_rent_lag1"] = df.groupby("CUSEC")["stock_residential_rent_all"].shift(1)
df["stock_ratio"] = df["stock_residential_sale_all"] / df["stock_residential_rent_all"].replace(0, np.nan)

df["month"] = df["period"].dt.month
df["quarter"] = df["period"].dt.quarter

BASE_FEATURES = [
    sale_col, rent_col, "rent_sale_ratio",
    "sale_change_curr", "rent_change_curr", "change_interaction",
    "case_encoded_feat",
    "is_case_A", "is_case_B", "is_case_C", "is_case_D", "is_case_None",
    "sale_lag1", "rent_lag1", "sale_lag2", "rent_lag2", "sale_lag4", "rent_lag4",
    "sale_change_lag1", "rent_change_lag1",
    "sale_roll3", "rent_roll3", "sale_roll_std3", "rent_roll_std3",
    "sale_momentum", "rent_momentum",
    "sale_vs_prov", "rent_vs_prov",
    "sale_yoy", "rent_yoy",
    "stock_residential_sale_all", "stock_residential_rent_all",
    "stock_sale_lag1", "stock_rent_lag1", "stock_ratio",
    "month", "quarter", "year",
]

SPATIAL_FEATURES = [
    "nb_rent_change", "nb_sale_change",
    "nb_pct_A", "nb_pct_B", "nb_pct_C", "nb_pct_D",
    "nb_sale_price", "nb_rent_price",
    "sale_vs_nb", "rent_vs_nb",
    "nb_sections_count",
    "mun_rent_change", "mun_sale_change",
]

log(f"Base: {len(BASE_FEATURES)}, Spatial: {len(SPATIAL_FEATURES)}")

# ── TRAIN/TEST ──
trainable = df.dropna(subset=["sale_lag1", "rent_lag1", "target_case"])
trainable = trainable[trainable["target_case"] != "Unknown"].copy()

le = LabelEncoder()
trainable["target_encoded"] = le.fit_transform(trainable["target_case"])

train_mask = trainable["year"] < 2020
test_mask = trainable["year"] >= 2020

# Baseline (no spatial)
log("\n" + "=" * 60)
log("BASELINE (no spatial features)")
imp = SimpleImputer(strategy="mean")
sc = StandardScaler()
X_tr = sc.fit_transform(imp.fit_transform(trainable.loc[train_mask, BASE_FEATURES].values))
X_te = sc.transform(imp.transform(trainable.loc[test_mask, BASE_FEATURES].values))
y_tr = trainable.loc[train_mask, "target_encoded"].values
y_te = trainable.loc[test_mask, "target_encoded"].values

clf = XGBClassifier(
    objective="multi:softprob", n_estimators=500, max_depth=8,
    learning_rate=0.05, subsample=0.8, colsample_bytree=0.7,
    eval_metric="mlogloss", random_state=42, n_jobs=-1,
)
clf.fit(X_tr, y_tr)
pred = clf.predict(X_te)
acc_base = accuracy_score(y_te, pred)
f1_base = f1_score(y_te, pred, average="weighted")
log(f"  Acc={acc_base:.4f}, F1={f1_base:.4f}")

# With spatial
log("\n" + "=" * 60)
log("BASE + SPATIAL FEATURES")
ALL = BASE_FEATURES + SPATIAL_FEATURES
imp2 = SimpleImputer(strategy="mean")
sc2 = StandardScaler()
X_tr2 = sc2.fit_transform(imp2.fit_transform(trainable.loc[train_mask, ALL].values))
X_te2 = sc2.transform(imp2.transform(trainable.loc[test_mask, ALL].values))

clf2 = XGBClassifier(
    objective="multi:softprob", n_estimators=500, max_depth=8,
    learning_rate=0.05, subsample=0.8, colsample_bytree=0.7,
    eval_metric="mlogloss", random_state=42, n_jobs=-1,
)
clf2.fit(X_tr2, y_tr)
pred2 = clf2.predict(X_te2)
acc_spatial = accuracy_score(y_te, pred2)
f1_spatial = f1_score(y_te, pred2, average="weighted")
log(f"  Acc={acc_spatial:.4f}, F1={f1_spatial:.4f}")
log(f"  Delta vs baseline: {(acc_spatial - acc_base)*100:+.2f}pp")

# Feature importance of spatial features
fi = sorted(zip(ALL, clf2.feature_importances_), key=lambda x: x[1], reverse=True)
log("\nTop 15 features:")
for name, imp_val in fi[:15]:
    tag = "[SPATIAL]" if name in SPATIAL_FEATURES else "[BASE]"
    log(f"  {tag} {name}: {imp_val:.4f}")

log("\nDONE")
