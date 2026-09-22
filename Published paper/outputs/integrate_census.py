"""
Integrate Census 2011 indicators with Idealista data.
Adds ~140 sociodemographic features per census section as STATIC baseline features.
These capture the pre-period state of each neighborhood (population, age, nationality,
education, housing) which predicts vulnerability to gentrification.

Output: enriched dataset with census features joined, ready for modeling.
"""
import sys, os, warnings, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from collections import Counter
from pathlib import Path
import glob

from src.data_loader import load_data
from src.gentrification import classify_cases

LOG = os.path.join(os.path.dirname(__file__), "census_integration.log")
open(LOG, "w").close()

def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")

log("=" * 60)
log("CENSUS 2011 INTEGRATION")
log("=" * 60)

# ── Load census data ──
CENSUS_DIR = Path("/Users/edenrochman/Documents/TFG/Base de datos/indicadores_seccion_censal_csv")
census_files = sorted(glob.glob(str(CENSUS_DIR / "C2011_ccaa*_Indicadores.csv")))

log(f"Loading {len(census_files)} census files...")
census_dfs = []
for f in census_files:
    df_c = pd.read_csv(f, dtype=str)
    census_dfs.append(df_c)

census = pd.concat(census_dfs, ignore_index=True)
log(f"Census rows: {len(census)}")
log(f"Census columns: {len(census.columns)}")

# Construct CUSEC from components: cpro(2) + cmun(3) + dist(2) + secc(3) = 10 digits
census["CUSEC"] = (
    census["cpro"].str.zfill(2) +
    census["cmun"].str.zfill(3) +
    census["dist"].str.zfill(2) +
    census["secc"].str.zfill(3)
)

# Convert numeric columns
skip_cols = {"ccaa", "cpro", "cmun", "dist", "secc", "CUSEC"}
for col in census.columns:
    if col not in skip_cols:
        census[col] = pd.to_numeric(census[col], errors="coerce")

log(f"Unique CUSEC in census: {census['CUSEC'].nunique()}")

# ── Identify most useful columns ──
# Census 2011 indicator codes (from INE documentation):
# t1_1: Total population
# t2_1/t2_2: Males/Females
# t3_1/t3_2/t3_3: Age groups (0-15, 16-64, 65+)
# t4_1-t4_8: Nationality (Spanish, EU, rest Europe, Africa, Americas, Asia, Oceania, Stateless)
# t5: Country of birth details
# t6_1/t6_2: Education (completed secondary+, less than secondary)
# t7: Economic activity
# t8: Dwelling type
# t9: Tenure
# t10: Building age
# t11-t22: Various housing and demographic indicators

# Create meaningful derived features from census
log("Engineering census features...")

# Population
census["pop_total"] = census["t1_1"]

# Age structure (proportions)
census["pct_young"] = census["t3_1"] / census["t1_1"]  # 0-15
census["pct_working"] = census["t3_2"] / census["t1_1"]  # 16-64
census["pct_elderly"] = census["t3_3"] / census["t1_1"]  # 65+

# Nationality
census["pct_foreign"] = 1 - (census["t4_1"] / census["t1_1"])  # non-Spanish
census["pct_eu_foreign"] = census["t4_2"] / census["t1_1"]  # EU foreigners
census["pct_africa"] = census["t4_4"] / census["t1_1"]
census["pct_americas"] = census["t4_5"] / census["t1_1"]

# Education (t6_1 = higher education, t6_2 = less)
if "t6_1" in census.columns and "t6_2" in census.columns:
    total_edu = census["t6_1"] + census["t6_2"]
    census["pct_higher_edu"] = census["t6_1"] / total_edu.replace(0, np.nan)

# Economic activity
if "t7_1" in census.columns:
    census["pct_employed"] = census["t7_1"] / census["t1_1"]
if "t7_3" in census.columns:
    census["pct_unemployed"] = census["t7_3"] / census["t1_1"]

# Housing tenure (t9_1=owned paid, t9_2=mortgage, t9_3=rented, etc.)
if all(f"t9_{i}" in census.columns for i in range(1, 7)):
    total_tenure = sum(census[f"t9_{i}"] for i in range(1, 7))
    census["pct_owned"] = (census["t9_1"] + census["t9_2"]) / total_tenure.replace(0, np.nan)
    census["pct_rented"] = census["t9_3"] / total_tenure.replace(0, np.nan)

# Building age (t10_1=before 1900, ..., t10_5=recent)
if all(f"t10_{i}" in census.columns for i in range(1, 6)):
    total_bld = sum(census[f"t10_{i}"] for i in range(1, 6))
    census["pct_old_buildings"] = (census["t10_1"] + census["t10_2"]) / total_bld.replace(0, np.nan)
    census["pct_new_buildings"] = census["t10_5"] / total_bld.replace(0, np.nan)

# Select features to keep
CENSUS_FEATURES = [
    "pop_total", "pct_young", "pct_working", "pct_elderly",
    "pct_foreign", "pct_eu_foreign", "pct_africa", "pct_americas",
    "pct_higher_edu", "pct_employed", "pct_unemployed",
    "pct_owned", "pct_rented",
    "pct_old_buildings", "pct_new_buildings",
]
# Keep only features that exist
CENSUS_FEATURES = [f for f in CENSUS_FEATURES if f in census.columns]
log(f"Census features engineered: {len(CENSUS_FEATURES)}")
for f in CENSUS_FEATURES:
    valid = census[f].notna().sum()
    log(f"  {f}: {valid} valid ({valid/len(census)*100:.1f}%)")

census_slim = census[["CUSEC"] + CENSUS_FEATURES].copy()

# ── Load main data ──
log("\nLoading Idealista data...")
df = load_data()
df = df.sort_values(["CUSEC", "period"])
df["year"] = df["period"].dt.year

log(f"Idealista rows: {len(df):,}")
log(f"Unique CUSEC in Idealista: {df['CUSEC'].nunique()}")

# ── Join census to main data ──
df = df.merge(census_slim, on="CUSEC", how="left")
matched = df[CENSUS_FEATURES[0]].notna().sum()
log(f"Census match rate: {matched:,}/{len(df):,} ({matched/len(df)*100:.1f}%)")

# ── Build features and target (same as model_leakfree.py) ──
sale_col = "unitprice_residential_sale_all"
rent_col = "unitprice_residential_rent_all"
grouped = df.groupby("CUSEC")

df["rent_change"] = grouped[rent_col].pct_change()
df["sale_change"] = grouped[sale_col].pct_change()
df = classify_cases(df)

df["target_case"] = df.groupby("CUSEC")["case"].shift(-1)

# Price features
df["sale_price"] = df[sale_col]
df["rent_price"] = df[rent_col]
df["rent_sale_ratio"] = df[rent_col] / df[sale_col].replace(0, np.nan)

# Current case
le_case = LabelEncoder()
df["case_encoded_feat"] = le_case.fit_transform(df["case"].fillna("Unknown"))

# Lags
df["sale_lag1"] = grouped[sale_col].shift(1)
df["rent_lag1"] = grouped[rent_col].shift(1)
df["sale_lag2"] = grouped[sale_col].shift(2)
df["rent_lag2"] = grouped[rent_col].shift(2)
df["sale_lag4"] = grouped[sale_col].shift(4)
df["rent_lag4"] = grouped[rent_col].shift(4)

# Changes
df["sale_change_curr"] = df["sale_change"]
df["rent_change_curr"] = df["rent_change"]
df["sale_change_lag1"] = grouped["sale_change"].shift(1)
df["rent_change_lag1"] = grouped["rent_change"].shift(1)

# Rolling
df["sale_roll3"] = grouped[sale_col].transform(lambda x: x.rolling(3, min_periods=1).mean())
df["rent_roll3"] = grouped[rent_col].transform(lambda x: x.rolling(3, min_periods=1).mean())
df["sale_roll_std3"] = grouped[sale_col].transform(lambda x: x.rolling(3, min_periods=1).std())
df["rent_roll_std3"] = grouped[rent_col].transform(lambda x: x.rolling(3, min_periods=1).std())
df["sale_roll6"] = grouped[sale_col].transform(lambda x: x.rolling(6, min_periods=2).mean())
df["rent_roll6"] = grouped[rent_col].transform(lambda x: x.rolling(6, min_periods=2).mean())

# Momentum
df["sale_momentum"] = df[sale_col] - df["sale_lag1"]
df["rent_momentum"] = df[rent_col] - df["rent_lag1"]
df["change_interaction"] = df["rent_change"] * df["sale_change"]

# Provincial context
prov_sale_mean = df.groupby(["NPRO", "period"])[sale_col].transform("mean")
prov_rent_mean = df.groupby(["NPRO", "period"])[rent_col].transform("mean")
df["sale_vs_prov"] = df[sale_col] / prov_sale_mean.replace(0, np.nan)
df["rent_vs_prov"] = df[rent_col] / prov_rent_mean.replace(0, np.nan)

# YoY
df["sale_yoy"] = (df[sale_col] - df["sale_lag4"]) / df["sale_lag4"].replace(0, np.nan)
df["rent_yoy"] = (df[rent_col] - df["rent_lag4"]) / df["rent_lag4"].replace(0, np.nan)

# Stock
df["stock_sale_lag1"] = grouped["stock_residential_sale_all"].shift(1)
df["stock_rent_lag1"] = grouped["stock_residential_rent_all"].shift(1)
df["stock_ratio"] = df["stock_residential_sale_all"] / df["stock_residential_rent_all"].replace(0, np.nan)

df["month"] = df["period"].dt.month
df["quarter"] = df["period"].dt.quarter

# ── Feature lists ──
PRICE_FEATURES = [
    "sale_price", "rent_price", "rent_sale_ratio", "case_encoded_feat",
    "sale_lag1", "rent_lag1", "sale_lag2", "rent_lag2",
    "sale_lag4", "rent_lag4",
    "sale_change_curr", "rent_change_curr",
    "sale_change_lag1", "rent_change_lag1",
    "sale_roll3", "rent_roll3", "sale_roll_std3", "rent_roll_std3",
    "sale_roll6", "rent_roll6",
    "sale_momentum", "rent_momentum", "change_interaction",
    "sale_vs_prov", "rent_vs_prov",
    "sale_yoy", "rent_yoy",
    "stock_residential_sale_all", "stock_residential_rent_all",
    "stock_sale_lag1", "stock_rent_lag1", "stock_ratio",
    "month", "quarter", "year",
]

ALL_FEATURES = PRICE_FEATURES + CENSUS_FEATURES

log(f"\nPrice features: {len(PRICE_FEATURES)}")
log(f"Census features: {len(CENSUS_FEATURES)}")
log(f"Total features: {len(ALL_FEATURES)}")

# ── Train/test ──
trainable = df.dropna(subset=["sale_lag1", "rent_lag1", "target_case"])
trainable = trainable[trainable["target_case"] != "Unknown"].copy()

le = LabelEncoder()
trainable["target_encoded"] = le.fit_transform(trainable["target_case"])

train_mask = trainable["year"] < 2020
X_train_all = trainable.loc[train_mask, ALL_FEATURES].values
X_test_all = trainable.loc[~train_mask, ALL_FEATURES].values
X_train_price = trainable.loc[train_mask, PRICE_FEATURES].values
X_test_price = trainable.loc[~train_mask, PRICE_FEATURES].values
y_train = trainable.loc[train_mask, "target_encoded"].values
y_test = trainable.loc[~train_mask, "target_encoded"].values

log(f"Train: {len(X_train_all):,}, Test: {len(X_test_all):,}")

imp = SimpleImputer(strategy="mean")
sc = StandardScaler()

# ── Model 1: Price only (baseline) ──
log("\n" + "=" * 60)
log("MODEL 1: Price features only (baseline)")
X_tr1 = sc.fit_transform(imp.fit_transform(X_train_price))
X_te1 = sc.transform(imp.transform(X_test_price))

clf1 = XGBClassifier(
    objective="multi:softprob", n_estimators=500, max_depth=8,
    learning_rate=0.05, subsample=0.8, colsample_bytree=0.7,
    eval_metric="mlogloss", random_state=42,
)
clf1.fit(X_tr1, y_train)
p1 = clf1.predict(X_te1)
acc1 = accuracy_score(y_test, p1)
f1_1 = f1_score(y_test, p1, average="weighted")

classes = list(le.classes_)
y_bin = np.isin(trainable.loc[~train_mask, "target_case"], ["A", "C"]).astype(int)
proba1 = clf1.predict_proba(X_te1)
a_idx, c_idx = classes.index("A"), classes.index("C")
auc1 = roc_auc_score(y_bin, proba1[:, a_idx] + proba1[:, c_idx])

log(f"  Acc={acc1:.4f}, F1={f1_1:.4f}, Disp AUC={auc1:.4f}")

# ── Model 2: Price + Census ──
log("\n" + "=" * 60)
log("MODEL 2: Price + Census 2011 features")
imp2 = SimpleImputer(strategy="mean")
sc2 = StandardScaler()
X_tr2 = sc2.fit_transform(imp2.fit_transform(X_train_all))
X_te2 = sc2.transform(imp2.transform(X_test_all))

clf2 = XGBClassifier(
    objective="multi:softprob", n_estimators=500, max_depth=8,
    learning_rate=0.05, subsample=0.8, colsample_bytree=0.7,
    eval_metric="mlogloss", random_state=42,
)
clf2.fit(X_tr2, y_train)
p2 = clf2.predict(X_te2)
acc2 = accuracy_score(y_test, p2)
f1_2 = f1_score(y_test, p2, average="weighted")

proba2 = clf2.predict_proba(X_te2)
auc2 = roc_auc_score(y_bin, proba2[:, a_idx] + proba2[:, c_idx])

log(f"  Acc={acc2:.4f}, F1={f1_2:.4f}, Disp AUC={auc2:.4f}")

# Feature importance for enriched model
fi = sorted(zip(ALL_FEATURES, clf2.feature_importances_), key=lambda x: x[1], reverse=True)
log("\nTop 20 feature importances (enriched model):")
for fname, imp_val in fi[:20]:
    source = "CENSUS" if fname in CENSUS_FEATURES else "PRICE"
    log(f"  [{source}] {fname}: {imp_val:.4f} ({imp_val*100:.1f}%)")

# ── Comparison ──
log("\n" + "=" * 60)
log("COMPARISON")
log("=" * 60)
log(f"Price only:    Acc={acc1:.4f}, F1={f1_1:.4f}, AUC={auc1:.4f} ({len(PRICE_FEATURES)} features)")
log(f"Price+Census:  Acc={acc2:.4f}, F1={f1_2:.4f}, AUC={auc2:.4f} ({len(ALL_FEATURES)} features)")
delta_acc = (acc2 - acc1) * 100
delta_auc = (auc2 - auc1) * 100
log(f"Delta:         Acc={delta_acc:+.2f}pp, AUC={delta_auc:+.2f}pp")
log("=" * 60)
