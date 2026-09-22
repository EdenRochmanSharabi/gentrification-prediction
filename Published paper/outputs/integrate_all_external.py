"""
Integrate ALL external datasets with Idealista data and test model improvement.
Datasets:
1. ADRH (income by census section, annual 2015-2023) — CUSEC join
2. Census 2011 (static demographics by census section) — CUSEC join
3. EPA (unemployment by CCAA, quarterly) — NCA join
4. Hipotecas (mortgages by province, monthly) — NPRO join
5. Turismo (tourism by province, monthly) — NPRO join
6. IPC (inflation by CCAA, monthly) — NCA join
7. Foreign population (by province, annual) — NPRO join
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
from collections import Counter
from pathlib import Path
import glob

from src.data_loader import load_data
from src.gentrification import classify_cases

LOG = os.path.join(os.path.dirname(__file__), "integrate_all.log")
open(LOG, "w").close()

def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")

EXT_DIR = Path("data/external")

log("=" * 60)
log("FULL EXTERNAL DATA INTEGRATION")
log("=" * 60)

# ══════════════════════════════════════════════════════════
# 1. ADRH — Income by Census Section
# ══════════════════════════════════════════════════════════
log("\n[1/7] ADRH (Income by Census Section)")
adrh = pd.read_parquet(EXT_DIR / "adrh_income_raw.parquet")
adrh = adrh[adrh["Secciones"].notna()].copy()
adrh["CUSEC"] = adrh["Secciones"].str[:10]
adrh["year"] = adrh["Periodo"].astype(int)
adrh["Total"] = pd.to_numeric(adrh["Total"].str.replace(".", "", regex=False).str.replace(",", ".", regex=False), errors="coerce")

# Pivot indicators
adrh_pivot = adrh.pivot_table(
    index=["CUSEC", "year"],
    columns="Indicadores de renta media y mediana",
    values="Total",
    aggfunc="first"
).reset_index()
adrh_pivot.columns = [c.replace(" ", "_").lower() if c not in ["CUSEC", "year"] else c for c in adrh_pivot.columns]
# Simplify column names
rename = {}
for c in adrh_pivot.columns:
    if "renta_neta_media_por_persona" in c: rename[c] = "income_net_person"
    elif "renta_neta_media_por_hogar" in c: rename[c] = "income_net_household"
    elif "renta_bruta_media_por_persona" in c: rename[c] = "income_gross_person"
    elif "media_de_la_renta_por_unidad" in c: rename[c] = "income_consumption_unit"
    elif "mediana_de_la_renta_por_unidad" in c: rename[c] = "income_median_consumption"
    elif "renta_bruta_media_por_hogar" in c: rename[c] = "income_gross_household"
adrh_pivot = adrh_pivot.rename(columns=rename)
income_cols = [c for c in adrh_pivot.columns if c.startswith("income_")]
log(f"  ADRH: {len(adrh_pivot)} rows, cols: {income_cols}")
log(f"  CUSEC: {adrh_pivot['CUSEC'].nunique()}, years: {sorted(adrh_pivot['year'].unique())}")

# ══════════════════════════════════════════════════════════
# 2. Census 2011 — Static Demographics
# ══════════════════════════════════════════════════════════
log("\n[2/7] Census 2011 (Static Demographics)")
CENSUS_DIR = Path("/Users/edenrochman/Documents/TFG/Base de datos/indicadores_seccion_censal_csv")
census_files = sorted(glob.glob(str(CENSUS_DIR / "C2011_ccaa*_Indicadores.csv")))
census = pd.concat([pd.read_csv(f, dtype=str) for f in census_files], ignore_index=True)
census["CUSEC"] = census["cpro"].str.zfill(2) + census["cmun"].str.zfill(3) + census["dist"].str.zfill(2) + census["secc"].str.zfill(3)
skip_cols = {"ccaa", "cpro", "cmun", "dist", "secc", "CUSEC"}
for col in census.columns:
    if col not in skip_cols:
        census[col] = pd.to_numeric(census[col], errors="coerce")

census["pop_total"] = census["t1_1"]
census["pct_young"] = census["t3_1"] / census["t1_1"]
census["pct_working"] = census["t3_2"] / census["t1_1"]
census["pct_elderly"] = census["t3_3"] / census["t1_1"]
census["pct_foreign"] = 1 - (census["t4_1"] / census["t1_1"])
census["pct_higher_edu"] = census["t6_1"] / (census["t6_1"] + census["t6_2"]).replace(0, np.nan)
census["pct_unemployed"] = census["t7_3"] / census["t1_1"]
total_tenure = sum(census[f"t9_{i}"] for i in range(1, 7))
census["pct_rented"] = census["t9_3"] / total_tenure.replace(0, np.nan)
census["pct_owned"] = (census["t9_1"] + census["t9_2"]) / total_tenure.replace(0, np.nan)

CENSUS_FEATS = ["pop_total", "pct_young", "pct_working", "pct_elderly",
                "pct_foreign", "pct_higher_edu", "pct_unemployed", "pct_rented", "pct_owned"]
census_slim = census[["CUSEC"] + CENSUS_FEATS].copy()
log(f"  Census: {len(census_slim)} sections, {len(CENSUS_FEATS)} features")

# ══════════════════════════════════════════════════════════
# 3. EPA — Unemployment by CCAA (quarterly)
# ══════════════════════════════════════════════════════════
log("\n[3/7] EPA (Unemployment Rate by CCAA)")
epa = pd.read_parquet(EXT_DIR / "epa_unemployment_raw.parquet")
epa = epa[(epa["Sexo"] == "Ambos sexos") & (epa["Edad"] == "Total")].copy()
epa = epa[~epa["Comunidades y Ciudades Autónomas"].str.contains("Total", na=False)].copy()
epa["NCA"] = epa["Comunidades y Ciudades Autónomas"].str.extract(r'\d+\s+(.*)')
epa["NCA"] = epa["NCA"].str.strip()
epa["year"] = epa["Periodo"].str.extract(r'(\d{4})').astype(int)
epa["quarter"] = epa["Periodo"].str.extract(r'T(\d)').astype(int)
epa["unemployment_rate"] = pd.to_numeric(epa["Total"].str.replace(",", "."), errors="coerce")
epa_slim = epa[["NCA", "year", "quarter", "unemployment_rate"]].dropna()
log(f"  EPA: {len(epa_slim)} rows, NCA: {epa_slim['NCA'].nunique()}")

# ══════════════════════════════════════════════════════════
# 4. Hipotecas — Mortgages by Province (monthly → quarterly)
# ══════════════════════════════════════════════════════════
log("\n[4/7] Hipotecas (Mortgages by Province)")
hip = pd.read_parquet(EXT_DIR / "hipotecas_raw.parquet")
hip = hip[hip["Entidad que concede el préstamo"] == "Total"].copy()
hip = hip[~hip["Provincias"].str.contains("Total", na=False)].copy()
hip["NPRO"] = hip["Provincias"].str.extract(r'\d+\s+(.*)')
hip["NPRO"] = hip["NPRO"].str.strip()
hip["Total"] = pd.to_numeric(hip["Total"].str.replace(".", "", regex=False).str.replace(",", ".", regex=False), errors="coerce")
hip_num = hip[hip["Número e importe"] == "Número de hipotecas"].copy()
hip_num["year"] = hip_num["Periodo"].str.extract(r'(\d{4})').astype(int)
hip_num["month"] = hip_num["Periodo"].str.extract(r'M(\d+)').astype(int)
hip_num["quarter"] = ((hip_num["month"] - 1) // 3) + 1
hip_q = hip_num.groupby(["NPRO", "year", "quarter"])["Total"].sum().reset_index()
hip_q.columns = ["NPRO", "year", "quarter", "mortgages_count"]
log(f"  Hipotecas: {len(hip_q)} quarterly rows, provinces: {hip_q['NPRO'].nunique()}")

# ══════════════════════════════════════════════════════════
# 5. Turismo — Tourism by Province (monthly → quarterly)
# ══════════════════════════════════════════════════════════
log("\n[5/7] Turismo (Tourism by Province)")
tur = pd.read_parquet(EXT_DIR / "turismo_raw.parquet")
tur = tur[tur["Provincias"].notna()].copy()
tur = tur[~tur["Provincias"].str.contains("Total", na=False)].copy()
tur = tur[tur["Viajeros y pernoctaciones"] == "Pernoctación"].copy()
tur = tur[tur["Residencia: Nivel 1"] == "Total"].copy()
tur["NPRO"] = tur["Provincias"].str.extract(r'\d+\s+(.*)')
tur["NPRO"] = tur["NPRO"].str.strip()
tur["Total"] = pd.to_numeric(tur["Total"].str.replace(".", "", regex=False).str.replace(",", ".", regex=False), errors="coerce")
tur["year"] = tur["Periodo"].str.extract(r'(\d{4})').astype(int)
tur["month"] = tur["Periodo"].str.extract(r'M(\d+)').astype(int)
tur["quarter"] = ((tur["month"] - 1) // 3) + 1
tur_q = tur.groupby(["NPRO", "year", "quarter"])["Total"].sum().reset_index()
tur_q.columns = ["NPRO", "year", "quarter", "tourism_overnight"]
log(f"  Turismo: {len(tur_q)} quarterly rows, provinces: {tur_q['NPRO'].nunique()}")

# ══════════════════════════════════════════════════════════
# 6. IPC — Inflation by CCAA (monthly → quarterly)
# ══════════════════════════════════════════════════════════
log("\n[6/7] IPC (Inflation by CCAA)")
ipc = pd.read_parquet(EXT_DIR / "ipc_raw.parquet")
ipc = ipc[ipc["Grupos ECOICOP"] == "Índice general"].copy()
ipc = ipc[ipc["Tipo de dato"] == "Índice"].copy()
ipc = ipc[~ipc["Comunidades y Ciudades Autónomas"].str.contains("Nacional", na=False)].copy()
ipc["NCA"] = ipc["Comunidades y Ciudades Autónomas"].str.strip()
ipc["Total"] = pd.to_numeric(ipc["Total"].str.replace(",", "."), errors="coerce")
ipc["year"] = ipc["Periodo"].str.extract(r'(\d{4})').astype(int)
ipc["month"] = ipc["Periodo"].str.extract(r'M(\d+)').astype(int)
ipc["quarter"] = ((ipc["month"] - 1) // 3) + 1
ipc_q = ipc.groupby(["NCA", "year", "quarter"])["Total"].mean().reset_index()
ipc_q.columns = ["NCA", "year", "quarter", "cpi_index"]
log(f"  IPC: {len(ipc_q)} quarterly rows, CCAA: {ipc_q['NCA'].nunique()}")

# ══════════════════════════════════════════════════════════
# LOAD MAIN DATA AND JOIN EVERYTHING
# ══════════════════════════════════════════════════════════
log("\n" + "=" * 60)
log("Loading Idealista + joining all external data")
log("=" * 60)

df = load_data()
df = df.sort_values(["CUSEC", "period"])
df["year"] = df["period"].dt.year
df["quarter"] = df["period"].dt.quarter
df["month"] = df["period"].dt.month

log(f"Base: {len(df):,} rows")

# Join 1: ADRH (CUSEC + year)
df = df.merge(adrh_pivot, on=["CUSEC", "year"], how="left")
matched = df["income_net_person"].notna().sum()
log(f"  ADRH joined: {matched:,}/{len(df):,} ({matched/len(df)*100:.1f}%)")

# Join 2: Census (CUSEC, static)
df = df.merge(census_slim, on="CUSEC", how="left")
matched = df["pop_total"].notna().sum()
log(f"  Census joined: {matched:,}/{len(df):,} ({matched/len(df)*100:.1f}%)")

# Join 3: EPA (NCA + year + quarter)
df = df.merge(epa_slim, on=["NCA", "year", "quarter"], how="left")
matched = df["unemployment_rate"].notna().sum()
log(f"  EPA joined: {matched:,}/{len(df):,} ({matched/len(df)*100:.1f}%)")

# Join 4: Hipotecas (NPRO + year + quarter)
df = df.merge(hip_q, on=["NPRO", "year", "quarter"], how="left")
matched = df["mortgages_count"].notna().sum()
log(f"  Hipotecas joined: {matched:,}/{len(df):,} ({matched/len(df)*100:.1f}%)")

# Join 5: Turismo (NPRO + year + quarter)
df = df.merge(tur_q, on=["NPRO", "year", "quarter"], how="left")
matched = df["tourism_overnight"].notna().sum()
log(f"  Turismo joined: {matched:,}/{len(df):,} ({matched/len(df)*100:.1f}%)")

# Join 6: IPC (NCA + year + quarter)
df = df.merge(ipc_q, on=["NCA", "year", "quarter"], how="left")
matched = df["cpi_index"].notna().sum()
log(f"  IPC joined: {matched:,}/{len(df):,} ({matched/len(df)*100:.1f}%)")

# ══════════════════════════════════════════════════════════
# BUILD FEATURES AND TARGET
# ══════════════════════════════════════════════════════════
log("\nEngineering features...")
sale_col = "unitprice_residential_sale_all"
rent_col = "unitprice_residential_rent_all"
grouped = df.groupby("CUSEC")

df["rent_change"] = grouped[rent_col].pct_change()
df["sale_change"] = grouped[sale_col].pct_change()
df = classify_cases(df)
df["target_case"] = df.groupby("CUSEC")["case"].shift(-1)

# Price features (same as before)
df["sale_price"] = df[sale_col]
df["rent_price"] = df[rent_col]
df["rent_sale_ratio"] = df[rent_col] / df[sale_col].replace(0, np.nan)
le_case = LabelEncoder()
df["case_encoded_feat"] = le_case.fit_transform(df["case"].fillna("Unknown"))
df["sale_lag1"] = grouped[sale_col].shift(1)
df["rent_lag1"] = grouped[rent_col].shift(1)
df["sale_lag2"] = grouped[sale_col].shift(2)
df["rent_lag2"] = grouped[rent_col].shift(2)
df["sale_lag4"] = grouped[sale_col].shift(4)
df["rent_lag4"] = grouped[rent_col].shift(4)
df["sale_change_curr"] = df["sale_change"]
df["rent_change_curr"] = df["rent_change"]
df["sale_change_lag1"] = grouped["sale_change"].shift(1)
df["rent_change_lag1"] = grouped["rent_change"].shift(1)
df["sale_roll3"] = grouped[sale_col].transform(lambda x: x.rolling(3, min_periods=1).mean())
df["rent_roll3"] = grouped[rent_col].transform(lambda x: x.rolling(3, min_periods=1).mean())
df["sale_roll_std3"] = grouped[sale_col].transform(lambda x: x.rolling(3, min_periods=1).std())
df["rent_roll_std3"] = grouped[rent_col].transform(lambda x: x.rolling(3, min_periods=1).std())
df["sale_roll6"] = grouped[sale_col].transform(lambda x: x.rolling(6, min_periods=2).mean())
df["rent_roll6"] = grouped[rent_col].transform(lambda x: x.rolling(6, min_periods=2).mean())
df["sale_momentum"] = df[sale_col] - df["sale_lag1"]
df["rent_momentum"] = df[rent_col] - df["rent_lag1"]
df["change_interaction"] = df["rent_change"] * df["sale_change"]
prov_sale_mean = df.groupby(["NPRO", "period"])[sale_col].transform("mean")
prov_rent_mean = df.groupby(["NPRO", "period"])[rent_col].transform("mean")
df["sale_vs_prov"] = df[sale_col] / prov_sale_mean.replace(0, np.nan)
df["rent_vs_prov"] = df[rent_col] / prov_rent_mean.replace(0, np.nan)
df["sale_yoy"] = (df[sale_col] - df["sale_lag4"]) / df["sale_lag4"].replace(0, np.nan)
df["rent_yoy"] = (df[rent_col] - df["rent_lag4"]) / df["rent_lag4"].replace(0, np.nan)
df["stock_sale_lag1"] = grouped["stock_residential_sale_all"].shift(1)
df["stock_rent_lag1"] = grouped["stock_residential_rent_all"].shift(1)
df["stock_ratio"] = df["stock_residential_sale_all"] / df["stock_residential_rent_all"].replace(0, np.nan)

# Derived external features
if "income_net_person" in df.columns:
    df["income_to_rent"] = df["income_net_person"] / (df[rent_col] * 12).replace(0, np.nan)
    df["income_to_sale"] = df["income_net_person"] / df[sale_col].replace(0, np.nan)

PRICE_FEATURES = [
    "sale_price", "rent_price", "rent_sale_ratio", "case_encoded_feat",
    "sale_lag1", "rent_lag1", "sale_lag2", "rent_lag2", "sale_lag4", "rent_lag4",
    "sale_change_curr", "rent_change_curr", "sale_change_lag1", "rent_change_lag1",
    "sale_roll3", "rent_roll3", "sale_roll_std3", "rent_roll_std3",
    "sale_roll6", "rent_roll6",
    "sale_momentum", "rent_momentum", "change_interaction",
    "sale_vs_prov", "rent_vs_prov", "sale_yoy", "rent_yoy",
    "stock_residential_sale_all", "stock_residential_rent_all",
    "stock_sale_lag1", "stock_rent_lag1", "stock_ratio",
    "month", "quarter", "year",
]

EXTERNAL_FEATURES = (
    [c for c in income_cols if c in df.columns] +
    CENSUS_FEATS +
    ["unemployment_rate", "mortgages_count", "tourism_overnight", "cpi_index"] +
    ["income_to_rent", "income_to_sale"]
)
EXTERNAL_FEATURES = [f for f in EXTERNAL_FEATURES if f in df.columns]

ALL_FEATURES = PRICE_FEATURES + EXTERNAL_FEATURES
log(f"Price features: {len(PRICE_FEATURES)}")
log(f"External features: {len(EXTERNAL_FEATURES)}")
log(f"Total features: {len(ALL_FEATURES)}")

# ══════════════════════════════════════════════════════════
# TRAIN / TEST
# ══════════════════════════════════════════════════════════
trainable = df.dropna(subset=["sale_lag1", "rent_lag1", "target_case"])
trainable = trainable[trainable["target_case"] != "Unknown"].copy()

le = LabelEncoder()
trainable["target_encoded"] = le.fit_transform(trainable["target_case"])

train_mask = trainable["year"] < 2020
y_train = trainable.loc[train_mask, "target_encoded"].values
y_test = trainable.loc[~train_mask, "target_encoded"].values

classes = list(le.classes_)
y_bin_test = np.isin(trainable.loc[~train_mask, "target_case"], ["A", "C"]).astype(int)
a_idx, c_idx = classes.index("A"), classes.index("C")

log(f"Train: {len(y_train):,}, Test: {len(y_test):,}")

def run_model(name, features):
    log(f"\n{'=' * 60}")
    log(f"MODEL: {name} ({len(features)} features)")
    X_tr = trainable.loc[train_mask, features].values
    X_te = trainable.loc[~train_mask, features].values
    imp = SimpleImputer(strategy="mean")
    sc = StandardScaler()
    X_tr_p = sc.fit_transform(imp.fit_transform(X_tr))
    X_te_p = sc.transform(imp.transform(X_te))

    clf = XGBClassifier(
        objective="multi:softprob", n_estimators=500, max_depth=8,
        learning_rate=0.05, subsample=0.8, colsample_bytree=0.7,
        eval_metric="mlogloss", random_state=42,
    )
    t0 = time.time()
    clf.fit(X_tr_p, y_train)
    elapsed = time.time() - t0

    pred = clf.predict(X_te_p)
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average="weighted")
    proba = clf.predict_proba(X_te_p)
    auc = roc_auc_score(y_bin_test, proba[:, a_idx] + proba[:, c_idx])

    log(f"  Acc={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}, Time={elapsed:.1f}s")

    # Top features
    fi = sorted(zip(features, clf.feature_importances_), key=lambda x: x[1], reverse=True)
    log("  Top 10 features:")
    for fname, imp_val in fi[:10]:
        source = "EXT" if fname in EXTERNAL_FEATURES else "PRICE"
        log(f"    [{source}] {fname}: {imp_val:.4f} ({imp_val*100:.1f}%)")

    return acc, f1, auc

# Run comparisons
acc1, f1_1, auc1 = run_model("Price only (baseline)", PRICE_FEATURES)
acc2, f1_2, auc2 = run_model("Price + ALL external", ALL_FEATURES)
acc3, f1_3, auc3 = run_model("Price + Income only", PRICE_FEATURES + [c for c in income_cols if c in df.columns] + ["income_to_rent", "income_to_sale"])
acc4, f1_4, auc4 = run_model("Price + Census only", PRICE_FEATURES + CENSUS_FEATS)
acc5, f1_5, auc5 = run_model("Price + Macro (EPA+Hip+Tur+IPC)", PRICE_FEATURES + ["unemployment_rate", "mortgages_count", "tourism_overnight", "cpi_index"])

log("\n" + "=" * 60)
log("FINAL COMPARISON")
log("=" * 60)
log(f"{'Model':<35} {'Acc':>8} {'F1':>8} {'AUC':>8}")
log("-" * 65)
for name, a, f, u in [
    ("Price only (baseline)", acc1, f1_1, auc1),
    ("Price + ALL external", acc2, f1_2, auc2),
    ("Price + Income (ADRH)", acc3, f1_3, auc3),
    ("Price + Census 2011", acc4, f1_4, auc4),
    ("Price + Macro", acc5, f1_5, auc5),
]:
    log(f"{name:<35} {a:>8.4f} {f:>8.4f} {u:>8.4f}")

log(f"\nBest delta vs baseline:")
best_acc = max(acc2, acc3, acc4, acc5)
best_auc = max(auc2, auc3, auc4, auc5)
log(f"  Acc: {(best_acc - acc1)*100:+.2f}pp")
log(f"  AUC: {(best_auc - auc1)*100:+.2f}pp")
log("DONE")
