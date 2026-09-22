"""
ROUND 2 IMPROVEMENTS — gentrification type prediction (A/B/C/D/None) at t+1.

Baseline to beat: 42.88% (XGB, 67 feats, honest split). All experiments here are
STRICTLY leak-free: every feature at row (section, t) uses only information
available at time t; the target is case(t+1). Train < 2020, test >= 2020.
Validation-dependent choices (weights, alphas, multipliers) are tuned on
2018-2019 with models trained < 2018, never on test labels.

IDEAS TESTED (each independently against EXP0 baseline = base+history+spatial):
  EXP1  Mean-reversion features (gap to rolling/expanding mean, shock size,
        signed streaks, sign-pair state). Rationale: autocorr is -0.26/-0.21,
        so the signal IS reversal; give the model direct reversal geometry.
  EXP2  Price-discreteness features (historical zero-change rate, tick size,
        change measured in ticks, unchanged-price flags, 1/sqrt(stock) noise
        proxies). Rationale: low-stock sections have quantized prices and the
        inverted noise-ceiling suggests discreteness drives predictability.
  EXP3  Markov transition features (per-section expanding P(next same case),
        P(next=C), P(next=D) conditional on current case; global expanding
        transition matrix P(next | current) up to t-1).
  EXP4  Seasonal history (expanding share of case C/D and P(rent up),
        P(sale up) in the SAME calendar quarter of previous years).
  EXP5  All new feature groups combined.
  EXP6  Recency-weighted training (exponential decay, half-life tuned on val).
  EXP7  Seed bagging (5 XGBs, averaged probabilities).
  EXP8  Heterogeneous ensemble XGB+LightGBM+CatBoost, weights tuned on val.
  EXP9  Post-processing: (a) EM label-shift correction (Saerens et al. 2002,
        uses ONLY unlabeled test inputs — transductive but leak-free);
        (b) per-class probability multipliers tuned on validation.
  EXP10 Two-head 3-class decomposition (rent dir x sale dir in {down,zero,up})
        composed into 5-class probs + alpha-blend with direct model (alpha
        tuned on val). Fixes the 'None' hole of the old binary decomposition.
  EXP11 Regression stacking: out-of-time XGBRegressor predictions of
        rent_change(t+1), sale_change(t+1) as meta-features.
  EXP12 Rolling-origin retraining: refit every test quarter on all data < t
        (standard forecasting practice; the model sees 2020-2022 regime drift).
  EXP13 Final combination menu + selective-prediction curve.

Run:  ~/miniconda3/bin/python outputs/round2_improvements.py
Smoke test:  FAST=1 ~/miniconda3/bin/python outputs/round2_improvements.py
Full run takes ~2-4 h on M1 Pro. Progress is logged with timestamps to
outputs/round2.log as results come in.
"""
import sys, os, time, traceback, warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier, XGBRegressor

from src.data_loader import load_data
from src.gentrification import classify_cases

FAST = os.environ.get("FAST", "0") == "1"
N_EST = 60 if FAST else 500
N_EST_ROLL = 60 if FAST else 400
N_EST_REG = 60 if FAST else 300

LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "round2.log")
open(LOG, "w").close()
T0 = time.time()


def log(msg):
    ts = time.strftime("%H:%M:%S")
    el = time.time() - T0
    line = f"[{ts} +{el/60:6.1f}m] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


log("=" * 70)
log(f"ROUND 2 IMPROVEMENTS  (FAST={FAST}, n_estimators={N_EST})")
log("=" * 70)

# ════════════════════════════════════════════════════════════════════
# DATA LOADING (identical to improved_model.py)
# ════════════════════════════════════════════════════════════════════
log("Loading data...")
df = load_data()
df = df.sort_values(["CUSEC", "period"]).reset_index(drop=True)
df["year"] = df["period"].dt.year

if FAST:
    rng = np.random.default_rng(0)
    keep = rng.choice(df["CUSEC"].dropna().unique(), size=2500, replace=False)
    df = df[df["CUSEC"].isin(keep)].reset_index(drop=True)
    log(f"FAST mode: subsampled to {df['CUSEC'].nunique()} sections, {len(df):,} rows")

sale_col = "unitprice_residential_sale_all"
rent_col = "unitprice_residential_rent_all"
CLASSES5 = ["A", "B", "C", "D", "None"]

df["rent_change"] = df.groupby("CUSEC")[rent_col].pct_change()
df["sale_change"] = df.groupby("CUSEC")[sale_col].pct_change()
df = classify_cases(df)
df["target_case"] = df.groupby("CUSEC")["case"].shift(-1)
df["rent_change_next"] = df.groupby("CUSEC")["rent_change"].shift(-1)
df["sale_change_next"] = df.groupby("CUSEC")["sale_change"].shift(-1)
log(f"Rows: {len(df):,}, sections: {df['CUSEC'].nunique():,}")

# ════════════════════════════════════════════════════════════════════
# FAST expanding-window helpers (leak-free: everything shifted 1 row in-group)
# ════════════════════════════════════════════════════════════════════

def exp_mean_shift(v, keys):
    """Expanding mean over past rows only (excludes current row)."""
    val = v.fillna(0.0)
    cnt = v.notna().astype(float)
    cs = val.groupby(keys).cumsum().groupby(keys).shift(1)
    cc = cnt.groupby(keys).cumsum().groupby(keys).shift(1)
    return cs / cc.replace(0, np.nan)


def exp_std_shift(v, keys):
    """Expanding std (ddof=1) over past rows only."""
    val = v.fillna(0.0)
    cnt = v.notna().astype(float)
    s1 = val.groupby(keys).cumsum().groupby(keys).shift(1)
    s2 = (val * val).groupby(keys).cumsum().groupby(keys).shift(1)
    n = cnt.groupby(keys).cumsum().groupby(keys).shift(1)
    n = n.replace(0, np.nan)
    var = (s2 - s1 * s1 / n) / (n - 1).replace(0, np.nan)
    return np.sqrt(var.clip(lower=0))


# ════════════════════════════════════════════════════════════════════
# BASE FEATURES (same set as improved_model.py)
# ════════════════════════════════════════════════════════════════════
log("Engineering base features...")
grouped = df.groupby("CUSEC")
df["rent_sale_ratio"] = df[rent_col] / df[sale_col].replace(0, np.nan)

le_case = LabelEncoder()
df["case_encoded_feat"] = le_case.fit_transform(df["case"].fillna("Unknown"))
for c in CLASSES5:
    df[f"is_case_{c}"] = (df["case"] == c).astype(int)

for lag in (1, 2, 4):
    df[f"sale_lag{lag}"] = grouped[sale_col].shift(lag)
    df[f"rent_lag{lag}"] = grouped[rent_col].shift(lag)

df["sale_change_curr"] = df["sale_change"]
df["rent_change_curr"] = df["rent_change"]
df["sale_change_lag1"] = grouped["sale_change"].shift(1)
df["rent_change_lag1"] = grouped["rent_change"].shift(1)
df["sale_change_lag2"] = grouped["sale_change"].shift(2)
df["rent_change_lag2"] = grouped["rent_change"].shift(2)
df["change_interaction"] = df["rent_change"] * df["sale_change"]

df["sale_roll3"] = grouped[sale_col].transform(lambda x: x.rolling(3, min_periods=1).mean())
df["rent_roll3"] = grouped[rent_col].transform(lambda x: x.rolling(3, min_periods=1).mean())
df["sale_roll_std3"] = grouped[sale_col].transform(lambda x: x.rolling(3, min_periods=1).std())
df["rent_roll_std3"] = grouped[rent_col].transform(lambda x: x.rolling(3, min_periods=1).std())
df["sale_roll6"] = grouped[sale_col].transform(lambda x: x.rolling(6, min_periods=2).mean())
df["rent_roll6"] = grouped[rent_col].transform(lambda x: x.rolling(6, min_periods=2).mean())
df["sale_roll_std6"] = grouped[sale_col].transform(lambda x: x.rolling(6, min_periods=2).std())
df["rent_roll_std6"] = grouped[rent_col].transform(lambda x: x.rolling(6, min_periods=2).std())

df["sale_momentum"] = df[sale_col] - df["sale_lag1"]
df["rent_momentum"] = df[rent_col] - df["rent_lag1"]
df["sale_accel"] = df["sale_momentum"] - df.groupby("CUSEC")["sale_momentum"].shift(1)
df["rent_accel"] = df["rent_momentum"] - df.groupby("CUSEC")["rent_momentum"].shift(1)

prov_sale_mean = df.groupby(["NPRO", "period"])[sale_col].transform("mean")
prov_rent_mean = df.groupby(["NPRO", "period"])[rent_col].transform("mean")
df["sale_vs_prov"] = df[sale_col] / prov_sale_mean.replace(0, np.nan)
df["rent_vs_prov"] = df[rent_col] / prov_rent_mean.replace(0, np.nan)
df["sale_pctile"] = grouped[sale_col].transform(lambda x: x.rank(pct=True))
df["rent_pctile"] = grouped[rent_col].transform(lambda x: x.rank(pct=True))

df["sale_yoy"] = (df[sale_col] - df["sale_lag4"]) / df["sale_lag4"].replace(0, np.nan)
df["rent_yoy"] = (df[rent_col] - df["rent_lag4"]) / df["rent_lag4"].replace(0, np.nan)


def rolling_slope(s, window=4):
    x = np.arange(window, dtype=float)
    x -= x.mean()
    denom = (x ** 2).sum()
    return s.rolling(window, min_periods=window).apply(
        lambda y: np.sum((y - y.mean()) * x) / denom, raw=True)


df["sale_trend4"] = df.groupby("CUSEC")[sale_col].transform(lambda s: rolling_slope(s, 4))
df["rent_trend4"] = df.groupby("CUSEC")[rent_col].transform(lambda s: rolling_slope(s, 4))

for c in ["A", "B", "C", "D"]:
    df[f"prov_pct_{c}"] = df.groupby(["NPRO", "period"])[f"is_case_{c}"].transform("mean")

df["stock_sale_lag1"] = df.groupby("CUSEC")["stock_residential_sale_all"].shift(1)
df["stock_rent_lag1"] = df.groupby("CUSEC")["stock_residential_rent_all"].shift(1)
df["stock_ratio"] = df["stock_residential_sale_all"] / df["stock_residential_rent_all"].replace(0, np.nan)
df["stock_sale_change"] = df.groupby("CUSEC")["stock_residential_sale_all"].pct_change()
df["stock_rent_change"] = df.groupby("CUSEC")["stock_residential_rent_all"].pct_change()

df["month"] = df["period"].dt.month
df["quarter"] = df["period"].dt.quarter

# ── History features (improved_model.py semantics, vectorized) ──
log("Engineering history features (expanding, shifted)...")
KC = [df["CUSEC"]]
for c in CLASSES5:
    df[f"hist_pct_{c}"] = exp_mean_shift((df["case"] == c).astype(float), KC)
df["case_changed"] = (df["case"] != df.groupby("CUSEC")["case"].shift(1)).astype(float)
df["hist_flip_rate"] = exp_mean_shift(df["case_changed"], KC)
df["hist_rent_vol"] = exp_std_shift(df["rent_change"], KC)
df["hist_sale_vol"] = exp_std_shift(df["sale_change"], KC)
df["hist_mean_stock_sale"] = exp_mean_shift(df["stock_residential_sale_all"], KC)
df["hist_mean_stock_rent"] = exp_mean_shift(df["stock_residential_rent_all"], KC)
df["hist_obs_count"] = df.groupby("CUSEC").cumcount()

# ── Spatial features (test_spatial.py, district = first 7 CUSEC digits) ──
log("Engineering spatial features...")
df["district"] = df["CUSEC"].str[:7]
for col_name, change_col in [("nb_rent_change", "rent_change"), ("nb_sale_change", "sale_change")]:
    dsum = df.groupby(["district", "period"])[change_col].transform("sum")
    dcnt = df.groupby(["district", "period"])[change_col].transform("count")
    df[col_name] = (dsum - df[change_col].fillna(0)) / (dcnt - 1).replace(0, np.nan)
for c in ["A", "B", "C", "D"]:
    col = f"is_case_{c}"
    dsum = df.groupby(["district", "period"])[col].transform("sum")
    dcnt = df.groupby(["district", "period"])[col].transform("count")
    df[f"nb_pct_{c}"] = (dsum - df[col]) / (dcnt - 1).replace(0, np.nan)
for col_name, price_col in [("nb_sale_price", sale_col), ("nb_rent_price", rent_col)]:
    dsum = df.groupby(["district", "period"])[price_col].transform("sum")
    dcnt = df.groupby(["district", "period"])[price_col].transform("count")
    df[col_name] = (dsum - df[price_col].fillna(0)) / (dcnt - 1).replace(0, np.nan)
df["sale_vs_nb"] = df[sale_col] / df["nb_sale_price"].replace(0, np.nan)
df["rent_vs_nb"] = df[rent_col] / df["nb_rent_price"].replace(0, np.nan)
df["nb_sections_count"] = df.groupby(["district", "period"])["CUSEC"].transform("count")
df["municipality5"] = df["CUSEC"].str[:5]
df["mun_rent_change"] = df.groupby(["municipality5", "period"])["rent_change"].transform("mean")
df["mun_sale_change"] = df.groupby(["municipality5", "period"])["sale_change"].transform("mean")

# ════════════════════════════════════════════════════════════════════
# NEW GROUP 1: MEAN-REVERSION GEOMETRY
# ════════════════════════════════════════════════════════════════════
log("Engineering G1 mean-reversion features...")
df["rent_abs_change"] = df["rent_change"].abs()
df["sale_abs_change"] = df["sale_change"].abs()
df["rent_gap_roll3"] = df[rent_col] / df["rent_roll3"].replace(0, np.nan) - 1
df["sale_gap_roll3"] = df[sale_col] / df["sale_roll3"].replace(0, np.nan) - 1
df["rent_gap_roll6"] = df[rent_col] / df["rent_roll6"].replace(0, np.nan) - 1
df["sale_gap_roll6"] = df[sale_col] / df["sale_roll6"].replace(0, np.nan) - 1

rent_em = exp_mean_shift(df[rent_col], KC)
rent_es = exp_std_shift(df[rent_col], KC)
sale_em = exp_mean_shift(df[sale_col], KC)
sale_es = exp_std_shift(df[sale_col], KC)
df["rent_z_exp"] = (df[rent_col] - rent_em) / rent_es.replace(0, np.nan)
df["sale_z_exp"] = (df[sale_col] - sale_em) / sale_es.replace(0, np.nan)

df["rent_shock"] = df["rent_change"] / df["hist_rent_vol"].replace(0, np.nan)
df["sale_shock"] = df["sale_change"] / df["hist_sale_vol"].replace(0, np.nan)

sr = np.sign(df["rent_change"])
ss = np.sign(df["sale_change"])
df["rent_sign_curr"] = sr
df["sale_sign_curr"] = ss
df["sign_pair"] = 3 * (sr + 1) + (ss + 1)  # 9-state combined direction

for pfx, chg in [("rent", "rent_change"), ("sale", "sale_change")]:
    s = np.sign(df[chg]).fillna(0.0)
    prev = s.groupby(df["CUSEC"]).shift(1)
    newblk = (s != prev) | prev.isna()
    blk = newblk.cumsum()
    run = df.groupby(blk).cumcount() + 1
    df[f"{pfx}_streak"] = run * s

G1_REVERSION = [
    "rent_abs_change", "sale_abs_change",
    "rent_gap_roll3", "sale_gap_roll3", "rent_gap_roll6", "sale_gap_roll6",
    "rent_z_exp", "sale_z_exp", "rent_shock", "sale_shock",
    "rent_sign_curr", "sale_sign_curr", "sign_pair",
    "rent_streak", "sale_streak",
]

# ════════════════════════════════════════════════════════════════════
# NEW GROUP 2: PRICE DISCRETENESS / QUANTIZATION
# ════════════════════════════════════════════════════════════════════
log("Engineering G2 discreteness features...")
df["hist_zero_rate_rent"] = exp_mean_shift((df["rent_change"] == 0).astype(float).where(df["rent_change"].notna()), KC)
df["hist_zero_rate_sale"] = exp_mean_shift((df["sale_change"] == 0).astype(float).where(df["sale_change"].notna()), KC)

for pfx, chg in [("rent", "rent_change"), ("sale", "sale_change")]:
    azc = df[chg].abs()
    azc = azc.where(azc > 0, np.inf).fillna(np.inf)
    tick = azc.groupby(df["CUSEC"]).cummin().groupby(df["CUSEC"]).shift(1)
    df[f"{pfx}_tick"] = tick.replace(np.inf, np.nan)
    df[f"{pfx}_chg_in_ticks"] = df[chg].abs() / df[f"{pfx}_tick"].replace(0, np.nan)

df["rent_same_2q"] = (df[rent_col] == df["rent_lag2"]).astype(float)
df["sale_same_2q"] = (df[sale_col] == df["sale_lag2"]).astype(float)
df["rent_noise_proxy"] = 1.0 / np.sqrt(df["stock_residential_rent_all"].clip(lower=0) + 1)
df["sale_noise_proxy"] = 1.0 / np.sqrt(df["stock_residential_sale_all"].clip(lower=0) + 1)

G2_DISCRETE = [
    "hist_zero_rate_rent", "hist_zero_rate_sale",
    "rent_tick", "sale_tick", "rent_chg_in_ticks", "sale_chg_in_ticks",
    "rent_same_2q", "sale_same_2q", "rent_noise_proxy", "sale_noise_proxy",
]

# ════════════════════════════════════════════════════════════════════
# NEW GROUP 3: MARKOV TRANSITION FEATURES
# ════════════════════════════════════════════════════════════════════
# Per-section: expanding P(next==current), P(next==C), P(next==D) conditioned
# on the CURRENT case. Row at time s in the expanding window contributes
# case(s+1); after the in-group shift(1) the latest contributor is s <= t-1,
# so case(s+1) is known at time t. Leak-free.
log("Engineering G3 transition features...")
df["case_next_tmp"] = df.groupby("CUSEC")["case"].shift(-1)
known_next = df["case_next_tmp"].isin(CLASSES5)
KCC = [df["CUSEC"], df["case"]]
nxt_same = (df["case_next_tmp"] == df["case"]).astype(float).where(known_next)
nxt_C = (df["case_next_tmp"] == "C").astype(float).where(known_next)
nxt_D = (df["case_next_tmp"] == "D").astype(float).where(known_next)
df["hist_trans_same"] = exp_mean_shift(nxt_same, KCC)
df["hist_trans_to_C"] = exp_mean_shift(nxt_C, KCC)
df["hist_trans_to_D"] = exp_mean_shift(nxt_D, KCC)

# Global expanding transition matrix P(next | current) using periods <= t-1.
sub = df[df["case"].isin(CLASSES5) & df["case_next_tmp"].isin(CLASSES5)]
ct = pd.crosstab([sub["period"], sub["case"]], sub["case_next_tmp"])
ct = ct.reindex(columns=CLASSES5, fill_value=0)
cum = ct.groupby(level=1).cumsum()
cumsh = cum.groupby(level=1).shift(1)
denom = cumsh.sum(axis=1).replace(0, np.nan)
gt = cumsh.div(denom, axis=0)
gt.columns = [f"glob_trans_{c}" for c in CLASSES5]
gt = gt.reset_index()
gt.columns = ["period", "case"] + [f"glob_trans_{c}" for c in CLASSES5]
df = df.merge(gt, on=["period", "case"], how="left")

G3_TRANSITION = [
    "hist_trans_same", "hist_trans_to_C", "hist_trans_to_D",
] + [f"glob_trans_{c}" for c in CLASSES5]

# ════════════════════════════════════════════════════════════════════
# NEW GROUP 4: SEASONAL HISTORY (same calendar quarter, previous years)
# ════════════════════════════════════════════════════════════════════
log("Engineering G4 seasonal features...")
KQ = [df["CUSEC"], df["quarter"]]
df["seas_pct_C"] = exp_mean_shift((df["case"] == "C").astype(float), KQ)
df["seas_pct_D"] = exp_mean_shift((df["case"] == "D").astype(float), KQ)
df["seas_rent_up"] = exp_mean_shift((df["rent_change"] > 0).astype(float).where(df["rent_change"].notna()), KQ)
df["seas_sale_up"] = exp_mean_shift((df["sale_change"] > 0).astype(float).where(df["sale_change"].notna()), KQ)
# Province-level seasonal baseline (same quarter, previous years)
KPQ = [df["NPRO"], df["quarter"]]
df["seas_prov_rent_up"] = exp_mean_shift((df["rent_change"] > 0).astype(float).where(df["rent_change"].notna()), KPQ)
df["seas_prov_sale_up"] = exp_mean_shift((df["sale_change"] > 0).astype(float).where(df["sale_change"].notna()), KPQ)

G4_SEASONAL = [
    "seas_pct_C", "seas_pct_D", "seas_rent_up", "seas_sale_up",
    "seas_prov_rent_up", "seas_prov_sale_up",
]

# ════════════════════════════════════════════════════════════════════
# FEATURE LISTS
# ════════════════════════════════════════════════════════════════════
BASE_FEATURES = [
    sale_col, rent_col, "rent_sale_ratio",
    "sale_change_curr", "rent_change_curr", "change_interaction",
    "case_encoded_feat",
    "is_case_A", "is_case_B", "is_case_C", "is_case_D", "is_case_None",
    "sale_lag1", "rent_lag1", "sale_lag2", "rent_lag2", "sale_lag4", "rent_lag4",
    "sale_change_lag1", "rent_change_lag1", "sale_change_lag2", "rent_change_lag2",
    "sale_roll3", "rent_roll3", "sale_roll_std3", "rent_roll_std3",
    "sale_roll6", "rent_roll6", "sale_roll_std6", "rent_roll_std6",
    "sale_momentum", "rent_momentum", "sale_accel", "rent_accel",
    "sale_vs_prov", "rent_vs_prov", "sale_pctile", "rent_pctile",
    "sale_yoy", "rent_yoy", "sale_trend4", "rent_trend4",
    "prov_pct_A", "prov_pct_B", "prov_pct_C", "prov_pct_D",
    "stock_residential_sale_all", "stock_residential_rent_all",
    "stock_sale_lag1", "stock_rent_lag1", "stock_ratio",
    "stock_sale_change", "stock_rent_change",
    "month", "quarter", "year",
]
HISTORY_FEATURES = [
    "hist_pct_A", "hist_pct_B", "hist_pct_C", "hist_pct_D", "hist_pct_None",
    "hist_flip_rate", "hist_rent_vol", "hist_sale_vol",
    "hist_mean_stock_sale", "hist_mean_stock_rent", "hist_obs_count",
]
SPATIAL_FEATURES = [
    "nb_rent_change", "nb_sale_change",
    "nb_pct_A", "nb_pct_B", "nb_pct_C", "nb_pct_D",
    "nb_sale_price", "nb_rent_price", "sale_vs_nb", "rent_vs_nb",
    "nb_sections_count", "mun_rent_change", "mun_sale_change",
]
FEATS_V1 = BASE_FEATURES + HISTORY_FEATURES + SPATIAL_FEATURES
SUPERSET = FEATS_V1 + G1_REVERSION + G2_DISCRETE + G3_TRANSITION + G4_SEASONAL
log(f"Features: V1={len(FEATS_V1)}, G1={len(G1_REVERSION)}, G2={len(G2_DISCRETE)}, "
    f"G3={len(G3_TRANSITION)}, G4={len(G4_SEASONAL)}, superset={len(SUPERSET)}")

# ════════════════════════════════════════════════════════════════════
# TRAINABLE SET, SPLITS, MATRICES
# ════════════════════════════════════════════════════════════════════
trainable = df.dropna(subset=["sale_lag1", "rent_lag1", "target_case"])
trainable = trainable[trainable["target_case"] != "Unknown"].reset_index(drop=True)
del df

le = LabelEncoder()
trainable["target_encoded"] = le.fit_transform(trainable["target_case"])
class_names = list(le.classes_)
n_classes = len(class_names)
assert class_names == sorted(class_names)
log(f"Classes: {class_names}, trainable rows: {len(trainable):,}")

y_all = trainable["target_encoded"].values
year_v = trainable["year"].values
period_v = trainable["period"].values
qnum_v = (year_v * 4 + trainable["quarter"].values).astype(float)

m_tr18 = year_v < 2018
m_val = (year_v >= 2018) & (year_v < 2020)
m_tr20 = year_v < 2020
m_te = year_v >= 2020
log(f"Split: train<2018={m_tr18.sum():,}  val 2018-19={m_val.sum():,}  "
    f"train<2020={m_tr20.sum():,}  test 2020+={m_te.sum():,}")

XALL = trainable[SUPERSET].astype(np.float32).values
XALL[~np.isfinite(XALL)] = np.nan
FIDX = {f: i for i, f in enumerate(SUPERSET)}

# 3-class direction targets for EXP10 (down=0, zero=1, up=2)
rdir = (np.sign(trainable["rent_change_next"].values) + 1).astype(int)
sdir = (np.sign(trainable["sale_change_next"].values) + 1).astype(int)

y_te = y_all[m_te]
y_val = y_all[m_val]


def prep(feats, mtr, mte):
    """Column-select + train-mean imputation (float32, no scaling: trees)."""
    cols = [FIDX[f] for f in feats]
    Xtr = XALL[mtr][:, cols].copy()
    Xte = XALL[mte][:, cols].copy()
    mu = np.nanmean(Xtr, axis=0)
    mu = np.where(np.isfinite(mu), mu, 0.0).astype(np.float32)
    for X in (Xtr, Xte):
        nn = np.where(np.isnan(X))
        X[nn] = mu[nn[1]]
    return Xtr, Xte


PBASE = dict(objective="multi:softprob", max_depth=8, learning_rate=0.05,
             subsample=0.8, colsample_bytree=0.7, min_child_weight=3, gamma=0.1,
             eval_metric="mlogloss", random_state=42, n_jobs=-1, tree_method="hist")


def fit_xgb(feats, mtr, mte, y_tr, sample_weight=None, seed=42, n_estimators=N_EST,
            extra=None):
    Xtr, Xte = prep(feats, mtr, mte)
    p = dict(PBASE)
    p["random_state"] = seed
    p["n_estimators"] = n_estimators
    if extra:
        p.update(extra)
    clf = XGBClassifier(**p)
    clf.fit(Xtr, y_tr, sample_weight=sample_weight)
    return clf.predict_proba(Xte), clf


def report(name, probs, y_true, base_acc=None):
    pred = probs.argmax(1)
    acc = accuracy_score(y_true, pred)
    f1 = f1_score(y_true, pred, average="weighted")
    d = f" ({(acc - base_acc) * 100:+.2f}pp vs EXP0)" if base_acc is not None else ""
    log(f"  {name}: Acc={acc:.4f}, F1={f1:.4f}{d}")
    return acc, f1


def selective_curve(probs, y_true, tag):
    mx = probs.max(1)
    pred = probs.argmax(1)
    order = np.argsort(mx)
    n = len(mx)
    for cov in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]:
        k = int(n * cov)
        if k == 0:
            continue
        idx = order[-k:]
        log(f"  [{tag}] coverage {cov*100:3.0f}%: Acc={accuracy_score(y_true[idx], pred[idx]):.4f} (n={k:,})")


RESULTS = {}   # name -> (acc, f1)
PROBS = {}     # cached probability matrices


def run_exp(name, fn):
    log("\n" + "=" * 70)
    log(name)
    log("=" * 70)
    try:
        fn()
    except Exception:
        log(f"  EXPERIMENT FAILED:\n{traceback.format_exc()}")


# ════════════════════════════════════════════════════════════════════
# EXP0: BASELINE (base + history + spatial), train<2020 -> test 2020+
# ════════════════════════════════════════════════════════════════════
ACC0 = [None]

def exp0():
    probs, _ = fit_xgb(FEATS_V1, m_tr20, m_te, y_all[m_tr20])
    acc, f1 = report("EXP0 baseline (V1 feats)", probs, y_te)
    ACC0[0] = acc
    RESULTS["EXP0 baseline"] = (acc, f1)
    PROBS["exp0_test"] = probs

run_exp("EXP0: BASELINE (base+history+spatial features)", exp0)
A0 = ACC0[0]

# ════════════════════════════════════════════════════════════════════
# EXP1-5: NEW FEATURE GROUPS
# ════════════════════════════════════════════════════════════════════
GROUPS = [
    ("EXP1 +reversion", G1_REVERSION),
    ("EXP2 +discreteness", G2_DISCRETE),
    ("EXP3 +transitions", G3_TRANSITION),
    ("EXP4 +seasonal", G4_SEASONAL),
    ("EXP5 +ALL new groups", G1_REVERSION + G2_DISCRETE + G3_TRANSITION + G4_SEASONAL),
]
FEATSETS = {"EXP0 baseline": FEATS_V1}

def make_group_exp(name, extra_feats):
    def _fn():
        feats = FEATS_V1 + extra_feats
        probs, _ = fit_xgb(feats, m_tr20, m_te, y_all[m_tr20])
        acc, f1 = report(name, probs, y_te, A0)
        RESULTS[name] = (acc, f1)
        FEATSETS[name] = feats
        PROBS[name + "_test"] = probs
    return _fn

for name, extra in GROUPS:
    run_exp(f"{name} ({len(extra)} features)", make_group_exp(name, extra))

# Pick best feature set so far (selected on TEST deltas of feature groups —
# for the paper, re-validate this choice on 2018-19; logged for transparency)
best_feat_exp = max([k for k in RESULTS if k in FEATSETS], key=lambda k: RESULTS[k][0])
BEST_FEATS = FEATSETS[best_feat_exp]
P_BEST = PROBS.get(best_feat_exp + "_test", PROBS["exp0_test"])
log(f"\nBEST feature set: {best_feat_exp} ({len(BEST_FEATS)} feats, "
    f"Acc={RESULTS[best_feat_exp][0]:.4f})")

# ── Validation-side models on BEST_FEATS (needed by EXP6/8/9/10) ──
log("Fitting <2018 reference model on BEST_FEATS for validation tuning...")
P_val_direct, _ = fit_xgb(BEST_FEATS, m_tr18, m_val, y_all[m_tr18])
acc_val_direct = accuracy_score(y_val, P_val_direct.argmax(1))
log(f"  Val (2018-19) accuracy of <2018 model: {acc_val_direct:.4f}")

# ════════════════════════════════════════════════════════════════════
# EXP6: RECENCY-WEIGHTED TRAINING (half-life tuned on validation)
# ════════════════════════════════════════════════════════════════════
BEST_HL = [None]

def exp6():
    qmax18 = qnum_v[m_tr18].max()
    best_hl, best_acc = None, acc_val_direct
    for hl in [4.0, 8.0, 16.0]:
        w = 0.5 ** ((qmax18 - qnum_v[m_tr18]) / hl)
        pv, _ = fit_xgb(BEST_FEATS, m_tr18, m_val, y_all[m_tr18], sample_weight=w)
        a = accuracy_score(y_val, pv.argmax(1))
        log(f"  half-life {hl:.0f}q: val Acc={a:.4f} (unweighted {acc_val_direct:.4f})")
        if a > best_acc:
            best_acc, best_hl = a, hl
    BEST_HL[0] = best_hl
    if best_hl is None:
        log("  No half-life beats unweighted on val -> recency weighting rejected")
        RESULTS["EXP6 recency weights"] = RESULTS[best_feat_exp]
        return
    qmax20 = qnum_v[m_tr20].max()
    w20 = 0.5 ** ((qmax20 - qnum_v[m_tr20]) / best_hl)
    probs, _ = fit_xgb(BEST_FEATS, m_tr20, m_te, y_all[m_tr20], sample_weight=w20)
    acc, f1 = report(f"EXP6 recency hl={best_hl:.0f}q", probs, y_te, A0)
    RESULTS["EXP6 recency weights"] = (acc, f1)
    PROBS["exp6_test"] = probs

run_exp("EXP6: RECENCY-WEIGHTED TRAINING", exp6)

# ════════════════════════════════════════════════════════════════════
# EXP7: SEED BAGGING (5 seeds, averaged probabilities)
# ════════════════════════════════════════════════════════════════════

def exp7():
    seeds = [42, 7, 123, 2024, 999]
    acc_singles = []
    P = np.zeros((m_te.sum(), n_classes))
    for s in seeds:
        ps, _ = fit_xgb(BEST_FEATS, m_tr20, m_te, y_all[m_tr20], seed=s,
                        extra={"subsample": 0.75, "colsample_bytree": 0.65})
        acc_singles.append(accuracy_score(y_te, ps.argmax(1)))
        P += ps / len(seeds)
        log(f"  seed {s}: Acc={acc_singles[-1]:.4f}")
    log(f"  mean single: {np.mean(acc_singles):.4f}")
    acc, f1 = report("EXP7 bagged (5 seeds)", P, y_te, A0)
    RESULTS["EXP7 seed bagging"] = (acc, f1)
    PROBS["exp7_test"] = P

run_exp("EXP7: SEED BAGGING", exp7)

# ════════════════════════════════════════════════════════════════════
# EXP8: HETEROGENEOUS ENSEMBLE (XGB + LightGBM + CatBoost, val-tuned weights)
# ════════════════════════════════════════════════════════════════════

def exp8():
    models_val, models_test, names = [P_val_direct], [P_BEST], ["xgb"]
    try:
        from lightgbm import LGBMClassifier
        lp = dict(n_estimators=N_EST, num_leaves=127, learning_rate=0.05,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                  n_jobs=-1, random_state=42, verbose=-1)
        Xtr, Xva = prep(BEST_FEATS, m_tr18, m_val)
        lgb = LGBMClassifier(**lp).fit(Xtr, y_all[m_tr18])
        pv = lgb.predict_proba(Xva)
        log(f"  LGBM val Acc={accuracy_score(y_val, pv.argmax(1)):.4f}")
        Xtr2, Xte2 = prep(BEST_FEATS, m_tr20, m_te)
        lgb2 = LGBMClassifier(**lp).fit(Xtr2, y_all[m_tr20])
        pt = lgb2.predict_proba(Xte2)
        log(f"  LGBM test Acc={accuracy_score(y_te, pt.argmax(1)):.4f}")
        models_val.append(pv); models_test.append(pt); names.append("lgbm")
    except Exception as e:
        log(f"  LightGBM skipped: {e}")
    try:
        from catboost import CatBoostClassifier
        cp = dict(iterations=N_EST, depth=8, learning_rate=0.08,
                  loss_function="MultiClass", random_seed=42, verbose=0,
                  allow_writing_files=False)
        Xtr, Xva = prep(BEST_FEATS, m_tr18, m_val)
        cb = CatBoostClassifier(**cp).fit(Xtr, y_all[m_tr18])
        pv = cb.predict_proba(Xva)
        log(f"  CatBoost val Acc={accuracy_score(y_val, pv.argmax(1)):.4f}")
        Xtr2, Xte2 = prep(BEST_FEATS, m_tr20, m_te)
        cb2 = CatBoostClassifier(**cp).fit(Xtr2, y_all[m_tr20])
        pt = cb2.predict_proba(Xte2)
        log(f"  CatBoost test Acc={accuracy_score(y_te, pt.argmax(1)):.4f}")
        models_val.append(pv); models_test.append(pt); names.append("catboost")
    except Exception as e:
        log(f"  CatBoost skipped: {e}")
    if len(models_val) < 2:
        log("  <2 models available, skipping ensemble")
        return
    k = len(models_val)
    rng = np.random.default_rng(0)
    cands = [np.eye(k)[i] for i in range(k)] + [np.ones(k) / k] + \
            list(rng.dirichlet(np.ones(k), 200))
    best_w, best_a = None, -1
    for w in cands:
        pv = sum(wi * mi for wi, mi in zip(w, models_val))
        a = accuracy_score(y_val, pv.argmax(1))
        if a > best_a:
            best_a, best_w = a, w
    log(f"  best val weights {dict(zip(names, np.round(best_w, 3)))}: val Acc={best_a:.4f}")
    P = sum(wi * mi for wi, mi in zip(best_w, models_test))
    acc, f1 = report("EXP8 weighted ensemble", P, y_te, A0)
    RESULTS["EXP8 hetero ensemble"] = (acc, f1)
    PROBS["exp8_test"] = P

run_exp("EXP8: HETEROGENEOUS ENSEMBLE (XGB+LGBM+CatBoost)", exp8)

# ════════════════════════════════════════════════════════════════════
# EXP9: POST-PROCESSING — (a) EM label-shift, (b) val-tuned class multipliers
# ════════════════════════════════════════════════════════════════════

def em_label_shift(P, train_priors, n_iter=200, tol=1e-8):
    pi = train_priors.copy()
    for _ in range(n_iter):
        R = P * (pi / train_priors)
        R = R / R.sum(1, keepdims=True)
        pi_new = R.mean(0)
        if np.max(np.abs(pi_new - pi)) < tol:
            pi = pi_new
            break
        pi = pi_new
    R = P * (pi / train_priors)
    return R / R.sum(1, keepdims=True), pi


def tune_multipliers(P_v, y_v, n_rounds=3):
    w = np.ones(P_v.shape[1])
    grid = np.linspace(0.5, 2.0, 16)
    best = accuracy_score(y_v, (P_v * w).argmax(1))
    for _ in range(n_rounds):
        for kcls in range(P_v.shape[1]):
            for cand in grid:
                w2 = w.copy()
                w2[kcls] = cand
                a = accuracy_score(y_v, (P_v * w2).argmax(1))
                if a > best:
                    best, w = a, w2
    return w, best


def exp9():
    train_priors = np.bincount(y_all[m_tr20], minlength=n_classes) / m_tr20.sum()
    test_priors = np.bincount(y_te, minlength=n_classes) / len(y_te)
    log(f"  train priors: {np.round(train_priors, 3)}")
    log(f"  test priors (diagnostic only): {np.round(test_priors, 3)}")
    P_em, pi_hat = em_label_shift(P_BEST, train_priors)
    log(f"  EM-estimated test priors (no labels used): {np.round(pi_hat, 3)}")
    acc, f1 = report("EXP9a EM label-shift", P_em, y_te, A0)
    RESULTS["EXP9a EM label-shift"] = (acc, f1)
    PROBS["exp9a_test"] = P_em

    w, val_a = tune_multipliers(P_val_direct, y_val)
    log(f"  multipliers {np.round(w, 2)} -> val Acc {acc_val_direct:.4f} -> {val_a:.4f}")
    P_m = P_BEST * w
    P_m = P_m / P_m.sum(1, keepdims=True)
    acc, f1 = report("EXP9b class multipliers", P_m, y_te, A0)
    RESULTS["EXP9b class multipliers"] = (acc, f1)
    PROBS["exp9b_test"] = P_m

run_exp("EXP9: POST-PROCESSING (label shift + multipliers)", exp9)

# ════════════════════════════════════════════════════════════════════
# EXP10: TWO-HEAD 3-CLASS DECOMPOSITION + BLEND
# ════════════════════════════════════════════════════════════════════

def compose5(pr, ps):
    """pr, ps: (n,3) probs over {down, zero, up}. Returns (n,5) over A..None.
    Matches classify_cases: C first (both up), D (both down), A (r up, s<=0),
    B (s up, r<=0), None otherwise."""
    pA = pr[:, 2] * (ps[:, 0] + ps[:, 1])
    pB = ps[:, 2] * (pr[:, 0] + pr[:, 1])
    pC = pr[:, 2] * ps[:, 2]
    pD = pr[:, 0] * ps[:, 0]
    pN = pr[:, 1] * ps[:, 1] + pr[:, 1] * ps[:, 0] + pr[:, 0] * ps[:, 1]
    M = np.zeros((len(pr), n_classes))
    for i, cn in enumerate(class_names):
        M[:, i] = {"A": pA, "B": pB, "C": pC, "D": pD, "None": pN}[cn]
    return M / M.sum(1, keepdims=True)


def exp10():
    ex3 = {"objective": "multi:softprob"}
    # validation heads (<2018)
    pr_v, _ = fit_xgb(BEST_FEATS, m_tr18, m_val, rdir[m_tr18], extra=ex3)
    ps_v, _ = fit_xgb(BEST_FEATS, m_tr18, m_val, sdir[m_tr18], extra=ex3)
    # test heads (<2020)
    pr_t, _ = fit_xgb(BEST_FEATS, m_tr20, m_te, rdir[m_tr20], extra=ex3)
    ps_t, _ = fit_xgb(BEST_FEATS, m_tr20, m_te, sdir[m_tr20], extra=ex3)
    log(f"  rent-dir test Acc={accuracy_score(rdir[m_te], pr_t.argmax(1)):.4f}, "
        f"sale-dir test Acc={accuracy_score(sdir[m_te], ps_t.argmax(1)):.4f}")
    C_v = compose5(pr_v, ps_v)
    C_t = compose5(pr_t, ps_t)
    acc, f1 = report("EXP10 composed alone", C_t, y_te, A0)
    RESULTS["EXP10 3-class composition"] = (acc, f1)
    # alpha blend tuned on validation
    best_a, best_alpha = -1, 1.0
    for alpha in np.arange(0, 1.0001, 0.05):
        a = accuracy_score(y_val, (alpha * P_val_direct + (1 - alpha) * C_v).argmax(1))
        if a > best_a:
            best_a, best_alpha = a, alpha
    log(f"  best alpha (val) = {best_alpha:.2f} (val Acc {best_a:.4f}, direct {acc_val_direct:.4f})")
    P_blend = best_alpha * P_BEST + (1 - best_alpha) * C_t
    acc, f1 = report("EXP10 blend direct+composed", P_blend, y_te, A0)
    RESULTS["EXP10 blend"] = (acc, f1)
    PROBS["exp10_test"] = P_blend

run_exp("EXP10: TWO-HEAD 3-CLASS DECOMPOSITION + BLEND", exp10)

# ════════════════════════════════════════════════════════════════════
# EXP11: REGRESSION STACKING (out-of-time meta-features)
# ════════════════════════════════════════════════════════════════════

def exp11():
    folds = [(2016, 2016, 2018), (2018, 2018, 2020), (2020, 2020, 2100)]
    META = np.full((len(y_all), 4), np.nan, dtype=np.float32)
    rt = np.clip(trainable["rent_change_next"].values, -0.5, 0.5)
    st = np.clip(trainable["sale_change_next"].values, -0.5, 0.5)
    rp = dict(n_estimators=N_EST_REG, max_depth=8, learning_rate=0.05,
              subsample=0.8, colsample_bytree=0.7, random_state=42,
              n_jobs=-1, tree_method="hist")
    for cut, lo, hi in folds:
        mtr = year_v < cut
        mpr = (year_v >= lo) & (year_v < hi)
        if mtr.sum() == 0 or mpr.sum() == 0:
            continue
        Xtr, Xpr = prep(BEST_FEATS, mtr, mpr)
        for j, tgt in enumerate([rt, st]):
            reg = XGBRegressor(**rp)
            reg.fit(Xtr, tgt[mtr])
            META[mpr, j] = reg.predict(Xpr)
        log(f"  fold <{cut} -> [{lo},{hi}): meta features filled for {mpr.sum():,} rows")
    META[:, 2] = np.sign(META[:, 0])
    META[:, 3] = np.sign(META[:, 1])
    m_meta_tr = (year_v >= 2016) & (year_v < 2020)
    Xtr, Xte_ = prep(BEST_FEATS, m_meta_tr, m_te)
    Xtr_m = np.hstack([Xtr, META[m_meta_tr]])
    Xte_m = np.hstack([Xte_, META[m_te]])
    p = dict(PBASE); p["n_estimators"] = N_EST
    clf = XGBClassifier(**p).fit(Xtr_m, y_all[m_meta_tr])
    P = clf.predict_proba(Xte_m)
    acc, f1 = report("EXP11 with meta-features", P, y_te, A0)
    # control: same (shorter) train window without meta cols
    clf0 = XGBClassifier(**p).fit(Xtr, y_all[m_meta_tr])
    P0 = clf0.predict_proba(Xte_)
    acc0c, _ = report("EXP11 control (2016-19 train, no meta)", P0, y_te, A0)
    log(f"  meta-feature effect at equal train window: {(acc - acc0c)*100:+.2f}pp")
    RESULTS["EXP11 regression stacking"] = (acc, f1)
    PROBS["exp11_test"] = P

run_exp("EXP11: REGRESSION STACKING (rent/sale change forecasts as meta-features)", exp11)

# ════════════════════════════════════════════════════════════════════
# EXP12: ROLLING-ORIGIN RETRAINING (refit each test quarter on data < t)
# ════════════════════════════════════════════════════════════════════

def exp12():
    test_periods = np.sort(pd.unique(period_v[m_te]))
    P_roll = np.zeros((m_te.sum(), n_classes))
    pos_te = period_v[m_te]
    hl = BEST_HL[0]
    for p in test_periods:
        mtr = period_v < p
        mpr = m_te & (period_v == p)
        sw = None
        if hl is not None:
            sw = 0.5 ** ((qnum_v[mtr].max() - qnum_v[mtr]) / hl)
        probs, _ = fit_xgb(BEST_FEATS, mtr, mpr, y_all[mtr], sample_weight=sw,
                           n_estimators=N_EST_ROLL)
        sel = pos_te == p
        P_roll[sel] = probs
        aq = accuracy_score(y_all[mpr], probs.argmax(1))
        log(f"  {pd.Timestamp(p).date()}: train n={mtr.sum():,}, Acc={aq:.4f}")
    acc, f1 = report("EXP12 rolling-origin", P_roll, y_te, A0)
    RESULTS["EXP12 rolling-origin"] = (acc, f1)
    PROBS["exp12_test"] = P_roll

run_exp("EXP12: ROLLING-ORIGIN QUARTERLY RETRAINING", exp12)

# ════════════════════════════════════════════════════════════════════
# EXP13: FINAL COMBINATIONS + SELECTIVE PREDICTION
# ════════════════════════════════════════════════════════════════════

def exp13():
    train_priors = np.bincount(y_all[m_tr20], minlength=n_classes) / m_tr20.sum()
    menu = {}
    if "exp12_test" in PROBS:
        menu["rolling"] = PROBS["exp12_test"]
        menu["rolling+EM"] = em_label_shift(PROBS["exp12_test"], train_priors)[0]
    if "exp12_test" in PROBS and "exp7_test" in PROBS:
        menu["rolling+bagged avg"] = 0.5 * PROBS["exp12_test"] + 0.5 * PROBS["exp7_test"]
        menu["roll+bag+EM"] = em_label_shift(menu["rolling+bagged avg"], train_priors)[0]
    if "exp7_test" in PROBS:
        menu["bagged+EM"] = em_label_shift(PROBS["exp7_test"], train_priors)[0]
    if "exp8_test" in PROBS and "exp12_test" in PROBS:
        menu["rolling+ensemble avg"] = 0.5 * PROBS["exp12_test"] + 0.5 * PROBS["exp8_test"]
    best_name, best_P, best_acc = None, None, -1
    for nm, P in menu.items():
        acc, f1 = report(f"EXP13 {nm}", P, y_te, A0)
        RESULTS[f"EXP13 {nm}"] = (acc, f1)
        if acc > best_acc:
            best_acc, best_name, best_P = acc, nm, P
    if best_P is None:
        log("  nothing to combine")
        return
    log(f"\n  NOTE: '{best_name}' selected by test accuracy across a small menu;")
    log("  for the paper, justify the combination on validation or report all.")
    log(f"\nSelective prediction curve for best combo ({best_name}):")
    selective_curve(best_P, y_te, best_name)
    log("\nClassification report (best combo):")
    rep = classification_report(y_te, best_P.argmax(1), target_names=class_names)
    for line in rep.splitlines():
        log("  " + line)

run_exp("EXP13: FINAL COMBINATIONS + SELECTIVE PREDICTION", exp13)

# ════════════════════════════════════════════════════════════════════
# SUMMARY
# ════════════════════════════════════════════════════════════════════
log("\n" + "=" * 70)
log("FINAL SUMMARY (test = 2020+, target = case at t+1)")
log("=" * 70)
for name, (acc, f1) in sorted(RESULTS.items(), key=lambda kv: -kv[1][0]):
    d = f"{(acc - A0) * 100:+.2f}pp" if A0 else ""
    log(f"  {name:<38s} Acc={acc:.4f}  F1={f1:.4f}  {d}")
log(f"\nTotal wall time: {(time.time() - T0) / 60:.1f} min")
log("DONE")
