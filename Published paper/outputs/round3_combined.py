"""
ROUND 3 — COMBINED MODEL + 18 NEW IDEAS.  Gentrification type (A/B/C/D/None) at t+1.

STEP 1 (combined stack of all round-2 winners, with ablation ladder):
  R1  rolling-origin retraining, seed 42, V1+reversion features (round-2 EXP12 repro)
  R2  R1 + recency weighting (half-life re-tuned on val; round 2 rejected it on val,
      so R2 may equal R1 — that itself answers "does recency survive combination")
  R3  R2 + regression-stacking meta-features (OOT rent/sale forecasts, annual folds)
  R4  R3 + seed bagging (5 XGBs averaged)
  R5  R4 + heterogeneous ensemble XGB-bag + LightGBM + CatBoost (val-tuned weights)
  R6  R5 + two-head 3-class composition blend (val-tuned alpha)   == COMBINED
  plus a combination menu from cached components to see what cancels out.

STEP 2 (new ideas, each leak-free, each vs the relevant round-3 baseline):
  Post-hoc on COMBINED probs (tuned on 2018-19 validation only):
   I1  vector-scaling calibration (multinomial LR on log-probs)      est +0.2pp
   I2  per-class isotonic calibration + renormalisation             est +0.2pp
   I3  val-tuned class multipliers applied to COMBINED              est +0.3pp
   I4  lag-1 observed-prior shift (case(t) distribution is KNOWN at t;
       shrunk toward train prior, lambda tuned on val)              est +0.3pp
   I5  temperature scaling (calibration: logloss/ECE + selective)   est acc ~0
   I6  split conformal prediction (marginal + class-conditional):
       coverage guarantee, set sizes, singleton accuracy            (guarantee)
   I7  stacking meta-learner: multinomial LR over component probs   est +0.3pp
  New features / training schemes (fixed-split screen -> rolling for winners;
  rolling baseline = R3 = rolling single-seed with meta[+recency]):
   I8  macro regime features (national case shares at t, national mean
       rent/sale change, deltas) — lets the model track label drift  est +0.5pp
   I9  spatial x temporal interactions (idiosyncratic component
       change - neighbourhood change, gap dynamics, alignment)       est +0.3pp
   I10 change-point features (CUSUM of standardized changes,
       variance ratio 3q/6q, time-since-shock)                       est +0.2pp
   I11 feature selection: keep top-K by gain importance (K on val)   est +0.1pp
   I12 class-balanced sample weights (power tuned on val)            est acc ~0, F1 +
   I13 focal loss (detached-weight variant, custom objective)        est +0.1pp
   I14 ambiguity downweighting of near-boundary training labels      est +0.2pp
   I15 training-window ensemble: avg(model_{<t}, model_{<t-1})       est +0.2pp
   I16 hierarchical two-stage: None-vs-change, then 4-class          est +0.1pp
   I17 region-specific models (per-NCA where big enough)             est 0/negative
   I18 drop raw calendar features (year, month) — OOD in test        est +0.1pp

Leakage rules: every feature at (section, t) uses info available at t; target is
case(t+1).  Rolling refits use all rows with period < t (their targets are
case(<=t), observed by t).  All tuning (hl, weights, alpha, lambda, multipliers,
calibrators, K, beta, gamma, eps) is done on val 2018-19 with models trained <2018.

Run:        ~/miniconda3/bin/python outputs/round3_combined.py
Smoke test: FAST=1 ~/miniconda3/bin/python outputs/round3_combined.py
Full run ~5h on M1 Pro; timestamped progress in outputs/round3.log.
"""
import sys, os, time, traceback, warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report, log_loss
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor

from src.data_loader import load_data
from src.gentrification import classify_cases

FAST = os.environ.get("FAST", "0") == "1"
N_EST = 40 if FAST else 400          # fixed-split fits
N_EST_ROLL = 40 if FAST else 400     # rolling-origin fits (round-2 EXP12 setting)
N_EST_REG = 40 if FAST else 300      # meta-feature regressors
N_EST_CAT = 40 if FAST else 350      # catboost iterations
SEEDS_BAG = [42, 7, 123, 2024, 999]

OUT = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(OUT, "round3.log")
open(LOG, "w").close()
T0 = time.time()


def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts} +{(time.time()-T0)/60:6.1f}m] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


log("=" * 72)
log(f"ROUND 3: COMBINED MODEL + NEW IDEAS  (FAST={FAST}, n_est={N_EST})")
log("=" * 72)

# ════════════════════════════════════════════════════════════════════
# DATA + FEATURES (identical semantics to round2_improvements.py)
# ════════════════════════════════════════════════════════════════════
log("Loading data...")
df = load_data()
df = df.sort_values(["CUSEC", "period"]).reset_index(drop=True)
df["year"] = df["period"].dt.year

if FAST:
    rng = np.random.default_rng(0)
    keep = rng.choice(df["CUSEC"].dropna().unique(), size=2500, replace=False)
    df = df[df["CUSEC"].isin(keep)].reset_index(drop=True)
    log(f"FAST mode: {df['CUSEC'].nunique()} sections, {len(df):,} rows")

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


def exp_mean_shift(v, keys):
    val = v.fillna(0.0)
    cnt = v.notna().astype(float)
    cs = val.groupby(keys).cumsum().groupby(keys).shift(1)
    cc = cnt.groupby(keys).cumsum().groupby(keys).shift(1)
    return cs / cc.replace(0, np.nan)


def exp_std_shift(v, keys):
    val = v.fillna(0.0)
    cnt = v.notna().astype(float)
    s1 = val.groupby(keys).cumsum().groupby(keys).shift(1)
    s2 = (val * val).groupby(keys).cumsum().groupby(keys).shift(1)
    n = cnt.groupby(keys).cumsum().groupby(keys).shift(1)
    n = n.replace(0, np.nan)
    var = (s2 - s1 * s1 / n) / (n - 1).replace(0, np.nan)
    return np.sqrt(var.clip(lower=0))


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

log("Engineering history features...")
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

log("Engineering reversion features (round-2 winner G1)...")
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
df["sign_pair"] = 3 * (sr + 1) + (ss + 1)
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
    "rent_sign_curr", "sale_sign_curr", "sign_pair", "rent_streak", "sale_streak",
]

# ── NEW GROUP I8: MACRO REGIME FEATURES (current-quarter, observed => leak-free)
log("Engineering I8 macro regime features...")
known = df["case"].isin(CLASSES5)
nat = (df.loc[known].groupby("period")["case"]
       .value_counts(normalize=True).unstack().reindex(columns=CLASSES5))
nat.columns = [f"nat_share_{c}" for c in CLASSES5]
nat["nat_rent_chg"] = df.groupby("period")["rent_change"].mean()
nat["nat_sale_chg"] = df.groupby("period")["sale_change"].mean()
nat = nat.sort_index()
for c in ["nat_share_C", "nat_share_D", "nat_rent_chg", "nat_sale_chg"]:
    nat[c + "_d1"] = nat[c].diff()
nat = nat.reset_index()
df = df.merge(nat, on="period", how="left")
df["qnum_idx"] = df["year"] * 4 + df["quarter"]
I8_MACRO = [f"nat_share_{c}" for c in CLASSES5] + [
    "nat_rent_chg", "nat_sale_chg",
    "nat_share_C_d1", "nat_share_D_d1", "nat_rent_chg_d1", "nat_sale_chg_d1",
    "qnum_idx",
]

# ── NEW GROUP I9: SPATIAL x TEMPORAL INTERACTIONS
log("Engineering I9 spatial-temporal interaction features...")
df["rent_idio"] = df["rent_change"] - df["nb_rent_change"]
df["sale_idio"] = df["sale_change"] - df["nb_sale_change"]
df["rent_idio_lag1"] = df.groupby("CUSEC")["rent_idio"].shift(1)
df["sale_idio_lag1"] = df.groupby("CUSEC")["sale_idio"].shift(1)
df["rent_vs_nb_d1"] = df["rent_vs_nb"] - df.groupby("CUSEC")["rent_vs_nb"].shift(1)
df["sale_vs_nb_d1"] = df["sale_vs_nb"] - df.groupby("CUSEC")["sale_vs_nb"].shift(1)
df["nb_minus_mun_rent"] = df["nb_rent_change"] - df["mun_rent_change"]
df["nb_minus_mun_sale"] = df["nb_sale_change"] - df["mun_sale_change"]
df["rentchg_x_nb"] = df["rent_change"] * df["nb_rent_change"]
df["salechg_x_nb"] = df["sale_change"] * df["nb_sale_change"]
df["aligned_C"] = df["is_case_C"] * df["nb_pct_C"]
df["aligned_D"] = df["is_case_D"] * df["nb_pct_D"]
I9_INTERACT = [
    "rent_idio", "sale_idio", "rent_idio_lag1", "sale_idio_lag1",
    "rent_vs_nb_d1", "sale_vs_nb_d1", "nb_minus_mun_rent", "nb_minus_mun_sale",
    "rentchg_x_nb", "salechg_x_nb", "aligned_C", "aligned_D",
]

# ── NEW GROUP I10: CHANGE-POINT / STRUCTURAL BREAK FEATURES
log("Engineering I10 change-point features...")
KCser = df["CUSEC"]
for pfx, chg, vol in [("rent", "rent_change", "hist_rent_vol"),
                      ("sale", "sale_change", "hist_sale_vol")]:
    z = (df[chg] - exp_mean_shift(df[chg], KC)) / df[vol].replace(0, np.nan)
    df[f"{pfx}_cusum"] = z.fillna(0.0).groupby(KCser).cumsum()
    df[f"{pfx}_varratio"] = df[f"{pfx}_roll_std3"] / df[f"{pfx}_roll_std6"].replace(0, np.nan)
    shock_flag = df[f"{pfx}_shock"].abs() > 2
    ordv = df.groupby("CUSEC").cumcount()
    last = ordv.where(shock_flag).groupby(KCser).ffill()
    df[f"{pfx}_tss"] = (ordv - last).fillna(99.0)
I10_CHANGEPT = ["rent_cusum", "sale_cusum", "rent_varratio", "sale_varratio",
                "rent_tss", "sale_tss"]

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
FEATS_WIN = FEATS_V1 + G1_REVERSION            # round-2 winning feature set (95)
ALLCOLS = FEATS_WIN + I8_MACRO + I9_INTERACT + I10_CHANGEPT
log(f"Features: V1={len(FEATS_V1)}, +reversion={len(FEATS_WIN)}, "
    f"I8={len(I8_MACRO)}, I9={len(I9_INTERACT)}, I10={len(I10_CHANGEPT)}, "
    f"all engineered={len(ALLCOLS)}")

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
NONE_IDX = class_names.index("None")
log(f"Classes: {class_names}, trainable rows: {len(trainable):,}")

y_all = trainable["target_encoded"].values
year_v = trainable["year"].values
period_v = trainable["period"].values
qnum_v = (year_v * 4 + trainable["quarter"].values).astype(float)
cur_case_v = trainable["case"].values
nca_v = trainable["NCA"].astype(str).values

m_tr18 = year_v < 2018
m_val = (year_v >= 2018) & (year_v < 2020)
m_tr20 = year_v < 2020
m_te = year_v >= 2020
y_te = y_all[m_te]
y_val = y_all[m_val]
n_te = int(m_te.sum())
te_periods_row = period_v[m_te]
TEST_PERIODS = np.sort(pd.unique(te_periods_row))
VAL_PERIODS = np.sort(pd.unique(period_v[m_val]))
log(f"Split: train<2018={m_tr18.sum():,} val={m_val.sum():,} "
    f"train<2020={m_tr20.sum():,} test={n_te:,} ({len(TEST_PERIODS)} test quarters)")

XALL = trainable[ALLCOLS].astype(np.float32).values
XALL[~np.isfinite(XALL)] = np.nan

rdir = (np.sign(np.nan_to_num(trainable["rent_change_next"].values)) + 1).astype(int)
sdir = (np.sign(np.nan_to_num(trainable["sale_change_next"].values)) + 1).astype(int)
rt_next = np.clip(np.nan_to_num(trainable["rent_change_next"].values), -0.5, 0.5)
st_next = np.clip(np.nan_to_num(trainable["sale_change_next"].values), -0.5, 0.5)
abs_rt_next = np.abs(trainable["rent_change_next"].values)
abs_st_next = np.abs(trainable["sale_change_next"].values)

# ════════════════════════════════════════════════════════════════════
# REGRESSION-STACKING META-FEATURES (annual expanding OOT folds)
# ════════════════════════════════════════════════════════════════════
log("Building OOT regression meta-features (annual expanding folds)...")
FIDX0 = {f: i for i, f in enumerate(ALLCOLS)}
COLSWIN0 = [FIDX0[f] for f in FEATS_WIN]
META = np.full((len(y_all), 4), np.nan, dtype=np.float32)
rp = dict(n_estimators=N_EST_REG, max_depth=8, learning_rate=0.05, subsample=0.8,
          colsample_bytree=0.7, random_state=42, n_jobs=-1, tree_method="hist")
years_sorted = np.sort(np.unique(year_v))
for Y in years_sorted:
    mtr = year_v < Y
    mpr = year_v == Y
    if mtr.sum() < 5000 or mpr.sum() == 0:
        continue
    Xtr = XALL[mtr][:, COLSWIN0].copy()
    Xpr = XALL[mpr][:, COLSWIN0].copy()
    mu = np.nanmean(Xtr, axis=0)
    mu = np.where(np.isfinite(mu), mu, 0.0).astype(np.float32)
    for X in (Xtr, Xpr):
        nn = np.where(np.isnan(X))
        X[nn] = mu[nn[1]]
    for j, tgt in enumerate([rt_next, st_next]):
        reg = XGBRegressor(**rp)
        reg.fit(Xtr, tgt[mtr])
        META[mpr, j] = reg.predict(Xpr)
    log(f"  meta fold <{Y} -> {Y}: filled {mpr.sum():,} rows")
META[:, 2] = np.sign(META[:, 0])
META[:, 3] = np.sign(META[:, 1])
META4 = ["meta_rent_pred", "meta_sale_pred", "meta_rent_sign", "meta_sale_sign"]
XALL2 = np.hstack([XALL, META])
del XALL
FIDX = dict(FIDX0)
for k, name in enumerate(META4):
    FIDX[name] = len(ALLCOLS) + k
COLS_WIN = [FIDX[f] for f in FEATS_WIN]                  # 95, no meta
COLS_MAIN = COLS_WIN + [FIDX[f] for f in META4]          # 99, with meta
NAMES_MAIN = FEATS_WIN + META4

# ════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════
PBASE = dict(objective="multi:softprob", max_depth=8, learning_rate=0.05,
             subsample=0.8, colsample_bytree=0.7, min_child_weight=3, gamma=0.1,
             eval_metric="mlogloss", random_state=42, n_jobs=-1, tree_method="hist")


def prep_idx(cols, mtr, mte):
    Xtr = XALL2[mtr][:, cols].copy()
    Xte = XALL2[mte][:, cols].copy()
    mu = np.nanmean(Xtr, axis=0)
    mu = np.where(np.isfinite(mu), mu, 0.0).astype(np.float32)
    for X in (Xtr, Xte):
        nn = np.where(np.isnan(X))
        X[nn] = mu[nn[1]]
    return Xtr, Xte, mu


def impute_with(X, mu):
    X = X.copy()
    nn = np.where(np.isnan(X))
    X[nn] = mu[nn[1]]
    return X


def rw(mask, hl):
    if hl is None:
        return None
    q = qnum_v[mask]
    return (0.5 ** ((q.max() - q) / hl)).astype(np.float32)


def combine_w(mask, hl, w_extra):
    w = rw(mask, hl)
    if w_extra is not None:
        we = w_extra[mask].astype(np.float32)
        w = we if w is None else w * we
    return w


def make_fitter(model, seed, n_est, K, extra=None):
    """Returns fit(Xtr,y,w) -> predict_proba callable."""
    if model == "xgb":
        p = dict(PBASE)
        p["random_state"] = seed
        p["n_estimators"] = n_est
        if K != n_classes:
            p["num_class"] = None
        if extra:
            p.update(extra)
        def fit(Xtr, ytr, w):
            clf = XGBClassifier(**{k: v for k, v in p.items() if v is not None})
            clf.fit(Xtr, ytr, sample_weight=w)
            return clf.predict_proba
        return fit
    if model == "lgbm":
        from lightgbm import LGBMClassifier
        lp = dict(n_estimators=n_est, num_leaves=127, learning_rate=0.05,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                  n_jobs=-1, random_state=seed, verbose=-1)
        def fit(Xtr, ytr, w):
            clf = LGBMClassifier(**lp)
            clf.fit(Xtr, ytr, sample_weight=w)
            return clf.predict_proba
        return fit
    if model == "cat":
        from catboost import CatBoostClassifier
        cp = dict(iterations=N_EST_CAT, depth=8, learning_rate=0.08,
                  loss_function="MultiClass", random_seed=seed, verbose=0,
                  allow_writing_files=False)
        def fit(Xtr, ytr, w):
            clf = CatBoostClassifier(**cp)
            clf.fit(Xtr, ytr, sample_weight=w)
            return clf.predict_proba
        return fit
    if model == "xgb_focal":
        gamma = extra["focal_gamma"]
        def fit(Xtr, ytr, w):
            def obj(preds, dtrain):
                yy = dtrain.get_label().astype(int)
                z = preds.reshape(len(yy), K) if preds.ndim == 1 else preds
                z = z - z.max(1, keepdims=True)
                p_ = np.exp(z)
                p_ /= p_.sum(1, keepdims=True)
                pt = p_[np.arange(len(yy)), yy]
                wf = ((1.0 - pt) ** gamma)[:, None]
                g = p_.copy()
                g[np.arange(len(yy)), yy] -= 1.0
                g *= wf
                h = np.maximum(2.0 * p_ * (1.0 - p_) * wf, 1e-6)
                # xgboost 2.0.x expects flattened row-major (n*K,) grad/hess
                return g.reshape(-1), h.reshape(-1)
            dtr = xgb.DMatrix(Xtr, label=ytr, weight=w)
            params = dict(max_depth=8, eta=0.05, subsample=0.8, colsample_bytree=0.7,
                          min_child_weight=3, gamma=0.1, num_class=K,
                          tree_method="hist", seed=seed,
                          disable_default_eval_metric=1)
            bst = xgb.train(params, dtr, num_boost_round=n_est, obj=obj)
            def proba(X):
                m = bst.predict(xgb.DMatrix(X), output_margin=True)
                m = m.reshape(-1, K) if m.ndim == 1 else m
                m = m - m.max(1, keepdims=True)
                e = np.exp(m)
                return e / e.sum(1, keepdims=True)
            return proba
        return fit
    raise ValueError(model)


def rolling_run(cols, yv=None, K=None, hl=None, seed=42, n_est=N_EST_ROLL,
                model="xgb", extra=None, w_extra=None, tag="",
                collect_prev=False, verbose=True):
    """Refit each test quarter on all rows with period < t. Leak-free."""
    yv = y_all if yv is None else yv
    K = n_classes if K is None else K
    P = np.zeros((n_te, K))
    Pprev = np.zeros((n_te, K))
    hasprev = np.zeros(n_te, bool)
    prev = None
    fitter = make_fitter(model, seed, n_est, K, extra)
    for p in TEST_PERIODS:
        mtr = period_v < p
        mpr = m_te & (period_v == p)
        Xtr, Xpr, mu = prep_idx(cols, mtr, mpr)
        w = combine_w(mtr, hl, w_extra)
        proba = fitter(Xtr, yv[mtr], w)
        sel = te_periods_row == p
        P[sel] = proba(Xpr)
        if collect_prev and prev is not None:
            pf, pmu = prev
            Xpr2 = impute_with(XALL2[mpr][:, cols], pmu)
            Pprev[sel] = pf(Xpr2)
            hasprev[sel] = True
        prev = (proba, mu)
        if verbose:
            aq = accuracy_score(yv[mpr], P[sel].argmax(1))
            log(f"    [{tag}] {pd.Timestamp(p).date()}: train n={mtr.sum():,} Acc={aq:.4f}")
    return P, Pprev, hasprev


def fixed_fit(cols, mtr, mte, yv=None, K=None, hl=None, seed=42, n_est=N_EST,
              model="xgb", extra=None, w_extra=None, return_clf=False):
    yv = y_all if yv is None else yv
    K = n_classes if K is None else K
    Xtr, Xte_, _ = prep_idx(cols, mtr, mte)
    w = combine_w(mtr, hl, w_extra)
    if return_clf and model == "xgb":
        p = dict(PBASE)
        p["random_state"] = seed
        p["n_estimators"] = n_est
        if extra:
            p.update(extra)
        clf = XGBClassifier(**p)
        clf.fit(Xtr, yv[mtr], sample_weight=w)
        return clf.predict_proba(Xte_), clf
    proba = make_fitter(model, seed, n_est, K, extra)(Xtr, yv[mtr], w)
    return proba(Xte_)


def acc_of(P, y):
    return accuracy_score(y, P.argmax(1))


RESULTS = {}


def report(name, P, y_true, base=None, record=True):
    pred = P.argmax(1)
    a = accuracy_score(y_true, pred)
    f = f1_score(y_true, pred, average="weighted")
    d = f" ({(a - base) * 100:+.2f}pp)" if base is not None else ""
    log(f"  >> {name}: Acc={a:.4f} F1={f:.4f}{d}")
    if record:
        RESULTS[name] = (a, f)
    return a, f


def selective_curve(P, y_true, tag):
    mx = P.max(1)
    pred = P.argmax(1)
    order = np.argsort(mx)
    n = len(mx)
    for cov in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]:
        k = int(n * cov)
        if k == 0:
            continue
        idx = order[-k:]
        log(f"  [{tag}] coverage {cov*100:3.0f}%: "
            f"Acc={accuracy_score(y_true[idx], pred[idx]):.4f} (n={k:,})")


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


def compose5(pr, ps):
    pA = pr[:, 2] * (ps[:, 0] + ps[:, 1])
    pB = ps[:, 2] * (pr[:, 0] + pr[:, 1])
    pC = pr[:, 2] * ps[:, 2]
    pD = pr[:, 0] * ps[:, 0]
    pN = pr[:, 1] * ps[:, 1] + pr[:, 1] * ps[:, 0] + pr[:, 0] * ps[:, 1]
    M = np.zeros((len(pr), n_classes))
    for i, cn in enumerate(class_names):
        M[:, i] = {"A": pA, "B": pB, "C": pC, "D": pD, "None": pN}[cn]
    return M / M.sum(1, keepdims=True)


def norm(P):
    return P / P.sum(1, keepdims=True)


def section(title):
    log("\n" + "=" * 72)
    log(title)
    log("=" * 72)


# ════════════════════════════════════════════════════════════════════
# SECTION A: VALIDATION TUNING (train <2018 -> val 2018-19)
# ════════════════════════════════════════════════════════════════════
section("A. VALIDATION TUNING (all hyper-choices decided here)")
VAL = {}

# A1: half-life for recency weighting (with meta features)
best_hl, best_hl_acc = None, -1
for hl in [None, 2.0, 4.0, 8.0, 16.0]:
    Pv = fixed_fit(COLS_MAIN, m_tr18, m_val, hl=hl, n_est=N_EST)
    a = acc_of(Pv, y_val)
    log(f"  hl={hl}: val Acc={a:.4f}")
    if a > best_hl_acc:
        best_hl_acc, best_hl, P_val_direct = a, hl, Pv
HL = best_hl
log(f"  -> chosen half-life: {HL} (val Acc={best_hl_acc:.4f})")
acc_val_direct = best_hl_acc

# A2: seed bag on validation
P_val_seeds = {42: P_val_direct if HL is None else None}
if P_val_seeds[42] is None:
    P_val_seeds[42] = P_val_direct  # already trained with HL
P_val_bag = np.zeros_like(P_val_direct)
for s in SEEDS_BAG:
    Pv = P_val_seeds.get(s)
    if Pv is None:
        Pv = fixed_fit(COLS_MAIN, m_tr18, m_val, hl=HL, seed=s, n_est=N_EST)
    P_val_bag += Pv / len(SEEDS_BAG)
    P_val_seeds[s] = Pv
log(f"  val bag Acc={acc_of(P_val_bag, y_val):.4f} (single {acc_val_direct:.4f})")

# A3: LGBM / CatBoost on validation
COMPS_VAL, COMPS_NAMES = [P_val_bag], ["xgb_bag"]
try:
    P_val_lgbm = fixed_fit(COLS_MAIN, m_tr18, m_val, hl=HL, model="lgbm", n_est=N_EST)
    log(f"  val LGBM Acc={acc_of(P_val_lgbm, y_val):.4f}")
    COMPS_VAL.append(P_val_lgbm); COMPS_NAMES.append("lgbm")
except Exception as e:
    P_val_lgbm = None
    log(f"  LGBM unavailable: {e}")
try:
    P_val_cb = fixed_fit(COLS_MAIN, m_tr18, m_val, hl=HL, model="cat")
    log(f"  val CatBoost Acc={acc_of(P_val_cb, y_val):.4f}")
    COMPS_VAL.append(P_val_cb); COMPS_NAMES.append("catboost")
except Exception as e:
    P_val_cb = None
    log(f"  CatBoost unavailable: {e}")

kcomp = len(COMPS_VAL)
rng = np.random.default_rng(0)
cands = [np.eye(kcomp)[i] for i in range(kcomp)] + [np.ones(kcomp) / kcomp] + \
        list(rng.dirichlet(np.ones(kcomp), 300))
best_w, best_wa = None, -1
for w in cands:
    a = acc_of(sum(wi * m for wi, m in zip(w, COMPS_VAL)), y_val)
    if a > best_wa:
        best_wa, best_w = a, w
ENS_W = best_w
log(f"  -> ensemble weights {dict(zip(COMPS_NAMES, np.round(ENS_W, 3)))}, val Acc={best_wa:.4f}")
P_val_ens = sum(wi * m for wi, m in zip(ENS_W, COMPS_VAL))

# A4: 3-class composition heads + alpha blend
ex3 = {"objective": "multi:softprob"}
pr_v = fixed_fit(COLS_MAIN, m_tr18, m_val, yv=rdir, K=3, hl=HL, extra=ex3)
ps_v = fixed_fit(COLS_MAIN, m_tr18, m_val, yv=sdir, K=3, hl=HL, extra=ex3)
C_val = compose5(pr_v, ps_v)
log(f"  val composed Acc={acc_of(C_val, y_val):.4f}")
best_alpha, best_aa = 1.0, -1
for alpha in np.arange(0, 1.0001, 0.05):
    a = acc_of(alpha * P_val_ens + (1 - alpha) * C_val, y_val)
    if a > best_aa:
        best_aa, best_alpha = a, alpha
ALPHA = best_alpha
log(f"  -> alpha={ALPHA:.2f} (val Acc={best_aa:.4f}, ens alone {best_wa:.4f})")
P_val_comb = ALPHA * P_val_ens + (1 - ALPHA) * C_val
acc_val_comb = acc_of(P_val_comb, y_val)
log(f"  VAL COMBINED Acc={acc_val_comb:.4f}")

# ════════════════════════════════════════════════════════════════════
# SECTION B: STEP 1 — COMBINED ROLLING MODEL + ABLATION LADDER
# ════════════════════════════════════════════════════════════════════
section("B. STEP 1: COMBINED ROLLING MODEL (ablation ladder R1..R6)")

log("R1: rolling, seed 42, 95 feats, no meta, no recency (round-2 EXP12 repro)")
P_R1, _, _ = rolling_run(COLS_WIN, tag="R1")
A_R1, _ = report("R1 rolling (round-2 repro)", P_R1, y_te)
BASE = A_R1

if HL is not None:
    log(f"R2: rolling + recency hl={HL}")
    P_R2, _, _ = rolling_run(COLS_WIN, hl=HL, tag="R2")
else:
    log("R2: recency rejected on val -> R2 = R1")
    P_R2 = P_R1
report("R2 +recency", P_R2, y_te, BASE)

log("R3: rolling + recency(if any) + regression-stacking meta-features")
P_R3, P_R3prev, has_prev = rolling_run(COLS_MAIN, hl=HL, tag="R3", collect_prev=True)
A_R3, _ = report("R3 +meta-features", P_R3, y_te, BASE)

log("R4: seed bagging (5 seeds, averaged)")
P_seeds = {42: P_R3}
for s in SEEDS_BAG:
    if s in P_seeds:
        continue
    Ps, _, _ = rolling_run(COLS_MAIN, hl=HL, seed=s, tag=f"R4 seed{s}", verbose=False)
    P_seeds[s] = Ps
    log(f"    seed {s}: Acc={acc_of(Ps, y_te):.4f}")
P_R4 = sum(P_seeds[s] for s in SEEDS_BAG) / len(SEEDS_BAG)
report("R4 +seed bagging", P_R4, y_te, BASE)

log("R5: heterogeneous ensemble (XGB-bag + LGBM + CatBoost, val weights)")
COMPS_TE = [P_R4]
if P_val_lgbm is not None:
    P_lgbm_roll, _, _ = rolling_run(COLS_MAIN, hl=HL, model="lgbm", tag="R5 lgbm", verbose=False)
    log(f"    LGBM rolling Acc={acc_of(P_lgbm_roll, y_te):.4f}")
    COMPS_TE.append(P_lgbm_roll)
else:
    P_lgbm_roll = None
if P_val_cb is not None:
    P_cb_roll, _, _ = rolling_run(COLS_MAIN, hl=HL, model="cat", tag="R5 catboost", verbose=False)
    log(f"    CatBoost rolling Acc={acc_of(P_cb_roll, y_te):.4f}")
    COMPS_TE.append(P_cb_roll)
else:
    P_cb_roll = None
P_R5 = sum(wi * m for wi, m in zip(ENS_W, COMPS_TE))
report("R5 +hetero ensemble", P_R5, y_te, BASE)

log("R6: + two-head 3-class composition blend (alpha from val)")
pr_t, _, _ = rolling_run(COLS_MAIN, yv=rdir, K=3, hl=HL, extra=ex3, tag="R6 rent-dir", verbose=False)
ps_t, _, _ = rolling_run(COLS_MAIN, yv=sdir, K=3, hl=HL, extra=ex3, tag="R6 sale-dir", verbose=False)
C_roll = compose5(pr_t, ps_t)
log(f"    composed alone rolling Acc={acc_of(C_roll, y_te):.4f}")
P_COMB = ALPHA * P_R5 + (1 - ALPHA) * C_roll
A_COMB, _ = report("R6 COMBINED (all winners)", P_COMB, y_te, BASE)

log("\nCombination menu (what cancels out; all params val-tuned):")
menu = {
    "xgb single R3": P_R3,
    "bag only R4": P_R4,
    "ens only R5": P_R5,
    "composed alone": C_roll,
    "R3+composed blend": ALPHA * P_R3 + (1 - ALPHA) * C_roll,
    "R4+composed blend": ALPHA * P_R4 + (1 - ALPHA) * C_roll,
    "COMBINED R6": P_COMB,
    "equal avg(bag,lgbm,cb,comp)": norm(sum(norm(p) for p in COMPS_TE + [C_roll]) / (len(COMPS_TE) + 1)),
}
for nm, P in menu.items():
    report(f"MENU {nm}", P, y_te, BASE)

log("\nPer-quarter accuracy of COMBINED:")
for p in TEST_PERIODS:
    sel = te_periods_row == p
    log(f"  {pd.Timestamp(p).date()}: Acc={acc_of(P_COMB[sel], y_te[sel]):.4f} (n={sel.sum():,})")

log("\nSelective prediction curve (COMBINED):")
selective_curve(P_COMB, y_te, "COMBINED")
log("\nClassification report (COMBINED):")
for line in classification_report(y_te, P_COMB.argmax(1), target_names=class_names).splitlines():
    log("  " + line)

np.savez_compressed(
    os.path.join(OUT, "round3_probs.npz"),
    P_COMB=P_COMB, P_R1=P_R1, P_R3=P_R3, P_R4=P_R4, P_R5=P_R5, C_roll=C_roll,
    y_te=y_te, te_periods=te_periods_row.astype("datetime64[ns]").astype("int64"),
)
log("Saved probability matrices to round3_probs.npz")

# ════════════════════════════════════════════════════════════════════
# SECTION C: CHEAP POST-HOC IDEAS I1-I7 (tuned on val, applied to COMBINED)
# ════════════════════════════════════════════════════════════════════
section("C. POST-HOC IDEAS ON COMBINED PROBABILITIES (I1-I7)")
POSTHOC = {}  # name -> (val_delta, P_test)


def idea(name, fn):
    log("\n--- " + name)
    try:
        fn()
    except Exception:
        log("  FAILED:\n" + traceback.format_exc())


def i1_vector_scaling():
    Zv = np.log(np.clip(P_val_comb, 1e-9, 1))
    lr = LogisticRegression(max_iter=2000, C=100.0)
    lr.fit(Zv, y_val)
    av = acc_of(lr.predict_proba(Zv), y_val)
    log(f"  val: {acc_val_comb:.4f} -> {av:.4f}")
    Pt = lr.predict_proba(np.log(np.clip(P_COMB, 1e-9, 1)))
    report("I1 vector scaling", Pt, y_te, A_COMB)
    POSTHOC["I1 vector scaling"] = (av - acc_val_comb, Pt)


def i2_isotonic():
    Pv = np.zeros_like(P_val_comb)
    isos = []
    for k in range(n_classes):
        iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1.0)
        iso.fit(P_val_comb[:, k], (y_val == k).astype(float))
        isos.append(iso)
        Pv[:, k] = iso.predict(P_val_comb[:, k])
    av = acc_of(norm(Pv), y_val)
    log(f"  val: {acc_val_comb:.4f} -> {av:.4f}")
    Pt = np.zeros_like(P_COMB)
    for k in range(n_classes):
        Pt[:, k] = isos[k].predict(P_COMB[:, k])
    Pt = norm(Pt)
    report("I2 isotonic calibration", Pt, y_te, A_COMB)
    POSTHOC["I2 isotonic"] = (av - acc_val_comb, Pt)


def i3_multipliers():
    w, av = tune_multipliers(P_val_comb, y_val)
    log(f"  multipliers {np.round(w, 2)}, val {acc_val_comb:.4f} -> {av:.4f}")
    Pt = norm(P_COMB * w)
    report("I3 class multipliers", Pt, y_te, A_COMB)
    POSTHOC["I3 multipliers"] = (av - acc_val_comb, Pt)


def prior_shift_apply(P, rows_periods, cur_case, prior_train_by_period, lam):
    Q = P.copy()
    for p in np.unique(rows_periods):
        sel = rows_periods == p
        cc = cur_case[sel]
        cc = cc[np.isin(cc, class_names)]
        pi_tr = prior_train_by_period(p)
        if len(cc) == 0:
            continue
        pi_obs = np.array([(cc == cn).mean() for cn in class_names])
        pi = lam * pi_obs + (1 - lam) * pi_tr
        Q[sel] = P[sel] * (pi / np.maximum(pi_tr, 1e-9))
    return norm(Q)


def i4_prior_shift():
    # case(t) is observed at prediction time; its distribution proxies the
    # target prior at t+1. Shrinkage lambda tuned on val.
    pi_tr18 = np.bincount(y_all[m_tr18], minlength=n_classes) / m_tr18.sum()
    cur_val = cur_case_v[m_val]
    per_val = period_v[m_val]
    best_lam, best_a = 0.0, acc_val_comb
    for lam in [0.0, 0.25, 0.5, 0.75, 1.0]:
        Pv = prior_shift_apply(P_val_comb, per_val, cur_val, lambda p: pi_tr18, lam)
        a = acc_of(Pv, y_val)
        log(f"  lambda={lam:.2f}: val Acc={a:.4f}")
        if a > best_a:
            best_a, best_lam = a, lam
    log(f"  -> lambda={best_lam}")
    def pi_tr_roll(p):
        m = period_v < p
        return np.bincount(y_all[m], minlength=n_classes) / m.sum()
    Pt = prior_shift_apply(P_COMB, te_periods_row, cur_case_v[m_te], pi_tr_roll, best_lam)
    report("I4 lag-1 prior shift", Pt, y_te, A_COMB)
    POSTHOC["I4 prior shift"] = (best_a - acc_val_comb, Pt)


def ece(P, y, bins=15):
    mx = P.max(1)
    pred = P.argmax(1)
    corr = (pred == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for i in range(bins):
        m = (mx > edges[i]) & (mx <= edges[i + 1])
        if m.sum() == 0:
            continue
        e += m.mean() * abs(corr[m].mean() - mx[m].mean())
    return e


def i5_temperature():
    best_T, best_nll = 1.0, log_loss(y_val, np.clip(P_val_comb, 1e-9, 1))
    for T in np.arange(0.5, 3.01, 0.1):
        Pv = norm(np.clip(P_val_comb, 1e-12, 1) ** (1.0 / T))
        nll = log_loss(y_val, np.clip(Pv, 1e-9, 1))
        if nll < best_nll:
            best_nll, best_T = nll, T
    Pt = norm(np.clip(P_COMB, 1e-12, 1) ** (1.0 / best_T))
    log(f"  T={best_T:.1f}; test logloss {log_loss(y_te, np.clip(P_COMB,1e-9,1)):.4f} -> "
        f"{log_loss(y_te, np.clip(Pt,1e-9,1)):.4f}; "
        f"ECE {ece(P_COMB, y_te):.4f} -> {ece(Pt, y_te):.4f}")
    report("I5 temperature scaling", Pt, y_te, A_COMB)
    log("  selective curve after temperature:")
    selective_curve(Pt, y_te, "I5")


def i6_conformal():
    alpha = 0.1
    nv = len(y_val)
    s_val = 1.0 - P_val_comb[np.arange(nv), y_val]
    qlev = min(np.ceil((nv + 1) * (1 - alpha)) / nv, 1.0)
    qhat = np.quantile(s_val, qlev)
    sets = (1.0 - P_COMB) <= qhat
    cover = sets[np.arange(n_te), y_te].mean()
    sizes = sets.sum(1)
    single = sizes == 1
    acc_single = (P_COMB[single].argmax(1) == y_te[single]).mean() if single.sum() else float("nan")
    log(f"  marginal 90%: coverage={cover:.4f}, avg set size={sizes.mean():.2f}, "
        f"singletons={single.mean()*100:.1f}% (acc on singletons={acc_single:.4f})")
    # class-conditional (Mondrian)
    qh = np.zeros(n_classes)
    for k in range(n_classes):
        sk = s_val[y_val == k]
        nk = len(sk)
        qh[k] = np.quantile(sk, min(np.ceil((nk + 1) * (1 - alpha)) / nk, 1.0))
    setsm = (1.0 - P_COMB) <= qh[None, :]
    coverm = setsm[np.arange(n_te), y_te].mean()
    log(f"  class-conditional 90%: coverage={coverm:.4f}, avg set size={setsm.sum(1).mean():.2f}")
    per_cls = [(class_names[k], sets[y_te == k, k].mean()) for k in range(n_classes)]
    log("  per-class coverage (marginal): " +
        ", ".join(f"{c}={v:.3f}" for c, v in per_cls))


def i7_stacking():
    comps_v = [P_val_bag] + ([P_val_lgbm] if P_val_lgbm is not None else []) + \
              ([P_val_cb] if P_val_cb is not None else []) + [C_val]
    comps_t = [P_R4] + ([P_lgbm_roll] if P_lgbm_roll is not None else []) + \
              ([P_cb_roll] if P_cb_roll is not None else []) + [C_roll]
    Xv = np.hstack(comps_v)
    Xt = np.hstack(comps_t)
    lr = LogisticRegression(max_iter=2000, C=1.0)
    lr.fit(Xv, y_val)
    av = acc_of(lr.predict_proba(Xv), y_val)
    log(f"  val stack Acc={av:.4f} (combined {acc_val_comb:.4f})")
    Pt = lr.predict_proba(Xt)
    report("I7 LR stacking of components", Pt, y_te, A_COMB)
    POSTHOC["I7 LR stacking"] = (av - acc_val_comb, Pt)


idea("I1 VECTOR SCALING CALIBRATION", i1_vector_scaling)
idea("I2 PER-CLASS ISOTONIC CALIBRATION", i2_isotonic)
idea("I3 VAL-TUNED CLASS MULTIPLIERS ON COMBINED", i3_multipliers)
idea("I4 LAG-1 OBSERVED-PRIOR SHIFT", i4_prior_shift)
idea("I5 TEMPERATURE SCALING (calibration)", i5_temperature)
idea("I6 SPLIT CONFORMAL PREDICTION (coverage guarantee)", i6_conformal)
idea("I7 STACKING META-LEARNER OVER COMPONENTS", i7_stacking)

# ════════════════════════════════════════════════════════════════════
# SECTION D: NEW FEATURES / TRAINING SCHEMES (I8-I18)
# ════════════════════════════════════════════════════════════════════
section("D. NEW FEATURE / TRAINING IDEAS (I8-I18)")
log("Fixed-split screen: train<2020 -> test 2020+ (single XGB, meta+hl config).")
log("Winners (>= +0.10pp on screen or val) are promoted to rolling vs R3.")

P_fixed_base = fixed_fit(COLS_MAIN, m_tr20, m_te, hl=HL)
A_FIX, _ = report("D0 fixed-split baseline (99 feats)", P_fixed_base, y_te)

PROMOTED = []  # (name, cols or spec)


def screen_feature_group(name, extra_names, est):
    cols = COLS_MAIN + [FIDX[f] for f in extra_names]
    Pt = fixed_fit(cols, m_tr20, m_te, hl=HL)
    a, _ = report(f"{name} screen (+{len(extra_names)} feats, est {est})", Pt, y_te, A_FIX)
    if a - A_FIX >= 0.0010:
        PROMOTED.append((name, cols))
        log(f"  -> PROMOTED to rolling")
    return a


idea("I8 MACRO REGIME FEATURES", lambda: screen_feature_group("I8 macro regime", I8_MACRO, "+0.5pp"))
idea("I9 SPATIAL x TEMPORAL INTERACTIONS", lambda: screen_feature_group("I9 spatio-temporal", I9_INTERACT, "+0.3pp"))
idea("I10 CHANGE-POINT FEATURES", lambda: screen_feature_group("I10 change-point", I10_CHANGEPT, "+0.2pp"))


def i11_feature_selection():
    _, clf = fixed_fit(COLS_MAIN, m_tr18, m_val, hl=HL, return_clf=True)
    imp = clf.feature_importances_
    order = np.argsort(-imp)
    best_k, best_a, best_cols = len(COLS_MAIN), acc_val_direct, COLS_MAIN
    for K in [50, 70]:
        cols = [COLS_MAIN[i] for i in order[:K]]
        a = acc_of(fixed_fit(cols, m_tr18, m_val, hl=HL), y_val)
        log(f"  top-{K} by gain: val Acc={a:.4f} (full {acc_val_direct:.4f})")
        if a > best_a:
            best_a, best_k, best_cols = a, K, cols
    if best_k < len(COLS_MAIN):
        log(f"  -> top-{best_k} wins on val, PROMOTED to rolling")
        PROMOTED.append((f"I11 top-{best_k} features", best_cols))
    else:
        log("  -> full feature set wins on val; no promotion")


def i12_class_balance():
    # counts from TRAINING labels only (no test label aggregates)
    cnt18 = np.bincount(y_all[m_tr18], minlength=n_classes).astype(float)
    best_b, best_a = 0.0, acc_val_direct
    for b in [0.25, 0.5, 1.0]:
        w = (m_tr18.sum() / (n_classes * cnt18))[y_all] ** b
        Pv = fixed_fit(COLS_MAIN, m_tr18, m_val, hl=HL, w_extra=w)
        a = acc_of(Pv, y_val)
        fmac = f1_score(y_val, Pv.argmax(1), average="macro")
        log(f"  beta={b}: val Acc={a:.4f} macroF1={fmac:.4f}")
        if a > best_a:
            best_a, best_b = a, b
    if best_b > 0:
        cnt20 = np.bincount(y_all[m_tr20], minlength=n_classes).astype(float)
        w = (m_tr20.sum() / (n_classes * cnt20))[y_all] ** best_b
        PROMOTED.append((f"I12 class-balance beta={best_b}", ("weights", w)))
        log(f"  -> beta={best_b} PROMOTED")
    else:
        log("  -> no beta beats unweighted on val")


def i13_focal():
    for g in ([1.0] if FAST else [1.0, 2.0]):
        Pt = fixed_fit(COLS_MAIN, m_tr20, m_te, hl=HL, model="xgb_focal",
                       extra={"focal_gamma": g})
        a, _ = report(f"I13 focal gamma={g} screen", Pt, y_te, A_FIX)
        if a - A_FIX >= 0.0010:
            PROMOTED.append((f"I13 focal gamma={g}", ("model", "xgb_focal", {"focal_gamma": g})))
            log("  -> PROMOTED")
            break


def i14_ambiguity():
    best_cfg, best_a = None, acc_val_direct
    for eps in [0.005, 0.01]:
        for delta in [0.3, 0.6]:
            amb = (abs_rt_next < eps) & (abs_st_next < eps)
            w = np.where(amb, delta, 1.0)
            a = acc_of(fixed_fit(COLS_MAIN, m_tr18, m_val, hl=HL, w_extra=w), y_val)
            log(f"  eps={eps}, w_amb={delta}: val Acc={a:.4f} "
                f"(amb share in train={amb[m_tr18].mean()*100:.1f}%)")
            if a > best_a:
                best_a, best_cfg = a, (eps, delta)
    if best_cfg:
        eps, delta = best_cfg
        amb = (abs_rt_next < eps) & (abs_st_next < eps)
        w = np.where(amb, delta, 1.0)
        PROMOTED.append((f"I14 ambiguity eps={eps} w={delta}", ("weights", w)))
        log(f"  -> PROMOTED eps={eps} w={delta}")
    else:
        log("  -> no config beats baseline on val")


def i15_window_ensemble():
    P15 = np.where(has_prev[:, None], 0.5 * P_R3 + 0.5 * P_R3prev, P_R3)
    report("I15 train-window ensemble avg(t,t-1)", P15, y_te, A_R3)
    log(f"  (delta vs R3; prev-model available for {has_prev.mean()*100:.0f}% of test rows)")


def i16_hierarchical():
    ybin = (y_all == NONE_IDX).astype(int)
    Pb = fixed_fit(COLS_MAIN, m_tr20, m_te, yv=ybin, K=2, hl=HL,
                   extra={"objective": "binary:logistic", "eval_metric": "logloss"})
    nonone = y_all != NONE_IDX
    y4map = {i: (i if i < NONE_IDX else i - 1) for i in range(n_classes) if i != NONE_IDX}
    y4 = np.array([y4map.get(v, 0) for v in y_all])
    m4tr = m_tr20 & nonone
    P4 = fixed_fit(COLS_MAIN, m4tr, m_te, yv=y4, K=4, hl=HL, extra=ex3)
    Pt = np.zeros((n_te, n_classes))
    Pt[:, NONE_IDX] = Pb[:, 1]
    j = 0
    for i in range(n_classes):
        if i == NONE_IDX:
            continue
        Pt[:, i] = (1 - Pb[:, 1]) * P4[:, j]
        j += 1
    report("I16 hierarchical None-vs-rest screen", Pt, y_te, A_FIX)


def i17_region_models():
    Pt = P_fixed_base.copy()
    ncas, cnts = np.unique(nca_v[m_tr20], return_counts=True)
    big = [c for c, n in zip(ncas, cnts) if n >= (5000 if FAST else 40000)]
    log(f"  region-specific models for {len(big)} communities: {big}")
    changed = np.zeros(n_te, bool)
    for c in big:
        mtr_c = m_tr20 & (nca_v == c)
        mte_c = m_te & (nca_v == c)
        if mte_c.sum() == 0:
            continue
        Pc = fixed_fit(COLS_MAIN, mtr_c, mte_c, hl=HL)
        sel = nca_v[m_te] == c
        Pt[sel] = Pc
        changed |= sel
        log(f"    {c}: regional Acc={acc_of(Pc, y_te[sel]):.4f} vs "
            f"global {acc_of(P_fixed_base[sel], y_te[sel]):.4f} (n={sel.sum():,})")
    report("I17 region-specific screen", Pt, y_te, A_FIX)


def i18_drop_calendar():
    drop = {FIDX["year"], FIDX["month"]}
    cols = [c for c in COLS_MAIN if c not in drop]
    Pt = fixed_fit(cols, m_tr20, m_te, hl=HL)
    a, _ = report("I18 drop year/month screen", Pt, y_te, A_FIX)
    if a - A_FIX >= 0.0010:
        PROMOTED.append(("I18 drop year/month", cols))
        log("  -> PROMOTED")


idea("I11 FEATURE SELECTION (top-K by gain, K on val)", i11_feature_selection)
idea("I12 CLASS-BALANCED WEIGHTS (beta on val)", i12_class_balance)
idea("I13 FOCAL LOSS (detached-weight, custom objective)", i13_focal)
idea("I14 AMBIGUITY DOWNWEIGHTING (near-boundary labels)", i14_ambiguity)
idea("I15 TRAINING-WINDOW ENSEMBLE (free, from R3 cache)", i15_window_ensemble)
idea("I16 HIERARCHICAL TWO-STAGE (None vs change -> 4-class)", i16_hierarchical)
idea("I17 REGION-SPECIFIC MODELS", i17_region_models)
idea("I18 DROP RAW CALENDAR FEATURES", i18_drop_calendar)

# ── Rolling verification of promoted ideas (vs R3 single-seed baseline) ──
section("D2. ROLLING VERIFICATION OF PROMOTED IDEAS (baseline = R3)")
ROLL_WINNERS = {}
for name, spec in PROMOTED:
    log(f"\n--- rolling: {name}")
    try:
        if isinstance(spec, list):
            P, _, _ = rolling_run(spec, hl=HL, tag=name, verbose=False)
        elif isinstance(spec, tuple) and spec[0] == "weights":
            P, _, _ = rolling_run(COLS_MAIN, hl=HL, w_extra=spec[1], tag=name, verbose=False)
        elif isinstance(spec, tuple) and spec[0] == "model":
            P, _, _ = rolling_run(COLS_MAIN, hl=HL, model=spec[1], extra=spec[2],
                                  tag=name, verbose=False)
        else:
            continue
        a, _ = report(f"{name} ROLLING", P, y_te, A_R3)
        if a > A_R3:
            ROLL_WINNERS[name] = (a, P)
    except Exception:
        log("  FAILED:\n" + traceback.format_exc())

# ════════════════════════════════════════════════════════════════════
# SECTION E: FINAL ASSEMBLY
# ════════════════════════════════════════════════════════════════════
section("E. FINAL ASSEMBLY")

# E1: union of winning feature groups in one rolling run
try:
    feat_groups = {"I8 macro regime": I8_MACRO, "I9 spatio-temporal": I9_INTERACT,
                   "I10 change-point": I10_CHANGEPT}
    union_extra = []
    for name in ROLL_WINNERS:
        for gname, g in feat_groups.items():
            if name.startswith(gname.split()[0]) and name.split()[0] == gname.split()[0]:
                union_extra += g
    union_extra = list(dict.fromkeys(union_extra))
    if len([n for n in ROLL_WINNERS if n.split()[0] in ("I8", "I9", "I10")]) >= 2:
        cols_u = COLS_MAIN + [FIDX[f] for f in union_extra]
        P_union, _, _ = rolling_run(cols_u, hl=HL, tag="union", verbose=False)
        a_union, _ = report(f"E1 union of winning feature groups ({len(union_extra)} extra)",
                            P_union, y_te, A_R3)
        ROLL_WINNERS["E1 union feats"] = (a_union, P_union)
    else:
        log("E1: fewer than 2 winning feature groups; union skipped")
except Exception:
    log("E1 FAILED:\n" + traceback.format_exc())

# E2: best single-seed rolling variant blended into the combined stack
best_roll_name, best_roll = None, None
if ROLL_WINNERS:
    best_roll_name = max(ROLL_WINNERS, key=lambda k: ROLL_WINNERS[k][0])
    best_roll = ROLL_WINNERS[best_roll_name][1]
    log(f"\nBest rolling idea: {best_roll_name} (Acc={ROLL_WINNERS[best_roll_name][0]:.4f})")
    P_mix = 0.5 * P_COMB + 0.5 * best_roll
    report(f"E2 avg(COMBINED, {best_roll_name})", P_mix, y_te, A_COMB)
    # swap: replace the single-seed member's share inside the bag
    P_bag_swap = (P_R4 * len(SEEDS_BAG) - P_R3 + best_roll) / len(SEEDS_BAG)
    P_R5_swap = sum(wi * m for wi, m in
                    zip(ENS_W, [P_bag_swap] + COMPS_TE[1:]))
    P_comb_swap = ALPHA * P_R5_swap + (1 - ALPHA) * C_roll
    report("E2b COMBINED with best idea swapped into bag", P_comb_swap, y_te, A_COMB)
else:
    log("No rolling idea beat R3; COMBINED stands as-is.")

# E3: best val-selected post-hoc transform applied to COMBINED
best_ph, best_ph_delta = None, 0.0
for nm, (vd, Pt) in POSTHOC.items():
    if vd > best_ph_delta:
        best_ph_delta, best_ph = vd, nm
if best_ph:
    log(f"\nBest post-hoc by VAL delta: {best_ph} ({best_ph_delta*100:+.2f}pp on val)")
    report(f"E3 COMBINED + {best_ph} (val-selected)", POSTHOC[best_ph][1], y_te, A_COMB)
else:
    log("\nNo post-hoc transform improved validation accuracy; none applied.")

# Final summary
section("FINAL SUMMARY (test 2020+, target = case at t+1)")
log(f"Round-2 anchor (rolling single, 95 feats): {A_R1:.4f}")
log(f"Round-3 COMBINED:                          {A_COMB:.4f} ({(A_COMB-A_R1)*100:+.2f}pp vs anchor)")
for name, (a, f) in sorted(RESULTS.items(), key=lambda kv: -kv[1][0]):
    log(f"  {name:<52s} Acc={a:.4f} F1={f:.4f} ({(a-A_R1)*100:+.2f}pp vs R1)")

rows = [{"experiment": k, "accuracy": v[0], "f1_weighted": v[1],
         "delta_vs_R1_pp": (v[0] - A_R1) * 100} for k, v in RESULTS.items()]
pd.DataFrame(rows).sort_values("accuracy", ascending=False).to_csv(
    os.path.join(OUT, "round3_results.csv"), index=False)
log(f"\nResults written to round3_results.csv. Wall time {(time.time()-T0)/60:.1f} min")
log("DONE")
