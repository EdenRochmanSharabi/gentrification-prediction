"""
Regenerate all paper figures with publication-quality styling.
Uses leak-free prediction model: predict case(t+1) from features(t).
"""
import sys, os, warnings
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import geopandas as gpd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier

from config import FIGURES_DIR
from src.data_loader import load_data
from src.gentrification import classify_cases

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.3, rc={
    "figure.dpi": 300, "savefig.dpi": 300, "font.family": "serif",
    "axes.edgecolor": "#333333", "axes.linewidth": 0.8,
    "grid.alpha": 0.3, "grid.linewidth": 0.5,
})

CASE_COLORS = {
    "A": "#E63946", "B": "#2A9D8F", "C": "#264653",
    "D": "#E9C46A", "None": "#AAAAAA",
}
CASE_LABELS = {
    "A": "A (Renter pressure)", "B": "B (Speculative)",
    "C": "C (Active gentrification)", "D": "D (Degradation)",
    "None": "None (No change)",
}
CASE_ORDER = ["A", "B", "C", "D", "None"]
MAINLAND_EXCLUDE = ["Las Palmas", "Santa Cruz de Tenerife"]
ERROR_CMAP = LinearSegmentedColormap.from_list(
    "purples_trunc", plt.cm.Purples(np.linspace(0.08, 0.95, 256)))


def _relative_luminance(hex_color):
    r, g, b = (int(hex_color[i:i+2], 16) / 255.0 for i in (1, 3, 5))
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
           for c in (r, g, b)]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]

def _mainland(gdf):
    return gdf[~gdf["name"].isin(MAINLAND_EXCLUDE)].copy()


# ════════════════════════════════════════
# DATA LOADING + LEAK-FREE MODEL
# ════════════════════════════════════════
print("Loading data...")
df = load_data()
df = df.sort_values(["CUSEC", "period"])
df["year"] = df["period"].dt.year

sale_col = "unitprice_residential_sale_all"
rent_col = "unitprice_residential_rent_all"
grouped = df.groupby("CUSEC")

df["rent_change"] = grouped[rent_col].pct_change()
df["sale_change"] = grouped[sale_col].pct_change()
df = classify_cases(df)

# Target: next quarter's case (leak-free)
df["target_case"] = df.groupby("CUSEC")["case"].shift(-1)

# Features
df["rent_sale_ratio"] = df[rent_col] / df[sale_col].replace(0, np.nan)
df["sale_change_curr"] = df["sale_change"]
df["rent_change_curr"] = df["rent_change"]
df["change_interaction"] = df["rent_change"] * df["sale_change"]

case_map = {"A": 0, "B": 1, "C": 2, "D": 3, "None": 4, "Unknown": -1}
df["case_encoded_feat"] = df["case"].map(case_map)
for c in ["A", "B", "C", "D", "None"]:
    df[f"is_case_{c}"] = (df["case"] == c).astype(int)

for lag in [1, 2, 4]:
    df[f"sale_lag{lag}"] = grouped[sale_col].shift(lag)
    df[f"rent_lag{lag}"] = grouped[rent_col].shift(lag)

df["sale_change_lag1"] = grouped["sale_change"].shift(1)
df["rent_change_lag1"] = grouped["rent_change"].shift(1)
df["sale_change_lag2"] = grouped["sale_change"].shift(2)
df["rent_change_lag2"] = grouped["rent_change"].shift(2)

for win in [3, 6]:
    mp = 1 if win == 3 else 2
    df[f"sale_roll{win}"] = grouped[sale_col].transform(lambda x: x.rolling(win, min_periods=mp).mean())
    df[f"rent_roll{win}"] = grouped[rent_col].transform(lambda x: x.rolling(win, min_periods=mp).mean())
    df[f"sale_roll_std{win}"] = grouped[sale_col].transform(lambda x: x.rolling(win, min_periods=mp).std())
    df[f"rent_roll_std{win}"] = grouped[rent_col].transform(lambda x: x.rolling(win, min_periods=mp).std())

df["sale_momentum"] = df[sale_col] - df["sale_lag1"]
df["rent_momentum"] = df[rent_col] - df["rent_lag1"]
df["sale_accel"] = df["sale_momentum"] - (df["sale_lag1"] - df["sale_lag2"])
df["rent_accel"] = df["rent_momentum"] - (df["rent_lag1"] - df["rent_lag2"])

prov_sale_mean = df.groupby(["NPRO", "period"])[sale_col].transform("mean")
prov_rent_mean = df.groupby(["NPRO", "period"])[rent_col].transform("mean")
df["sale_vs_prov"] = df[sale_col] / prov_sale_mean.replace(0, np.nan)
df["rent_vs_prov"] = df[rent_col] / prov_rent_mean.replace(0, np.nan)

df["sale_pctile"] = grouped[sale_col].transform(lambda x: x.rank(pct=True))
df["rent_pctile"] = grouped[rent_col].transform(lambda x: x.rank(pct=True))

df["sale_yoy"] = (df[sale_col] - df["sale_lag4"]) / df["sale_lag4"].replace(0, np.nan)
df["rent_yoy"] = (df[rent_col] - df["rent_lag4"]) / df["rent_lag4"].replace(0, np.nan)

df["stock_sale_lag1"] = grouped["stock_residential_sale_all"].shift(1)
df["stock_rent_lag1"] = grouped["stock_residential_rent_all"].shift(1)
df["stock_ratio"] = df["stock_residential_sale_all"] / df["stock_residential_rent_all"].replace(0, np.nan)
df["stock_sale_change"] = grouped["stock_residential_sale_all"].pct_change()
df["stock_rent_change"] = grouped["stock_residential_rent_all"].pct_change()

df["month"] = df["period"].dt.month
df["quarter"] = df["period"].dt.quarter

FEATURES = [
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
    "sale_yoy", "rent_yoy",
    "stock_residential_sale_all", "stock_residential_rent_all",
    "stock_sale_lag1", "stock_rent_lag1", "stock_ratio",
    "stock_sale_change", "stock_rent_change",
    "month", "quarter", "year",
]

FEATURE_LABELS = {
    "case_encoded_feat": "Current case type",
    "is_case_A": "Is case A", "is_case_B": "Is case B",
    "is_case_C": "Is case C", "is_case_D": "Is case D",
    "is_case_None": "Is case None",
    sale_col: "Sale price", rent_col: "Rent price",
    "rent_sale_ratio": "Rent/sale ratio",
    "sale_change_curr": "Sale change (current)", "rent_change_curr": "Rent change (current)",
    "change_interaction": "Change interaction",
    "sale_lag1": "Sale price (lag 1)", "rent_lag1": "Rent price (lag 1)",
    "sale_lag2": "Sale price (lag 2)", "rent_lag2": "Rent price (lag 2)",
    "sale_lag4": "Sale price (lag 4)", "rent_lag4": "Rent price (lag 4)",
    "sale_change_lag1": "Sale change (lag 1)", "rent_change_lag1": "Rent change (lag 1)",
    "sale_change_lag2": "Sale change (lag 2)", "rent_change_lag2": "Rent change (lag 2)",
    "sale_roll3": "Sale rolling mean (3Q)", "rent_roll3": "Rent rolling mean (3Q)",
    "sale_roll_std3": "Sale volatility (3Q)", "rent_roll_std3": "Rent volatility (3Q)",
    "sale_roll6": "Sale rolling mean (6Q)", "rent_roll6": "Rent rolling mean (6Q)",
    "sale_roll_std6": "Sale volatility (6Q)", "rent_roll_std6": "Rent volatility (6Q)",
    "sale_momentum": "Sale momentum", "rent_momentum": "Rent momentum",
    "sale_accel": "Sale acceleration", "rent_accel": "Rent acceleration",
    "sale_vs_prov": "Sale vs province", "rent_vs_prov": "Rent vs province",
    "sale_pctile": "Sale percentile", "rent_pctile": "Rent percentile",
    "sale_yoy": "Sale YoY change", "rent_yoy": "Rent YoY change",
    "stock_residential_sale_all": "Sale stock", "stock_residential_rent_all": "Rent stock",
    "stock_sale_lag1": "Sale stock (lag 1)", "stock_rent_lag1": "Rent stock (lag 1)",
    "stock_ratio": "Stock ratio",
    "stock_sale_change": "Sale stock change", "stock_rent_change": "Rent stock change",
    "month": "Month", "quarter": "Quarter", "year": "Year",
}

print(f"Data loaded: {len(df):,} rows, {len(FEATURES)} features")

# Train leak-free model inline
trainable = df.dropna(subset=["sale_lag1", "rent_lag1", "target_case"]).copy()
trainable = trainable[trainable["target_case"] != "Unknown"]

le = LabelEncoder()
trainable["target_encoded"] = le.fit_transform(trainable["target_case"])

train_mask = trainable["year"] < 2020
X_train = trainable.loc[train_mask, FEATURES].values
X_test = trainable.loc[~train_mask, FEATURES].values
y_train = trainable.loc[train_mask, "target_encoded"].values
y_test = trainable.loc[~train_mask, "target_encoded"].values

print(f"Training model: {len(X_train):,} train, {len(X_test):,} test")
imp = SimpleImputer(strategy="mean")
sc = StandardScaler()
X_train_p = sc.fit_transform(imp.fit_transform(X_train))
X_test_p = sc.transform(imp.transform(X_test))

clf = XGBClassifier(
    objective="multi:softprob", n_estimators=500, max_depth=8,
    learning_rate=0.05, subsample=0.8, colsample_bytree=0.7,
    min_child_weight=3, gamma=0.1,
    eval_metric="mlogloss", random_state=42, n_jobs=-1,
)
clf.fit(X_train_p, y_train)
xgb_pred = clf.predict(X_test_p)
from sklearn.metrics import accuracy_score, f1_score
acc = accuracy_score(y_test, xgb_pred)
f1 = f1_score(y_test, xgb_pred, average="weighted")
print(f"Model: Acc={acc:.4f}, F1={f1:.4f}")

test_data = trainable[~train_mask].copy()
test_data["predicted"] = le.inverse_transform(xgb_pred)
test_data["actual"] = test_data["target_case"]

# Descriptive data (for case distribution plots, uses case at t, not t+1)
clean = df[df["case"].isin(["A", "B", "C", "D", "None"])].copy()

# ════════════════════════════════════════
# FIGURE 1: Case Distribution Over Time
# ════════════════════════════════════════
print("Generating case distribution over time...")
case_time = clean.groupby(["year", "case"]).size().unstack(fill_value=0)
case_order_stack = ["D", "A", "None", "B", "C"]
case_time = case_time[[c for c in case_order_stack if c in case_time.columns]]

fig, ax = plt.subplots(figsize=(14, 7))
bottom = np.zeros(len(case_time))
for case in case_time.columns:
    vals = case_time[case].values
    ax.bar(case_time.index.astype(str), vals, bottom=bottom,
           color=CASE_COLORS[case], label=CASE_LABELS.get(case, case),
           edgecolor="white", linewidth=0.5)
    bottom += vals
ax.set_xlabel("Year", fontsize=14, fontweight="bold")
ax.set_ylabel("Number of Census Sections", fontsize=14, fontweight="bold")
ax.set_title("Gentrification Case Distribution Over Time", fontsize=16, fontweight="bold", pad=15)
ax.legend(title="Case Type", title_fontsize=12, fontsize=11,
          loc="upper left", bbox_to_anchor=(1.02, 1), framealpha=0.95, edgecolor="#cccccc")
ax.tick_params(axis="both", labelsize=12)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{int(x):,}"))
sns.despine(left=True, bottom=True)
plt.tight_layout()
fig.savefig(FIGURES_DIR / "case_distribution_over_time.png", bbox_inches="tight")
plt.close(fig)
print("  Saved case_distribution_over_time.png")

# ════════════════════════════════════════
# FIGURE 2: Confusion Matrix (predicting t+1)
# ════════════════════════════════════════
print("Generating confusion matrix...")
cm = confusion_matrix(y_test, xgb_pred)
fig, ax = plt.subplots(figsize=(10, 8))
cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=0.6)

short_labels = list(le.classes_)
ax.set_xticks(range(len(short_labels)))
ax.set_yticks(range(len(short_labels)))
ax.set_xticklabels(short_labels, fontsize=13, fontweight="bold")
ax.set_yticklabels(short_labels, fontsize=13, fontweight="bold")
ax.set_xlabel("Predicted (t+1)", fontsize=14, fontweight="bold", labelpad=10)
ax.set_ylabel("Actual (t+1)", fontsize=14, fontweight="bold", labelpad=10)
ax.set_title("XGBoost Confusion Matrix: Predicting Next-Quarter Case\n(Normalized)",
             fontsize=16, fontweight="bold", pad=15)

for i in range(len(le.classes_)):
    for j in range(len(le.classes_)):
        val = cm_norm[i, j]
        count = cm[i, j]
        text_color = "white" if val > 0.4 else "black"
        ax.text(j, i, f"{val:.2f}\n({count:,})", ha="center", va="center",
                color=text_color, fontsize=11, fontweight="bold")

cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
cbar.set_label("Proportion", fontsize=12)
plt.tight_layout()
fig.savefig(FIGURES_DIR / "xgboost_confusion_matrix.png", bbox_inches="tight")
plt.close(fig)
print("  Saved xgboost_confusion_matrix.png")

# ════════════════════════════════════════
# FIGURE 3: Feature Importance (top 20)
# ════════════════════════════════════════
print("Generating feature importance...")
importances = clf.feature_importances_
idx = np.argsort(importances)
top_n = 20
idx = idx[-top_n:]

feat_names = [FEATURE_LABELS.get(FEATURES[i], FEATURES[i]) for i in idx]
feat_vals = importances[idx]
colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(feat_names)))

fig, ax = plt.subplots(figsize=(12, 10))
bars = ax.barh(range(len(feat_names)), feat_vals, color=colors, edgecolor="white", linewidth=0.5)
ax.set_yticks(range(len(feat_names)))
ax.set_yticklabels(feat_names, fontsize=12)
ax.set_xlabel("Feature Importance", fontsize=14, fontweight="bold")
ax.set_title("XGBoost Feature Importance (Predicting Next-Quarter Case)\nTop 20 Features",
             fontsize=16, fontweight="bold", pad=15)
ax.tick_params(axis="x", labelsize=12)

for bar, val in zip(bars, feat_vals):
    ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
            f"{val:.1%}", va="center", fontsize=10, color="#333333")
sns.despine(left=True, bottom=True)
plt.tight_layout()
fig.savefig(FIGURES_DIR / "xgboost_feature_importance.png", bbox_inches="tight")
plt.close(fig)
print("  Saved xgboost_feature_importance.png")

# ════════════════════════════════════════
# FIGURE 4: Per-Class Performance
# ════════════════════════════════════════
print("Generating per-class performance...")
report = classification_report(y_test, xgb_pred, target_names=le.classes_,
                                output_dict=True, zero_division=0)
classes = list(le.classes_)
f1s = [report[c]["f1-score"] for c in classes]
precisions = [report[c]["precision"] for c in classes]
recalls = [report[c]["recall"] for c in classes]

x = np.arange(len(classes))
width = 0.25
fig, ax = plt.subplots(figsize=(12, 7))
b1 = ax.bar(x - width, precisions, width, label="Precision",
            color="#264653", edgecolor="white", linewidth=0.5)
b2 = ax.bar(x, recalls, width, label="Recall",
            color="#E76F51", edgecolor="white", linewidth=0.5)
b3 = ax.bar(x + width, f1s, width, label="F1 Score",
            color="#2A9D8F", edgecolor="white", linewidth=0.5)

ax.set_xticks(x)
labels = [CASE_LABELS.get(c, c) for c in classes]
ax.set_xticklabels(labels, fontsize=11, rotation=15, ha="right")
ax.set_ylabel("Score", fontsize=14, fontweight="bold")
ax.set_title("XGBoost Per-Class Performance (Predicting Next-Quarter Case)",
             fontsize=16, fontweight="bold", pad=15)
ax.legend(fontsize=12, framealpha=0.95, edgecolor="#cccccc")
ax.set_ylim(0, 0.65)
ax.tick_params(axis="y", labelsize=12)
for bars_group in [b1, b2, b3]:
    for bar in bars_group:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                f"{h:.2f}", ha="center", va="bottom", fontsize=9, color="#333333")
sns.despine(left=True, bottom=True)
plt.tight_layout()
fig.savefig(FIGURES_DIR / "xgboost_per_class_performance.png", bbox_inches="tight")
plt.close(fig)
print("  Saved xgboost_per_class_performance.png")

# ════════════════════════════════════════
# FIGURES 5+: Province-level choropleth maps
# ════════════════════════════════════════
print("Generating Spain choropleth maps...")
ne_url = "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_1_states_provinces.zip"
try:
    spain_geo = gpd.read_file(ne_url)
    spain_geo = spain_geo[spain_geo["admin"] == "Spain"].copy()
except Exception as e:
    print(f"  Could not load Natural Earth data: {e}")
    spain_geo = None

if spain_geo is not None and "NPRO" in df.columns:
    name_map = {
        "Álava": "Araba/Álava", "Alicante": "Alicante/Alacant",
        "Castellón": "Castellón/Castelló", "Valencia": "Valencia/Valéncia",
        "La Coruña": "Coruña, A", "Gerona": "Girona", "Lérida": "Lleida",
        "Orense": "Ourense", "Vizcaya": "Bizkaia", "Guipúzcoa": "Gipuzkoa",
        "Baleares": "Balears, Illes", "Las Palmas": "Palmas, Las",
        "Santa Cruz de Tenerife": "Santa Cruz de Tenerife",
    }
    spain_geo["NPRO_match"] = spain_geo["name"].map(name_map).fillna(spain_geo["name"])

    # Case C intensity map
    prov_dom = clean.groupby(["NPRO", "case"]).size().unstack(fill_value=0)
    prov_dom["total"] = prov_dom[["A", "B", "C", "D", "None"]].sum(axis=1)
    for c in ["A", "B", "C", "D"]:
        if c in prov_dom.columns:
            prov_dom[f"pct_{c}"] = prov_dom[c] / prov_dom["total"]
    spain_merged = spain_geo.merge(prov_dom, left_on="NPRO_match", right_index=True, how="left")
    mainland = spain_merged[~spain_merged["name"].isin(MAINLAND_EXCLUDE)].copy()

    fig, ax = plt.subplots(figsize=(14, 10))
    mainland.plot(column="pct_C", ax=ax, cmap="YlOrRd", edgecolor="#333333",
                  linewidth=0.5, legend=True, missing_kwds={"color": "#f0f0f0", "edgecolor": "#999"},
                  legend_kwds={"label": "Proportion of Case C (Active Gentrification)",
                               "orientation": "horizontal", "pad": 0.02, "shrink": 0.6})
    ax.set_title("Active Gentrification Intensity by Province\n(Proportion of Case C, 2012--2022)",
                 fontsize=16, fontweight="bold", pad=15)
    ax.set_axis_off()
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "spain_case_c_intensity_map.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved spain_case_c_intensity_map.png")

    # Temporal shift map
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    for ax_i, (period_name, year_range) in enumerate([
        ("2012--2015", range(2012, 2016)), ("2018--2021", range(2018, 2022))
    ]):
        ax = axes[ax_i]
        period_data = clean[clean["year"].isin(year_range)]
        period_dom = period_data.groupby(["NPRO", "case"]).size().unstack(fill_value=0)
        period_dom["dominant"] = period_dom.idxmax(axis=1)
        period_merged = spain_geo.merge(period_dom[["dominant"]], left_on="NPRO_match",
                                         right_index=True, how="left")
        period_mainland = period_merged[~period_merged["name"].isin(MAINLAND_EXCLUDE)]
        for case_val, color in CASE_COLORS.items():
            subset = period_mainland[period_mainland["dominant"] == case_val]
            if len(subset) > 0:
                subset.plot(ax=ax, color=color, edgecolor="#333333", linewidth=0.5)
        no_data = period_mainland[period_mainland["dominant"].isna()]
        if len(no_data) > 0:
            no_data.plot(ax=ax, color="#f0f0f0", edgecolor="#999999", linewidth=0.5)
        ax.set_title(period_name, fontsize=18, fontweight="bold", pad=10)
        ax.set_axis_off()
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=CASE_COLORS[c], edgecolor="#333")
               for c in CASE_ORDER]
    labels = [CASE_LABELS[c] for c in CASE_ORDER]
    fig.legend(handles, labels, title="Dominant Case", title_fontsize=13,
               fontsize=12, loc="lower center", ncol=5, framealpha=0.95,
               edgecolor="#cccccc", bbox_to_anchor=(0.5, 0.02))
    fig.suptitle("Temporal Shift in Dominant Gentrification Type", fontsize=20,
                 fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    fig.savefig(FIGURES_DIR / "spain_temporal_shift_map.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved spain_temporal_shift_map.png")

    # Agreement map (uses test_data with predictions)
    if "NPRO" in test_data.columns:
        print("Generating prediction-agreement figures...")

        # Agreement map
        actual_dom = (test_data.groupby(["NPRO", "actual"]).size()
                      .unstack(fill_value=0).idxmax(axis=1).rename("dom_actual"))
        pred_dom = (test_data.groupby(["NPRO", "predicted"]).size()
                    .unstack(fill_value=0).idxmax(axis=1).rename("dom_pred"))

        merged = spain_geo.merge(actual_dom, left_on="NPRO_match", right_index=True, how="left")
        merged = merged.merge(pred_dom, left_on="NPRO_match", right_index=True, how="left")
        mainland_a = _mainland(merged)

        has_data = mainland_a["dom_actual"].notna() & mainland_a["dom_pred"].notna()
        match = has_data & (mainland_a["dom_actual"] == mainland_a["dom_pred"])
        n_total, n_match = int(has_data.sum()), int(match.sum())

        old_hatch_lw = matplotlib.rcParams["hatch.linewidth"]
        matplotlib.rcParams["hatch.linewidth"] = 1.2
        fig, ax = plt.subplots(figsize=(14, 10))
        for case_val, color in CASE_COLORS.items():
            subset = mainland_a[mainland_a["dom_actual"] == case_val]
            if len(subset) > 0:
                subset.plot(ax=ax, color=color, edgecolor="#333333", linewidth=0.5)
        no_data = mainland_a[~has_data]
        if len(no_data) > 0:
            no_data.plot(ax=ax, color="#f0f0f0", edgecolor="#999999", linewidth=0.5)
        mismatch = mainland_a[has_data & ~match]
        for _, row in mismatch.iterrows():
            fill = CASE_COLORS.get(row["dom_actual"], "#f0f0f0")
            hatch_col = "#FFFFFF" if _relative_luminance(fill) < 0.45 else "#1a1a1a"
            gpd.GeoSeries([row.geometry], crs=mainland_a.crs).plot(
                ax=ax, facecolor="none", edgecolor=hatch_col, hatch="///", linewidth=0)
        if len(mismatch) > 0:
            mismatch.boundary.plot(ax=ax, color="#1a1a1a", linewidth=1.6)

        handles = [plt.Rectangle((0, 0), 1, 1, facecolor=CASE_COLORS[c], edgecolor="#333333")
                   for c in CASE_ORDER]
        labels = [CASE_LABELS[c] for c in CASE_ORDER]
        handles.append(plt.Rectangle((0, 0), 1, 1, facecolor="white",
                                     edgecolor="#1a1a1a", hatch="///", linewidth=1.2))
        labels.append("Predicted dominant type differs")
        ax.legend(handles, labels, title="Actual Dominant Case (Test Set)",
                  title_fontsize=12, fontsize=11, loc="lower left",
                  framealpha=0.95, edgecolor="#cccccc")
        pct = 100.0 * n_match / max(n_total, 1)
        ax.text(0.99, 0.99,
                f"Dominant type correctly predicted\nin {n_match} of {n_total} provinces ({pct:.0f}%)",
                transform=ax.transAxes, ha="right", va="top", fontsize=13,
                fontweight="bold", color="#1a1a1a",
                bbox=dict(facecolor="white", edgecolor="#cccccc",
                          boxstyle="round,pad=0.5", alpha=0.95))
        ax.set_title("Predicted vs. Actual Dominant Gentrification Type by Province\n"
                     "(XGBoost, Predicting t+1, Test Set 2020--2022)",
                     fontsize=16, fontweight="bold", pad=15)
        ax.set_axis_off()
        plt.tight_layout()
        fig.savefig(FIGURES_DIR / "spain_agreement_map.png", bbox_inches="tight")
        plt.close(fig)
        matplotlib.rcParams["hatch.linewidth"] = old_hatch_lw
        print(f"  Saved spain_agreement_map.png ({n_match}/{n_total} provinces agree)")

        # Calibration scatter
        rows = []
        for prov, g in test_data.groupby("NPRO"):
            n = len(g)
            for c in CASE_ORDER:
                rows.append({
                    "province": prov, "case": c, "n": n,
                    "actual": (g["actual"] == c).mean(),
                    "predicted": (g["predicted"] == c).mean(),
                })
        cal = pd.DataFrame(rows)
        mae_pp = (cal["predicted"] - cal["actual"]).abs().mean() * 100
        r = np.corrcoef(cal["actual"], cal["predicted"])[0, 1]
        n_prov = cal["province"].nunique()
        lim = float(max(cal["actual"].max(), cal["predicted"].max()) * 1.08)

        fig, ax = plt.subplots(figsize=(10, 9))
        ax.plot([0, lim], [0, lim], linestyle="--", color="#999999",
                linewidth=1.2, zorder=1)
        for c in CASE_ORDER:
            sub = cal[cal["case"] == c]
            ax.scatter(sub["actual"], sub["predicted"],
                       s=np.sqrt(sub["n"]) * 2.0 + 20,
                       color=CASE_COLORS[c], edgecolor="white", linewidth=1.2,
                       alpha=0.85, zorder=3, label=CASE_LABELS[c])
        ax.text(0.03, 0.97,
                f"MAE = {mae_pp:.1f} pp\nPearson r = {r:.3f}\n"
                f"{n_prov} provinces x 5 classes",
                transform=ax.transAxes, ha="left", va="top", fontsize=12,
                bbox=dict(facecolor="white", edgecolor="#cccccc",
                          boxstyle="round,pad=0.5", alpha=0.95))
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_aspect("equal")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
        ax.set_xlabel("Actual Class Share per Province", fontsize=14, fontweight="bold")
        ax.set_ylabel("Predicted Class Share per Province", fontsize=14, fontweight="bold")
        ax.set_title("Province-Level Calibration of Predicted Class Shares\n"
                     "(XGBoost, Predicting t+1, Test Set 2020--2022)",
                     fontsize=16, fontweight="bold", pad=15)
        ax.legend(title="Case Type", title_fontsize=12, fontsize=10,
                  loc="lower right", framealpha=0.95, edgecolor="#cccccc", markerscale=0.7)
        ax.tick_params(axis="both", labelsize=12)
        sns.despine()
        plt.tight_layout()
        fig.savefig(FIGURES_DIR / "province_calibration_scatter.png", bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved province_calibration_scatter.png (MAE {mae_pp:.1f} pp, r {r:.3f})")

print("\nAll figures regenerated.")
